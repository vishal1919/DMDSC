import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def l2_distance(features, centers):
    """
    features: [B, D]
    centers : [C, D]
    returns : [B, C] squared Euclidean distance
    """
    features = features.float().contiguous()
    centers = centers.float().contiguous()

    if features.dim() != 2:
        features = features.view(features.size(0), -1)

    if centers.dim() != 2:
        centers = centers.view(centers.size(0), -1)

    if features.size(1) != centers.size(1):
        raise RuntimeError(
            f"Feature-center dim mismatch: "
            f"features={features.shape}, centers={centers.shape}"
        )

    if torch.isnan(features).any() or torch.isinf(features).any():
        raise RuntimeError("NaN/Inf found in features")

    if torch.isnan(centers).any() or torch.isinf(centers).any():
        raise RuntimeError("NaN/Inf found in centers")

    distmat = (
        torch.sum(features ** 2, dim=1, keepdim=True)
        + torch.sum(centers ** 2, dim=1).unsqueeze(0)
        - 2.0 * torch.matmul(features, centers.t())
    )

    return torch.clamp(distmat, min=0.0)


class NirvanaOpenset_loss(nn.Module):
    def __init__(
        self,
        num_classes=10,
        feat_dim=128,
        precalc_centers=None,
        m_min=38.0,
        m_max=71.0,
        Expand=100,
        outlier_weight=1.0,
        inter_weight=1.0,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.num_centers = num_classes
        self.feat_dim = int(feat_dim)

        self.m_min = float(m_min)
        self.m_max = float(m_max)

        self.E = Expand
        self.outlier_weight = float(outlier_weight)
        self.inter_weight = float(inter_weight)

        if precalc_centers:
            precalculated_centers = FindCenters(
                self.feat_dim,
                self.E
            )[:self.num_classes, :]

        self.centers = nn.Parameter(
            torch.randn(self.num_classes, self.feat_dim),
            requires_grad=False
        )

        if precalc_centers:
            with torch.no_grad():
                self.centers.copy_(
                    torch.from_numpy(precalculated_centers).float()
                )
            print("Centers loaded.")

        self.register_buffer(
            "class_counts",
            torch.zeros(self.num_classes, dtype=torch.float32)
        )

        self.register_buffer(
            "m_per_class",
            torch.full(
                (self.num_classes,),
                float(self.m_min),
                dtype=torch.float32
            )
        )

        self.register_buffer(
            "_counts_initialized",
            torch.tensor(False)
        )

    def set_class_counts(self, counts):
        counts = counts.to(
            dtype=torch.float32,
            device=self.class_counts.device
        )

        counts = torch.clamp(counts, min=0.0)

        if counts.numel() != self.num_classes:
            raise RuntimeError(
                f"class_counts length mismatch: "
                f"got {counts.numel()}, expected {self.num_classes}"
            )

        self.class_counts.copy_(counts)
        self._counts_initialized.fill_(True)

        self.compute_margins()

        print(f"[NirvanaOpenset_loss] Class counts updated: {counts.tolist()}")
        print(
            f"[NirvanaOpenset_loss] Dynamic margins: "
            f"{self.m_per_class.tolist()}"
        )
        print(
            f"[NirvanaOpenset_loss] Margin range: "
            f"[{self.m_per_class.min():.2f}, "
            f"{self.m_per_class.max():.2f}]"
        )

    def update_counts_online(self, labels):
        binc = torch.bincount(
            labels.long(),
            minlength=self.num_classes
        ).to(self.class_counts)

        self.class_counts.add_(binc)
        self.compute_margins()

    def compute_margins(self):
        total_samples = torch.sum(self.class_counts)

        if total_samples <= 0:
            self.m_per_class.fill_(self.m_min)
            return

        class_fraction = self.class_counts / total_samples

        self.m_per_class.copy_(
            self.m_min
            + (self.m_max - self.m_min) * (1.0 - class_fraction)
        )

    def forward(
        self,
        labels,
        x,
        x_out=None,
        ramp=False,
        update_counts=False
    ):
        if labels.max() >= self.num_classes or labels.min() < 0:
            raise ValueError(
                "Labels out of valid range for NirvanaOpenset_loss."
            )

        if (
            not self._counts_initialized.item()
            and self.class_counts.sum() == 0
        ):
            import warnings

            warnings.warn(
                "Dynamic margins are not initialized. "
                "Call criterion.set_class_counts(counts) before training.",
                UserWarning
            )

            self._counts_initialized.fill_(True)

        if update_counts:
            self.update_counts_online(labels.detach())

        dtype = x.dtype
        device = x.device

        labels = labels.long().to(device)
        centers = self.centers.to(dtype=dtype, device=device)

        batch_size = x.size(0)

        dist_in = l2_distance(x, centers).to(
            dtype=dtype,
            device=device
        )

        intra = dist_in.gather(
            dim=1,
            index=labels.unsqueeze(1)
        ).squeeze(1)

        intraclass_loss = intra.sum() / (
            batch_size * self.feat_dim * 2.0
        )

        m_batch = self.m_per_class[labels].to(
            dtype=dtype,
            device=device
        )

        m_batch_sq = m_batch.pow(2)

        centers_dist_inter = intra.unsqueeze(1) - dist_in

        mask = ~F.one_hot(
            labels,
            num_classes=self.num_classes
        ).bool()

        interclass_hinge = (
            m_batch_sq.unsqueeze(1) + centers_dist_inter
        ).clamp(min=0.0)

        interclass_loss_triplet = (
            (interclass_hinge * mask).sum()
            / (self.num_centers * batch_size * 2.0)
        )

        interclass_loss_triplet = (
            self.inter_weight * interclass_loss_triplet
        )


        if x_out is not None:
            x_out = x_out.to(dtype=dtype, device=device)
            batch_size_out = x_out.size(0)

            dist_out = l2_distance(x_out, centers).to(
                dtype=dtype,
                device=device
            )

            outlier_cols = dist_out.index_select(1, labels)

            m_pair_sq = m_batch_sq.unsqueeze(0).expand(
                batch_size_out,
                batch_size
            )

            hinge = m_pair_sq + (
                intra.unsqueeze(0) - outlier_cols
            )

            hinge = hinge.clamp(min=0.0)

            if ramp:
                hinge = hinge.clamp(max=self.m_max ** 2)

            outlier_triplet_multi_loss = hinge.sum() / (
                batch_size * batch_size_out * 2.0
            )

            weighted_outlier_loss = (
                self.outlier_weight * outlier_triplet_multi_loss
            )

            return (
                intraclass_loss,
                interclass_loss_triplet,
                weighted_outlier_loss
            )

        return intraclass_loss, interclass_loss_triplet, None

def FindCenters(k, E=1):
    Centers = np.empty((k + 1, k), dtype=np.float32)
    CC = np.empty((k, k), dtype=np.float32)

    Unit_Vector = np.identity(k)

    c = -(
        (1 + np.sqrt(k + 1))
        / np.power(k, 3 / 2)
    )

    CC.fill(c)

    d = np.sqrt((k + 1) / k)
    DU = d * Unit_Vector

    Centers[0, :].fill(1 / np.sqrt(k))
    Centers[1:, :] = CC + DU

    Distances = np.empty((k + 1, k), dtype=np.float32)

    for idx, rows in enumerate(Centers):
        Distances[idx, :] = np.linalg.norm(
            rows - np.delete(Centers, idx, axis=0),
            axis=1
        )

    assert np.allclose(
        np.random.choice(Distances.flatten(), size=1),
        Distances,
        rtol=1e-05,
        atol=1e-08,
        equal_nan=False
    ), "Distances are not equal"

    return Centers * E


def get_l2_pred(features, centers, return_logits=False):
    with torch.no_grad():
        _, feat_dim = centers.shape
        serialized_centers = centers.view(-1, feat_dim)

        distmat = l2_distance(
            features,
            serialized_centers
        )

        pred = distmat.argmin(dim=1)

        if return_logits:
            logits = 1.0 / (1.0 + distmat)
            return pred, logits

        return pred


def get_l2_pred_b9(features, centers, return_logits=False):
    with torch.no_grad():
        _, feat_dim = centers.shape
        serialized_centers = centers.view(-1, feat_dim)

        distmat = l2_distance(
            features,
            serialized_centers
        )

        pred = distmat.argmin(dim=1)

        if return_logits:
            logits = 1.0 / (1.0 + distmat)

            distmat_norm = F.normalize(
                distmat,
                p=2,
                dim=1
            )

            logits_b9 = 1.0 / (1.0 + distmat_norm)

            return pred, logits, logits_b9

        return pred


def accuracy_l2(features, centers, targets):
    with torch.no_grad():
        targets = targets.long()

        _, feat_dim = centers.shape
        serialized_centers = centers.view(-1, feat_dim)

        distmat = l2_distance(
            features,
            serialized_centers
        )

        pred = distmat.argmin(dim=1)
        correct = pred.eq(targets)

        correct_k = correct.flatten().sum(dtype=torch.float32)
        batch_size = targets.size(0)

        return correct_k * (100.0 / batch_size)


def accuracy_l2_nosubcenter(features, centers, targets):
    with torch.no_grad():
        targets = targets.long()

        _, feat_dim = centers.shape
        serialized_centers = centers.view(-1, feat_dim)

        distmat = torch.cdist(
            features.float().contiguous(),
            serialized_centers.float().contiguous(),
            p=2
        )

        pred = distmat.argmin(dim=1)
        correct = pred.eq(targets)

        correct_k = correct.flatten().sum(dtype=torch.float32)
        batch_size = targets.size(0)

        return correct_k * (100.0 / batch_size)


def get_l2_pred_nosubcenter(features, centers, target=None):
    with torch.no_grad():
        _, feat_dim = centers.shape
        serialized_centers = centers.view(-1, feat_dim)

        distmat = torch.cdist(
            features.float().contiguous(),
            serialized_centers.float().contiguous(),
            p=2
        )

        pred = distmat.argmin(dim=1)

        return pred


def cosine_similarity(features, centers, target=None):
    with torch.no_grad():
        _, feat_dim = centers.shape
        serialized_centers = centers.view(-1, feat_dim)

        features = features.float().contiguous()
        serialized_centers = serialized_centers.float().contiguous()

        sim = F.cosine_similarity(
            features.unsqueeze(1),
            serialized_centers.unsqueeze(0),
            dim=2
        )

        pred = sim.argmax(dim=1)

        return pred


def euc_cos(features, centers, target=None):
    with torch.no_grad():
        _, feat_dim = centers.shape
        serialized_centers = centers.view(-1, feat_dim)

        features = features.float().contiguous()
        serialized_centers = serialized_centers.float().contiguous()

        disteuc = torch.cdist(
            features,
            serialized_centers,
            p=2
        )

        distcos = F.cosine_similarity(
            features.unsqueeze(1),
            serialized_centers.unsqueeze(0),
            dim=2
        )

        score = (1.0 / (2.0 + distcos)) * disteuc
        pred = score.argmin(dim=1)

        return pred
