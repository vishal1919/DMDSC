"""
Utilities for creating imbalanced datasets dynamically.
Supports IR-based imbalance (mild, moderate, severe) for ASC dataset.
"""

import numpy as np
from collections import Counter
from typing import Dict, Optional


def create_imbalanced_indices(
    labels: np.ndarray, 
    imbalance_ratio: int = 10, 
    random_state: int = 42
) -> np.ndarray:
    """
    Create indices for imbalanced dataset using IR-based approach.
    
    Parameters:
    -----------
    labels : array-like, shape (n_samples,)
        Original labels
    imbalance_ratio : int
        IR = max_samples / min_samples
        IR=2 (very mild), IR=5 (mild), IR=10 (mild), IR=50 (moderate), IR=100 (severe)
    random_state : int
        Random seed for reproducibility
    
    Returns:
    --------
    imbalanced_indices : np.ndarray
        Indices to sample from original dataset
    """
    np.random.seed(random_state)
    
    labels = np.array(labels).flatten()
    unique_classes = np.unique(labels)
    n_classes = len(unique_classes)
    
    # Get original class counts
    class_counts = Counter(labels)
    max_samples = max(class_counts.values())
    
    # Calculate target samples per class based on IR
    target_samples = get_ir_based_samples(
        n_classes, max_samples, imbalance_ratio
    )
    
    # Sample indices for each class
    imbalanced_indices = []
    
    for class_label in sorted(unique_classes):
        class_indices = np.where(labels == class_label)[0]
        n_target = target_samples[class_label]
        
        # Ensure we don't sample more than available
        n_target = min(n_target, len(class_indices))
        
        if n_target > 0:
            # Randomly sample indices
            sampled_indices = np.random.choice(
                class_indices, 
                size=n_target, 
                replace=False
            )
            imbalanced_indices.extend(sampled_indices)
    
    # Shuffle to mix classes
    imbalanced_indices = np.array(imbalanced_indices)
    np.random.shuffle(imbalanced_indices)
    
    return imbalanced_indices


def get_ir_based_samples(
    n_classes: int, 
    max_samples: int, 
    imbalance_ratio: int
) -> Dict[int, int]:
    """
    Calculate target samples per class for IR-based imbalance.
    
    Strategy for ASC (6 classes):
    - IR=2: 3 majority (100%), 2 medium (80%), 1 minority (50%)
    - IR=5: 2 majority (100%), 2 medium (70%), 2 minority (20%)
    - IR=10: 2 majority (100%), 2 medium (50%), 2 minority (10%)
    - IR=50: 2 majority (100%), 2 medium (20%), 2 minority (2%)
    - IR=100: 1 majority (100%), 3 medium (10%), 2 minority (1%)
    
    Parameters:
    -----------
    n_classes : int
        Number of classes (should be 6 for ASC)
    max_samples : int
        Maximum samples available per class
    imbalance_ratio : int
        Target IR (2, 5, 10, 50, or 100)
    
    Returns:
    --------
    target_samples : Dict[int, int]
        Mapping of class_id -> target_sample_count
    """
    target_samples = {}
    
    if imbalance_ratio == 2:  # Very mild imbalance
        # Strategy: 3 majority (100%), 2 medium (80%), 1 minority (50%)
        if n_classes == 6:
            samples = [
                max_samples,              # Class 0: 100%
                int(max_samples * 0.90),              # Class 1: 100%
                int(max_samples * 0.80),           # Class 2: 100%
                int(max_samples * 0.70),   # Class 3: 80%
                int(max_samples * 0.60),   # Class 4: 80%
                int(max_samples * 0.50)    # Class 5: 50%
            ]
        else:
            samples = _get_generic_samples(n_classes, max_samples, 2)
    
    elif imbalance_ratio == 5:  # Mild imbalance
        # Strategy: 2 majority (100%), 2 medium (70%), 2 minority (20%)
        if n_classes == 6:
            samples = [
                max_samples,              # Class 0: 100%
                int(max_samples * 0.84),              # Class 1: 100%
                int(max_samples * 0.68),   # Class 2: 70%
                int(max_samples * 0.52),   # Class 3: 70%
                int(max_samples * 0.36),   # Class 4: 20%
                int(max_samples * 0.2)    # Class 5: 20%
            ]
        else:
            samples = _get_generic_samples(n_classes, max_samples, 5)
    
    elif imbalance_ratio == 10:  # Mild imbalance
        # Strategy: 2 majority (100%), 2 medium (50%), 2 minority (10%)
        if n_classes == 6:
            samples = [
                max_samples,              # Class 0: 100%
                int(max_samples * 0.82),             # Class 1: 100%
                int(max_samples * 0.64),   # Class 2: 50%
                int(max_samples * 0.46),   # Class 3: 50%
                int(max_samples * 0.28),   # Class 4: 10%
                int(max_samples * 0.1)    # Class 5: 10%
            ]
        else:
            # Generic strategy for other class counts
            samples = _get_generic_samples(n_classes, max_samples, 10)
    
    elif imbalance_ratio == 50:  # Moderate imbalance
        # Strategy: 2 majority (100%), 2 medium (20%), 2 minority (2%)
        if n_classes == 6:
            samples = [
                max_samples,              # Class 0: 100%
                int(max_samples * 0.805),             # Class 1: 100%
                int(max_samples * 0.61),   # Class 2: 20%
                int(max_samples * 0.415),   # Class 3: 20%
                int(max_samples * 0.22),  # Class 4: 2%
                int(max_samples * 0.02)   # Class 5: 2%
            ]
        else:
            samples = _get_generic_samples(n_classes, max_samples, 50)
    
    elif imbalance_ratio == 100:  # Severe imbalance
        # Strategy: 1 majority (100%), 3 medium (10%), 2 minority (1%)
        if n_classes == 6:
            samples = [
                max_samples,              # Class 0: 100%
                int(max_samples * 0.1),   # Class 1: 10%
                int(max_samples * 0.1),   # Class 2: 10%
                int(max_samples * 0.1),   # Class 3: 10%
                int(max_samples * 0.01),  # Class 4: 1%
                int(max_samples * 0.01)   # Class 5: 1%
            ]
        else:
            samples = _get_generic_samples(n_classes, max_samples, 100)
    
    else:
        raise ValueError(
            f"Unsupported imbalance ratio: {imbalance_ratio}. "
            f"Use 2 (very mild), 5 (mild), 10 (mild), 50 (moderate), or 100 (severe)."
        )
    
    # Ensure at least 1 sample per class
    samples = [max(1, s) for s in samples]
    
    # Create dictionary mapping
    for i in range(n_classes):
        target_samples[i] = samples[i]
    
    return target_samples


def _get_generic_samples(n_classes: int, max_samples: int, ir: int) -> list:
    """Generate generic imbalance pattern for arbitrary number of classes."""
    min_samples = max(1, int(max_samples / ir))
    
    # Exponential decay from max to min
    samples = []
    for i in range(n_classes):
        ratio = (n_classes - i - 1) / (n_classes - 1) if n_classes > 1 else 1
        s = int(min_samples + (max_samples - min_samples) * ratio)
        samples.append(s)
    
    return samples


def get_imbalance_info(
    labels: np.ndarray, 
    indices: Optional[np.ndarray] = None
) -> Dict:
    """
    Get information about dataset imbalance.
    
    Parameters:
    -----------
    labels : np.ndarray
        All labels
    indices : Optional[np.ndarray]
        Indices to subset (if None, use all labels)
    
    Returns:
    --------
    info : Dict
        Dictionary with imbalance statistics
    """
    if indices is not None:
        labels = labels[indices]
    
    labels = np.array(labels).flatten()
    class_counts = Counter(labels)
    
    max_count = max(class_counts.values())
    min_count = min(class_counts.values())
    ir = max_count / min_count if min_count > 0 else float('inf')
    
    return {
        'class_counts': dict(class_counts),
        'total_samples': len(labels),
        'max_samples': max_count,
        'min_samples': min_count,
        'imbalance_ratio': ir
    }


def print_imbalance_summary(
    labels: np.ndarray, 
    indices: Optional[np.ndarray] = None, 
    dataset_name: str = "Dataset"
):
    """
    Print formatted imbalance summary.
    
    Parameters:
    -----------
    labels : np.ndarray
        All labels
    indices : Optional[np.ndarray]
        Indices to subset
    dataset_name : str
        Name for display
    """
    info = get_imbalance_info(labels, indices)
    
    print(f"\n{'='*60}")
    print(f"{dataset_name} Imbalance Summary")
    print(f"{'='*60}")
    print(f"Total samples: {info['total_samples']}")
    print(f"Imbalance Ratio (IR): {info['imbalance_ratio']:.2f}")
    print(f"\nClass distribution:")
    
    for class_id in sorted(info['class_counts'].keys()):
        count = info['class_counts'][class_id]
        percentage = (count / info['total_samples']) * 100
        print(f"  Class {class_id}: {count:4d} samples ({percentage:5.2f}%)")
    
    print(f"{'='*60}\n")


def compute_class_weights(labels: np.ndarray) -> np.ndarray:
    """
    Compute inverse frequency class weights for imbalanced data.
    
    Parameters:
    -----------
    labels : np.ndarray
        Class labels
    
    Returns:
    --------
    weights : np.ndarray
        Weight for each class (shape: num_classes)
    """
    labels = np.array(labels).flatten()
    class_counts = Counter(labels)
    n_samples = len(labels)
    n_classes = len(class_counts)
    
    weights = np.zeros(n_classes)
    for class_id, count in class_counts.items():
        weights[class_id] = n_samples / (n_classes * count)
    
    return weights
