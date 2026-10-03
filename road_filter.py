"""Keep the largest 8-connected road components without changing ignore pixels."""
import hashlib

import cv2
import numpy as np


def mask_digest(mask):
    return hashlib.sha256(np.ascontiguousarray(mask, dtype=np.uint8).tobytes()).hexdigest()


def keep_largest_road_blobs(mask, max_blobs=2):
    if type(max_blobs) is not int or max_blobs < 1:
        raise ValueError('max_blobs must be a positive integer')
    if mask.ndim != 2 or mask.dtype != np.uint8 or not np.isin(mask, [0, 1, 255]).all():
        raise ValueError('Expected a 2D uint8 mask with values 0, 1, 255')
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        (mask == 1).astype(np.uint8), connectivity=8)
    # Stable label ordering breaks area ties; background is never a candidate.
    ranked = sorted(range(1, count), key=lambda i: (-int(stats[i, cv2.CC_STAT_AREA]), i))
    kept = ranked[:max_blobs]
    lookup = np.zeros(count, dtype=bool)
    lookup[kept] = True
    removed = (mask == 1) & ~lookup[labels]
    result = mask.copy()
    result[removed] = 0
    info = {
        'method': 'largest_road_components', 'connectivity': 8, 'max_blobs': max_blobs,
        'components_before': count - 1, 'components_after': len(kept),
        'components_removed': max(0, count - 1 - len(kept)),
        'pixels_removed': int(removed.sum()),
        'kept_areas': [int(stats[i, cv2.CC_STAT_AREA]) for i in kept],
        'before_mask_sha256': mask_digest(mask), 'after_mask_sha256': mask_digest(result),
    }
    return result, info


def matches_teacher_mask(mask, probability_u16, threshold, margin):
    """Compare legacy masks with quantized probabilities, allowing rounding at edges.

    The generator writes float32 predictions as round(p * 65535). At a decision
    boundary, either neighboring label can be valid within this quantization bin.
    """
    if probability_u16.shape != mask.shape or not 0 < threshold < 1:
        return False
    if not 0 <= margin < min(threshold, 1 - threshold):
        return False
    p = probability_u16.astype(np.float64) / 65535
    tolerance = 0.5 / 65535 + np.finfo(np.float32).eps
    low, high = p - tolerance, p + tolerance
    valid = ((mask == 0) & (low < threshold - margin)) | ((mask == 1) & (high >= threshold + margin))
    if margin:
        valid |= (mask == 255) & (high > threshold - margin) & (low < threshold + margin)
    return bool(valid.all())
