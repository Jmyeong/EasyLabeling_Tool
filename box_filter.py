"""Deterministic same-class NMS for near-identical pseudo detection boxes."""
import math


def box_iou(a, b):
    intersection = max(0., min(a[2], b[2])-max(a[0], b[0])) * max(0., min(a[3], b[3])-max(a[1], b[1]))
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - intersection
    return intersection / union if union > 0 else 0.


def filter_duplicate_boxes(boxes, threshold=0.9):
    """Keep highest confidence per class when IoU >= threshold; ties keep first.

    Output preserves the input order among survivors. Removed records refer to
    input indices, so every deletion can be traced to a retained box.
    """
    if not isinstance(threshold, (int, float)) or not math.isfinite(threshold) or not 0 < threshold <= 1:
        raise ValueError('IoU threshold must be in (0, 1]')
    for box in boxes:
        xy = box.get('bbox_xyxy')
        score = box.get('confidence')
        cls = box.get('class_id')
        if type(cls) is not int or cls < 0:
            raise ValueError('class_id must be a nonnegative integer')
        if not isinstance(xy, (list, tuple)) or len(xy) != 4 or not all(type(v) in (int,float) and math.isfinite(v) for v in xy):
            raise ValueError('Boxes need four finite coordinates')
        if xy[2] <= xy[0] or xy[3] <= xy[1]:
            raise ValueError('Box width and height must be positive')
        if type(score) not in (int,float) or not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError('Boxes need valid confidence scores')
    keep, removed = [], []
    for index in sorted(range(len(boxes)), key=lambda i:(-boxes[i]['confidence'],i)):
        candidate = boxes[index]
        for retained in keep:
            winner = boxes[retained]
            if candidate['class_id'] != winner['class_id']:
                continue
            overlap = box_iou(candidate['bbox_xyxy'], winner['bbox_xyxy'])
            if overlap >= threshold:
                removed.append({'removed_index':index, 'kept_index':retained,
                                'class_id':candidate['class_id'], 'iou':overlap,
                                'removed_confidence':candidate['confidence'], 'kept_confidence':winner['confidence']})
                break
        else:
            keep.append(index)
    return [boxes[i] for i in sorted(keep)], sorted(removed,key=lambda r:r['removed_index'])
