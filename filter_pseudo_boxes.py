#!/usr/bin/env python3
"""Filter existing unreviewed pseudo boxes; dry-run by default, --apply to commit."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import secrets

from box_filter import filter_duplicate_boxes
from pseudo_index import index_contents
from prepare_raw_dy_from_instseg import yolo_line
from review_gui import ReviewStore, atomic_bytes, inside, json_bytes


def label_bytes(boxes, width, height):
    lines = [yolo_line(box,width,height) for box in boxes]
    return ('\n'.join(lines)+ ('\n' if lines else '')).encode()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def filter_existing(store, threshold=0.9, dates=('260916',), apply=False):
    filter_duplicate_boxes([],threshold)  # Validate even if no frames qualify.
    with store.lock, store.generation_lock():
        store.recover()
        counts, changes, updates, records, protected = Counter(), [], {}, [], {}
        timestamp = datetime.now(timezone.utc).isoformat()
        for frame in store.frames.values():
            paths = frame['paths']
            annotation = inside(store.root,paths['annotation_file'])
            old_record = annotation.read_bytes()
            record = json.loads(old_record)
            records.append(record)
            counts['frames_scanned'] += 1
            if record['date'] not in dates:
                counts['other_date'] += 1
                continue
            # Never replace annotations saved/approved/rejected by a human.
            if record.get('review_status','pending') != 'pending' or record.get('review'):
                counts['reviewed_preserved'] += 1
                if apply:
                    for key in ('annotation_file','detection_file','road_file'):
                        rel = paths[key]
                        protected[rel] = digest(inside(store.root,rel).read_bytes())
                continue
            counts['unreviewed'] += 1
            boxes = record.get('filtered_detections',record['detections'])
            old_label = inside(store.root,paths['detection_file']).read_bytes()
            if old_label != label_bytes(boxes,record['width'],record['height']):
                counts['changed_labels_preserved'] += 1
                continue
            filtered, removed = filter_duplicate_boxes(boxes,threshold)
            if not removed:
                counts['already_clean'] += 1
                continue
            change = {'image_file':record['image_file'],'before':len(boxes),'after':len(filtered),'removed':removed}
            changes.append(change)
            counts['frames_changed'] += 1
            counts['boxes_before'] += len(boxes)
            counts['boxes_after'] += len(filtered)
            counts['boxes_removed'] += len(removed)
            # Keep original teacher detections, previews and confidence maps intact.
            record['filtered_detections'] = filtered
            record['detection_count'] = len(filtered)
            record['box_dedup'] = {'method':'same_class_nms','iou_threshold':threshold,
                                   'applied_utc':timestamp, **{k:v for k,v in change.items() if k!='image_file'}}
            reasons = [r for r in record.get('review_reasons',[]) if r not in ('no_detections','low_confidence_detection')]
            if not filtered: reasons.append('no_detections')
            if any(b['confidence'] < .4 for b in filtered): reasons.append('low_confidence_detection')
            record['review_reasons'] = reasons
            updates[paths['detection_file']] = (old_label,label_bytes(filtered,record['width'],record['height']))
            updates[paths['annotation_file']] = (old_record,json_bytes(record))
            if apply:
                protected[paths['road_file']] = digest(inside(store.root,paths['road_file']).read_bytes())
        report = {'applied':False,'method':'same_class_nms','iou_threshold':threshold,
                  'created_utc':timestamp,'counts':dict(counts),'changes':changes}
        if not apply or not changes:
            return report
        indices, _ = index_contents(records)
        for name, new in indices.items():
            path = store.root/name
            if path.is_file():
                updates[name] = (path.read_bytes(),new)
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')+'-'+secrets.token_hex(3)
        backup_root = inside(store.root,f'_box_filter_history/{stamp}')
        backup_root.mkdir(parents=True,exist_ok=False)
        report['backup_directory'] = str(backup_root)
        report['protected_sha256'] = protected
        report['files'] = {rel:{'before_sha256':digest(old),'after_sha256':digest(new)} for rel,(old,new) in updates.items()}
        report_path = backup_root/'report.json'
        atomic_bytes(report_path,json_bytes(report))
        transaction = {'files':{}}
        for rel,(old,new) in updates.items():
            target = inside(store.root,rel)
            if target.read_bytes() != old:
                raise RuntimeError(f'File changed during planning; no labels updated: {rel}')
            backup = backup_root/'files'/rel
            backup.parent.mkdir(parents=True,exist_ok=True)
            atomic_bytes(backup,old)
            transaction['files'][rel] = backup.relative_to(store.root).as_posix()
        atomic_bytes(store.journal,json_bytes(transaction))
        try:
            for rel,(old,new) in updates.items():
                atomic_bytes(inside(store.root,rel),new)
            for rel,expected in protected.items():
                if digest(inside(store.root,rel).read_bytes()) != expected:
                    raise RuntimeError(f'Protected labels changed during filtering: {rel}')
            store.journal.unlink()
        except Exception:
            store.recover()
            report['rolled_back'] = True
            atomic_bytes(report_path,json_bytes(report))
            raise
        store.refresh()
        report.update(applied=True,protected_files_unchanged=True,protected_file_count=len(protected))
        atomic_bytes(report_path,json_bytes(report))
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root',type=Path,default=Path('datasets'))
    parser.add_argument('--pseudo-root',type=Path)
    parser.add_argument('--dates',nargs='+',default=['260916'])
    parser.add_argument('--iou',type=float,default=.9)
    parser.add_argument('--apply',action='store_true',help='Back up and update untouched pending pseudo labels')
    args = parser.parse_args()
    store = ReviewStore(args.dataset_root,args.pseudo_root or args.dataset_root/'pseudo_labels')
    report = filter_existing(store,args.iou,args.dates,args.apply)
    print(json.dumps({k:v for k,v in report.items() if k not in ('changes','files','protected_sha256')},indent=2,ensure_ascii=False))


if __name__=='__main__':
    main()
