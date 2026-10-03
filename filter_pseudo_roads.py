#!/usr/bin/env python3
"""Keep the largest road blobs in untouched pending pseudo labels; dry-run by default."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import secrets
import sys

import cv2
import numpy as np
from PIL import Image

from filter_pseudo_boxes import digest
from pseudo_index import index_contents
from review_gui import ReviewStore, atomic_bytes, inside, json_bytes
from road_filter import keep_largest_road_blobs, mask_digest, matches_teacher_mask


def png_bytes(mask):
    buffer = io.BytesIO()
    Image.fromarray(mask).save(buffer, format='PNG', compress_level=1)
    return buffer.getvalue()


def filter_existing_roads(store, max_blobs=2, dates=('260916',), apply=False, progress=None):
    keep_largest_road_blobs(np.zeros((1, 1), np.uint8), max_blobs)
    with store.lock, store.generation_lock():
        store.recover()
        run_path = store.root / 'run.json'
        run = json.loads(run_path.read_text()) if run_path.is_file() else {}
        counts, changes, updates, records, protected = Counter(), [], {}, [], {}
        timestamp = datetime.now(timezone.utc).isoformat()
        for frame in store.frames.values():
            paths = frame['paths']
            old_record = inside(store.root, paths['annotation_file']).read_bytes()
            record = json.loads(old_record)
            records.append(record)
            counts['frames_scanned'] += 1
            if progress and counts['frames_scanned'] % 500 == 0:
                progress(dict(counts))
            if record['date'] not in dates:
                counts['other_date'] += 1
                continue
            if record.get('review_status', 'pending') != 'pending' or record.get('review'):
                counts['reviewed_preserved'] += 1
                if apply:
                    for key in ('annotation_file', 'detection_file', 'road_file'):
                        protected[paths[key]] = digest(inside(store.root, paths[key]).read_bytes())
                continue
            counts['unreviewed'] += 1
            old_mask = inside(store.root, paths['road_file']).read_bytes()
            with Image.open(io.BytesIO(old_mask)) as image:
                mask = np.array(image)
            if mask.shape != (record['height'], record['width']):
                raise ValueError(f"Unexpected mask dimensions: {paths['road_file']}")
            prior_filter = record.get('road_blob_filter')
            if prior_filter:
                untouched = mask_digest(mask) == prior_filter.get('after_mask_sha256')
            else:
                probability_path = inside(store.root, paths['road_probability_file'])
                if (not probability_path.is_file() or not run.get('generation_signature')
                        or record.get('generation_signature') != run['generation_signature']):
                    counts['unverifiable_masks_preserved'] += 1
                    continue
                with Image.open(probability_path) as image:
                    probability = np.array(image)
                untouched = matches_teacher_mask(mask, probability, run['road_threshold'], run['road_ignore_margin'])
            if not untouched:
                counts['changed_masks_preserved'] += 1
                continue
            filtered, info = keep_largest_road_blobs(mask, max_blobs)
            if not info['components_removed']:
                counts['already_clean'] += 1
                continue
            counts['frames_changed'] += 1
            for key in ('components_before', 'components_after', 'components_removed', 'pixels_removed'):
                counts[key] += info[key]
            change = {'image_file': record['image_file'], **info}
            changes.append(change)
            record.setdefault('original_road_fraction', record.get('road_fraction', float((mask == 1).mean())))
            record['road_fraction'] = float((filtered == 1).mean())
            record['road_blob_filter'] = {**info, 'applied_utc': timestamp}
            updates[paths['road_file']] = (old_mask, png_bytes(filtered))
            updates[paths['annotation_file']] = (old_record, json_bytes(record))
            if apply:
                for key in ('detection_file', 'road_probability_file'):
                    protected[paths[key]] = digest(inside(store.root, paths[key]).read_bytes())
        report = {'applied': False, 'method': 'largest_road_components', 'max_blobs': max_blobs,
                  'connectivity': 8, 'created_utc': timestamp, 'counts': dict(counts), 'changes': changes}
        if not apply or not changes:
            return report
        indices, _ = index_contents(records)
        for name, new in indices.items():
            path = store.root / name
            if path.is_file():
                updates[name] = (path.read_bytes(), new)
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ') + '-' + secrets.token_hex(3)
        backup_root = inside(store.root, f'_road_filter_history/{stamp}')
        backup_root.mkdir(parents=True, exist_ok=False)
        report['backup_directory'] = str(backup_root)
        report['protected_sha256'] = protected
        report['files'] = {rel: {'before_sha256': digest(old), 'after_sha256': digest(new)}
                           for rel, (old, new) in updates.items()}
        report_path = backup_root / 'report.json'
        atomic_bytes(report_path, json_bytes(report))
        transaction = {'files': {}}
        for rel, (old, new) in updates.items():
            if inside(store.root, rel).read_bytes() != old:
                raise RuntimeError(f'File changed during planning; no labels updated: {rel}')
            backup = backup_root / 'files' / rel
            backup.parent.mkdir(parents=True, exist_ok=True)
            atomic_bytes(backup, old)
            transaction['files'][rel] = backup.relative_to(store.root).as_posix()
        atomic_bytes(store.journal, json_bytes(transaction))
        try:
            for rel, (old, new) in updates.items():
                atomic_bytes(inside(store.root, rel), new)
            for rel, expected in protected.items():
                if digest(inside(store.root, rel).read_bytes()) != expected:
                    raise RuntimeError(f'Protected labels changed during filtering: {rel}')
            store.journal.unlink()
        except Exception:
            store.recover()
            report['rolled_back'] = True
            atomic_bytes(report_path, json_bytes(report))
            raise
        store.refresh()
        report.update(applied=True, protected_files_unchanged=True, protected_file_count=len(protected))
        atomic_bytes(report_path, json_bytes(report))
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', type=Path, default=Path('datasets'))
    parser.add_argument('--pseudo-root', type=Path)
    parser.add_argument('--dates', nargs='+', default=['260916'])
    parser.add_argument('--max-blobs', type=int, default=2)
    parser.add_argument('--apply', action='store_true', help='Back up and update untouched pending pseudo labels')
    args = parser.parse_args()
    if args.max_blobs < 1:
        parser.error('max-blobs must be positive')
    cv2.setNumThreads(1)
    store = ReviewStore(args.dataset_root, args.pseudo_root or args.dataset_root / 'pseudo_labels')
    def progress(counts):
        print(f"Scanned {counts['frames_scanned']}/{len(store.frames)} frames; "
              f"{counts.get('frames_changed', 0)} need filtering", file=sys.stderr, flush=True)
    report = filter_existing_roads(store, args.max_blobs, args.dates, args.apply, progress)
    print(json.dumps({k: v for k, v in report.items() if k not in ('changes', 'files', 'protected_sha256')},
                     indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
