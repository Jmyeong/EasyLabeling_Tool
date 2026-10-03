"""Run with: python -m unittest test_dated_dataset -v (no GPU required)."""

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from dated_dataset import annotation_paths, read_depth
from generate_pseudo_labels import restore_predictions
from train import ThreeTaskDataset, letterbox_all


class DatedDatasetTests(unittest.TestCase):
    def test_reverse_letterbox(self):
        for h, w in [(720, 1280), (481, 643), (901, 400)]:
            rgb = np.zeros((h, w, 3), np.uint8)
            _, _, _, _, boxes = letterbox_all(rgb, rgb[:, :, 0], np.zeros((h, w)), rgb[:, :, 0], [(1, .5, .5, .6, .8)], (352, 640))
            cls, cx, cy, bw, bh = boxes[0]
            det = [(cls, (cx-bw/2)*640, (cy-bh/2)*352, (cx+bw/2)*640, (cy+bh/2)*352, .8)]
            scale = min(352/h, 640/w)
            nh, nw = round(h*scale), round(w*scale)
            py, px = (352-nh)//2, (640-nw)//2
            prob = np.zeros((352, 640), np.float32)
            prob[py:py+nh, px:px+nw] = .8
            result, restored = restore_predictions(det, prob, h, w, (352, 640))
            np.testing.assert_allclose(result[0]['bbox_xyxy'], [.2*w, .1*h, .8*w, .9*h], atol=1e-4)
            np.testing.assert_allclose(restored, .8, atol=1e-6)

    def test_review_policy_and_depth_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = []
            for split in ['train', 'test']:
                row = dict(date='260916', split=split, sequence='seq', image_file=f'{split}.png', nir_file=f'{split}_nir.png',
                           semseg_file='', instseg_file='', completed_depth_file=f'{split}.npz', image_sha256='fixture')
                Image.fromarray(np.zeros((32, 32, 3), np.uint8)).save(root / row['image_file'])
                Image.fromarray(np.zeros((16, 16), np.uint8)).save(root / row['nir_file'])
                depth = np.full((32, 32), 3, np.float32)
                source = np.ones((32, 32), np.uint8)
                source[16:] = 2
                np.savez(root / row['completed_depth_file'], depth_m=depth, source=source)
                paths = annotation_paths(row)
                for path in paths.values():
                    (root / 'pseudo_labels' / path).parent.mkdir(parents=True, exist_ok=True)
                (root / 'pseudo_labels' / paths['detection_file']).write_text('')
                road = np.zeros((32, 32), np.uint8)
                road[:8] = 255
                road[16:] = 1
                Image.fromarray(road).save(root / 'pseudo_labels' / paths['road_file'])
                (root / 'pseudo_labels' / paths['annotation_file']).write_text(json.dumps({**row, 'review_status': 'pending'}))
                rows.append(row)
            with (root / 'manifest.csv').open('w', encoding='utf-8-sig', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)

            def load(split, policy='reviewed'):
                return ThreeTaskDataset(root, split, (32, 32), 1, 35, False, pseudo_label_policy=policy)

            with self.assertRaisesRegex(ValueError, 'No labeled train'):
                load('train')
            with self.assertRaisesRegex(ValueError, 'No labeled test'):
                load('test', 'all')
            train = load('train', 'all')[0]
            self.assertEqual(train['boxes'], [])  # Existing empty file is a valid negative.
            self.assertEqual(int(train['valid_depth'].sum()), 1024)
            self.assertEqual(set(train['road'].unique().tolist()), {0, 1, 255})
            for row in rows:
                path = root / 'pseudo_labels' / annotation_paths(row)['annotation_file']
                record = json.loads(path.read_text())
                record['review_status'] = 'approved'
                path.write_text(json.dumps(record))
            self.assertEqual(len(load('train')), 1)
            self.assertEqual(int(load('test')[0]['valid_depth'].sum()), 512)
            path = root / 'pseudo_labels' / annotation_paths(rows[0])['detection_file']
            path.unlink()
            with self.assertRaises(FileNotFoundError):
                load('train')

    def test_nonfinite_depth_is_ignored(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'depth.npz'
            np.savez(path, depth_m=np.array([[np.nan, np.inf, 3]], np.float32), source=np.array([[1, 1, 2]], np.uint8))
            np.testing.assert_array_equal(read_depth(path), [[0, 0, 3]])
            np.testing.assert_array_equal(read_depth(path, measured_only=True), [[0, 0, 0]])


if __name__ == '__main__':
    unittest.main()
