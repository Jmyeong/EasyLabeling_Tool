import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from filter_pseudo_roads import filter_existing_roads, png_bytes
from pseudo_index import index_contents
from review_gui import atomic_bytes
from road_filter import keep_largest_road_blobs, matches_teacher_mask
import test_review_gui


def sample_mask():
    mask = np.zeros((32, 48), np.uint8)
    mask[1:6, 1:9] = 1       # 40 pixels
    mask[10:13, 10:16] = 1   # 18 pixels
    mask[20:22, 20:23] = 1   # 6 pixels
    mask[27, 27] = 1        # 1 pixel
    mask[30, :] = 255
    return mask


class ComponentTests(unittest.TestCase):
    def test_top_two_preserve_ignore_and_input(self):
        mask = sample_mask()
        original = mask.copy()
        result, info = keep_largest_road_blobs(mask)
        self.assertEqual(info['components_before'], 4)
        self.assertEqual(info['components_after'], 2)
        self.assertEqual(info['pixels_removed'], 7)
        self.assertEqual(info['kept_areas'], [40, 18])
        np.testing.assert_array_equal(mask, original)
        np.testing.assert_array_equal(result == 255, mask == 255)
        self.assertEqual(int((result == 1).sum()), 58)
        self.assertEqual(keep_largest_road_blobs(result)[1]['pixels_removed'], 0)

    def test_empty_one_two_blobs_and_diagonal_connectivity(self):
        for value in (0, 1, 255):
            mask = np.full((8, 8), value, np.uint8)
            np.testing.assert_array_equal(keep_largest_road_blobs(mask)[0], mask)
        mask = np.eye(8, dtype=np.uint8)
        self.assertEqual(keep_largest_road_blobs(mask)[1]['components_before'], 1)
        mask = np.zeros((10, 10), np.uint8)
        mask[1, 1] = mask[5, 5] = 1
        np.testing.assert_array_equal(keep_largest_road_blobs(mask)[0], mask)

    def test_tied_areas_and_invalid_input(self):
        mask = np.zeros((8, 8), np.uint8)
        mask[1, 1] = mask[3, 3] = mask[5, 5] = 1
        a, _ = keep_largest_road_blobs(mask)
        b, _ = keep_largest_road_blobs(mask)
        np.testing.assert_array_equal(a, b)
        self.assertEqual(int((a == 1).sum()), 2)
        for limit in (0, -1, 1.5):
            with self.assertRaises(ValueError):
                keep_largest_road_blobs(mask, limit)
        with self.assertRaises(ValueError):
            keep_largest_road_blobs(np.full((2, 2), 2, np.uint8))

    def test_probability_quantization_and_manual_edit_detection(self):
        probability = np.array([[.1, .4, .400001, .5, .599999, .6, .9]], np.float32)
        for margin in (.1, 0):
            mask = (probability >= .5).astype(np.uint8)
            mask[np.abs(probability - .5) < margin] = 255
            quantized = np.rint(probability * 65535).astype(np.uint16)
            self.assertTrue(matches_teacher_mask(mask, quantized, .5, margin))
            mask[0, 0] = 1
            self.assertFalse(matches_teacher_mask(mask, quantized, .5, margin))


class ExistingRoadTests(unittest.TestCase):
    def setUp(self):
        self.fixture = test_review_gui.ReviewTests()
        self.fixture.setUp()
        self.store, self.root, self.paths = self.fixture.store, self.fixture.pseudo, self.fixture.paths
        self.mask = sample_mask()
        self.probability = np.where(self.mask == 255, .5, np.where(self.mask == 1, .9, .1)).astype(np.float32)
        (self.root / self.paths['road_file']).write_bytes(png_bytes(self.mask))
        Image.fromarray(np.rint(self.probability * 65535).astype(np.uint16)).save(self.root / self.paths['road_probability_file'])
        self.record = self.fixture.record
        self.record['detections'][0]['class_name'] = 'Golfcart'
        self.record['road_fraction'] = float((self.mask == 1).mean())
        (self.root / self.paths['annotation_file']).write_text(json.dumps(self.record))
        (self.root / 'run.json').write_text(json.dumps({'generation_signature': 'teacher-fixture',
                                                      'road_threshold': .5, 'road_ignore_margin': .1}))
        for name, data in index_contents([self.record])[0].items():
            (self.root / name).write_bytes(data)
        self.store.refresh()

    def tearDown(self):
        self.fixture.tearDown()

    def snapshot(self):
        return {str(p.relative_to(self.root)): p.read_bytes() for p in self.root.rglob('*')
                if p.is_file() and not any(part.startswith('_') for part in p.relative_to(self.root).parts)}

    def test_dry_run_apply_backup_and_idempotence(self):
        before = self.snapshot()
        report = filter_existing_roads(self.store)
        self.assertFalse(report['applied'])
        self.assertEqual(report['counts']['pixels_removed'], 7)
        self.assertEqual(self.snapshot(), before)
        report = filter_existing_roads(self.store, apply=True)
        self.assertTrue(report['applied'])
        self.assertTrue(report['protected_files_unchanged'])
        record = json.loads((self.root / self.paths['annotation_file']).read_text())
        self.assertEqual(record['review_status'], 'pending')
        self.assertNotIn('review', record)
        self.assertEqual(record['road_fraction'], 58 / self.mask.size)
        self.assertEqual(record['original_road_fraction'], 65 / self.mask.size)
        self.assertEqual(record['detections'], self.record['detections'])
        np.testing.assert_array_equal(np.array(Image.open(self.root / self.paths['road_file'])), keep_largest_road_blobs(self.mask)[0])
        for key in ('detection_file', 'road_probability_file'):
            self.assertEqual((self.root / self.paths[key]).read_bytes(), before[self.paths[key]])
        for rel in report['files']:
            self.assertEqual((Path(report['backup_directory']) / 'files' / rel).read_bytes(), before[rel])
        after = self.snapshot()
        self.assertEqual(filter_existing_roads(self.store, apply=True)['counts']['already_clean'], 1)
        self.assertEqual(self.snapshot(), after)

    def test_human_reviews_and_untracked_edits_preserved(self):
        annotation = self.root / self.paths['annotation_file']
        for metadata in ({'review_status': 'approved'}, {'review_status': 'rejected'},
                         {'review_status': 'pending', 'review': {'note': 'manual'}}):
            annotation.write_text(json.dumps({**self.record, **metadata}))
            before = self.snapshot()
            report = filter_existing_roads(self.store, apply=True)
            self.assertEqual(report['counts']['reviewed_preserved'], 1)
            self.assertEqual(self.snapshot(), before)
        annotation.write_text(json.dumps(self.record))
        edited = self.mask.copy()
        edited[0, 0] = 1
        (self.root / self.paths['road_file']).write_bytes(png_bytes(edited))
        before = self.snapshot()
        self.assertEqual(filter_existing_roads(self.store, apply=True)['counts']['changed_masks_preserved'], 1)
        self.assertEqual(self.snapshot(), before)

    def test_unverifiable_and_changed_filtered_mask_preserved(self):
        report = filter_existing_roads(self.store, apply=True)
        self.assertTrue(report['applied'])
        (self.root / self.paths['road_file']).write_bytes(png_bytes(self.mask))
        self.assertEqual(filter_existing_roads(self.store, apply=True)['counts']['changed_masks_preserved'], 1)
        (self.root / self.paths['annotation_file']).write_text(json.dumps(self.record))
        (self.root / 'run.json').unlink()
        self.assertEqual(filter_existing_roads(self.store, apply=True)['counts']['unverifiable_masks_preserved'], 1)

    def test_failure_restores_mask_metadata_and_indices(self):
        before = self.snapshot()
        target = self.root / self.paths['annotation_file']
        failed = False
        def fail(path, data):
            nonlocal failed
            if path == target and not failed:
                failed = True
                raise OSError('simulated failure')
            atomic_bytes(path, data)
        with patch('filter_pseudo_roads.atomic_bytes', side_effect=fail):
            with self.assertRaises(OSError):
                filter_existing_roads(self.store, apply=True)
        self.assertTrue(failed)
        self.assertEqual(self.snapshot(), before)
        self.assertFalse(self.store.journal.exists())

    def test_generator_default_and_disabled_filter(self):
        from generate_pseudo_labels import write_sample
        args = SimpleNamespace(box_dedup=True, box_dedup_iou=.9, previews=False,
                               road_threshold=.5, road_ignore_margin=.1, road_max_blobs=2,
                               dataset_root=self.fixture.root)
        row = self.store.frame(self.fixture.frame_id)['source']
        for limit in (2, 0):
            args.road_max_blobs = limit
            record = write_sample(args, self.root, row, [], self.probability, 'test')
            actual = np.array(Image.open(self.root / self.paths['road_file']))
            expected = keep_largest_road_blobs(self.mask)[0] if limit else self.mask
            np.testing.assert_array_equal(actual, expected)
            self.assertEqual(record['road_fraction'], float((expected == 1).mean()))
            self.assertEqual('road_blob_filter' in record, bool(limit))


if __name__ == '__main__':
    unittest.main()
