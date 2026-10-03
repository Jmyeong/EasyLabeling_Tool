import copy
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from box_filter import box_iou, filter_duplicate_boxes
from filter_pseudo_boxes import filter_existing, label_bytes
from pseudo_index import index_contents
from review_gui import atomic_bytes
import test_review_gui


def box(score=.5, cls=0, xy=(0,0,10,10)):
    return {'class_id':cls,'class_name':['Golfcart','Person','Tree','Undef_obj'][cls],
            'bbox_xyxy':list(xy),'confidence':score}


class NMSTests(unittest.TestCase):
    def test_highest_score_same_class_only_and_stable_output(self):
        boxes=[box(.2),box(.9),box(.99,1),box(.4,xy=(20,20,25,25))]
        old=copy.deepcopy(boxes)
        kept,removed=filter_duplicate_boxes(boxes)
        self.assertEqual(kept,boxes[1:])
        self.assertEqual(removed[0]['removed_index'],0)
        self.assertEqual(removed[0]['kept_index'],1)
        self.assertEqual(boxes,old)

    def test_exact_threshold_and_iou_not_containment(self):
        a,b=box(.9),box(.8,xy=(0,0,9,10))
        self.assertEqual(box_iou(a['bbox_xyxy'],b['bbox_xyxy']),.9)
        self.assertEqual(len(filter_duplicate_boxes([a,b],.9)[0]),1)
        self.assertEqual(len(filter_duplicate_boxes([a,b],.91)[0]),2)
        self.assertEqual(len(filter_duplicate_boxes([a,box(.8,xy=(0,0,1,1))])[0]),2)

    def test_ties_and_nontransitive_overlap(self):
        boxes=[box(.8,xy=(0,0,100,100)),box(.8,xy=(5,0,105,100)),box(.7,xy=(10,0,110,100))]
        kept,removed=filter_duplicate_boxes(boxes)
        self.assertEqual(kept,[boxes[0],boxes[2]])
        self.assertEqual(removed[0]['kept_index'],0)
        self.assertEqual(filter_duplicate_boxes([]),([],[]))

    def test_invalid_boxes_and_thresholds(self):
        for threshold in (0,-1,1.1,float('nan')):
            with self.assertRaises(ValueError):filter_duplicate_boxes([],threshold)
        for invalid in (box(float('nan')),box(1.1),box(xy=(1,0,1,2)),{**box(),'confidence':None}):
            with self.assertRaises(ValueError):filter_duplicate_boxes([invalid])


class ExistingLabelTests(unittest.TestCase):
    def setUp(self):
        self.fixture=test_review_gui.ReviewTests();self.fixture.setUp()
        self.store=self.fixture.store;self.root=self.fixture.pseudo
        self.boxes=[box(.2),box(.9),box(.99,1),box(.4,xy=(20,20,25,25))]
        self.record={**self.fixture.record,'detections':self.boxes,'detection_count':len(self.boxes),
                     'review_reasons':['low_confidence_detection']}
        self.fixture.record=self.record
        (self.root/self.fixture.paths['annotation_file']).write_text(json.dumps(self.record))
        (self.root/self.fixture.paths['detection_file']).write_bytes(label_bytes(self.boxes,48,32))
        contents,_=index_contents([self.record])
        for name,data in contents.items():(self.root/name).write_bytes(data)
        self.store.refresh()

    def tearDown(self):self.fixture.tearDown()

    def snapshot(self):return {str(p.relative_to(self.root)):p.read_bytes() for p in self.root.rglob('*') if p.is_file() and not any(part.startswith('_') for part in p.relative_to(self.root).parts)}

    def test_dry_run_apply_backup_teacher_preservation_and_idempotence(self):
        before=self.snapshot()
        planned=filter_existing(self.store)
        self.assertFalse(planned['applied']);self.assertEqual(planned['counts']['boxes_removed'],1)
        self.assertEqual(before,self.snapshot())
        result=filter_existing(self.store,apply=True)
        self.assertTrue(result['applied']);self.assertTrue(result['protected_files_unchanged'])
        record=json.loads((self.root/self.fixture.paths['annotation_file']).read_text())
        self.assertEqual(record['detections'],self.boxes)
        self.assertEqual(record['filtered_detections'],self.boxes[1:])
        self.assertEqual(record['review_status'],'pending');self.assertNotIn('review',record)
        self.assertNotIn('low_confidence_detection',record['review_reasons'])
        self.assertEqual(self.store.load(self.fixture.frame_id)['box_count'],3)
        self.assertEqual((self.root/self.fixture.paths['road_file']).read_bytes(),before[self.fixture.paths['road_file']])
        backup=Path(result['backup_directory'])/'files'
        for name in result['files']:self.assertEqual((backup/name).read_bytes(),before[name])
        after=self.snapshot();again=filter_existing(self.store,apply=True)
        self.assertEqual(again['counts'].get('boxes_removed',0),0);self.assertEqual(after,self.snapshot())

    def test_reviewed_frames_and_untracked_edits_preserved(self):
        self.fixture.add_frame()
        path=self.root/self.fixture.paths['annotation_file']
        reviewed={**self.record,'review_status':'approved','review':{'note':'수동 검수','save_count':2}}
        path.write_text(json.dumps(reviewed));self.store.refresh()
        before=self.store.current(self.store.frame(self.fixture.frame_id))[0]
        result=filter_existing(self.store,apply=True)
        self.assertEqual(result['counts']['reviewed_preserved'],1)
        self.assertEqual(before,self.store.current(self.store.frame(self.fixture.frame_id))[0])
        path.write_text(json.dumps(self.record))
        label=self.root/self.fixture.paths['detection_file'];label.write_text('1 0.5 0.5 0.2 0.2\n')
        self.store.refresh();before=self.snapshot()
        result=filter_existing(self.store,apply=True)
        self.assertEqual(result['counts']['changed_labels_preserved'],1)
        self.assertEqual(before,self.snapshot())

    def test_write_failure_rolls_back_every_label_and_index(self):
        before=self.snapshot();target=self.root/self.fixture.paths['annotation_file'];failed=False
        def fail(path,data):
            nonlocal failed
            if path==target and not failed:
                failed=True;raise OSError('simulated failure')
            atomic_bytes(path,data)
        with patch('filter_pseudo_boxes.atomic_bytes',side_effect=fail):
            with self.assertRaises(OSError):filter_existing(self.store,apply=True)
        self.assertTrue(failed);self.assertEqual(before,self.snapshot());self.assertFalse(self.store.journal.exists())

    def test_generator_writes_filtered_labels_and_preserves_raw_predictions(self):
        from generate_pseudo_labels import write_sample
        args=SimpleNamespace(box_dedup=True,box_dedup_iou=.9,previews=False,
                             road_threshold=.5,road_ignore_margin=.1,road_max_blobs=2,dataset_root=self.fixture.root)
        row=self.store.frame(self.fixture.frame_id)['source']
        result=write_sample(args,self.root,row,self.boxes,np.full((32,48),.8,np.float32),'test')
        self.assertEqual(result['detection_count'],3)
        self.assertEqual(result['detections'],self.boxes[1:]);self.assertEqual(result['raw_detections'],self.boxes)
        self.assertEqual((self.root/self.fixture.paths['detection_file']).read_bytes(),label_bytes(self.boxes[1:],48,32))


if __name__=='__main__':unittest.main()
