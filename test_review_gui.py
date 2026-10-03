"""Backend regression tests; only temporary fixtures are modified."""

import base64
import csv
import fcntl
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from http.server import ThreadingHTTPServer

import numpy as np
from PIL import Image

from dated_dataset import annotation_paths
from review_gui import ReviewError, ReviewStore, atomic_bytes, make_handler


class ReviewTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.pseudo = self.root / "pseudo_labels"
        self.pseudo.mkdir()
        row = {"date": "260916", "split": "train", "sequence": "seq",
               "image_file": "image.png", "nir_file": "nir.png", "image_sha256": "fixture",
               "semseg_file": "", "instseg_file": "", "completed_depth_file": "depth.npz"}
        for name in ("image.png", "nir.png"):
            Image.fromarray(np.zeros((32, 48, 3), np.uint8)).save(self.root / name)
        np.savez(self.root / "depth.npz", depth_m=np.ones((32, 48), np.float32)*3, source=np.ones((32, 48), np.uint8))
        with (self.root / "manifest.csv").open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=row.keys())
            writer.writeheader()
            writer.writerow(row)
        self.paths = annotation_paths(row)
        for path in self.paths.values():
            (self.pseudo / path).parent.mkdir(parents=True, exist_ok=True)
        (self.pseudo / self.paths["detection_file"]).write_text("0 0.5 0.5 0.5 0.5\n")
        Image.fromarray(np.zeros((32, 48), np.uint8)).save(self.pseudo / self.paths["road_file"])
        self.record = {**row, **self.paths, "width": 48, "height": 32, "review_status": "pending",
                       "detections": [{"confidence": .8, "class_id": 0}], "detection_count": 1,
                       "road_fraction": 0, "uncertain_fraction": 0, "generation_signature": "teacher-fixture"}
        (self.pseudo / self.paths["annotation_file"]).write_text(json.dumps(self.record))
        self.store = ReviewStore(self.root, self.pseudo)
        self.frame_id = next(iter(self.store.frames))

    def tearDown(self):
        self.temp.cleanup()

    def payload(self):
        frame = self.store.load(self.frame_id)
        mask = np.zeros((32, 48), np.uint8)
        mask[:8] = 1
        mask[8:12] = 255
        return {"revision": frame["revision"], "status": "approved", "note": "수동 보정",
                "boxes": [{"class_id": 2, "bbox_xyxy": [1, 2, 20, 25]}],
                "mask": base64.b64encode(mask.tobytes()).decode()}

    def add_frame(self):
        row = dict(self.store.frame(self.frame_id)["source"], image_file="other.png")
        (self.root / "other.png").write_bytes((self.root / "image.png").read_bytes())
        with (self.root / "manifest.csv").open("a", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=row.keys()).writerow(row)
        paths = annotation_paths(row)
        for key in ("detection_file", "road_file"):
            (self.pseudo / paths[key]).write_bytes((self.pseudo / self.paths[key]).read_bytes())
        (self.pseudo / paths["annotation_file"]).write_text(json.dumps({**self.record, **row, **paths}))
        self.store.refresh()
        return next(key for key in self.store.frames if key != self.frame_id)

    def batch_payload(self):
        return {"frames": [{"id": f["id"], "annotation_revision": f["annotation_revision"]}
                           for f in self.store.listing()]}

    def test_batch_exclusion_preserves_labels_notes_and_teacher(self):
        self.add_frame()
        self.store.save(self.frame_id, self.payload())
        before = {fid: self.store.current(frame)[0] for fid, frame in self.store.frames.items()}
        result = self.store.reject_many(self.batch_payload())
        self.assertEqual(result["count"], 2)
        for fid, old in before.items():
            frame = self.store.load(fid)
            self.assertEqual(frame["status"], "rejected")
            after = self.store.current(self.store.frame(fid))[0]
            for key in ("detection_file", "road_file"):
                self.assertEqual(after[key], old[key])
            previous = json.loads(old["annotation_file"])
            record = json.loads(after["annotation_file"])
            self.assertEqual(record["detections"], previous["detections"])
            self.assertEqual(record["review"].get("note"), previous.get("review", {}).get("note"))
            backups = list((self.pseudo / "_review_history" / fid).glob("*/*.json"))
            self.assertTrue(any(p.read_bytes() == old["annotation_file"] for p in backups))

    def test_batch_conflict_and_bad_ids_write_nothing(self):
        other = self.add_frame()
        payload = self.batch_payload()
        self.store.save(self.frame_id, self.payload())
        before = {fid: self.store.current(f)[1] for fid, f in self.store.frames.items()}
        with self.assertRaises(ReviewError) as error:
            self.store.reject_many(payload)
        self.assertEqual(error.exception.status, 409)
        valid = self.batch_payload()["frames"]
        for frames in ([], [valid[0], valid[0]], [valid[0], {"id":"unknown"}], [None]):
            with self.assertRaises(ReviewError):
                self.store.reject_many({"frames": frames})
        self.assertEqual(before, {fid:self.store.current(f)[1] for fid,f in self.store.frames.items()})
        self.assertEqual(self.store.load(other)["status"], "pending")

    def test_batch_write_failure_rolls_back_whole_batch_and_cache(self):
        other = self.add_frame()
        before = {fid: self.store.current(f)[1] for fid, f in self.store.frames.items()}
        target = self.pseudo / self.store.frame(other)["paths"]["annotation_file"]
        failed = False
        def failing_write(path, data):
            nonlocal failed
            if path == target and not failed:
                failed = True
                raise OSError("simulated batch failure")
            atomic_bytes(path, data)
        with patch("review_gui.atomic_bytes", side_effect=failing_write):
            with self.assertRaises(OSError):
                self.store.reject_many(self.batch_payload())
        self.assertTrue(failed)
        self.assertFalse(self.store.journal.exists())
        self.assertEqual(before, {fid:self.store.current(f)[1] for fid,f in self.store.frames.items()})
        self.assertTrue(all(f["status"] == "pending" for f in self.store.listing()))

    def test_save_roundtrip_and_training_loader(self):
        old = self.store.current(self.store.frame(self.frame_id))[0]
        result = self.store.save(self.frame_id, self.payload())
        loaded = self.store.load(self.frame_id)
        self.assertEqual(loaded["status"], "approved")
        self.assertEqual(loaded["boxes"][0]["class_id"], 2)
        self.assertEqual(loaded["note"], "수동 보정")
        self.assertEqual(loaded["teacher_boxes"], self.record["detections"])
        backup = self.pseudo / result["backup"]
        for key, data in old.items():
            self.assertEqual((backup / Path(self.paths[key]).name).read_bytes(), data)
        from train import ThreeTaskDataset
        dataset = ThreeTaskDataset(self.root, "train", (32, 64), 1, 35, False)
        self.assertEqual(dataset.rows[0]["label_source"], "reviewed")
        self.assertEqual(dataset[0]["boxes"][0][0], 2)
        self.assertEqual(set(dataset[0]["road"].unique().tolist()), {0, 1, 255})

    def test_invalid_payload_and_stale_revision_do_not_write(self):
        before = self.store.load(self.frame_id)["revision"]
        for update in [{"boxes": [{"class_id": 4, "bbox_xyxy": [0, 0, 1, 1]}]},
                       {"boxes": [{"class_id": 0, "bbox_xyxy": [-1, 0, 1, 1]}]},
                       {"boxes": [{"class_id": 0, "bbox_xyxy": [0, 0, float("nan"), 1]}]},
                       {"mask": base64.b64encode(bytes([2])*32*48).decode()},
                       {"mask": "AA=="}, {"status": "arbitrary"}]:
            payload = self.payload()
            payload.update(update)
            with self.assertRaises(ReviewError):
                self.store.save(self.frame_id, payload)
            self.assertEqual(self.store.load(self.frame_id)["revision"], before)
        payload = self.payload()
        self.store.save(self.frame_id, payload)
        with self.assertRaises(ReviewError) as context:
            self.store.save(self.frame_id, payload)
        self.assertEqual(context.exception.status, 409)

    def test_empty_boxes_and_exclusion(self):
        payload = self.payload()
        payload.update(boxes=[], status="rejected")
        self.store.save(self.frame_id, payload)
        self.assertEqual((self.pseudo / self.paths["detection_file"]).read_text(), "")
        self.assertEqual(self.store.listing()[0]["box_count"], 0)
        self.assertEqual(self.store.load(self.frame_id)["status"], "rejected")

    def test_write_failure_rolls_back_all_files(self):
        before = self.store.load(self.frame_id)["revision"]
        target = self.pseudo / self.paths["road_file"]
        failed = False

        def failing_write(path, data):
            nonlocal failed
            if path == target and not failed:
                failed = True
                raise OSError("simulated disk failure")
            atomic_bytes(path, data)

        with patch("review_gui.atomic_bytes", side_effect=failing_write):
            with self.assertRaises(OSError):
                self.store.save(self.frame_id, self.payload())
        self.assertTrue(failed)
        self.assertEqual(self.store.load(self.frame_id)["revision"], before)
        self.assertFalse(self.store.journal.exists())

    def test_startup_recovers_interrupted_transaction(self):
        before = self.store.load(self.frame_id)["revision"]
        result = self.store.save(self.frame_id, self.payload())
        transaction = {"files": {path: str(Path(result["backup"]) / Path(path).name)
                                  for key, path in self.paths.items() if key in ("road_file", "detection_file", "annotation_file")}}
        self.store.journal.write_text(json.dumps(transaction))
        recovered = ReviewStore(self.root, self.pseudo)
        self.assertEqual(recovered.load(self.frame_id)["revision"], before)

    def test_generation_lock(self):
        with (self.pseudo / ".generation.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaises(ReviewError) as context:
                self.store.save(self.frame_id, self.payload())
            self.assertEqual(context.exception.status, 409)

    def test_http_endpoints_and_save_token(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(self.store))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        url = f"http://127.0.0.1:{server.server_port}"
        try:
            for route in ("/", "/app.js", "/style.css", "/api/config", "/api/frames", f"/api/frame/{self.frame_id}", f"/api/frame/{self.frame_id}/rgb", "/api/export"):
                with urlopen(url + route) as response:
                    self.assertEqual(response.status, 200)
                    self.assertTrue(response.read())
            request = Request(url + f"/api/frame/{self.frame_id}", data=json.dumps(self.payload()).encode(), headers={"Content-Type": "application/json"})
            with self.assertRaises(HTTPError) as context:
                urlopen(request)
            self.assertEqual(context.exception.code, 403)
            request.add_header("X-Review-Token", self.store.token)
            with urlopen(request) as response:
                self.assertEqual(response.status, 200)
            self.assertEqual(self.store.load(self.frame_id)["status"], "approved")
            request = Request(url + "/api/frames/reject", data=json.dumps(self.batch_payload()).encode(),
                              headers={"Content-Type":"application/json"})
            with self.assertRaises(HTTPError) as error:
                urlopen(request)
            self.assertEqual(error.exception.code, 403)
            request.add_header("X-Review-Token", self.store.token)
            with urlopen(request) as response:
                self.assertEqual(json.load(response)["count"], 1)
            self.assertEqual(self.store.load(self.frame_id)["status"], "rejected")
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


if __name__ == "__main__":
    unittest.main()
