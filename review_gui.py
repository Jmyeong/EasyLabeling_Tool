#!/usr/bin/env python3
"""Local browser editor for detection and binary-road pseudo annotations."""

import argparse
import base64
import binascii
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import math
import mimetypes
import os
from pathlib import Path
import secrets
import threading
from urllib.parse import parse_qs, urlparse

import numpy as np
from PIL import Image

from dated_dataset import annotation_paths, read_manifest
from prepare_raw_dy_from_instseg import CLASS_NAMES, yolo_line
from review_sam import SamService, SamError, validate_points, DEFAULT_PYTHON, DEFAULT_CHECKPOINT, DEFAULT_CONFIG

PROJECT = Path(__file__).resolve().parent
STATUSES = {"pending", "approved", "rejected"}


class ReviewError(Exception):
    def __init__(self, message, status=400):
        super().__init__(message)
        self.status = status


def inside(root, relative):
    path = (root / relative).resolve()
    if not relative or not path.is_relative_to(root):
        raise ReviewError("허용되지 않은 파일 경로입니다.")
    return path


def atomic_bytes(path, data):
    temporary = path.with_name(path.name + ".review-tmp")
    try:
        with temporary.open("wb") as handle:
            handle.write(data)
            handle.flush()
            import os
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()


class ReviewStore:
    def __init__(self, dataset_root, pseudo_root):
        self.dataset_root = Path(dataset_root).expanduser().resolve()
        self.root = Path(pseudo_root).expanduser().resolve()
        if not self.root.is_dir():
            raise FileNotFoundError(f"Pseudo label 폴더가 없습니다: {self.root}")
        self.lock = threading.RLock()
        self.frames = {}
        self.token = secrets.token_urlsafe(32)
        self.journal = self.root / ".review_transaction.json"
        with self.generation_lock():
            self.recover()
        self.refresh()

    def generation_lock(self):
        from contextlib import contextmanager

        @contextmanager
        def acquire():
            with (self.root / ".generation.lock").open("a") as handle:
                try:
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError as error:
                    raise ReviewError("라벨 생성 또는 다른 저장이 진행 중입니다. 완료 후 다시 시도하세요.", 409) from error
                try:
                    yield
                finally:
                    fcntl.flock(handle, fcntl.LOCK_UN)
        return acquire()

    def recover(self):
        if not self.journal.exists():
            return
        transaction = json.loads(self.journal.read_text())
        for target, backup in transaction["files"].items():
            atomic_bytes(inside(self.root, target), inside(self.root, backup).read_bytes())
        self.journal.unlink()

    def refresh(self):
        with self.lock:
            frames = {}
            for row in read_manifest(self.dataset_root):
                paths = annotation_paths(row)
                annotation = inside(self.root, paths["annotation_file"])
                if not annotation.is_file():
                    continue
                annotation_data = annotation.read_bytes()
                record = json.loads(annotation_data)
                if record["image_file"] != row["image_file"] or record.get("image_sha256") != row.get("image_sha256"):
                    raise ReviewError(f"원본과 annotation이 일치하지 않습니다: {annotation}")
                frame_id = hashlib.sha256(row["image_file"].encode()).hexdigest()[:24]
                if frame_id in frames:
                    raise ReviewError("중복 frame ID가 있습니다.")
                frames[frame_id] = {"id": frame_id, "source": row, "paths": paths, "record": record,
                                    "annotation_revision": hashlib.sha256(annotation_data).hexdigest()}
            self.frames = frames
            if not frames:
                raise ReviewError("편집할 pseudo label이 없습니다. 먼저 라벨을 생성하세요.")

    def frame(self, frame_id):
        try:
            return self.frames[frame_id]
        except KeyError as error:
            raise ReviewError("프레임을 찾을 수 없습니다.", 404) from error

    def current(self, frame):
        paths = frame["paths"]
        data = {key: inside(self.root, paths[key]).read_bytes()
                for key in ("detection_file", "road_file", "annotation_file")}
        revision = hashlib.sha256()
        for key, value in data.items():
            revision.update(key.encode())
            revision.update(len(value).to_bytes(8, "big"))
            revision.update(value)
        return data, revision.hexdigest()

    @staticmethod
    def summary(frame):
        row, record = frame["source"], frame["record"]
        review = record.get("review", {})
        return {
            "id": frame["id"], "date": row["date"], "split": row["split"],
            "sequence": row["sequence"], "name": Path(row["image_file"]).name,
            "image_file": row["image_file"], "status": record.get("review_status", "pending"),
            "box_count": review.get("box_count", record.get("detection_count", 0)),
            "uncertain_fraction": review.get("uncertain_fraction", record.get("uncertain_fraction", 0)),
            "review_reasons": record.get("review_reasons", []),
            "edited": bool(review), "note": review.get("note", ""),
            "annotation_revision": frame["annotation_revision"],
        }

    def listing(self):
        with self.lock:
            return [self.summary(f) for f in self.frames.values()]

    def load(self, frame_id):
        with self.lock:
            frame = self.frame(frame_id)
            data, revision = self.current(frame)
            record = json.loads(data["annotation_file"])
            frame["record"] = record
            frame["annotation_revision"] = hashlib.sha256(data["annotation_file"]).hexdigest()
            w, h = int(record["width"]), int(record["height"])
            with Image.open(io.BytesIO(data["road_file"])) as image:
                mask = np.array(image)
            if mask.shape != (h, w) or mask.dtype != np.uint8 or not np.isin(mask, [0, 1, 255]).all():
                raise ReviewError("마스크는 원본 해상도의 0/1/255 단일 채널 PNG여야 합니다.")
            boxes = []
            for line in data["detection_file"].decode().splitlines():
                if not line.strip():
                    continue
                cls, cx, cy, bw, bh = map(float, line.split())
                boxes.append({"class_id": cls, "bbox_xyxy": [max(0, (cx-bw/2)*w), max(0, (cy-bh/2)*h),
                                                                           min(w, (cx+bw/2)*w), min(h, (cy+bh/2)*h)]})
            boxes = self.validate_boxes(boxes, w, h)
            with Image.open(inside(self.dataset_root, frame["source"]["image_file"])) as image:
                if image.size != (w, h):
                    raise ReviewError("RGB와 마스크 해상도가 다릅니다.")
            return {
                **self.summary(frame), "width": w, "height": h, "revision": revision,
                "boxes": boxes, "mask": base64.b64encode(mask.tobytes()).decode(),
                "nir_available": bool(frame["source"].get("nir_file")),
                "teacher_boxes": record.get("detections", []),
            }

    @staticmethod
    def validate_boxes(boxes, w, h):
        if not isinstance(boxes, list) or len(boxes) > 2000:
            raise ReviewError("박스 수가 올바르지 않습니다.")
        result = []
        for box in boxes:
            if not isinstance(box, dict):
                raise ReviewError("잘못된 박스 형식입니다.")
            cls, xyxy = box.get("class_id"), box.get("bbox_xyxy")
            if isinstance(cls, bool) or not isinstance(cls, (int, float)) or cls not in range(len(CLASS_NAMES)):
                raise ReviewError("검출 클래스는 0~3이어야 합니다.")
            if not isinstance(xyxy, list) or len(xyxy) != 4:
                raise ReviewError("박스 좌표가 올바르지 않습니다.")
            if not all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) for x in xyxy):
                raise ReviewError("박스 좌표는 유한한 숫자여야 합니다.")
            x1, y1, x2, y2 = xyxy
            if not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h):
                raise ReviewError("박스는 이미지 안에 있어야 하며 너비와 높이가 양수여야 합니다.")
            result.append({"class_id": int(cls), "bbox_xyxy": list(map(float, xyxy))})
        return result

    def save(self, frame_id, payload):
        if not isinstance(payload, dict):
            raise ReviewError("JSON 객체가 필요합니다.")
        with self.lock, self.generation_lock():
            self.recover()
            frame = self.frame(frame_id)
            old_data, revision = self.current(frame)
            if payload.get("revision") != revision:
                raise ReviewError("다른 창에서 이 프레임을 변경했습니다. 현재 작업을 복사하거나 새로 불러온 뒤 다시 저장하세요.", 409)
            record = json.loads(old_data["annotation_file"])
            w, h = int(record["width"]), int(record["height"])
            status = payload.get("status")
            if status not in STATUSES:
                raise ReviewError("검수 상태가 올바르지 않습니다.")
            note = payload.get("note", "")
            if not isinstance(note, str) or len(note) > 4000:
                raise ReviewError("메모는 4,000자 이내로 입력하세요.")
            boxes = self.validate_boxes(payload.get("boxes"), w, h)
            try:
                raw_mask = base64.b64decode(payload.get("mask", ""), validate=True)
            except (ValueError, TypeError, binascii.Error) as error:
                raise ReviewError("마스크 인코딩이 잘못되었습니다.") from error
            if len(raw_mask) != w*h:
                raise ReviewError("마스크 크기가 원본 해상도와 다릅니다.")
            mask = np.frombuffer(raw_mask, dtype=np.uint8).reshape(h, w)
            if not np.isin(mask, [0, 1, 255]).all():
                raise ReviewError("마스크 값은 0, 1, 255만 허용합니다.")
            now = datetime.now(timezone.utc)
            record["review_status"] = status
            record["review"] = {
                "edited_at": now.isoformat(), "note": note, "box_count": len(boxes),
                "road_fraction": float((mask == 1).mean()), "uncertain_fraction": float((mask == 255).mean()),
                "editor": "DY review GUI", "save_count": record.get("review", {}).get("save_count", 0) + 1,
            }
            # Teacher detections, confidence maps and previews remain the original proposal.
            png = io.BytesIO()
            Image.fromarray(mask).save(png, format="PNG", compress_level=1)
            lines = [yolo_line(box, w, h) for box in boxes]
            new_data = {
                "detection_file": ("\n".join(lines) + ("\n" if lines else "")).encode(),
                "road_file": png.getvalue(), "annotation_file": json_bytes(record),
            }
            stamp = now.strftime("%Y%m%dT%H%M%S.%fZ") + "-" + secrets.token_hex(3)
            backup_dir = inside(self.root, (Path("_review_history") / frame_id / stamp).as_posix())
            backup_dir.mkdir(parents=True)
            transaction = {"files": {}}
            for key, content in old_data.items():
                backup = backup_dir / Path(frame["paths"][key]).name
                atomic_bytes(backup, content)
                transaction["files"][frame["paths"][key]] = backup.relative_to(self.root).as_posix()
            atomic_bytes(self.journal, json_bytes(transaction))
            try:
                for key, content in new_data.items():
                    atomic_bytes(inside(self.root, frame["paths"][key]), content)
                self.journal.unlink()
            except Exception:
                self.recover()
                raise
            frame["record"] = record
            frame["annotation_revision"] = hashlib.sha256(new_data["annotation_file"]).hexdigest()
            _, new_revision = self.current(frame)
            return {"revision": new_revision, "frame": self.summary(frame),
                    "backup": backup_dir.relative_to(self.root).as_posix()}

    def reject_many(self, payload):
        """Change only review metadata, with whole-batch conflict checks and rollback."""
        entries = payload.get("frames") if isinstance(payload, dict) else None
        if not isinstance(entries, list) or not 1 <= len(entries) <= len(self.frames):
            raise ReviewError("제외할 프레임 목록이 올바르지 않습니다.")
        with self.lock, self.generation_lock():
            self.recover()
            changes, seen = [], set()
            now = datetime.now(timezone.utc)
            stamp = now.strftime("%Y%m%dT%H%M%S.%fZ") + "-" + secrets.token_hex(3)
            for entry in entries:
                if not isinstance(entry, dict) or not isinstance(entry.get("id"), str) or entry["id"] in seen:
                    raise ReviewError("중복되거나 잘못된 프레임 ID입니다.")
                seen.add(entry["id"])
                frame = self.frame(entry["id"])
                path = inside(self.root, frame["paths"]["annotation_file"])
                old = path.read_bytes()
                if entry.get("annotation_revision") != hashlib.sha256(old).hexdigest():
                    raise ReviewError("다른 창에서 선택한 프레임을 변경했습니다. 아무 프레임도 제외하지 않았습니다. 목록을 새로고침한 뒤 다시 선택하세요.", 409)
                record = json.loads(old)
                record["review_status"] = "rejected"
                review = dict(record.get("review", {}))
                review.update(edited_at=now.isoformat(), editor="DY review GUI",
                              save_count=review.get("save_count", 0) + 1, last_action="batch_reject")
                record["review"] = review
                changes.append((frame, path, old, record, json_bytes(record)))
            transaction = {"files": {}}
            for frame, path, old, record, new in changes:
                backup = inside(self.root, f"_review_history/{frame['id']}/{stamp}/{path.name}")
                backup.parent.mkdir(parents=True)
                atomic_bytes(backup, old)
                transaction["files"][frame["paths"]["annotation_file"]] = backup.relative_to(self.root).as_posix()
            atomic_bytes(self.journal, json_bytes(transaction))
            try:
                for frame, path, old, record, new in changes:
                    atomic_bytes(path, new)
                self.journal.unlink()
            except Exception:
                self.recover()
                raise
            for frame, path, old, record, new in changes:
                frame["record"] = record
                frame["annotation_revision"] = hashlib.sha256(new).hexdigest()
            return {"frames": [self.summary(frame) for frame, *_ in changes], "count": len(changes)}

    def export_csv(self):
        output = io.StringIO(newline="")
        fields = ["image_file", "date", "split", "sequence", "status", "box_count", "uncertain_fraction", "edited", "note"]
        writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(self.listing())
        return output.getvalue().encode("utf-8-sig")


def make_handler(store, sam=None):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            if args and str(args[0]).startswith("POST"):
                super().log_message(fmt, *args)

        def send_data(self, content, content_type="application/json; charset=utf-8", status=200, extra=None):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'self'; img-src 'self' blob: data:; style-src 'self'; script-src 'self'; connect-src 'self'; frame-ancestors 'none'")
            for key, value in (extra or {}).items():
                self.send_header(key, value)
            self.end_headers()
            self.wfile.write(content)

        def json(self, value, status=200):
            self.send_data(json_bytes(value), status=status)

        def do_GET(self):
            try:
                path = urlparse(self.path).path
                if path in ("/", "/app.js", "/style.css"):
                    filename = {"/": "index.html", "/app.js": "app.js", "/style.css": "style.css"}[path]
                    file = PROJECT / "review_gui_web" / filename
                    return self.send_data(file.read_bytes(), mimetypes.guess_type(str(file))[0] + "; charset=utf-8")
                if path == "/api/config":
                    return self.json({"token": store.token, "classes": CLASS_NAMES, "root": str(store.root),
                                      "sam": sam.info() if sam else {"available":False,"message":"SAM이 연결되지 않았습니다."}})
                if path == "/api/frames":
                    if parse_qs(urlparse(self.path).query).get("refresh") == ["1"]:
                        store.refresh()
                    return self.json({"frames": store.listing()})
                if path == "/api/export":
                    return self.send_data(store.export_csv(), "text/csv; charset=utf-8", extra={"Content-Disposition": 'attachment; filename="review_status.csv"'})
                parts = path.strip("/").split("/")
                if len(parts) >= 3 and parts[:2] == ["api", "frame"]:
                    frame_id = parts[2]
                    if len(parts) == 3:
                        return self.json(store.load(frame_id))
                    if len(parts) == 4 and parts[3] in ("rgb", "nir"):
                        frame = store.frame(frame_id)
                        key = "image_file" if parts[3] == "rgb" else "nir_file"
                        file = inside(store.dataset_root, frame["source"][key])
                        return self.send_data(file.read_bytes(), mimetypes.guess_type(str(file))[0] or "application/octet-stream")
                raise ReviewError("찾을 수 없는 주소입니다.", 404)
            except ReviewError as error:
                self.json({"error": str(error)}, error.status)
            except (ValueError, KeyError, OSError) as error:
                self.json({"error": str(error)}, 500)

        def do_POST(self):
            try:
                if not secrets.compare_digest(self.headers.get("X-Review-Token", ""), store.token):
                    raise ReviewError("세션이 만료되었습니다. 페이지를 새로고침하세요.", 403)
                origin = self.headers.get("Origin")
                if origin and urlparse(origin).netloc != self.headers.get("Host"):
                    raise ReviewError("다른 사이트에서 보낸 저장 요청입니다.", 403)
                if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
                    raise ReviewError("JSON 요청만 지원합니다.", 415)
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 32*1024*1024:
                    raise ReviewError("저장 요청 크기가 올바르지 않습니다.", 413)
                parts = urlparse(self.path).path.strip("/").split("/")
                batch = parts == ["api", "frames", "reject"]
                sam_request = len(parts) == 4 and parts[:2] == ["api", "frame"] and parts[3] == "sam"
                if not batch and not sam_request and (len(parts) != 3 or parts[:2] != ["api", "frame"]):
                    raise ReviewError("찾을 수 없는 주소입니다.", 404)
                payload = json.loads(self.rfile.read(length))
                if sam_request:
                    if sam is None:
                        raise SamError("SAM이 연결되지 않았습니다.")
                    with store.lock:
                        frame = store.frame(parts[2])
                        w, h = int(frame["record"]["width"]), int(frame["record"]["height"])
                        points = validate_points(payload.get("points") if isinstance(payload,dict) else None, w, h)
                        image_path = inside(store.dataset_root, frame["source"]["image_file"])
                    return self.json(sam.predict(image_path, w, h, points))
                self.json(store.reject_many(payload) if batch else store.save(parts[2], payload))
            except SamError as error:
                self.json({"error":str(error)}, error.status)
            except ReviewError as error:
                self.json({"error": str(error)}, error.status)
            except (ValueError, TypeError, KeyError) as error:
                self.json({"error": str(error)}, 400)
            except OSError as error:
                self.json({"error": f"파일 저장 실패: {error}"}, 500)
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets"))
    parser.add_argument("--pseudo-root", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--sam-python", default=DEFAULT_PYTHON)
    parser.add_argument("--sam-checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--sam-config", default=DEFAULT_CONFIG)
    parser.add_argument("--sam-device", default="auto", help="auto, cpu, or logical CUDA device such as cuda:1")
    args = parser.parse_args()
    store = ReviewStore(args.dataset_root, args.pseudo_root or args.dataset_root / "pseudo_labels")
    # Optional handoff when restarting the local server: existing tabs can still
    # save unsaved edits before reloading to pick up the new UI.
    store.token = os.environ.pop("DY_REVIEW_SESSION_TOKEN", None) or store.token
    sam = SamService(args.sam_python, args.sam_checkpoint, args.sam_config, args.sam_device)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(store, sam))
    print(f"DY label review: http://{args.host}:{server.server_port} | {len(store.frames):,} frames", flush=True)
    print(f"Labels: {store.root}\n종료: Ctrl+C", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        sam.close()


if __name__ == "__main__":
    main()
