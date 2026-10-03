"""Read-only SAM requests through a lazy, isolated local Python worker."""
import json
import math
import os
from pathlib import Path
import select
import subprocess
import sys
import threading
import time

DEFAULT_PYTHON = os.environ.get('SAM_PYTHON', sys.executable)
DEFAULT_CHECKPOINT = os.environ.get('SAM_CHECKPOINT', str(Path(__file__).resolve().parent / 'checkpoints/sam2.1_hiera_large.pt'))
DEFAULT_CONFIG = 'configs/sam2.1/sam2.1_hiera_l.yaml'


class SamError(Exception):
    def __init__(self, message, status=503):
        super().__init__(message)
        self.status = status


def validate_points(value, width, height):
    if not isinstance(value, list) or not 1 <= len(value) <= 64:
        raise SamError('SAM 점은 1~64개로 지정하세요.', 400)
    result, seen = [], {}
    for point in value:
        if not isinstance(point, dict) or type(point.get('label')) is not int or point['label'] not in (0,1):
            raise SamError('SAM 점의 label은 포함 1 또는 제외 0이어야 합니다.', 400)
        x, y = point.get('x'), point.get('y')
        if not all(type(v) in (int,float) and math.isfinite(v) for v in (x,y)) or not (0 <= x <= width-1 and 0 <= y <= height-1):
            raise SamError('SAM 점은 원본 이미지 안의 유효한 좌표여야 합니다.', 400)
        pixel = (round(x), round(y))
        if pixel in seen:
            if seen[pixel] != point['label']:
                raise SamError('같은 위치에 포함점과 제외점을 함께 지정할 수 없습니다.', 400)
            continue
        seen[pixel] = point['label']
        result.append({'x':float(x), 'y':float(y), 'label':point['label']})
    if not any(p['label'] == 1 for p in result):
        raise SamError('분할할 영역 안에 포함점(+)을 하나 이상 찍으세요.', 400)
    return result


class SamService:
    def __init__(self, python=DEFAULT_PYTHON, checkpoint=DEFAULT_CHECKPOINT,
                 config=DEFAULT_CONFIG, device='auto', timeout=120, idle_seconds=600):
        self.python, self.checkpoint, self.config, self.device = str(python), str(checkpoint), config, device
        self.timeout, self.idle_seconds = timeout, idle_seconds
        self.process = None
        self.lock = threading.Lock()

    def info(self):
        enabled = Path(self.python).is_file() and Path(self.checkpoint).is_file()
        return {'available':enabled, 'model':'SAM 2.1',
                'message':'포함점과 제외점으로 영역을 선택하세요.' if enabled else 'SAM 실행 환경 또는 체크포인트를 찾을 수 없습니다.'}

    def close(self):
        process, self.process = self.process, None
        if process is not None:
            if process.poll() is None:
                process.terminate()
                try: process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill(); process.wait()
            for stream in (process.stdin, process.stdout):
                stream.close()

    def predict(self, image_path, width, height, points):
        points = validate_points(points, width, height)
        if not self.lock.acquire(blocking=False):
            raise SamError('다른 SAM 요청을 처리 중입니다. 잠시 후 다시 실행하세요.', 409)
        try:
            if not self.info()['available']:
                raise SamError(self.info()['message'])
            if self.process is None or self.process.poll() is not None:
                self.close()
                env = os.environ.copy()
                env.update(OMP_NUM_THREADS='4', TOKENIZERS_PARALLELISM='false')
                self.process = subprocess.Popen([
                    self.python, '-u', str(Path(__file__).with_name('review_sam_worker.py')),
                    '--checkpoint', self.checkpoint, '--config', self.config, '--device', self.device,
                    '--idle-seconds', str(self.idle_seconds),
                ], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=None, env=env, bufsize=0)
            message = {'image_path':str(image_path), 'width':width, 'height':height, 'points':points}
            self.process.stdin.write((json.dumps(message)+'\n').encode())
            self.process.stdin.flush()
            deadline = time.monotonic()+self.timeout
            data = bytearray()
            while not data.endswith(b'\n'):
                remaining = deadline-time.monotonic()
                if remaining <= 0 or not select.select([self.process.stdout], [], [], remaining)[0]:
                    raise TimeoutError('SAM 응답 시간이 초과되었습니다. 다시 실행하세요.')
                chunk = os.read(self.process.stdout.fileno(), 65536)
                if not chunk:
                    raise RuntimeError('SAM 실행이 종료되었습니다. 서버 로그를 확인한 뒤 다시 실행하세요.')
                data.extend(chunk)
                if len(data) > 32*1024*1024:
                    raise RuntimeError('SAM 응답 크기가 올바르지 않습니다.')
            result = json.loads(data)
            if 'error' in result:
                raise RuntimeError(result['error'])
            if result.get('width') != width or result.get('height') != height:
                raise RuntimeError('SAM 결과의 해상도가 원본과 다릅니다.')
            return result
        except (OSError, RuntimeError, ValueError) as error:
            self.close()
            raise SamError(str(error)) from error
        finally:
            self.lock.release()
