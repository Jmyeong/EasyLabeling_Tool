#!/usr/bin/env python3
"""SAM-only worker: JSON lines on stdin/stdout, diagnostics on stderr; no writes."""
import argparse
import base64
from contextlib import nullcontext, redirect_stdout
import json
from pathlib import Path
import select
import sys
import time
import traceback


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--idle-seconds', type=int, default=600)
    args = parser.parse_args()
    predictor, image_key, device = None, None, None
    while select.select([sys.stdin], [], [], args.idle_seconds)[0]:
        line = sys.stdin.readline()
        if not line:
            break
        try:
            request = json.loads(line)
            started = time.perf_counter()
            with redirect_stdout(sys.stderr):
                import numpy as np
                from PIL import Image
                import torch
                from sam2.build_sam import build_sam2
                from sam2.sam2_image_predictor import SAM2ImagePredictor
                from review_sam import validate_points
                if predictor is None:
                    torch.set_num_threads(4)
                    device = args.device
                    if device == 'auto':
                        if torch.cuda.is_available():
                            free = [torch.cuda.mem_get_info(i)[0] for i in range(torch.cuda.device_count())]
                            index = max(range(len(free)), key=free.__getitem__)
                            if free[index] < 3*1024**3:
                                raise RuntimeError('SAM에 사용할 GPU 메모리가 부족합니다. 여유 GPU를 지정하세요.')
                            device = f'cuda:{index}'
                        else:
                            device = 'cpu'
                    predictor = SAM2ImagePredictor(build_sam2(args.config, args.checkpoint, device=device, apply_postprocessing=False))
                    print(f'GUI SAM loaded on {device}', file=sys.stderr, flush=True)
                path = Path(request['image_path'])
                stat = path.stat()
                key = (str(path.resolve()),stat.st_mtime_ns,stat.st_size,request['width'],request['height'])
                points = validate_points(request['points'],request['width'],request['height'])
                coords = np.array([[p['x'],p['y']] for p in points], dtype=np.float32)
                labels = np.array([p['label'] for p in points], dtype=np.int32)
                cached = key == image_key
                # Explicit device context avoids initializing autocast on a busy
                # default GPU when the auto-selected device has a different index.
                cuda = str(device).startswith('cuda')
                with torch.inference_mode(), (torch.cuda.device(device) if cuda else nullcontext()), (torch.autocast('cuda',dtype=torch.bfloat16) if cuda else nullcontext()):
                    if not cached:
                        with Image.open(path) as image:
                            if image.size != (request['width'],request['height']):
                                raise ValueError('원본 RGB 해상도가 annotation과 다릅니다.')
                            predictor.set_image(np.array(image.convert('RGB'),copy=True))
                        image_key = key
                    masks, scores, _ = predictor.predict(point_coords=coords,point_labels=labels,multimask_output=True)
                masks = masks.astype(bool)
                pixels = np.rint(coords).astype(int)
                candidates = []
                for i,(mask,score) in enumerate(zip(masks,scores)):
                    if not np.isfinite(score):
                        raise ValueError('SAM 점수가 유효하지 않습니다.')
                    at_points = mask[pixels[:,1],pixels[:,0]]
                    misses = int(((labels==1) & ~at_points).sum())
                    leaks = int(((labels==0) & at_points).sum())
                    candidates.append({'index':i,'prompt_violations':misses+leaks,'sam_score':float(score)})
                best = min(candidates,key=lambda c:(c['prompt_violations'],-c['sam_score']))
                result = {'width':request['width'],'height':request['height'],
                          'mask':base64.b64encode(masks[best['index']].astype(np.uint8).tobytes()).decode(),
                          'prompt_violations':best['prompt_violations'], 'candidates':candidates,
                          'cached_image':cached,'device':str(device),'elapsed_seconds':time.perf_counter()-started}
            print(json.dumps(result,allow_nan=False),flush=True)
        except Exception as error:
            traceback.print_exc(file=sys.stderr)
            print(json.dumps({'error':str(error)},ensure_ascii=False),flush=True)
            break  # Parent restarts after errors; release GPU memory.


if __name__ == '__main__':
    main()
