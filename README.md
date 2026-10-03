# EasyLabeling Tool

기존 RGB/NIR YOLO26 멀티태스크 모델의 사전학습 가중치로 detection·도로 segmentation
pseudo label을 생성하고, 브라우저 GUI에서 사람이 수정·승인하는 도구입니다.

## 포함 기능

- RGB/NIR 추론 결과를 원본 이미지 좌표로 복원하여 YOLO TXT, 도로 PNG, 확률 PNG,
  confidence·출처·검수 상태 JSON, 미리보기와 검수 목록을 저장합니다.
- 같은 클래스의 IoU ≥ 0.90 박스는 confidence가 높은 박스를 유지합니다.
- 도로는 8방향 연결 영역 중 가장 큰 2개만 유지합니다. 불확실 영역(255)은 보존합니다.
- GUI에서 박스 추가·이동·크기·클래스 수정, 브러시·다각형 마스크 편집, 실행 취소·재실행,
  RGB/NIR 보기, 확대·이동을 지원합니다.
- SAM 2.1 포함점/제외점으로 도로 추가·삭제·교체를 미리 보고 적용할 수 있습니다.
- Shift 범위 선택, Ctrl/⌘ 개별 선택으로 프레임을 일괄 제외하고,
  Ctrl/⌘+S로 저장·승인합니다. 백업, 동시 편집 충돌 검사, 중단된 저장 복구를 지원합니다.

현재 클래스는 `Golfcart`, `Person`, `Tree`, `Undef_obj`이며, segmentation은
도로/비도로 이진 분할입니다. 임의의 YOLO 가중치가 아니라 이 저장소의
`train_yolo26_mtl.py` 모델 구조로 학습한 joint checkpoint를 사용합니다.
VLM 기반 자동 보정 실험은 포함하지 않습니다.

## 설치

Linux 또는 WSL, Python 3.11 환경을 기준으로 합니다. 파일 잠금과 SAM worker에
POSIX 기능을 사용하므로 Windows에서는 WSL에서 서버를 실행하세요.

```bash
git clone https://github.com/Jmyeong/EasyLabeling_Tool.git
cd EasyLabeling_Tool
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

GPU 추론은 GPU와 드라이버에 맞는 PyTorch 빌드가 필요합니다.
검증 환경은 Python 3.11, PyTorch 2.7.0+cu128, Ultralytics 8.4.118입니다.
이미 생성된 라벨의 GUI 편집과 후처리만 사용하면 `requirements-gui.txt`만 설치해도 됩니다.
전체 회귀 테스트는 `requirements.txt` 환경에서 실행합니다.

데이터, 모델 가중치, 생성 결과, 편집 백업은 저장소에 포함하지 않습니다.
아래 가중치를 직접 준비하거나 명령행 옵션으로 기존 위치를 지정하세요.

```text
checkpoints/
  best_joint.pt           # 학습된 RGB/NIR joint checkpoint
  yolo26s.pt              # 해당 checkpoint와 호환되는 YOLO26 detector template
  sam2.1_hiera_large.pt   # 선택 사항: GUI SAM 보정
```

## 1. 사전학습 가중치로 pseudo label 생성

데이터 루트의 `manifest.csv`를 읽습니다. 이미 사용 중인 DY datasets manifest를
그대로 지정할 수 있습니다. 필수 열과 경로 규약은 아래 데이터 형식을 참고하세요.

```bash
DATA_ROOT=/path/to/datasets DEVICE=0 \
  bash run_pseudo_labels.sh \
  --checkpoint /path/to/best_joint.pt \
  --model-template /path/to/yolo26s.pt \
  --dates 260916 --splits train test

# 먼저 8장만 별도 폴더에 생성
DATA_ROOT=/path/to/datasets DEVICE=cpu \
  bash run_pseudo_labels.sh \
  --checkpoint /path/to/best_joint.pt \
  --model-template /path/to/yolo26s.pt \
  --limit 8 --output-root /tmp/pseudo_preview
```

기본 출력은 `<dataset-root>/pseudo_labels`입니다. GT·깊이 없이 RGB/NIR만 사용하며
학습을 실행하지 않습니다. 기본 confidence는 0.20, 도로 임계값은 0.50,
불확실 범위는 ±0.10입니다. 생성 설정과 checkpoint 해시를 기록하고, 같은 설정으로
재실행하면 완료된 프레임을 보존합니다. 설정이 다른 경우 새 출력 폴더를 사용하세요.

기본 모델은 `gated` RGB/NIR 모델입니다. 과거 `hsvnet` checkpoint는 별도
Pixel_aligned_RGB_NIR_Stereo 소스·가중치가 필요하며 `--fusion-repo`,
`--fusion-checkpoint`로 지정합니다. 이 외부 저장소는 포함하지 않습니다.

## 2. GUI에서 편집·승인

```bash
DATA_ROOT=/path/to/datasets bash run_review_gui.sh
# 별도 위치에 생성한 결과
DATA_ROOT=/path/to/datasets bash run_review_gui.sh --pseudo-root /tmp/pseudo_preview
```

브라우저에서 **http://127.0.0.1:8765/** 를 엽니다. 서버 프로세스는 실행 상태로
유지해야 합니다. 원격 서버라면 브라우저를 여는 PC에서 포트를 연결하세요.

```bash
ssh -N -L 8765:127.0.0.1:8765 USER@SERVER
```

박스와 마스크를 수정한 뒤 **Ctrl+S → 저장·승인**합니다. 임시 저장은 `pending`,
승인은 `approved`, 학습 제외는 `rejected`로 기록합니다.
**Shift+클릭**으로 연속 프레임을 선택한 뒤 한 번에 제외할 수 있습니다.

선택 사항인 SAM 보정은 [SAM 2](https://github.com/facebookresearch/sam2)가 설치된
Python 환경과 SAM 2.1 checkpoint가 필요합니다.

```bash
DATA_ROOT=/path/to/datasets bash run_review_gui.sh \
  --sam-python /path/to/sam-env/bin/python \
  --sam-checkpoint /path/to/sam2.1_hiera_large.pt \
  --sam-config configs/sam2.1/sam2.1_hiera_l.yaml \
  --sam-device auto
```

포함점/제외점 → SAM 미리보기 → 추가/지우기/전체 교체 → 미리보기 적용 → Ctrl+S
순으로 사용합니다. SAM은 처음 요청할 때 로드하며, 없어도 일반 GUI 편집은 동작합니다.

## 이미 생성한 라벨 후처리

```bash
# --apply 없이 실행하면 변경 대상만 검사합니다.
python filter_pseudo_boxes.py --dataset-root /path/to/datasets --dates 260916
python filter_pseudo_roads.py --dataset-root /path/to/datasets --dates 260916

# 백업 후 적용
python filter_pseudo_boxes.py --dataset-root /path/to/datasets --dates 260916 --apply
python filter_pseudo_roads.py --dataset-root /path/to/datasets --dates 260916 --apply
```

미검수·미수정 결과만 처리하고 수정·승인·제외 프레임은 보존합니다.
박스는 `--iou`, 도로는 `--max-blobs`로 기준을 조절할 수 있습니다.
변경 전 파일과 보고서는 `_box_filter_history` 또는 `_road_filter_history`에 보관합니다.
GUI에서 **목록 새로고침 후 프레임을 다시 열면** 적용 결과를 확인할 수 있습니다.

## 데이터 형식

생성용 `manifest.csv`의 최소 열은 다음과 같습니다. 경로는 dataset root 기준이며,
라벨이 없는 프레임의 `semseg_file`, `instseg_file`은 빈 문자열로 둡니다.

```csv
date,split,sequence,image_file,nir_file,image_sha256,semseg_file,instseg_file
260916,train,sequence_01,260916/train/sequence_01/rgb/000001.png,260916/train/sequence_01/nir/000001.png,<RGB 파일 SHA256>,,
```

실제 RGB 파일 해시를 넣고 manifest와 원본 이미지를 함께 관리하세요.
GUI는 manifest와 annotation의 이미지 식별자·해시가 일치하는지 확인합니다.
원본 이미지와 NIR가 모두 있어야 하며, 출력 좌표는 원본 RGB 기준입니다.
학습 로더를 함께 쓸 때만 `depth_file`, `filtered_depth_file`, `completed_depth_file`
등 깊이 데이터 열이 추가로 필요합니다.

```text
pseudo_labels/<date>/<split>/<sequence>/
  labels/<frame>.txt            # YOLO: class cx cy width height, 정규화 좌표
  road/<frame>.png              # uint8: 0 비도로 / 1 도로 / 255 불확실
  road_probability/<frame>.png  # uint16: P(road) × 65535
  annotations/<frame>.json      # confidence, 출처, 검수 상태, 후처리 정보
  previews/<frame>.jpg          # 생성 당시 preview; GUI는 현재 라벨로 다시 그림
```

프레임은 `date/split/sequence/image stem`으로 구분하므로 이 조합이 고유해야 합니다.
수정된 TXT/PNG가 학습용 라벨이고, JSON의 교사 예측과 확률 PNG는 최초 제안으로
보존합니다. 후처리 전후 예측의 구분은 [생성·후처리 문서](PSEUDO_LABELS.md)를 참고하세요.

## 검증 및 코드 구성

```bash
python -m unittest test_dated_dataset test_review_gui test_review_sam test_box_filter test_road_filter -v
```

테스트는 임시 데이터만 수정합니다. 저장·복구·동시 편집 충돌, 일괄 제외,
SAM 요청 검증, 박스 중복 제거, 도로 blob 정리, 학습 로더 연동을 검사합니다.

- `generate_pseudo_labels.py`: checkpoint 추론과 라벨 생성
- `review_gui.py`, `review_gui_web/`: 로컬 HTTP 서버와 GUI
- `review_sam.py`, `review_sam_worker.py`: 별도 Python worker를 통한 SAM 보정
- `filter_pseudo_boxes.py`, `filter_pseudo_roads.py`: 기존 결과의 안전한 후처리
- `train*.py`, `test.py`, `test_yolo26_mtl.py`, `visualize_yolo_test.py`: 기존 checkpoint와
  호환되는 모델 정의·전처리·로더 및 추론 유틸리티. 생성기 import에 필요한 코드도 포함합니다.
- [PSEUDO_LABELS.md](PSEUDO_LABELS.md): 생성 설정, 저장 규약, 후처리와 학습 연결
- [REVIEW_GUI.md](REVIEW_GUI.md): 편집 도구, 단축키, SAM, 백업과 복구
