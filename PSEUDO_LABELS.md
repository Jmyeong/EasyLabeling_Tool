# 날짜별 데이터 로딩 및 pseudo label 생성

## 현재 범위

- 데이터: `datasets`, 원본 manifest의 train/test 분할 유지.
- 교사 모델: `checkpoints/best_joint.pt`.
- Detection: `0 Golfcart`, `1 Person`, `2 Tree`, `3 Undef_obj`.
- Segmentation: **도로/비도로 이진 분할**. 기존 모델에 다중 클래스/instance segmentation head가 없으므로 그 라벨은 생성하지 않습니다.
- 생성 도구는 학습을 실행하지 않습니다. 원본 이미지/라벨/manifest는 수정하지 않습니다.
- 생성 후 [라벨 검수 GUI](REVIEW_GUI.md)에서 마스크와 박스를 편집할 수 있습니다: `bash run_review_gui.sh`.

## 실행

```bash
cd /path/to/EasyLabeling_Tool
# 260916의 train + test 전체. 기본 GPU=0, batch=8.
bash run_pseudo_labels.sh

# 별도 폴더에 먼저 소량 생성
bash run_pseudo_labels.sh --limit 8 --output-root /tmp/dy_pseudo_preview

# 원하는 split, 임계값, checkpoint 선택
DEVICE=0 BATCH=8 CHECKPOINT="$PWD/checkpoints/best_road.pt" \
  bash run_pseudo_labels.sh --splits train --conf 0.15 \
  --output-root datasets/pseudo_labels_road_teacher
```

기본 출력은 `datasets/pseudo_labels`입니다. RGB/NIR만 읽어
기존 모델과 같은 352×640 letterbox로 추론한 후 padding을 제거하고 원본 RGB
좌표/해상도로 복원합니다. NIR 전처리도 기존 모델의 resize 방식을 따릅니다.
GT나 깊이값은 pseudo label 추론에 사용하지 않습니다.

동일 명령을 재실행하면 완료된 프레임은 보존하고 나머지만 처리합니다.
checkpoint/manifest SHA256, 입력 크기, 임계값 등 생성 설정이 다르면 같은 출력
폴더에 혼합하지 않고 오류를 냅니다. 설정 변경 실험에는 새 `--output-root`를
지정하세요. 완료 JSON이 있으나 파일이 누락된 경우에도 덮어쓰지 않고 오류를 냅니다.
동일 출력 폴더에서 동시 실행은 잠금으로 막습니다.

## 출력 형식

```text
pseudo_labels/
  run.json                       # checkpoint/manifest SHA256와 추론 설정
  model.json                     # 교사 epoch, architecture, input mode
  manifest.csv                   # 생성된 프레임 인덱스
  summary.json                   # 프레임/클래스/검수 상태 집계
  review_queue.csv               # 검수가 필요한 신호로 정렬한 목록
  260916/<train|test>/<sequence>/
    labels/<frame>.txt           # YOLO: class cx cy width height (정규화 좌표)
    road/<frame>.png             # uint8: 0 비도로, 1 도로, 255 불확실/ignore
    road_probability/<frame>.png # uint16 / 65535 = 도로 확률
    annotations/<frame>.json     # 원본 좌표 box, confidence, 경로, review_status
    previews/<frame>.jpg         # 녹색 도로, 주황 불확실, box 및 confidence
```

기본 detection confidence는 0.20입니다. 도로 임계값은 0.50이고 그 주변 ±0.10은
ignore 처리합니다 (`--road-threshold`, `--road-ignore-margin`으로 조절).
새로 생성하는 박스는 기본적으로 **같은 클래스끼리 IoU ≥ 0.90**이면 confidence가
높은 박스부터 유지합니다. confidence가 같으면 먼저 나온 박스를 유지하며,
클래스가 다르면 제거하지 않습니다. IoU는 교집합 면적 / 합집합 면적입니다.
`--box-dedup-iou 0.95`로 기준을 조절하거나 `--no-box-dedup`으로 끌 수 있습니다.
제거가 발생하면 JSON의 `raw_detections`에 필터 전 예측을 보관합니다.
`--road-ignore-margin 0`은 모든 픽셀에 이진 라벨을 부여합니다.
도로 픽셀(값 1)은 **8방향 연결 영역 중 면적이 큰 2개만** 남깁니다.
대각선으로 닿은 픽셀도 같은 영역이며, 나머지 도로 영역은 비도로(0)로 바꿉니다.
불확실 픽셀(255)과 큰 영역 내부의 구멍은 그대로 둡니다. 영역이 2개 이하면
변경하지 않습니다. `--road-max-blobs N`으로 개수를 바꾸고, `0`이면 끕니다.
면적에 따른 정리이므로 큰 오검출이나 누락 영역의 정답 여부까지 판별하지는 않습니다.
검출이 없는 이미지에는 **빈 TXT를 실제로 생성**합니다. 누락된 TXT와 구별됩니다.
낮은 confidence 검출, 검출 없음, 불확실 도로 비율은 review_queue 정렬 신호입니다.
이 신호가 모든 오류/누락을 찾아내거나 높은 confidence가 정답임을 보장하지는 않습니다.

경로 기준: JSON/CSV의 `image_file`, `nir_file`은 dataset root 기준이며,
`detection_file`, `road_file`, `annotation_file` 등 출력 경로는 pseudo root 기준입니다.

## 이미 생성한 박스 중복 제거

```bash
cd /path/to/EasyLabeling_Tool
# 읽기 전용 검사: 기본 날짜 260916, 같은 클래스 IoU ≥ 0.90
python filter_pseudo_boxes.py
# 원본 백업 후 적용
python filter_pseudo_boxes.py --apply
```

`--dates`, `--dataset-root`, `--pseudo-root`, `--iou`로 범위를 지정할 수 있습니다.
검수 상태가 pending이고 수정 이력이 없는 프레임만 처리합니다. 승인/제외된
프레임과 GUI에서 저장한 프레임, TXT가 교사 예측과 다른 프레임은 보존합니다.
세그멘테이션과 검수 상태는 변경하지 않습니다. 반복 실행해도 추가 중복이 없으면
라벨을 다시 쓰지 않습니다.

변경 전 TXT/JSON 및 인덱스는 pseudo root의 `_box_filter_history/<실행 시각>/files/`에
보관합니다. 같은 폴더의 `report.json`에는 제거 근거, 변경 전후 해시, 보존 검증
결과가 남습니다. 적용 중 GUI 저장은 잠금으로 막으며, 쓰기 실패 시 복구합니다.
기존 JSON의 `detections`와 preview는 최초 생성 결과로 보존하고, 제거 후 목록은
`filtered_detections`와 학습용 TXT에 저장합니다. 인덱스와 검수 목록도 갱신합니다.
GUI에서 **목록 새로고침** 후 프레임을 다시 열면 변경된 박스가 표시됩니다.

이 기능 추가 전 생성한 출력 폴더는 기존 생성 설정을 유지합니다. 기본 생성 명령을
그 폴더에 재실행하면 설정 불일치 오류가 납니다. 기존 결과 정리는 위 도구를 사용하고,
새 설정으로 생성할 때는 별도 `--output-root`를 지정하세요. 예전 설정으로 이어서
생성하려면 `--no-box-dedup --road-max-blobs 0`을 명시하고 이후 정리 도구를
사용할 수 있습니다. 박스 필터만 켜진 설정으로 생성한 폴더에는 `--road-max-blobs 0`을
사용하세요. 실행 당시 설정은 `run.json`에 기록됩니다.

## 이미 생성한 도로 영역 정리

```bash
cd /path/to/EasyLabeling_Tool
# 기본 260916, 도로 영역을 면적순 최대 2개 유지: 변경 없이 검사
python filter_pseudo_roads.py
# 원본 백업 후 적용
python filter_pseudo_roads.py --apply
```

`--max-blobs`, `--dates`, `--dataset-root`, `--pseudo-root`로 범위를 조절합니다.
검수 상태가 pending이고 GUI 수정 이력이 없는 프레임만 처리합니다. 최초 마스크는
보관된 확률 PNG와 생성 임계값을 비교하고, 이미 필터한 마스크는 기록된 해시를
비교하여 수동 변경이 발견되거나 출처를 검증할 수 없으면 보존합니다.
확률 PNG는 uint16 양자화에 따른 경계 오차를 허용해 비교합니다.

변경 전 마스크/JSON 및 인덱스는 `_road_filter_history/<실행 시각>/files/`에
백업하며, 같은 폴더의 `report.json`에 제거 영역 수·픽셀 수·파일 해시를 저장합니다.
JSON에는 `road_blob_filter`와 갱신된 `road_fraction`을 기록합니다. 박스와 확률
PNG, 검수 상태는 보존합니다. 적용 중 저장은 잠금으로 막고, 쓰기 실패 시 복구합니다.
반복 실행해도 조건을 만족하는 마스크는 다시 쓰지 않습니다.

GUI에서 **목록 새로고침 후 프레임을 다시 열면** 적용된 마스크가 표시됩니다.
기존 preview JPG는 생성 당시 결과를 보존합니다. 새로 생성하는 경우에는 필터된
마스크로 preview도 생성하며, 필터 전 확률 PNG를 함께 보관합니다.

## GUI 검수 및 연결 규약

`bash run_review_gui.sh`를 실행하고 `http://127.0.0.1:8765`에서 브러시/다각형으로
마스크를 수정하거나 박스를 추가·이동·크기 조절·삭제합니다. 저장 및 검수 승인,
자동 백업, 실행 취소를 지원합니다. 자세한 사용법은 [REVIEW_GUI.md](REVIEW_GUI.md)를
참고하세요. GUI의 ‘검수 현황 CSV’에는 현재 수정한 라벨 수와 검수 상태가 반영됩니다.

생성 직후 모든 프레임은 JSON의 `review_status: "pending"`입니다.
GUI는 원본 RGB와 `labels/*.txt`, `road/*.png`를 읽어 수정하고 저장한 다음 해당
JSON을 `"approved"`로 바꾸면 됩니다. 제외할 프레임은 `"rejected"`로 표시합니다.
TXT와 PNG가 학습 타깃의 원본이며, JSON의 detection 목록/confidence와 probability
PNG는 교사 모델의 최초 제안입니다. 사람이 수정한 결과를 이 최초 예측과 구분할 수 있습니다.
마스크는 팔레트 시각화 이미지가 아니라 값 0/1/255인 단일 채널 PNG로 저장하세요.
ignore 영역도 사람이 확인하면 0 또는 1로 수정할 수 있습니다.
검수 상태의 기준은 프레임별 JSON입니다. CSV는 인덱스 스냅샷이며 같은 생성
명령을 재실행하면 수정된 JSON 상태로 인덱스를 갱신합니다.
기존 preview는 생성 당시 결과입니다. GUI에서는 수정 라벨로 다시 그려야 합니다.

## 학습/평가 로더

활성 `train_yolo26_mtl.py`, `test_yolo26_mtl.py`와 실행 스크립트의 기본 데이터
경로를 새 데이터셋으로 바꿨습니다. 기존 raw/prepared manifest도 계속 지원합니다.
새 데이터셋의 260618/260724는 기존 semantic/instance mask에서 road/box를 얻습니다.
원본 mask가 세로로 두 장 붙은 형식이면 기존 변환 규칙대로 하단 mask를 사용합니다.

| 옵션 | 의미 |
|---|---|
| `--pseudo-root PATH` | 기본 `<dataset-root>/pseudo_labels` |
| `--pseudo-label-policy reviewed` | 기본값. 원본 라벨 + approved 프레임 |
| `--pseudo-label-policy all` | train에 pending도 포함. test는 approved만 |
| `--pseudo-label-policy exclude` | 원본 라벨만 사용 |
| `--depth-source completed` | 기본. train은 보완 깊이 포함, test는 source==1만 |
| `--depth-source filtered` | overlap filtering한 LiDAR 깊이 |
| `--depth-source raw` | 원본 sparse LiDAR 깊이 |

라벨이 없는 프레임과 pending/rejected 프레임 제외 수는 로더가 출력합니다.
사용 대상 라벨의 파일이 없거나 이미지가 다르면 오류를 냅니다. 승인 전 기본
로딩 수는 train 1,409장 / test 225장입니다. pending까지 명시적으로 사용하면
train 5,539장이고, test의 397장은 승인 전까지 평가에 포함되지 않습니다.
approved도 원래 pseudo label에서 출발한 데이터라는 출처를 보존해야 합니다.

```bash
# 검수 이후 실행할 학습 명령 (생성 도구는 학습을 자동 실행하지 않음)
bash run_train.sh
# 의도적으로 미검수 pseudo label을 학습에 포함하는 실험
bash run_train.sh --pseudo-label-policy all
# CPU 회귀 검사
python -m unittest test_dated_dataset -v
```
