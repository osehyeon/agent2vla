# 따라 하기

SO-101 팔 하나로 시연을 자동으로 모으고, SmolVLA를 학습해서 돌려 본다.

## 준비물

| 물건 | 비고 |
|---|---|
| SO-101 팔로워 팔, 12 V 전원 | 리더 팔은 필요 없음 |
| USB 카메라 2대 | 위에서 내려다보는 카메라 1대, 그리퍼에 붙인 손목 카메라 1대. 1280×720 |
| 3 cm 큐브 | 한 면이 빨강 |
| `robot/sheet_a4.pdf` 인쇄본 | A4 가로, 배율 100 % |
| Linux PC (NVIDIA GPU) 또는 Jetson Orin | 수집과 실행 |
| 학습용 GPU | 메모리 24 GB 이상 |

## 1. 설치

```
git clone https://github.com/osehyeon/agent2vla && cd agent2vla
python3.12 -m venv .venv && . .venv/bin/activate
pip install "lerobot[smolvla,feetech]==0.6.1" trimesh
sudo usermod -aG dialout,video $USER    # 한 번 로그아웃했다 들어오기
```

Jetson Orin(JetPack 7.2)은 기본 PyTorch가 Orin GPU를 지원하지 않는다. CUDA 13.2 판으로 바꾼다.

```
pip install "torch==2.12.1" "torchvision==0.27.1" --index-url https://download.pytorch.org/whl/cu132
```

## 2. 팔 캘리브레이션

```
ls /dev/serial/by-id/           # 팔 포트 확인
lerobot-calibrate --robot.type=so101_follower --robot.port=<포트> --robot.id=my_follower
```

화면의 안내대로 관절을 가운데 자세에 두고, 각 관절을 끝까지 한 번씩 움직인다.

## 3. 경로와 카메라

```
export AGENT2VLA_WORK=$HOME/agent2vla-work
export AGENT2VLA_FOLLOWER_PORT=<2번의 포트>
mkdir -p $AGENT2VLA_WORK
ls -l /dev/v4l/by-path/          # 카메라 경로 확인
```

`$AGENT2VLA_WORK/robot_cams.json`을 만든다. 두 카메라 중 어느 쪽이 위이고 어느 쪽이 손목인지는 한 장씩 찍어 보고 정한다.

```
{"top": "/dev/v4l/by-path/...-video-index0", "wrist": "/dev/v4l/by-path/...-video-index0"}
```

## 4. 종이 놓기

`robot/sheet_a4.pdf`를 배율 100 %로 인쇄한다. 부채꼴의 중심이 팔의 어깨 회전축에 오게 놓는다. 회전축은 종이 아래 변에서 90.8 mm 아래, 왼쪽 변에서 144.3 mm 오른쪽이다. 테이프로 고정한다.

위 카메라에 종이 네 귀퉁이의 마커가 모두 보여야 한다.

## 5. 위 카메라와 로봇 좌표 맞추기

```
python robot/calib_table.py
```

팔 토크가 꺼진 상태로 진행한다. 손으로 팔을 잡고 안내대로 한다.

1. 팔을 위 카메라 화면 밖으로 치운다
2. 그리퍼 끝을 마커 0~3의 가운데에 차례로 댄다
3. 팔을 접힌 자세로 돌려 놓는다

`$AGENT2VLA_WORK/homography.json`이 생긴다. 카메라나 종이를 옮기면 다시 한다.

## 6. 집는 위치 맞추기

큐브를 빨간 면이 위로 오게 아무 칸에나 놓는다.

```
python robot/collect.py calib
```

팔이 큐브를 몇 번 집어 보고, 손목 카메라에서 큐브가 있어야 할 위치를 저장한다.

## 7. 시연 모으기

```
python robot/collect.py collect 20
```

큐브를 다른 칸으로 옮기기를 20번 한다. 5번마다 저장한다. 멈출 때는 Ctrl+C를 한 번 누른다. 팔이 접힐 때까지 기다린다. 큐브가 칸 밖으로 나가면 멈추니 다시 놓고 같은 명령을 실행한다.

100개가 될 때까지 반복한 뒤 하나로 합친다.

```
python robot/merge_relabel.py
```

결과는 `$AGENT2VLA_WORK/collect/merged/so101_cube_regions_v1`이다.

## 8. 학습

```
lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --dataset.repo_id=local/so101_cube_regions_v1 \
  --dataset.root=$AGENT2VLA_WORK/collect/merged/so101_cube_regions_v1 \
  --rename_map='{"observation.images.top": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2"}' \
  --batch_size=32 --steps=20000 --save_freq=5000 \
  --output_dir=outputs/smolvla_cube --policy.push_to_hub=false
```

Slurm 클러스터라면 `train/smolvla_train.sbatch`를 쓴다.

## 9. 실행

학습한 체크포인트나 공개 모델로 돌린다.

```
python robot/rollout.py --policy somae/smolvla-so101-cube --task "Move the cube to region 2."
python robot/trials.py --policy somae/smolvla-so101-cube --n 10
```

Ctrl+C를 누르면 정책이 멈추고 팔이 HOME으로 돌아가 접힌다. 이상하면 12 V 전원을 뽑는다.
