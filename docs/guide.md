# 따라 하기

Jetson Orin에 팔, 카메라, 실행 환경, 학습한 모델이 준비되어 있다. `ai` 계정으로 접속해서 쓴다.

## 접속

```
ssh ai@<Orin 주소>
. /ssd/agent2vla/env.sh
```

`env.sh`가 실행 환경을 켜고 코드 폴더(`/ssd/agent2vla/code`)로 옮긴다. 결과는 `/ssd/agent2vla/work/<계정>`에 쌓인다. 접속할 때마다 `env.sh`부터 실행한다.

팔은 한 번에 한 프로그램만 쓸 수 있다. 다른 사람이 쓰고 있으면 "The robot is in use"가 나온다.

## 큐브 놓기

빨간 면이 위로 오게, 1~3번 칸 중 한 칸의 가운데에 놓는다.

## 학습한 모델 돌리기

```
python robot/rollout.py --policy $POLICY --task "Move the cube to region 2."
```

팔이 HOME으로 간 뒤 25초 동안 지시를 따르고, 돌아와 접힌다. 지시문은 바꿔도 된다.

| 종류 | 예 |
|---|---|
| 숫자 | Move the cube to region 3. |
| 색 | Put the cube on the yellow region. |
| 숫자와 색 | Place the cube in region 1, the blue one. |

여러 번 이어서 돌리고 성공을 세려면

```
python robot/trials.py --policy $POLICY --n 10
```

큐브가 놓인 칸과 다른 칸으로 옮기라고 매번 바꿔 지시한다. 결과는 `/ssd/agent2vla/work/<계정>/rollouts/<이름>_results.jsonl`에 남는다.

## 시연 모으기

```
python robot/collect.py collect 20
```

스크립트가 큐브를 다른 칸으로 옮기기를 20번 하며 녹화한다. 5번마다 저장한다. 큐브가 칸 밖으로 나가면 멈추니 다시 놓고 같은 명령을 실행한다.

다 모았으면 하나로 합친다.

```
python robot/merge_relabel.py
```

결과는 `/ssd/agent2vla/work/<계정>/collect/merged/so101_cube_regions_v1`이다. 학습은 담당자에게 이 폴더를 알려 준다.

## 멈추기

- 터미널에서 Ctrl+C를 한 번 누른다. 팔이 돌아와 접힐 때까지 기다린다
- 위험하면 팔의 12 V 전원을 뽑는다

## 처음 세팅 (담당자)

새 책상이나 다른 기계에서 처음 세팅할 때 한다.

1. 설치: `pip install "lerobot[smolvla,feetech]==0.6.1" trimesh`. Jetson Orin(JetPack 7.2)은 이어서 `pip install "torch==2.12.1" "torchvision==0.27.1" --index-url https://download.pytorch.org/whl/cu132`
2. 팔 캘리브레이션: `lerobot-calibrate --robot.type=so101_follower --robot.port=<포트> --robot.id=my_follower`. 역기구학의 기준 자세(`kin.py`)는 이 결과에서 자동으로 계산된다
3. 카메라: `/dev/v4l/by-path/`에서 두 카메라 경로를 찾아 `$AGENT2VLA_WORK/robot_cams.json`에 `{"top": ..., "wrist": ...}`로 적는다
4. 종이: `robot/sheet_a4.pdf`를 배율 100 %로 인쇄한다. 부채꼴 중심이 어깨 회전축에 오게 놓는다. 회전축은 종이 아래 변에서 90.8 mm 아래, 왼쪽 변에서 144.3 mm 오른쪽이다
5. 위 카메라와 로봇 좌표: `python robot/calib_table.py`
6. 손목 카메라 방향: 큐브를 빨간 면이 위로 오게 놓고 `python robot/calib_wrist.py`
7. 집는 위치: 큐브를 놓고 `python robot/collect.py calib`
8. 학습: `train/smolvla_train.sbatch` 또는 `lerobot-train` (설정은 그 파일 참고)
