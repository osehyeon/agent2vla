# agent2vla

A frontier model (Claude) writes and debugs a scripted data collector for an SO-101 arm. The collector records demonstrations without teleoperation, and SmolVLA is fine-tuned on them to do the task on its own.

Task: move a cube to the region named in an instruction (region 1, 2, 3 = blue, green, yellow).

Videos: [scripted collection](https://youtu.be/1S-19LJiTYw), [SmolVLA on the robot](https://youtu.be/xV32gTi1n2Q)

## Layout

| Folder | Contents |
|---|---|
| `robot/` | Data collector, policy rollout, kinematics, cube and sheet detection |
| `train/` | SmolVLA fine-tuning job |
| `results/` | Instructions per episode, real-robot trials |

## Install

```
pip install "lerobot[smolvla,feetech,viz]==0.6.1" trimesh
```

Paths and the serial port are set in `robot/config.py` and can be overridden with `AGENT2VLA_WORK`, `AGENT2VLA_FOLLOWER_PORT` and `AGENT2VLA_FOLLOWER_ID`.

## Pipeline

0. `robot/ctl.py` is a command server for direct control: Claude reads the camera images and sends target poses. The collector grew out of it.
1. `robot/collect.py collect N` records LeRobotDataset episodes.
2. `robot/merge_relabel.py` merges them and assigns instructions.
3. `train/smolvla_train.sbatch` fine-tunes `lerobot/smolvla_base`.
4. `robot/rollout.py` runs the policy on the arm. Ctrl+C stops it; the arm then returns home and folds.

```
python robot/rollout.py --policy somae/smolvla-so101-cube --task "Put the cube on the green region." --display
```

5. `robot/trials.py` repeats rollouts and checks where the cube ended up.

`robot/so101.urdf` is `Simulation/SO101/so101_new_calib.urdf` from [TheRobotStudio/SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100) (Apache-2.0).
