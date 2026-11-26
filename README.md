# agent2vla

Claude wrote the data-collection script for my SO-101 arm. The script collected 100 demos without teleoperation, and I fine-tuned SmolVLA on them.

The task is to move a cube to region 1, 2 or 3 (blue, green, yellow) as the instruction says.

- Videos: [collection, 8x](https://youtu.be/1S-19LJiTYw), [SmolVLA, 2x](https://youtu.be/xV32gTi1n2Q)
- Model: [somae/smolvla-so101-cube](https://huggingface.co/somae/smolvla-so101-cube)

## Run

```
pip install "lerobot[smolvla,feetech]==0.6.1" trimesh

python robot/collect.py collect 20
python robot/merge_relabel.py
cd train && mkdir -p logs && STEPS=20000 sbatch smolvla_train.sbatch && cd ..
python robot/rollout.py --policy somae/smolvla-so101-cube --task "Put the cube on the green region."
python robot/trials.py --policy somae/smolvla-so101-cube --n 10
```

Ctrl+C stops a run, and the arm goes home and folds. Paths and the serial port are in `robot/config.py`.

`robot/ctl.py` is how Claude drove the arm directly, before the collection script existed.

`robot/so101.urdf` comes from [TheRobotStudio/SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100) (Apache-2.0).
