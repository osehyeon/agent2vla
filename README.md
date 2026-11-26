# agent2vla

SmolVLA on an SO-101 arm, trained on demos from a collection script written by Claude.

```
pip install "lerobot[smolvla,feetech]==0.6.1" trimesh

python robot/collect.py collect 20     # record demos
python robot/merge_relabel.py          # merge and add instructions
sbatch train/smolvla_train.sbatch      # fine-tune
python robot/rollout.py --policy somae/smolvla-so101-cube --task "Move the cube to region 2."
```

Model: https://huggingface.co/somae/smolvla-so101-cube  
Video: https://youtu.be/xV32gTi1n2Q  
URDF: [TheRobotStudio/SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100) (Apache-2.0)
