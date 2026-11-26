import glob, json, os, random, shutil, sys
from pathlib import Path
from collections import Counter
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.dataset_tools import merge_datasets, modify_tasks
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import WORK
C = WORK / "collect"; OUT = C / "merged" / "so101_cube_regions_v1"
roots = sorted(glob.glob(str(C / "datasets" / "cube_*")))
dss = [LeRobotDataset(repo_id="local/" + Path(r).name, root=r, video_backend="pyav") for r in roots]
print("chunks", len(dss), "episodes", sum(d.meta.total_episodes for d in dss))
if OUT.exists(): shutil.rmtree(OUT)
m = merge_datasets(dss, output_repo_id="local/so101_cube_regions_v1", output_dir=OUT)
print("merged episodes", m.meta.total_episodes, "frames", m.meta.total_frames)
COL = {1: "blue", 2: "green", 3: "yellow"}
T = [("number", "Move the cube to region {n}."), ("number", "Put the cube in region {n}."),
     ("color", "Move the cube to the {c} area."), ("color", "Put the cube on the {c} region."),
     ("both", "Place the cube in region {n}, the {c} one."), ("both", "Move the cube to the {c} region {n}.")]
rng = random.Random(0); ep_tasks = {}; info = {}
for ep in range(m.meta.total_episodes):
    t = m.meta.episodes[ep]["tasks"][0]; n = int(t.strip(".").split()[-1])
    kind, tmpl = T[ep % len(T)] if False else rng.choice(T)
    ep_tasks[ep] = tmpl.format(n=n, c=COL[n]); info[ep] = {"region": n, "kind": kind, "task": ep_tasks[ep]}
modify_tasks(m, episode_tasks=ep_tasks)
json.dump(info, open(OUT / "task_labels.json", "w"), indent=1)
print(Counter(v["kind"] for v in info.values()), Counter(v["region"] for v in info.values()))
m2 = LeRobotDataset(repo_id="local/so101_cube_regions_v1", root=OUT, video_backend="pyav")
it = m2[len(m2) // 2]; print("check:", m2.meta.total_episodes, m2.meta.total_tasks, "tasks |", it["task"], tuple(it["observation.images.top"].shape))
