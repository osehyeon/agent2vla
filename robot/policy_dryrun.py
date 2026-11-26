import sys, time, json, torch, numpy as np, cv2
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.factory import make_pre_post_processors
CK = sys.argv[1]
dev = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
t0 = time.time()
pol = SmolVLAPolicy.from_pretrained(CK).to(dev).eval()
pre, post = make_pre_post_processors(pol.config, CK, preprocessor_overrides={"device_processor": {"device": dev}})
print("load s", round(time.time() - t0, 1), "device", dev, "image feats", list(pol.config.image_features), flush=True)
def frame(path):
    im = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB); im = cv2.resize(im, (640, 360))
    return torch.from_numpy(im).permute(2, 0, 1).float() / 255.0
obs = {"observation.images.camera1": frame(sys.argv[2]), "observation.images.camera2": frame(sys.argv[3]),
       "observation.state": torch.tensor([0, -64, 65, 0, 0, 3], dtype=torch.float32), "task": "Move the cube to region 2."}
for k in range(3):
    pol.reset(); t = time.time()
    with torch.no_grad():
        b = pre(obs); a = pol.predict_action_chunk(b) if hasattr(pol, "predict_action_chunk") else pol.select_action(b)
        if dev == "mps": torch.mps.synchronize()
    a = post(a)
    print(f"infer {k}: {time.time() - t:.2f}s shape {tuple(a.shape)}", flush=True)
print("first action", np.round(a.reshape(-1, a.shape[-1])[0].cpu().numpy(), 1).tolist())
