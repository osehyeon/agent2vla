"""Web page to start a policy run and stop it. usage: python gui.py [PORT], then open http://localhost:PORT"""
import html, json, os, subprocess, sys, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import HERE, WORK

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
CKPT = WORK / "ckpt"; LOGS = WORK / "gui"; LOGS.mkdir(parents=True, exist_ok=True)
TASKS = ["Move the cube to region 1.", "Put the cube on the green region.", "Place the cube in region 3, the yellow one.",
         "Bring the cube to the blue zone."]
run = {"p": None, "tag": None, "log": None}


def running(): return run["p"] is not None and run["p"].poll() is None


def page():
    cks = sorted(p.name for p in CKPT.iterdir() if (p / "model.safetensors").exists())
    tail = open(run["log"]).read().splitlines()[-15:] if run["log"] and os.path.exists(run["log"]) else []
    img = ""
    if run["tag"]:
        d = WORK / "rollouts" / run["tag"]; shots = sorted(d.glob("top_*.jpg")) + sorted(d.glob("final_top.jpg"))
        if shots: img = f'<img src="/img?p={shots[-1].relative_to(WORK)}&t={time.time()}" width="640">'
    opts = "".join(f"<option{' selected' if c.endswith('20k') else ''}>{c}</option>" for c in cks)
    tasks = "".join(f"<option>{html.escape(t)}</option>" for t in TASKS)
    state = f"running ({run['tag']})" if running() else "idle"
    return f"""<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="2"><title>agent2vla</title>
<body style="font-family:sans-serif;max-width:720px;margin:24px auto">
<h2>SmolVLA on SO-101</h2><p>State: <b>{state}</b></p>
<form action="/start"><p>Checkpoint <select name="ck">{opts}</select></p>
<p>Instruction <input name="task" list="tasks" size="48" value="{html.escape(TASKS[1])}"><datalist id="tasks">{tasks}</datalist></p>
<p>Seconds <input name="secs" value="25" size="4"></p>
<button style="font-size:20px;padding:8px 24px" {'disabled' if running() else ''}>Start</button></form>
<form action="/stop"><button style="font-size:20px;padding:8px 24px;background:#B51700;color:#fff">Stop</button></form>
<p>{img}</p><pre style="background:#eee;padding:8px">{html.escape(chr(10).join(tail))}</pre></body>"""


class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass

    def send(self, body, ctype="text/html; charset=utf-8", code=200):
        self.send_response(code); self.send_header("Content-Type", ctype); self.end_headers(); self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path); q = {k: v[0] for k, v in parse_qs(u.query).items()}
        if u.path == "/start" and not running():
            ck = CKPT / os.path.basename(q.get("ck", "")); task = q.get("task", TASKS[0]); secs = str(float(q.get("secs", 25)))
            run["tag"] = "gui_" + time.strftime("%m%d_%H%M%S"); run["log"] = str(LOGS / f"{run['tag']}.log")
            (WORK / "STOP").unlink(missing_ok=True)
            run["p"] = subprocess.Popen([sys.executable, str(HERE / "rollout.py"), str(ck), task, secs, run["tag"]],
                                        cwd=WORK, stdout=open(run["log"], "w"), stderr=subprocess.STDOUT)
            json.dump({"tag": run["tag"], "ck": ck.name, "task": task, "secs": secs}, open(LOGS / f"{run['tag']}.json", "w"))
        elif u.path == "/stop" and running():
            (WORK / "STOP").touch()
        elif u.path == "/img":
            p = (WORK / q.get("p", "")).resolve()
            if WORK.resolve() in p.parents and p.suffix == ".jpg" and p.exists(): return self.send(p.read_bytes(), "image/jpeg")
            return self.send(b"", code=404)
        if u.path in ("/start", "/stop"):
            self.send_response(303); self.send_header("Location", "/"); self.end_headers(); return
        self.send(page().encode())


print(f"open http://localhost:{PORT}", flush=True)
ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()
