import json, time, requests, sys
from concurrent.futures import ThreadPoolExecutor

LOG=open("_bench_result.txt","w",encoding="utf-8")
def log(*a):
    s=" ".join(str(x) for x in a); LOG.write(s+"\n"); LOG.flush(); print(s)

import io, contextlib
nb=json.load(open("feature_extraction.ipynb",encoding="utf-8"))
cc=["".join(c["source"]) for c in nb["cells"] if c["cell_type"]=="code"]
ns={}
with contextlib.redirect_stdout(io.StringIO()):   # nuốt output ồn của cell 0,1
    for i in (0,1): exec(cc[i],ns)
    exec(cc[7].split("# Test thử")[0], ns)
df=ns["df"]; SYS=ns["SYSTEM_PROMPT"]; bp=ns["build_prompt"]
H=ns["LLM_HEADERS"]; M=ns["LLM_MODEL"]; SERVERS=ns["LLM_SERVERS"]

qids=[7,9,10,11,22,8,12,45]
prompts=[bp(df[df.question_id==q].iloc[0]) for q in qids if (df.question_id==q).any()]
MAXTOK=768

def one(server, prompt):
    p={"model":M,"messages":[{"role":"system","content":SYS},{"role":"user","content":prompt}],
       "temperature":0.0,"max_tokens":MAXTOK}
    t0=time.time()
    try:
        j=requests.post(f"{server}/chat/completions",headers=H,json=p,timeout=300).json()
        return time.time()-t0, j.get("usage",{}).get("completion_tokens",0)
    except Exception as e:
        return time.time()-t0, 0

def sweep(server, levels):
    log(f"\n=== Server: {server} ===")
    for c in levels:
        nreq=2*c
        batch=[prompts[i%len(prompts)] for i in range(nreq)]
        t0=time.time()
        with ThreadPoolExecutor(max_workers=c) as ex:
            res=list(ex.map(lambda pr: one(server,pr), batch))
        dt=time.time()-t0; toks=sum(r[1] for r in res); lat=sum(r[0] for r in res)/len(res)
        log(f"  concurrency={c:2d} | {nreq} req in {dt:5.1f}s | {toks/dt:6.1f} tok/s | "
            f"{nreq/dt:5.2f} req/s | avg latency {lat:5.1f}s")

LEVELS=[1,2,4,8]
for s in SERVERS:
    sweep(s, LEVELS)

log("\n=== KET HOP 2 server (round-robin, tong 8 luong) ===")
batch=[(SERVERS[i%2], prompts[i%len(prompts)]) for i in range(16)]
t0=time.time()
with ThreadPoolExecutor(max_workers=8) as ex:
    res=list(ex.map(lambda sp: one(sp[0],sp[1]), batch))
dt=time.time()-t0; toks=sum(r[1] for r in res)
log(f"  16 req in {dt:5.1f}s | {toks/dt:6.1f} tok/s | {16/dt:5.2f} req/s")
log("\nDONE")
LOG.close()
