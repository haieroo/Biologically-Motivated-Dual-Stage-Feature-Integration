import argparse,hashlib,json
from pathlib import Path
def h(p):
    q=hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda:f.read(1048576),b""):q.update(b)
    return q.hexdigest()
p=argparse.ArgumentParser();p.add_argument("paths",nargs="+");p.add_argument("--json",default="");a=p.parse_args()
r=[{"path":str(Path(x).resolve()),"sha256":h(x)} for x in a.paths];print(json.dumps(r,indent=2))
if a.json:Path(a.json).write_text(json.dumps(r,indent=2),encoding="utf-8")
