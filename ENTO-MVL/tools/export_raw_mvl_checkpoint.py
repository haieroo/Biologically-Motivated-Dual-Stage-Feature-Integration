from __future__ import annotations
import argparse, json, hashlib
from pathlib import Path
import torch


def max_abs_diff(a, b):
    if set(a) != set(b):
        raise RuntimeError("state_dict keys differ")
    m = 0.0
    for k in a:
        x, y = a[k], b[k]
        if torch.is_tensor(x):
            d = (x.detach().cpu().float() - y.detach().cpu().float()).abs().max().item()
            m = max(m, float(d))
    return m


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--checkpoint",required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--seed",type=int,required=True)
    a=p.parse_args()
    src=Path(a.checkpoint); out=Path(a.output)
    ck=torch.load(src,map_location="cpu")
    raw=ck.get("raw_model")
    model=ck.get("model")
    ema=ck.get("ema_model")
    if raw is None: raise RuntimeError("best.pth has no raw_model")
    d_model=max_abs_diff(model,raw) if model is not None else None
    d_ema=max_abs_diff(ema,raw) if ema is not None else None
    decay=ck.get("args",{}).get("ema_decay",None)
    if decay is not None and float(decay)!=0.0:
        raise RuntimeError(f"ema_decay={decay}, expected 0.0")
    if d_model not in (None,0.0): raise RuntimeError(f"model != raw: {d_model}")
    if d_ema not in (None,0.0): raise RuntimeError(f"ema != raw: {d_ema}")
    export=dict(ck)
    export["model"]=raw
    export["raw_model"]=raw
    export["ema_model"]=None
    export["best_is_ema"]=False
    export["weights_kind"]="raw_only"
    export["ema_smoothing_used"]=False
    export["ema_decay_requested"]=0.0
    out.parent.mkdir(parents=True,exist_ok=True)
    torch.save(export,out)
    audit={
      "seed":a.seed,
      "epoch":int(ck.get("epoch",-1)),
      "ema_decay_requested":decay,
      "model_vs_raw_max_abs":d_model,
      "ema_vs_raw_max_abs":d_ema,
      "raw_equivalence_certified":True,
      "export":str(out),
      "export_sha256":sha256(out),
    }
    (out.parent/"raw_export_audit.json").write_text(json.dumps(audit,indent=2),encoding="utf-8")
    print(json.dumps(audit,indent=2))
    print("[PASS] ema_decay=0 checkpoint is exactly RAW-equivalent")

if __name__=="__main__": main()
