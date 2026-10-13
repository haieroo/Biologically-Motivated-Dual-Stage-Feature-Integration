from __future__ import annotations
import argparse,os,shutil
from pathlib import Path
def mp(p):
    d={}
    for l in Path(p).read_text(encoding="utf-8").splitlines():
        if l.strip(): k,v=l.split(maxsplit=1); d[int(k)]=v.strip()
    return d
def put(s,d,m):
    d.parent.mkdir(parents=True,exist_ok=True)
    if d.exists():return
    if m=="copy":shutil.copy2(s,d)
    elif m=="hardlink":os.link(s,d)
    else:d.symlink_to(s.resolve())
def main():
    p=argparse.ArgumentParser();p.add_argument("--cub-root",required=True);p.add_argument("--output-root",required=True)
    p.add_argument("--mode",choices=["copy","hardlink","symlink"],default="copy");a=p.parse_args()
    r=Path(a.cub_root).resolve();o=Path(a.output_root).resolve();imgs=mp(r/"images.txt");lab={k:int(v) for k,v in mp(r/"image_class_labels.txt").items()}
    spl={k:int(v) for k,v in mp(r/"train_test_split.txt").items()};cls=mp(r/"classes.txt");nt=nv=0
    for i,rel in imgs.items():
        sub="train" if spl[i]==1 else "val";src=r/"images"/rel;dst=o/sub/cls[lab[i]]/Path(rel).name;put(src,dst,a.mode)
        if sub=="train":nt+=1
        else:nv+=1
    print(f"train={nt}, val={nv}, classes={len(cls)}")
    if (nt,nv,len(cls))!=(5994,5794,200):raise RuntimeError("Unexpected CUB counts")
if __name__=="__main__":main()
