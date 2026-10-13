from __future__ import annotations
import argparse,csv,json,random,sys
from pathlib import Path
import numpy as np, torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
REPO_ROOT=Path(__file__).resolve().parents[1]; CORE=REPO_ROOT/"core"; sys.path.insert(0,str(CORE))
from seed_utils import resolve_seed
from data_pipeline import build_datasets
from models import CleanResNet50
def args():
    p=argparse.ArgumentParser("Reference ResNet50 CUB448 baseline")
    p.add_argument("--data-root",required=True); p.add_argument("--output-dir",required=True)
    p.add_argument("--seed",type=int,default=None,help="Optional; OS-random if omitted"); p.add_argument("--img-size",type=int,default=448)
    p.add_argument("--epochs",type=int,default=40); p.add_argument("--batch-size",type=int,default=2)
    p.add_argument("--accum-steps",type=int,default=4); p.add_argument("--num-workers",type=int,default=2)
    p.add_argument("--lr",type=float,default=1e-4); p.add_argument("--weight-decay",type=float,default=1e-4)
    p.add_argument("--label-smoothing",type=float,default=.05); p.add_argument("--grad-clip",type=float,default=.5)
    p.add_argument("--amp",action="store_true"); p.add_argument("--no-amp",dest="amp",action="store_false"); p.set_defaults(amp=True)
    return p.parse_args()
def seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
def wi(_):
    s=torch.initial_seed()%(2**32); random.seed(s); np.random.seed(s)
def dl(ds,b,sh,nw,s):
    g=torch.Generator().manual_seed(s)
    return DataLoader(ds,batch_size=b,shuffle=sh,drop_last=sh,num_workers=nw,pin_memory=torch.cuda.is_available(),
                      persistent_workers=nw>0,worker_init_fn=wi,generator=g)
def ac(on):
    if torch.cuda.is_available():
        try:return torch.amp.autocast("cuda",enabled=on)
        except (AttributeError,TypeError):return torch.cuda.amp.autocast(enabled=on)
    from contextlib import nullcontext
    return nullcontext()
def scaler(on):
    try:return torch.amp.GradScaler("cuda",enabled=on)
    except (AttributeError,TypeError):return torch.cuda.amp.GradScaler(enabled=on)
@torch.inference_mode()
def ev(m,L,d,amp):
    m.eval(); n=c1=c5=0
    for x,y in L:
        x=x.to(d,non_blocking=True); y=y.to(d,non_blocking=True)
        with ac(amp and d.type=="cuda"): z=m(x)["logits"]
        n+=y.numel(); c1+=int(z.argmax(1).eq(y).sum()); c5+=int(z.topk(min(5,z.size(1)),1).indices.eq(y[:,None]).any(1).sum())
    return {"top1":100*c1/n,"top5":100*c5/n,"correct_count":c1,"n":n}
def main():
    a=args(); a.seed=resolve_seed(a.seed,"ResNet50"); out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True); seed(a.seed)
    tr,va=build_datasets(a.data_root,img_size=a.img_size,random_erasing=True)
    T=dl(tr,a.batch_size,True,a.num_workers,a.seed); V=dl(va,max(a.batch_size,4),False,a.num_workers,a.seed+1)
    d=torch.device("cuda" if torch.cuda.is_available() else "cpu"); m=CleanResNet50(len(tr.classes),pretrained=True).to(d)
    o=torch.optim.AdamW(m.parameters(),lr=a.lr,weight_decay=a.weight_decay)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(o,T_max=max(a.epochs,1)); sc=scaler(a.amp and d.type=="cuda")
    log=out/"train_log.csv"
    for ep in range(1,a.epochs+1):
        m.train(); o.zero_grad(set_to_none=True); N=0; S=0.
        for st,(x,y) in enumerate(T,1):
            x=x.to(d,non_blocking=True); y=y.to(d,non_blocking=True)
            with ac(a.amp and d.type=="cuda"): loss=F.cross_entropy(m(x)["logits"],y,label_smoothing=a.label_smoothing)
            sc.scale(loss/a.accum_steps).backward()
            if st%a.accum_steps==0 or st==len(T):
                sc.unscale_(o); torch.nn.utils.clip_grad_norm_(m.parameters(),a.grad_clip); sc.step(o); sc.update(); o.zero_grad(set_to_none=True)
            N+=y.numel(); S+=float(loss.detach())*y.numel()
        sch.step(); row={"epoch":ep,"train_ce":S/max(N,1),"lr":o.param_groups[0]["lr"]}; first=not log.exists()
        with log.open("a",encoding="utf-8-sig",newline="") as f:
            w=csv.DictWriter(f,fieldnames=list(row))
            if first:w.writeheader()
            w.writerow(row)
        print(f"Epoch {ep:03d}/{a.epochs} | CE={row['train_ce']:.5f}")
    metric=ev(m,V,d,a.amp)
    ck={"model":{k:v.detach().cpu() for k,v in m.state_dict().items()},"epoch":a.epochs,"seed":a.seed,"classes":list(tr.classes),
        "metrics":metric,"recipe_scope":"reference_only_exact_paper_baseline_requires_released_checkpoint",
        "official_eval_used_for_checkpoint_selection":False,"args":vars(a)}
    torch.save(ck,out/"final_epoch40.pth")
    (out/"summary.json").write_text(json.dumps({k:v for k,v in ck.items() if k!="model"},indent=2),encoding="utf-8")
    print(json.dumps(metric,indent=2))
if __name__=="__main__":main()
