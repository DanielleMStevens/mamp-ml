"""Phase 1 — train the per-residue pLDDT head on ESM-2 embeddings.

Pipeline: dataset.jsonl (from extract_dataset.py) -> ESM-2 per-residue
embeddings (cached) -> train PLDDTHead -> evaluate on held-out receptors ->
save head + a report. Run on a GPU node.

    python train_head.py dataset.jsonl --esm-model esm2_t33_650M_UR50D \
        --cache-dir emb_cache --epochs 40 --out head.pt --report report.md

Notes
-----
* `--esm-model esm2_t6_8M_UR50D` is the cheap option that matches the embeddings
  the pipeline already computes; `esm2_t33_650M_UR50D` (default) gives a stronger
  signal. Comparing them IS the experiment.
* The held-out split is by *receptor* (random). For a rigorous estimate, cluster
  receptors by sequence identity (mmseqs/cd-hit) and pass `--groups groups.tsv`
  so related receptors never split across train/val — random splits overstate
  in-family performance and hide the out-of-family failure mode.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from model import PLDDTHead, per_residue_metrics


# --- data ----------------------------------------------------------------
def load_dataset(path: Path) -> List[dict]:
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    for r in rows:
        r["plddt"] = np.asarray(r["plddt"], dtype=np.float32)
    return rows


def load_groups(path: "Optional[Path]") -> Dict[str, str]:
    """Optional ``receptor<TAB>group`` map so related receptors stay together."""
    if path is None:
        return {}
    out = {}
    for line in Path(path).read_text().splitlines():
        if line.strip():
            recv, grp = line.split("\t")[:2]
            out[recv.strip()] = grp.strip()
    return out


# --- embeddings ----------------------------------------------------------
def embed_dataset(rows, esm_model: str, cache_dir: Path, device: str) -> None:
    """Compute + cache per-residue ESM-2 embeddings (one .npy per receptor)."""
    import esm
    import torch

    cache_dir.mkdir(parents=True, exist_ok=True)
    short = esm_model.replace("esm2_", "").replace("_UR50D", "")
    todo = [r for r in rows if not (cache_dir / f"{short}_{r['receptor']}.npy").exists()]
    if not todo:
        return
    model, alphabet = esm.pretrained.load_model_and_alphabet(esm_model)
    model = model.eval().to(device)
    layer = model.num_layers
    bc = alphabet.get_batch_converter()
    with torch.no_grad():
        for i, r in enumerate(todo, 1):
            _, _, toks = bc([(r["receptor"], r["sequence"])])
            toks = toks.to(device)
            rep = model(toks, repr_layers=[layer])["representations"][layer]
            emb = rep[0, 1 : len(r["sequence"]) + 1].float().cpu().numpy()
            np.save(cache_dir / f"{short}_{r['receptor']}.npy", emb)
            if i % 25 == 0 or i == len(todo):
                print(f"  embedded {i}/{len(todo)}")


def load_emb(r, esm_model: str, cache_dir: Path) -> np.ndarray:
    short = esm_model.replace("esm2_", "").replace("_UR50D", "")
    return np.load(cache_dir / f"{short}_{r['receptor']}.npy")


# --- train / eval --------------------------------------------------------
def split_receptors(rows, groups, val_frac, seed):
    keys = {}
    for r in rows:
        keys.setdefault(groups.get(r["receptor"], r["receptor"]), []).append(r)
    grp_ids = sorted(keys)
    random.Random(seed).shuffle(grp_ids)
    n_val = max(1, int(round(len(grp_ids) * val_frac)))
    val_ids = set(grp_ids[:n_val])
    train = [r for g in grp_ids if g not in val_ids for r in keys[g]]
    val = [r for g in val_ids for r in keys[g]]
    return train, val


def run(args) -> int:
    import torch

    rows = load_dataset(args.dataset)
    if args.max_receptors:
        rows = rows[: args.max_receptors]
    groups = load_groups(args.groups)
    print(f"{len(rows)} receptor(s); embedding with {args.esm_model} …")
    embed_dataset(rows, args.esm_model, args.cache_dir, args.device)

    train, val = split_receptors(rows, groups, args.val_frac, args.seed)
    print(f"train={len(train)}  val={len(val)} receptor(s)")
    in_dim = load_emb(rows[0], args.esm_model, args.cache_dir).shape[1]

    torch.manual_seed(args.seed)
    head = PLDDTHead(in_dim, hidden=args.hidden, kernel=args.kernel).to(args.device)
    opt = torch.optim.Adam(head.parameters(), lr=args.lr)
    lossf = torch.nn.MSELoss()

    def batches(split):
        order = list(range(len(split)))
        random.Random(args.seed).shuffle(order)
        for i in order:
            r = split[i]
            x = torch.from_numpy(load_emb(r, args.esm_model, args.cache_dir)).float()[None].to(args.device)
            y = torch.from_numpy(r["plddt"]).float()[None].to(args.device)
            yield r, x, y

    for ep in range(1, args.epochs + 1):
        head.train()
        tot = 0.0
        for _, x, y in batches(train):
            opt.zero_grad()
            loss = lossf(head(x), y)
            loss.backward()
            opt.step()
            tot += float(loss)
        if ep % 5 == 0 or ep == args.epochs:
            m = evaluate(head, val, args)
            print(f"epoch {ep:3d}  train_mse={tot/max(1,len(train)):8.2f}  "
                  f"val_pearson={m['pearson']:.3f}  val_spearman={m['spearman']:.3f}  val_mae={m['mae']:.2f}")

    metrics = evaluate(head, val, args)
    torch.save({"state_dict": head.state_dict(), "in_dim": in_dim,
                "esm_model": args.esm_model, "hidden": args.hidden, "kernel": args.kernel},
               args.out)
    report = _report(args, train, val, metrics)
    print("\n" + report)
    if args.report:
        args.report.write_text(report, encoding="utf-8")
    return 0


def evaluate(head, split, args) -> dict:
    import torch

    head.eval()
    preds, trues = [], []
    with torch.no_grad():
        for r in split:
            x = torch.from_numpy(load_emb(r, args.esm_model, args.cache_dir)).float()[None].to(args.device)
            preds.append(head(x)[0].cpu().numpy())
            trues.append(r["plddt"])
    return per_residue_metrics(np.concatenate(preds), np.concatenate(trues))


def _report(args, train, val, m) -> str:
    return (
        f"# Phase 1 — pLDDT head\n\n"
        f"- ESM-2 model: `{args.esm_model}`\n"
        f"- receptors: train={len(train)} val={len(val)} (val_frac={args.val_frac}, seed={args.seed})\n"
        f"- split: {'identity groups' if args.groups else 'random by receptor (optimistic — see README)'}\n\n"
        f"## held-out per-residue accuracy\n"
        f"- residues: {m['n']}\n"
        f"- Pearson : {m['pearson']:.3f}\n"
        f"- Spearman: {m['spearman']:.3f}\n"
        f"- MAE     : {m['mae']:.2f} pLDDT\n\n"
        f"Spearman is the headline: the B-factor bandpass keys on the *shape* of "
        f"the pLDDT profile, so rank-correlation matters more than absolute MAE.\n"
    )


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dataset", type=Path, help="dataset.jsonl from extract_dataset.py")
    p.add_argument("--esm-model", default="esm2_t33_650M_UR50D")
    p.add_argument("--cache-dir", type=Path, default=Path("emb_cache"))
    p.add_argument("--groups", type=Path, default=None, help="optional receptor<TAB>group map")
    p.add_argument("--device", default="cuda")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--kernel", type=int, default=5)
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-receptors", type=int, default=None, help="cap for a quick smoke run")
    p.add_argument("--out", type=Path, default=Path("plddt_head.pt"))
    p.add_argument("--report", type=Path, default=None)
    return run(p.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
