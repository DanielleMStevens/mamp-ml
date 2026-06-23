"""Phase 1 — build the (sequence, per-residue pLDDT) training set.

Walks a directory of folded structures (ColabFold/ESMFold PDBs, which store
per-residue pLDDT in the CA atom's B-factor column) and writes one JSON object
per receptor: ``{"key", "receptor", "sequence", "plddt": [...]}``. That is the
teacher signal for the pLDDT head — no labels to annotate, it's already in the
structures you've folded.

    python extract_dataset.py STRUCT_DIR -o dataset.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Optional, Tuple

from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1

# ColabFold/ESMFold filenames look like
# "<receptor>_unrelaxed_rank_001_...pdb" — strip the folding suffix to a key.
_SUFFIX_RE = re.compile(r"_(unrelaxed|relaxed|rank)_.*$")


def normalise_receptor(stem: str) -> str:
    return _SUFFIX_RE.sub("", stem)


def extract_one(pdb_path: Path) -> "Optional[Tuple[str, list]]":
    """Return ``(sequence, plddt_list)`` for the first chain, or None if empty."""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("x", str(pdb_path))
    try:
        chain = next(next(structure.get_models()).get_chains())
    except StopIteration:
        return None
    seq, plddt = [], []
    for res in chain.get_residues():
        if "CA" not in res:
            continue
        seq.append(seq1(res.resname, undef_code="X"))
        plddt.append(round(float(res["CA"].get_bfactor()), 3))
    if not seq:
        return None
    return "".join(seq), plddt


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("struct_dir", type=Path, help="directory of folded PDBs (pLDDT in B-factor)")
    p.add_argument("-o", "--out", type=Path, required=True, help="output JSONL path")
    p.add_argument("--glob", default="*.pdb", help="filename pattern (default: *.pdb)")
    args = p.parse_args(argv)

    pdbs = sorted(args.struct_dir.rglob(args.glob))
    n = 0
    seen = set()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        for pdb in pdbs:
            receptor = normalise_receptor(pdb.stem)
            if receptor in seen:  # one structure per receptor (best rank already)
                continue
            res = extract_one(pdb)
            if res is None:
                continue
            seq, plddt = res
            seen.add(receptor)
            fh.write(json.dumps({"key": pdb.stem, "receptor": receptor,
                                 "sequence": seq, "plddt": plddt}) + "\n")
            n += 1
    print(f"wrote {n} receptor(s) to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
