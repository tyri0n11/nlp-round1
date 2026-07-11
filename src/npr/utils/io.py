"""Read raw records and write predictions in the required output layout."""
from __future__ import annotations

import json
import os
import re
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from .schema import Concept


def _record_id(path: Path) -> int:
    m = re.match(r"(\d+)", path.stem)
    return int(m.group(1)) if m else 0


def read_inputs(input_dir: str | os.PathLike) -> List[Tuple[str, str]]:
    """Return [(record_id_str, text), ...] sorted numerically by id.

    Each input file `N.txt` becomes record `N`. Text is read as UTF-8 and
    kept verbatim (positions are char offsets into this exact string).
    """
    d = Path(input_dir)
    files = sorted(d.glob("*.txt"), key=_record_id)
    out: List[Tuple[str, str]] = []
    for f in files:
        text = f.read_text(encoding="utf-8")
        out.append((f.stem, text))
    return out


def write_outputs(
    predictions: Dict[str, List[Concept]],
    output_dir: str | os.PathLike,
    zip_path: str | os.PathLike | None = None,
) -> Path:
    """Write one `N.json` per record into `output_dir`, optionally zip it.

    The zip, if requested, has the required top-level `output/` folder.
    Returns the output directory path.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for rid, concepts in predictions.items():
        payload = [c.to_dict() for c in concepts]
        (out / f"{rid}.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    if zip_path is not None:
        zp = Path(zip_path)
        with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as zf:
            for rid in predictions:
                zf.write(out / f"{rid}.json", arcname=f"output/{rid}.json")
    return out


def read_gold(gold_dir: str | os.PathLike) -> Dict[str, List[Concept]]:
    """Load reference `N.json` files into Concept lists (for local eval)."""
    d = Path(gold_dir)
    gold: Dict[str, List[Concept]] = {}
    for f in sorted(d.glob("*.json"), key=_record_id):
        data = json.loads(f.read_text(encoding="utf-8"))
        gold[f.stem] = [Concept.from_dict(x) for x in data]
    return gold
