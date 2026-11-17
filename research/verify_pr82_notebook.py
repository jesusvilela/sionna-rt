#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 Jesus Vilela Jato
# SPDX-License-Identifier: Apache-2.0
"""Preflight PR #82 without GPU simulations or falsifying stored outputs.

Only the existing Python/numpy constellation cell is executed. Other code
cells are syntax checked, not imported/executed. The old notebook is read
from the archived Git branch, not from a generated expectation.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import subprocess
from pathlib import Path

import numpy as np


DIAGNOSTIC_ID = "f0c74cc2"
CONSTELLATION_ID = "123d30b0"
SER_ID = "650f2233"
DISCLAIMER_A = (
    "print(\"In this sample, rank-1 has ~2x the worst-channel minimum squared distance,\")\n"
)
DISCLAIMER_B = (
    "print(\"but this diagnostic does not establish the cause of its SER advantage.\")"
)
STDOUT_A = (
    "In this sample, rank-1 has ~2x the worst-channel minimum squared distance,\n"
)
STDOUT_B = (
    "but this diagnostic does not establish the cause of its SER advantage.\n"
)
LICENSE = [
    "<!--\n",
    "SPDX-FileCopyrightText: Copyright (c) 2026 Jesús Vilela Jato\n",
    "SPDX-License-Identifier: Apache-2.0\n",
    "-->\n",
    "\n",
]


def _source(ref: str, path: str) -> str:
    return subprocess.check_output(
        ["git", "show", f"{ref}:{path}"], text=True
    )


def _validate(notebook: dict) -> None:
    assert notebook["nbformat"] == 4
    assert notebook["nbformat_minor"] == 5
    cells = notebook["cells"]
    assert len(cells) == 11
    ids = [cell["id"] for cell in cells]
    assert len(set(ids)) == len(ids)
    assert sum(cell["cell_type"] == "code" for cell in cells) == 5
    assert all(isinstance(cell["source"], list) for cell in cells)
    for cell in cells:
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"cell-{cell['id']}", "exec")


def _no_unapproved_change(old: dict, new: dict) -> None:
    a, b = old["cells"], new["cells"]
    assert len(a) == len(b)
    for i, (before, after) in enumerate(zip(a, b)):
        assert before["id"] == after["id"]
        if i == 0:
            adjusted = json.loads(json.dumps(after))
            assert adjusted["source"][:len(LICENSE)] == LICENSE
            adjusted["source"] = adjusted["source"][len(LICENSE):]
            # Only the empirical-claims paragraph can differ; introduction
            # and all original claims about the algebraic object are frozen.
            assert adjusted["source"][:13] == before["source"][:13]
            text = "".join(adjusted["source"])
            assert "results reproduced below for Munich" in text
            assert "Florence scene is not evaluated in this notebook" in text
            assert "minimum effective squared distance" in text
            assert "not directly ablate this 16-point experiment" in text
            assert "Florence scene it wins" not in text
            assert "2x larger worst-channel effective distance" not in text
            adjusted["source"] = before["source"]
            assert adjusted == before, "opening cell changed outside editorial text"
        elif i in (8, 10):
            assert before["cell_type"] == after["cell_type"] == "markdown"
            text = "".join(after["source"])
            assert "squared" in text.lower(), "squared-distance units missing"
            assert "16-point" in text and "64-point" in text
            adjusted = json.loads(json.dumps(after))
            adjusted["source"] = before["source"]
            assert adjusted == before, "unexpected markdown metadata change"
        elif after["id"] == DIAGNOSTIC_ID:
            assert before["cell_type"] == after["cell_type"] == "code"
            assert len(before["outputs"]) == len(after["outputs"]) == 1
            adjusted = json.loads(json.dumps(after))
            assert adjusted["source"][-2:] == [DISCLAIMER_A, DISCLAIMER_B]
            assert adjusted["outputs"][0]["text"][-2:] == [
                STDOUT_A, STDOUT_B
            ]
            adjusted["source"][-2:] = before["source"][-2:]
            adjusted["outputs"][0]["text"][-2:] = (
                before["outputs"][0]["text"][-2:]
            )
            assert adjusted == before, "diagnostic changed more than prose"
        else:
            assert before == after, f"unrelated cell changed: {i}"


def _constellation_check(cells: list[dict]) -> dict:
    cell = next(c for c in cells if c["id"] == CONSTELLATION_ID)
    source = "".join(cell["source"])
    scope = {"np": np}
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        exec(compile(source, "constellation-cell", "exec"), scope)
    assert output.getvalue() == "".join(cell["outputs"][0]["text"])

    points = scope["CONSTS"]
    assert set(points.keys()) == {"qpsk2", "rank1"}
    metrics = {}
    for label, pts in points.items():
        assert pts.shape == (16, 2)
        energy = float(np.mean(np.sum(np.abs(pts) ** 2, axis=1)))
        assert math.isclose(energy, 1.0, rel_tol=0, abs_tol=1e-13)
        pair = np.linalg.norm(pts[:, None] - pts[None], axis=-1)
        nontrivial = pair[np.triu_indices(16, k=1)]
        min_distance = float(np.min(nontrivial))
        assert min_distance > 0
        metrics[label] = {
            "min_distance": min_distance,
            "average_energy": energy,
        }
    assert math.isclose(
        metrics["qpsk2"]["min_distance"], 1.0, abs_tol=1e-12
    )
    assert math.isclose(
        metrics["rank1"]["min_distance"],
        math.sqrt(2.0 - math.sqrt(2.0)),
        abs_tol=1e-12,
    )
    arr = points["rank1"]
    real_matrices = np.stack(
        (
            np.stack((arr[:, 0].real, arr[:, 0].imag), axis=1),
            np.stack((arr[:, 1].real, arr[:, 1].imag), axis=1),
        ),
        axis=1,
    )
    assert real_matrices.shape == (16, 2, 2)
    determinant = np.linalg.det(real_matrices)
    assert np.abs(determinant).max() < 1e-13
    metrics["rank1"]["max_abs_determinant"] = float(
        np.max(np.abs(determinant))
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archived-ref", required=True)
    parser.add_argument("--result", required=True, type=Path)
    args = parser.parse_args()

    target_path = "tutorials/Null-Cone-Constellations.ipynb"
    index_path = "doc/source/tutorials.rst"
    old_text = _source(args.archived_ref, target_path)
    new_text = Path(target_path).read_text(encoding="utf-8")
    old_index = _source(args.archived_ref, index_path)
    new_index = Path(index_path).read_text(encoding="utf-8")
    old, new = json.loads(old_text), json.loads(new_text)

    _validate(old)
    _validate(new)
    _no_unapproved_change(old, new)
    assert old_index.replace("\r\n", "\n") == new_index.replace(
        "\r\n", "\n"
    ), "tutorial index changed beyond line-ending normalization"
    assert new_index.count("tutorials/Null-Cone-Constellations.ipynb") == 1
    assert "\r" not in new_index
    assert new_text.count("\r\n") in (0, 366)
    assert "Rank-1 wins at high SNR because" not in new_text
    assert "on the Florence scene it wins" not in new_text
    assert len(new["cells"]) == 11
    assert old["cells"][7] == new["cells"][7], "SER results changed"
    metrics = _constellation_check(new["cells"])
    result = {
        "schema": "rt-pr82-preflight-v1",
        "status": "pass",
        "notebook_cells": len(new["cells"]),
        "compiled_python_cells": 5,
        "numerical_outputs_unchanged": True,
        "stored_SER_outputs_unchanged": True,
        "diagnostic_numeric_outputs_unchanged": True,
        "only_text_changes": (
            "SPDX notice, noncausal diagnostic prose, and evidence/units "
            "clarification in Markdown cells 0, 8 and 10"
        ),
        "tutorial_index_one_entry_only": True,
        "constellation_checks": metrics,
        "scope_limit": (
            "No GPU ray-tracing notebook execution, ray-traced SER reproduction, "
            "or backend CUDA validation performed by this preflight."
        ),
    }
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.result.write_text(
        json.dumps(result, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
