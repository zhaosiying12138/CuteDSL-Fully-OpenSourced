"""M2b: public cute dialect layout algebra agrees with our reference evaluator.

Emits cute.make_layout / cute.size / cute.cosize / cute.layout_eval probes,
folds with `cute-opt -cute-fold-static`, and compares the folded
`cute.static` values against the Python reference in
self_cutedsl.frontend.layout.
"""
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "python/cutlass_compat"))

from self_cutedsl.frontend.layout import CuteLayout  # noqa: E402

CUTE_OPT = ROOT / "build-compiler/cute_ir/tools/cute-opt/cute-opt"

LAYOUTS = [
    ((4,), (1,)),
    ((4, 8), (1, 4)),
    ((4, 3, 2), (1, 4, 12)),
    ((2, 3), (4, 1)),
    ((64, 32), (32, 1)),
    ((128, 64, 64), (64 * 64, 64, 1)),
    (((2, 3), 4), ((1, 6), 2)),   # nested shape
    ((5, 7), (7, 1)),             # non-power-of-two
]


def _fold_probe(body_lines: list[str], ret_type: str) -> str:
    mlir = "module {\n" + "\n".join(body_lines) + "\n}\n"
    proc = subprocess.run([str(CUTE_OPT), "--cute-fold-static", "-"],
                          input=mlir, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    return proc.stdout


def _static_int(folded: str) -> int:
    m = re.search(r'cute\.static : !cute\.int_tuple<"([^"]+)"', folded)
    assert m, folded
    return int(m.group(1))


_LAYOUT_IDS = [str(s).replace(" ", "") for s, _ in LAYOUTS]


@pytest.mark.parametrize("shape,stride", LAYOUTS, ids=_LAYOUT_IDS)
def test_size_cosize(shape, stride):
    lay = CuteLayout(shape, stride)

    body = [f"func.func @probe() -> !cute.int_tuple<\"{lay.size}\"> {{"]
    body += lay._decl("l")
    body.append(f'  %r = cute.size(%l_layout) : (!cute.layout<"{_rt(lay)}">) -> !cute.int_tuple<"{lay.size}">')
    body.append(f'  return %r : !cute.int_tuple<"{lay.size}">')
    body.append("}")
    folded = _fold_probe(body, "size")
    assert _static_int(folded) == lay.size, folded

    body = [f"func.func @probe() -> !cute.int_tuple<\"{lay.cosize}\"> {{"]
    body += lay._decl("l")
    body.append(f'  %r = cute.cosize(%l_layout) : (!cute.layout<"{_rt(lay)}">) -> !cute.int_tuple<"{lay.cosize}">')
    body.append(f'  return %r : !cute.int_tuple<"{lay.cosize}">')
    body.append("}")
    folded = _fold_probe(body, "cosize")
    assert _static_int(folded) == lay.cosize, folded


@pytest.mark.parametrize("shape,stride", LAYOUTS[:6], ids=[f"ev{i}" for i in range(6)])
@pytest.mark.parametrize("sample", [0, 1], ids=["first", "mid"])
def test_layout_eval(shape, stride, sample):
    lay = CuteLayout(shape, stride)
    idx = min(sample, max(lay.size - 1, 0))
    coord = _unravel_py(idx, shape)
    expect = lay.eval(coord)

    coord_s = "(" + ",".join(str(c) for c in coord) + ")"
    body = [f'func.func @probe() -> !cute.int_tuple<"{expect}"> {{']
    body += lay._decl("l")
    body.append(f'  %c = cute.make_coord () : () -> !cute.coord<"{coord_s}">')
    body.append(f'  %r = cute.layout_eval(%c, %l_layout)')
    body.append(f'         : (!cute.coord<"{coord_s}">, {lay.layout_type})')
    body.append(f'        -> !cute.int_tuple<"{expect}">')
    body.append(f'  return %r : !cute.int_tuple<"{expect}">')
    body.append("}")
    folded = _fold_probe(body, "eval")
    assert _static_int(folded) == expect, folded


def _rt(lay: CuteLayout) -> str:
    from self_cutedsl.frontend.layout import _render

    return f"{_render(lay.shape)}:{_render(lay.stride)}"


def _unravel_py(idx: int, shape) -> tuple:
    from self_cutedsl.frontend.layout import _flatten

    flat = _flatten(shape)
    coords, acc = [], idx
    for d in reversed(flat):
        coords.append(acc % d)
        acc //= d
    return tuple(reversed(coords))
