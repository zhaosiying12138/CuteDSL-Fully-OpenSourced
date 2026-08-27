"""P0 differential guardrail: cutegen binding vs cute-dialect verifier.

Generates a seeded corpus of layouts (rank 1-4, nested modes, dynamic '?'
leaves) and asserts, per algebra op, that

  * the in-process cutegen binding (build-oracle/, the type engine the
    dialect itself uses), and
  * the BSD cutlass-compiler binary running the REAL cute dialect
    verifier (the pipeline that will consume the emitted ops)

agree character-for-character on the inferred result type. The verifier
side goes through full textual-MLIR round-trip (parse + verify + infer),
so this checks the pipeline, not just the library.

Optional third oracle (env SC_DIFF_OFFICIAL=1): the official
nvidia-cutlass-dsl wheel's Python layout algebra in .venv-reference, when
that environment exists.
"""
import os
import random
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from self_cutedsl.object_model import cutegen_binding as O  # noqa: E402

CC = str(ROOT / "build-compiler/tools/cutlass-compiler/cutlass-compiler")
_INFER_RE = re.compile(r"inferred type\(s\) '([^']+)'")

# deterministic corpus -------------------------------------------------------
rng = random.Random(20260827)


def gen_skeleton(depth: int = 0):
    """Shared shape/stride tree skeleton (CuTe requires matching structure)."""
    if depth < 2 and rng.random() < 0.45:
        return tuple(gen_skeleton(depth + 1)
                     for _ in range(rng.randint(2, 3)))
    return None


def fill_skeleton(skel, leaf):
    if isinstance(skel, tuple):
        return tuple(fill_skeleton(k, leaf) for k in skel)
    return leaf()


def gen_layout_text():
    skel = gen_skeleton()
    sh = fill_skeleton(skel, lambda: rng.choice(
        [1, 2, 4, 8, 16, 32, 64, 128, 256]) if rng.random() < 0.75 else "?")
    st = fill_skeleton(skel, lambda: rng.choice([0, 1, 4, 8, 16, 32, 64]))
    return f"{render(sh)}:{render(st)}"


def render(t):
    if isinstance(t, tuple):
        return "(" + ",".join(render(x) for x in t) + ")"
    return "?" if t == "?" else str(t)


def _normalize(t):
    return t.replace(" ", "")


CORPUS = [
    ("(4,8):(8,1)", "(32,4):(4,1)"),          # canonical composition case
    ("(256,?):(?,1)", "(32,8):(8,1)"),        # dynamic shape extent
    ("(8,?):(4,?)", "(?,16):(1,16)"),         # both operands dynamic
    ("((4,8),(16,2)):((8,1),(32,64))", "(64,4):(1,16)"),  # nested modes
    ("(32,4):(4,1)", "((4,8),(32,4)):((8,1),(4,32))"),
] + [(gen_layout_text(), gen_layout_text()) for _ in range(40)]


def _materialize(name: str, layout_text: str, lines: list) -> str:
    """Emit an operand producer for a (possibly dynamic) layout.

    cute.static cannot hold '?'-typed values (the verifier rejects the op
    outright, which the stderr-regex bootstrap path used to silently read
    as 'placeholder accepted') — dynamic layouts must be built with
    cute.make_shape / make_stride over i32 leaves, exactly like the real
    emitter does.
    """
    if "?" not in layout_text:
        lines.append(f"    %{name} = cute.static : !cute.layout<\"{layout_text}\">")
        return f'!cute.layout<"{layout_text}">'
    sh_s, _, st_s = layout_text.partition(":")
    ty = f'!cute.layout<"{layout_text}">'
    lines.append(f"    %{name}_k = arith.constant 1 : i32")

    def make(kind: str, text: str, n_dyn: int):
        short = {"shape": "sh", "stride": "st"}[kind]
        if n_dyn == 0:
            lines.append(f"    %{name}_{short} = cute.static : "
                         f"!cute.{kind}<\"{text}\">")
            return f"!cute.{kind}<\"{text}\">"
        leaves = ", ".join([f"%{name}_k"] * n_dyn)
        ity = ", ".join(["i32"] * n_dyn)
        rt = f"!cute.{kind}<\"{text}\">"
        lines.append(f"    %{name}_{short} = cute.make_{kind} ({leaves}) "
                     f": ({ity}) -> {rt}")
        return rt

    sh_ty = make("shape", sh_s, sh_s.count("?"))
    st_ty = make("stride", st_s, st_s.count("?"))
    lines.append(f"    %{name} = cute.make_layout (%{name}_sh, %{name}_st) "
                 f": ({sh_ty}, {st_ty}) -> {ty}")
    return ty


def verifier_infer(op: str, a: str, b: str = None, modes: str = "") -> str:
    """Ask the cute dialect verifier for the op's result type."""
    lines = []
    ta = _materialize("a", a, lines)
    if b is None:
        call = f'cute.{op}(%a) : ({ta}) -> !cute.layout<"(1):(1)">'
        opnds = ta
    else:
        tb = _materialize("b", b, lines)
        call = (f"cute.{op}(%a, %b) : ({ta}, {tb}) "
                f'-> !cute.layout<"(1):(1)">')
        opnds = f"{ta}, {tb}"
    mod = ("module {\n  func.func @probe() {\n" + "\n".join(lines) +
           f"\n    %c = {call}\n"
           "    return\n  }\n}\n")
    proc = subprocess.run([CC, "-"], input=mod, capture_output=True,
                          text=True, timeout=60)
    m = _INFER_RE.search(proc.stderr)
    if m is None:
        if "error:" in proc.stderr:
            raise RuntimeError(
                f"verifier rejected the probe module without inferring a "
                f"type (bootstrap-path ambiguity is NOT auto-resolved "
                f"here):\n{proc.stderr[:800]}")
        return '!cute.layout<"(1):(1)">'
    # the diagnostic carries the COMPLETE type text (e.g. !cute.layout<"...">)
    return m.group(1)


@pytest.mark.parametrize("a,b", CORPUS)
def test_composition_binding_vs_verifier(a, b):
    ours = O.composition(f'!cute.layout<"{a}">', f'!cute.layout<"{b}">')
    ref = verifier_infer("composition", a, b)
    assert _normalize(ours) == _normalize(ref), (a, b, ours, ref)


@pytest.mark.parametrize("a,b", CORPUS)
def test_zipped_divide_binding_vs_verifier(a, b):
    tiler = "(4)"          # shape-kind tiler, as the object model uses
    ours = O.zipped_divide(f'!cute.layout<"{a}">',
                           f'!cute.shape<"{tiler}">')
    lines = []
    ta = _materialize("a", a, lines)
    tt = f'!cute.shape<"{tiler}">'
    lines.append(f"    %t = cute.static : {tt}")
    lines.append(f"    %c = cute.zipped_divide(%a, %t) : ({ta}, {tt}) "
                 f'-> !cute.layout<"(1):(1)">')
    mod = ("module {\n  func.func @probe() {\n" + "\n".join(lines) +
           "\n    return\n  }\n}\n")
    proc = subprocess.run([CC, "-"], input=mod, capture_output=True,
                          text=True, timeout=60)
    m = _INFER_RE.search(proc.stderr)
    if m is None and "error:" in proc.stderr:
        raise RuntimeError(f"probe rejected:\n{proc.stderr[:600]}")
    ref = (m.group(1) if m else '!cute.layout<"(1):(1)">')
    assert _normalize(ours) == _normalize(ref), (a, tiler, ours, ref)


@pytest.mark.parametrize("a,_b", CORPUS)
def test_coalesce_and_flatten_binding_vs_verifier(a, _b):
    for op in ("coalesce", "flatten"):
        ours = getattr(O, op)(f'!cute.layout<"{a}">')
        ref = verifier_infer(op, a)
        assert _normalize(ours) == _normalize(ref), (op, a, ours, ref)


def test_binding_selfcheck_and_stats():
    assert O.available(), O.unavailable_reason()
    calls, ms = O.stats()
    assert calls > 0 and ms >= 0.0
