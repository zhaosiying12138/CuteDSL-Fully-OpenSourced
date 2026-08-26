"""emitter.py — MLIR text emitter for traced kernels (M2 scope).

Emits gpu.func bodies in the base-dialect boundary (arith/scf/gpu/LLVM),
which cutlass-compiler's one-shot-convert-to-llvm lowers to NVVM/PTX.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(eq=False)
class SSA:
    """A dynamic (runtime) value: an SSA result with an MLIR type."""
    type: str
    id: int
    dtype_name: str = "int32"

    @property
    def name(self) -> str:
        return f"%v{self.id}"

    def __repr__(self):
        return f"<SSA {self.name}:{self.type}>"


class KernelEmitter:
    """Accumulates the body of one gpu.func."""

    def __init__(self, name: str, indent: str = "    "):
        self.name = name
        self._indent = indent
        self._lines: list[str] = []
        self._depth = 1  # inside gpu.func
        self._next_id = 0
        self.params: list[tuple[str, str]] = []  # (mlir_name, type)
        self.smem_globals: list[str] = []         # module-level shared decls

    # -- plumbing ----------------------------------------------------------
    def raw(self, line: str) -> None:
        self._lines.append(self._indent * self._depth + line)

    def ssa(self, type_: str, op: str, dtype_name: str = "int32") -> SSA:
        v = SSA(type_, self._next_id, dtype_name)
        self._next_id += 1
        self.raw(f"{v.name} = {op}")
        return v

    def const_i32(self, value: int) -> SSA:
        return self.ssa("i32", f"arith.constant {value} : i32")

    def const_i64(self, value: int) -> SSA:
        return self.ssa("i64", f"arith.constant {value} : i64")

    # -- structured control flow --------------------------------------------
    def open_if(self, cond: SSA) -> None:
        assert cond.type == "i1"
        self.raw(f"scf.if {cond.name} {{")
        self._depth += 1

    def close_if(self) -> None:
        self._depth -= 1
        self.raw("}")

    def open_for(self, lb: SSA, ub: SSA, step: SSA, iter_args=None) -> tuple[SSA, list[SSA]]:
        """iter_args: list of SSA initial values (non-index loop-carried)."""
        iv = SSA("index", self._next_id, "index")
        self._next_id += 1
        args = []
        results = []
        if iter_args:
            decls, res_names = [], []
            for init in iter_args:
                a = SSA(init.type, self._next_id, init.dtype_name)
                self._next_id += 1
                r = SSA(init.type, self._next_id, init.dtype_name)
                self._next_id += 1
                decls.append(f"{a.name} = {init.name}")
                res_names.append(r.name)
                args.append(a)
                results.append(r)
            tys = ", ".join(a.type for a in args)
            header = (", ".join(res_names) + " = "
                      f"scf.for {iv.name} = {lb.name} to {ub.name} step {step.name}"
                      + " iter_args(" + ", ".join(decls) + f") -> ({tys})")
        else:
            header = f"scf.for {iv.name} = {lb.name} to {ub.name} step {step.name}"
        self.raw(header + " {")
        self._depth += 1
        return iv, args, results

    def yield_for(self, vals: list[SSA]) -> None:
        tys = ", ".join(v.type for v in vals)
        self.raw(f"scf.yield {', '.join(v.name for v in vals)} : {tys}")

    def close_for(self) -> None:
        self._depth -= 1
        self.raw("}")

    # -- index arithmetic (for partition pointer math) ------------------------
    def idx_const(self, v: int) -> SSA:
        return self.ssa("index", f"arith.constant {int(v)} : index")

    def idx_binop(self, op: str, a: SSA, b) -> SSA:
        if not isinstance(b, SSA):
            b = self.idx_const(int(b))
        assert a.type == "index" and b.type == "index"
        return self.ssa("index", f"{op} {a.name}, {b.name} : index")

    def cmpi_slt_const(self, a: SSA, bound: int) -> SSA:
        c = self.idx_const(bound)
        return self.ssa("i1", f"arith.cmpi slt, {a.name}, {c.name} : index")

    # -- memory (raw pointer ABI; tensors lower to !llvm.ptr<1>) --------------
    def index_to_i64(self, v: SSA) -> SSA:
        if v.type == "i64":
            return v
        assert v.type == "index", f"index_to_i64 on {v.type}"
        return self.ssa("i64", f"arith.index_cast {v.name} : index to i64")

    def gep(self, base: SSA, offset: SSA, elem_type: str = "f32") -> SSA:
        off = self.index_to_i64(offset) if offset.type in ("index", "i32") else offset
        return self.ssa(
            base.type,
            f"llvm.getelementptr {base.name}[{off.name}] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, {elem_type}",
        )

    def load_f32(self, p: SSA) -> SSA:
        return self.ssa("f32", f"llvm.load {p.name} : !llvm.ptr<1> -> f32", "float32")

    def store_f32(self, v: SSA, p: SSA) -> None:
        self.raw(f"llvm.store {v.name}, {p.name} : f32, !llvm.ptr<1>")

    # -- vectorized access (128-bit path for aligned contiguous values) ------
    def load_vec_f32(self, p: SSA, width: int) -> SSA:
        # explicit alignment lets NVPTX emit ld.global.v4 instead of 4x b32
        return self.ssa(f"vector<{width}xf32>",
                        f"llvm.load {p.name} {{alignment = {width * 4} : i64}}"
                        f" : !llvm.ptr<1> -> vector<{width}xf32>")

    def store_vec_f32(self, v: SSA, p: SSA, width: int) -> None:
        self.raw(f"llvm.store {v.name}, {p.name} {{alignment = {width * 4} : i64}}"
                 f" : vector<{width}xf32>, !llvm.ptr<1>")

    def lane_f32(self, vec: SSA, lane: int) -> SSA:
        c = self.ssa("i32", f"arith.constant {lane} : i32")
        return self.ssa("f32", f"llvm.extractelement {vec.name}[{c.name} : i32] : vector<4xf32>"
                        if width_of(vec) == 4 else
                        f"llvm.extractelement {vec.name}[{c.name} : i32] : vector<{width_of(vec)}xf32>",
                        "float32")

    def undef_vec_f32(self, width: int) -> SSA:
        return self.ssa(f"vector<{width}xf32>", f"llvm.mlir.undef : vector<{width}xf32>")

    def insert_lane_f32(self, v: SSA, vec: SSA, lane: int) -> SSA:
        c = self.ssa("i32", f"arith.constant {lane} : i32")
        return self.ssa(vec.type,
                        f"llvm.insertelement {v.name}, {vec.name}[{c.name} : i32] : {vec.type}")

    # -- builtins -------------------------------------------------------------
    def thread_id(self, axis: str) -> SSA:
        return self.ssa("index", f"gpu.thread_id {axis}", "index")

    def block_id(self, axis: str) -> SSA:
        return self.ssa("index", f"gpu.block_id {axis}", "index")

    def block_dim(self, axis: str) -> SSA:
        return self.ssa("index", f"gpu.block_dim {axis}", "index")

    uses_printf = False

    def printf(self, fmt_c: str, args: list[SSA]) -> None:
        self.uses_printf = True
        escaped = fmt_c.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\0A")
        if args:
            ops = ", ".join(a.name for a in args)
            types = ", ".join(a.type for a in args)
            self.raw(f'gpu.printf "{escaped}", {ops} : {types}')
        else:
            self.raw(f'gpu.printf "{escaped}"')

    # -- shared memory ---------------------------------------------------------
    def smem_global_declare(self, name: str, elems: int, elem_mlir: str = "f16",
                            align: int = 16) -> None:
        line = (f"    llvm.mlir.global internal @{name}() "
                f"{{addr_space = 3 : i32, alignment = {align} : i64}} "
                f": !llvm.array<{elems} x {elem_mlir}>")
        if not any(name in g for g in self.smem_globals):
            self.smem_globals.append(line)

    def smem_ptr(self, name: str, elem_mlir: str = "f16") -> SSA:
        return self.ssa(f"!llvm.ptr<3>",
                        f"llvm.mlir.addressof @{name} : !llvm.ptr<3>")

    def barrier(self) -> None:
        self.raw("gpu.barrier")

    # -- ldmatrix / mma ----------------------------------------------------------
    def ldmatrix(self, ptr: SSA, num: int, trans: bool):
        layout = "col" if trans else "row"
        n = num
        if n == 1:
            ty = "i32"
        elif n == 2:
            ty = "!llvm.struct<(i32, i32)>"
        else:
            ty = "!llvm.struct<(i32, i32, i32, i32)>"
            n = 4
        return self.ssa(ty, f"nvvm.ldmatrix {ptr.name} {{num = {n} : i32, "
                            f"layout = #nvvm.mma_layout<{layout}>, "
                            f"eltType = #nvvm.ld_st_matrix_elt_type<b16>, "
                            f"shape = #nvvm.ld_st_matrix_shape<m = 8, n = 8>}} "
                            f": (!llvm.ptr<3>) -> {ty}")

    def mma_f16(self, a: list, b: list, c: list):
        """One mma.sync m16n8k16 f16->f32. a: 4 SSA (i32 or vector<2xf16>),
        b: 2 SSA, c: 4 SSA f32. Returns 4 SSA f32."""
        aops = ", ".join(x.name for x in a)
        bops = ", ".join(x.name for x in b)
        cops = ", ".join(x.name for x in c)
        r = self.ssa("!llvm.struct<(f32, f32, f32, f32)>",
                     f"nvvm.mma.sync A[{aops}] B[{bops}] C[{cops}] "
                     f"{{layoutA = #nvvm.mma_layout<row>, layoutB = #nvvm.mma_layout<col>, "
                     f"shape = #nvvm.shape<m = 16, n = 8, k = 16>}} "
                     f": (vector<2xf16>, vector<2xf16>, f32) -> !llvm.struct<(f32, f32, f32, f32)>")
        outs = []
        for i in range(4):
            outs.append(self.ssa("f32",
                       f"llvm.extractvalue {r.name}[{i}] : !llvm.struct<(f32, f32, f32, f32)>"))
        return outs

    def mma_mxf4nvf4(self, a, b, sfa, sfb, c):
        """m16n8k64 kind::mxf4nvf4 (SM120 NVFP4). A: 4xi32, B: 2xi32,
        SFA/SFB: 1xi32 each; C: 4xf32; D via explicit asm outputs
        ($w form — the rw-tied form gets DCEd by the conversion)."""
        for x in a + b + sfa + sfb:
            if x.type != "i32":
                raise TypeError("mxf4nvf4 packed operands must be i32")
        for x in c:
            if x.type != "f32":
                raise TypeError("mxf4nvf4 accumulators must be f32")
        c_i32 = [self.ssa("i32", f"llvm.bitcast {x.name} : f32 to i32")
                 for x in c]
        def _zreg():
            # force a real 16-bit register (immediates are rejected by
            # ptxas inside mma brace vectors)
            return self.ssa(
                "i16",
                'nvvm.inline_ptx "mov.b16 {$w0}, 0;" -> i16')
        z16 = [_zreg() for _ in range(4)]
        ro = a + b + sfa + [z16[0], z16[1]] + sfb + [z16[2], z16[3]] + c_i32
        ro_ops = ", ".join(x.name for x in ro)
        ro_types = ", ".join(x.type for x in ro)
        sty = "!llvm.struct<(f32, f32, f32, f32)>"
        line = ('nvvm.inline_ptx "' +
                'mma.sync.aligned.kind::mxf4nvf4' +
                '.block_scale.scale_vec::4X.m16n8k64.row.col.f32' +
                '.e2m1.e2m1.f32.ue4m3 ' +
                '{$0, $1, $2, $3}, ' +
                '{$4, $5, $6, $7}, {$8, $9}, ' +
                '{$16, $17, $18, $19}, ' +
                '{$10}, {$11, $12}, {$13}, {$14, $15};" ' +
                f'ro ({ro_ops} : {ro_types}) -> {sty}')
        r = self.ssa(sty, line)
        return [self.ssa("f32",
                f"llvm.extractvalue {r.name}[{i}] : " + sty)
                for i in range(4)]

    def extract_i32(self, struct: SSA, idx: int) -> SSA:
        n = _struct_len(struct.type)
        inner = ", ".join(["i32"] * n)
        return self.ssa("i32",
                        f"llvm.extractvalue {struct.name}[{idx}] : !llvm.struct<({inner})>")

    def bitcast_f16x2(self, v: SSA) -> SSA:
        return self.ssa("vector<2xf16>", f"llvm.bitcast {v.name} : i32 to vector<2xf16>")

    def gep_smem(self, base: SSA, offset: SSA, elem: str = "f16") -> SSA:
        off = self.index_to_i64(offset) if getattr(offset, "type", "") in ("index", "i32") else offset
        return self.ssa("!llvm.ptr<3>",
                        f"llvm.getelementptr {base.name}[{off.name}] "
                        f": (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, {elem}")

    def store_smem_f16(self, v: SSA, p: SSA) -> None:
        self.raw(f"llvm.store {v.name}, {p.name} : f16, !llvm.ptr<3>")

    def load_gmem_f16(self, p: SSA) -> SSA:
        return self.ssa("f16", f"llvm.load {p.name} : !llvm.ptr<1> -> f16")

    def store_gmem_f32(self, v: SSA, p: SSA) -> None:
        self.raw(f"llvm.store {v.name}, {p.name} : f32, !llvm.ptr<1>")

    # -- TMA / mbarrier (M5) ----------------------------------------------------
    def mbarrier_ptr(self, name: str) -> SSA:
        """Declare an mbarrier (8B) in shared memory; returns ptr."""
        line = (f"    llvm.mlir.global internal @{name}() "
                f"{{addr_space = 3 : i32, alignment = 8 : i64}} : i64")
        if not any(name in g for g in self.smem_globals):
            self.smem_globals.append(line)
        return self.ssa("!llvm.ptr<3>", f"llvm.mlir.addressof @{name} : !llvm.ptr<3>")

    def mbarrier_init(self, bar: SSA, count: int) -> None:
        c = self.ssa("i32", f"arith.constant {int(count)} : i32")
        self.raw(f"nvvm.mbarrier.init {bar.name}, {c.name} : !llvm.ptr<3>, i32")

    def mbarrier_init_single_thread(self, bar: SSA, count: int,
                                    tid: SSA) -> None:
        """mbarrier.init executed by thread 0 only (PTX: init by multiple
        threads on the same object is UB). tid: index SSA of thread_id x."""
        c = self.ssa("i32", f"arith.constant {int(count)} : i32")
        z = self.ssa("index", "arith.constant 0 : index")
        p = self.ssa("i1", f"arith.cmpi eq, {tid.name}, {z.name} : index")
        self.raw(f"scf.if {p.name} {{")
        self._depth += 1
        self.raw(f"nvvm.mbarrier.init {bar.name}, {c.name} : !llvm.ptr<3>, i32")
        self._depth -= 1
        self.raw("}")

    def fence_mbarrier_init(self) -> None:
        self.raw("nvvm.fence.mbarrier.init")

    def mbarrier_arrive_expect_tx(self, bar: SSA, tx_bytes: int) -> None:
        c = self.ssa("i32", f"arith.constant {int(tx_bytes)} : i32")
        self.raw(f"nvvm.mbarrier.arrive.expect_tx {bar.name}, {c.name} "
                 f": !llvm.ptr<3>, i32 -> i64")

    def mbarrier_try_wait_parity(self, bar: SSA, phase) -> None:
        """Wait for the phase parity via a polling test_wait loop. The
        nvvm try_wait op's suspend-time hint proved unreliable on this
        driver (missed wakeups -> hangs / 100ms sleeps); the plain spin
        is the standard CUTLASS wait idiom."""
        if isinstance(phase, SSA):
            p = phase
            if p.type != "i32":
                p = self.ssa("i32",
                             f"arith.index_cast {phase.name} : {phase.type} to i32")
        else:
            p = self.ssa("i32", f"arith.constant {int(phase)} : i32")
        bar32 = self.ssa("i32", f"llvm.ptrtoint {bar.name} : "
                                f"!llvm.ptr<3> to i32")
        self._bw_id = getattr(self, "_bw_id", 0) + 1
        i = self._bw_id
        self.raw(
            'nvvm.inline_ptx "{'
            '.reg .pred P1; '
            f'LABW{i}: '
            'mbarrier.test_wait.parity.shared.b64 P1, [$0], $1; '
            f'@!P1 bra.uni LABW{i}; '
            f'DN{i}: add.u32 $0, $0, 0;}}" ro (' +
            f"{bar32.name}, {p.name} : i32, i32)")

    def tma_load(self, smem: SSA, tma_ptr: SSA, bar: SSA, coords: list) -> None:
        """cp.async.bulk.tensor.<dim>d G2S with mbarrier complete_tx."""
        cs = []
        for c in coords:
            if isinstance(c, int):
                cs.append(self.ssa("i32", f"arith.constant {int(c)} : i32"))
            elif isinstance(c, SSA) and c.type != "i32":
                cs.append(self.ssa("i32",
                        f"arith.index_cast {c.name} : {c.type} to i32"))
            else:
                cs.append(c)
        ops = ", ".join(x.name for x in cs)
        self.raw(f"nvvm.cp.async.bulk.tensor.shared.cluster.global "
                 f"{smem.name}, {tma_ptr.name}, {bar.name}, box[{ops}] "
                 f"{{isCTAOnly = true}} : !llvm.ptr<3>, !llvm.ptr")

    def tma_store(self, tma_ptr: SSA, smem: SSA, coords: list) -> None:
        cs = []
        for c in coords:
            if isinstance(c, int):
                cs.append(self.ssa("i32", f"arith.constant {int(c)} : i32"))
            elif isinstance(c, SSA) and c.type != "i32":
                cs.append(self.ssa("i32",
                        f"arith.index_cast {c.name} : {c.type} to i32"))
            else:
                cs.append(c)
        ops = ", ".join(x.name for x in cs)
        self.raw(f"nvvm.cp.async.bulk.tensor.global.shared.cta "
                 f"{tma_ptr.name}, {smem.name}, box[{ops}] "
                 f": !llvm.ptr, !llvm.ptr<3>")
        self.raw("nvvm.cp.async.bulk.commit.group")
        self.raw("nvvm.cp.async.bulk.wait_group 0")

    def smem_tile_declare(self, name: str, elems: int, elem_mlir: str = "f32",
                          align: int = 128) -> SSA:
        line = (f"    llvm.mlir.global internal @{name}() "
                f"{{addr_space = 3 : i32, alignment = {align} : i64}} "
                f": !llvm.array<{elems} x {elem_mlir}>")
        if not any(name in g for g in self.smem_globals):
            self.smem_globals.append(line)
        return self.ssa("!llvm.ptr<3>", f"llvm.mlir.addressof @{name} : !llvm.ptr<3>")

    def setmaxregister(self, value: int, increase: bool) -> None:
        act = "increase" if increase else "decrease"
        self.raw(f"nvvm.setmaxregister {act} {int(value)}")

    def named_barrier_arrive(self, name: str, id_: int, count: int) -> None:
        """bar.arrive with an immediate barrier id (warp specialization)."""
        self.raw(f'nvvm.inline_ptx "bar.arrive {int(id_)}, {int(count)};"')

    def named_barrier_sync(self, id_: int, count: int) -> None:
        self.raw(f'nvvm.inline_ptx "bar.sync {int(id_)}, {int(count)};"')

    def fence_proxy_async_shared(self) -> None:
        """Order generic SMEM accesses vs the async (TMA) proxy."""
        self.raw("nvvm.fence.proxy {kind = #nvvm.proxy_kind<async.shared>, "
                 "space = #nvvm.shared_space<cta>}")

    def tma_desc_param(self, name: str) -> SSA:
        """Mark a kernel param as a TMA descriptor pointer (!llvm.ptr)."""
        return SSA("!llvm.ptr", 0, "tma_desc")

    # -- output ---------------------------------------------------------------
    def module_text(self) -> str:
        # param SSAs are pre-allocated as %v{i} by the interpreter; keep names in sync
        params = ", ".join(f"%v{i} : {ty}" for i, (_, ty) in enumerate(self.params))
        lines = [
            "module attributes {gpu.container_module} {",
            f"  gpu.module @{self.name}_module {{",
            *self.smem_globals,
            f"    gpu.func @{self.name}({params}) kernel {{",
            *self._lines,
            "      gpu.return",
            "    }",
            "  }",
            "}",
        ]
        return "\n".join(lines) + "\n"


def width_of(vec: SSA) -> int:
    # "vector<4xf32>" -> 4
    t = vec.type
    return int(t.split("<")[1].split("x")[0])

def _struct_len(ty: str) -> int:
    return ty.count("i32")
