"""interp.py — typed AST interpreter for @cute.kernel device code (M2 scope).

Walks the kernel function's Python AST and emits MLIR via KernelEmitter.
Value model:
  * Python int/float/bool/str  -> compile-time constant (constexpr)
  * SSA                        -> dynamic runtime value
Dynamic control flow (if on SSA conditions) lowers to scf.if; constexpr
control flow stays Python. This mirrors the official DSL's partial-evaluation
model: Constexpr values decide Python-level branches, dynamic values decide
device branches.
"""
from __future__ import annotations

import ast
import textwrap
from dataclasses import dataclass

from .emitter import KernelEmitter, SSA



class InterpError(Exception):
    pass


@dataclass
class KernelParam:
    name: str
    kind: str          # "constexpr" | "dynamic"
    dtype: _DType | None
    default: object = None


class KernelInterpreter:
    def __init__(self, fn, params: list[KernelParam], arg_values: dict):
        self.fn = fn
        self.params = params
        self.env: dict[str, object] = {}
        self.emitter = KernelEmitter(fn.__name__)

        dyn_idx = 0
        for p in params:
            val = arg_values.get(p.name, p.default)
            if p.kind == "constexpr":
                self.env[p.name] = _unwrap_constexpr(val)
            else:
                mlir_ty = p.dtype.mlir if p.dtype else "i32"
                self.emitter.params.append((p.name, mlir_ty))
                ssa = SSA(mlir_ty, dyn_idx, p.dtype.name if p.dtype else "int32")
                # dynamic params are already %pN in the signature; alias them
                self.env[p.name] = ssa
                dyn_idx += 1
        # renumber ssa ids after params
        self.emitter._next_id = dyn_idx
        # bind param SSAs to signature names
        for i, (pname, _) in enumerate(self.emitter.params):
            self.env[pname] = SSA(self.emitter.params[i][1], i,
                                  getattr(self.env.get(pname), "dtype_name", "int32"))

        self.tree = ast.parse(textwrap.dedent(_get_source(fn)))
        self.body = self.tree.body[0].body  # FunctionDef body

    # ------------------------------------------------------------------ run
    def run(self) -> KernelEmitter:
        for stmt in self.body:
            self.exec_stmt(stmt)
        return self.emitter

    # --------------------------------------------------------------- stmts
    def exec_stmt(self, node: ast.stmt) -> None:
        if isinstance(node, ast.Assign):
            value = self.eval(node.value)
            self.assign(node.targets[0], value)
        elif isinstance(node, ast.AugAssign):
            cur = self.eval(node.target)
            rhs = self.eval(node.value)
            self.assign_ast(node.target, self.binop(node.op, cur, rhs))
        elif isinstance(node, ast.Expr):
            self.eval(node.value)
        elif isinstance(node, ast.If):
            self.exec_if(node)
        elif isinstance(node, (ast.Pass,)):
            pass
        elif isinstance(node, ast.Return):
            pass  # implicit gpu.return appended by emitter
        else:
            raise InterpError(f"unsupported statement {ast.dump(node)[:80]}")

    def exec_if(self, node: ast.If) -> None:
        cond = self.eval(node.test)
        if isinstance(cond, SSA):
            self.emitter.open_if(cond)
            for s in node.body:
                self.exec_stmt(s)
            if node.orelse:
                self.emitter.raw("} else {")
                for s in node.orelse:
                    self.exec_stmt(s)
            self.emitter.close_if()  # closes the else/then block
        else:
            # constexpr branch: evaluate in Python
            if cond:
                for s in node.body:
                    self.exec_stmt(s)
            else:
                for s in node.orelse:
                    self.exec_stmt(s)

    def assign(self, target: ast.expr, value) -> None:
        if isinstance(target, ast.Tuple):
            vals = _explode(value, len(target.elts))
            for t, v in zip(target.elts, vals):
                self.assign(t, v)
        elif isinstance(target, ast.Name):
            self.env[target.id] = value
        else:
            raise InterpError(f"unsupported assign target {ast.dump(target)[:60]}")

    def assign_ast(self, target: ast.expr, value) -> None:
        if isinstance(target, ast.Name):
            self.env[target.id] = value

    # --------------------------------------------------------------- exprs
    def eval(self, node: ast.expr):
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            if node.id in self.env:
                return self.env[node.id]
            g = getattr(self.fn, "__globals__", {})
            if node.id in g:
                return g[node.id]
            import builtins as _b

            if hasattr(_b, node.id):
                return getattr(_b, node.id)
            raise InterpError(f"unbound name {node.id!r}")
        if isinstance(node, ast.Tuple):
            return tuple(self.eval(e) for e in node.elts)
        if isinstance(node, ast.BoolOp):
            parts = [self.eval(v) for v in node.values]
            if any(isinstance(p, SSA) for p in parts):
                ssa_parts = [self.to_i1(p) for p in parts]
                op = "arith.ori" if isinstance(node.op, ast.Or) else "arith.andi"
                acc = ssa_parts[0]
                for p in ssa_parts[1:]:
                    acc = self.emitter.ssa("i1", f"{op} {acc.name}, {p.name} : i1")
                return acc
            return any(parts) if isinstance(node.op, ast.Or) else all(parts)
        if isinstance(node, ast.UnaryOp):
            v = self.eval(node.operand)
            if isinstance(v, SSA) and isinstance(node.op, ast.USub):
                zero = self.emitter.const_i32(0)
                return self.emitter.ssa(v.type, f"arith.subi {zero.name}, {v.name} : {v.type}")
            if isinstance(node.op, ast.Not) and not isinstance(v, SSA):
                return not v
            raise InterpError("unsupported unary op on this value")
        if isinstance(node, ast.BinOp):
            return self.binop(node.op, self.eval(node.left), self.eval(node.right))
        if isinstance(node, ast.Compare):
            return self.compare(node)
        if isinstance(node, ast.Call):
            return self.call(node)
        if isinstance(node, ast.Attribute):
            base = self.eval(node.value)
            return getattr(base, node.attr)
        raise InterpError(f"unsupported expression {ast.dump(node)[:80]}")

    def binop(self, op, lhs, rhs):
        py_only = isinstance(lhs, (int, float)) and isinstance(rhs, (int, float))
        arith = {
            ast.Add: "arith.addi", ast.Sub: "arith.subi", ast.Mult: "arith.muli",
        }.get(type(op))
        if arith is None:
            if isinstance(op, ast.Add) and not py_only:
                arith = "arith.addf"
            else:
                if py_only:
                    return _py_binop(op, lhs, rhs)
                raise InterpError(f"unsupported binop {op}")
        if py_only:
            return _py_binop(op, lhs, rhs)
        a, b = self._pair(lhs, rhs)
        fop = arith[:-1] + "f" if a.type.startswith("f") and arith.endswith("i") else arith
        return self.emitter.ssa(a.type, f"{fop} {a.name}, {b.name} : {a.type}")

    def compare(self, node: ast.Compare):
        assert len(node.ops) == 1 and len(node.comparators) == 1, "chained compare unsupported"
        lhs, rhs = self.eval(node.left), self.eval(node.comparators[0])
        op = node.ops[0]
        if isinstance(lhs, SSA) or isinstance(rhs, SSA):
            a, b = self._pair(lhs, rhs)
            pred = {
                ast.Eq: "eq", ast.NotEq: "ne", ast.Lt: "slt", ast.LtE: "sle",
                ast.Gt: "sgt", ast.GtE: "sge",
            }[type(op)]
            if a.type.startswith("f"):
                pred = pred.replace("s", "o")  # ordered float compare
            return self.emitter.ssa("i1", f"arith.cmpi {pred}, {a.name}, {b.name} : {a.type}"
                                    if not a.type.startswith("f")
                                    else f"arith.cmpf {pred}, {a.name}, {b.name} : {a.type}")
        return _py_compare(op, lhs, rhs)

    def call(self, node: ast.Call):
        func = self.eval(node.func)
        args = [self.eval(a) for a in node.args]
        return func(*args) if not isinstance(func, _DelayedCall) else func(*args)

    # --------------------------------------------------------------- helpers
    def to_i1(self, v) -> SSA:
        if isinstance(v, SSA) and v.type == "i1":
            return v
        if isinstance(v, SSA):
            raise InterpError("dynamic value in boolean context needs explicit compare")
        return self.emitter.ssa("i1", f"arith.constant {bool(v)} : i1")

    def _pair(self, lhs, rhs) -> tuple[SSA, SSA]:
        if not isinstance(lhs, SSA):
            lhs = self._const_like(rhs, lhs)
        if not isinstance(rhs, SSA):
            rhs = self._const_like(lhs, rhs)
        if lhs.type != rhs.type:
            raise InterpError(f"type mismatch {lhs.type} vs {rhs.type}")
        return lhs, rhs

    def _const_like(self, like: SSA, value) -> SSA:
        if like.type == "i32":
            return self.emitter.const_i32(int(value))
        if like.type == "i64":
            return self.emitter.const_i64(int(value))
        if like.type == "index":
            return self.emitter.ssa("index", f"arith.constant {int(value)} : index")
        if like.type == "f32":
            return self.emitter.ssa("f32", f"arith.constant {float(value)} : f32")
        raise InterpError(f"no constant materializer for {like.type}")


class _DelayedCall:
    pass


def _explode(value, n: int) -> list:
    if isinstance(value, (tuple, list)) and len(value) == n:
        return list(value)
    raise InterpError(f"cannot unpack {value!r} into {n} targets")


def _get_source(fn) -> str:
    import inspect

    src = inspect.getsource(fn)
    # strip decorator lines
    lines = src.splitlines()
    while lines and lines[0].strip().startswith("@"):
        lines = lines[1:]
    return "\n".join(lines)


def _unwrap_constexpr(v):
    from cutlass.dtypes import Constexpr

    if isinstance(v, Constexpr):
        return v.value
    return v


def _py_binop(op, a, b):
    import operator as _op

    return {
        ast.Add: _op.add, ast.Sub: _op.sub, ast.Mult: _op.mul,
        ast.Div: _op.truediv, ast.FloorDiv: _op.floordiv, ast.Mod: _op.mod,
    }[type(op)](a, b)


def _py_compare(op, a, b):
    import operator as _op

    return {
        ast.Eq: _op.eq, ast.NotEq: _op.ne, ast.Lt: _op.lt,
        ast.LtE: _op.le, ast.Gt: _op.gt, ast.GtE: _op.ge,
    }[type(op)](a, b)
