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

import ast
import textwrap
from dataclasses import dataclass

from .emitter import KernelEmitter, SSA



class InterpError(Exception):
    pass


from dataclasses import dataclass as _dc


@_dc
class KernelParam:
    name: str
    kind: str          # "constexpr" | "dynamic"
    dtype: object = None
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
            elif p.kind == "tensor" and (_is_coord_meta(val) or _is_tma_view(val)):
                # coordinate/tma-view tensors are compile-time meta (no device
                # pointer: data flows via the TMA descriptor param)
                self.env[p.name] = val
                continue
            elif p.kind == "tma":
                mlir_ty = "!llvm.ptr"
                self.emitter.params.append((p.name, mlir_ty))
                desc = SSA(mlir_ty, dyn_idx, "tma_desc")
                dyn_idx += 1
                # keep the host object (box/recipe meta) with the desc SSA
                # attached: cute.copy(atom, ...) lowers through desc_ssa
                if val is not None and not isinstance(val, (bool, int, float)):
                    val.desc_ssa = desc
                    self.env[p.name] = val
                else:
                    self.env[p.name] = desc
                continue
            elif p.kind == "tensor":
                from .kernel_objects import KernelTensor

                mlir_ty = "!llvm.ptr<1>"
                self.emitter.params.append((p.name, mlir_ty))
                ptr = SSA(mlir_ty, dyn_idx, "ptr")
                dyn_idx += 1
                meta = getattr(val, "tiled", None) or val
                elem = getattr(getattr(meta, "base", meta), "element_type", None)
                self.env[p.name] = KernelTensor(ptr, meta, elem)
                continue
            else:
                mlir_ty = p.dtype.mlir if p.dtype else "i32"
                self.emitter.params.append((p.name, mlir_ty))
                ssa = SSA(mlir_ty, dyn_idx, p.dtype.name if p.dtype else "int32")
                # dynamic params are already %pN in the signature; alias them
                self.env[p.name] = ssa
                dyn_idx += 1
        # renumber ssa ids after params; max over params actually emitted
        n_params = len(self.emitter.params)
        self.emitter._next_id = max(dyn_idx, n_params)

        self.tidx_ssa = None  # recorded by cute.arch.thread_idx()
        # non-param extras (e.g. method receiver) pre-seeded into env
        for k, v in (arg_values or {}).items():
            if k not in (p.name for p in params):
                self.env[k] = v

        self.tree = ast.parse(textwrap.dedent(_get_source(fn)))
        self.body = self.tree.body[0].body  # FunctionDef body

    # ------------------------------------------------------------------ run
    def run(self) -> KernelEmitter:
        self._did_return = False
        self._return_value = None
        for stmt in self.body:
            self.exec_stmt(stmt)
            if self._did_return:
                break
        return self.emitter

    def inline_call(self, fn, bound):
        """Interpret a nested @cute.jit body into this kernel's emitter."""
        child = object.__new__(KernelInterpreter)
        child.fn = fn
        child.params = []
        child.env = dict(bound)
        child.emitter = self.emitter
        child.tidx_ssa = self.tidx_ssa
        child.tree = ast.parse(textwrap.dedent(_get_source(fn)))
        child.body = child.tree.body[0].body
        child._did_return = False
        child._return_value = None
        from . import builtins as _builtins

        previous = _builtins._active
        _builtins._active = child
        try:
            for stmt in child.body:
                child.exec_stmt(stmt)
                if child._did_return:
                    break
            return child._return_value
        finally:
            _builtins._active = previous

    # --------------------------------------------------------------- stmts
    def exec_stmt(self, node: ast.stmt) -> None:
        import os as _os
        if _os.environ.get("SC_TRACE2"):
            import sys as _s
            print("STMT:", type(node).__name__,
                  getattr(getattr(node, "targets", [None])[0], "lineno", "?"), file=_s.stderr)
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Subscript):
            tgt = node.targets[0]
            base = self.eval(tgt.value)
            from .kernel_objects import Fragment as _Frag, _FragSlice
            from .meta import TensorMeta as _TM

            if isinstance(base, (_Frag, _FragSlice)):
                idx = self.eval(tgt.slice)
                val = self.eval(node.value)
                base[idx] = val
            elif isinstance(base, list):
                idx = int(_const_index(self.eval(tgt.slice)))
                val = self.eval(node.value)
                base[idx] = val
            else:
                from .kernel_objects import KernelTensor
                from . import builtins as _b

                idx = self.eval(tgt.slice) if not isinstance(tgt.slice, ast.Tuple) else self.eval(tgt.slice)
                val = self.eval(node.value)
                if isinstance(base, _b.SmemArray):
                    ety = "f16" if base.elem.name.lower() in ("f16", "float16") else "f32"
                    soff = getattr(base, "stage_offset", None)
                    if soff is not None:
                        idx = self.emitter.idx_binop("arith.addi", idx, soff)
                    p = self.emitter.gep_smem(base.ptr, idx, ety)
                    if ety == "f16":
                        self.emitter.store_smem_f16(val, p)
                    else:
                        self.emitter.raw(
                            f"llvm.store {val.name}, {p.name} : f32, !llvm.ptr<3>")
                elif isinstance(base, _TM):
                    # host-elementwise write through the meta's torch tensor
                    coord = idx if isinstance(idx, (tuple, list)) else (idx,)
                    base[tuple(int(c) for c in coord)] = val
                elif base.__class__.__module__ == \
                        "self_cutedsl.frontend.cute_objects" and \
                        base.__class__.__name__ == "Tensor":
                    base[idx] = val
                else:
                    ptr = base.ptr if isinstance(base, KernelTensor) else base
                    meta = getattr(base, "meta", None)
                    elem = getattr(meta, "element_type", None)
                    self._store_elem(ptr, idx, val, elem, meta)
        elif isinstance(node, ast.Assign):
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
        elif isinstance(node, ast.For):
            self.exec_for(node)
        elif isinstance(node, ast.While):
            self.exec_while(node)
        elif isinstance(node, (ast.Pass,)):
            pass
        elif isinstance(node, ast.ImportFrom):
            import importlib

            if node.level:
                raise InterpError("relative imports inside kernels unsupported")
            module = importlib.import_module(node.module)
            for alias in node.names:
                self.env[alias.asname or alias.name] = getattr(module, alias.name)
        elif isinstance(node, ast.Import):
            import importlib

            for alias in node.names:
                module = importlib.import_module(alias.name)
                self.env[alias.asname or alias.name.split(".")[0]] = module
        elif isinstance(node, ast.ClassDef):
            # Kernel-local @cute.struct declarations are compile-time Python
            # metadata. Execute just the class statement with the trace env.
            namespace = dict(getattr(self.fn, "__globals__", {}))
            namespace.update(self.env)
            module = ast.fix_missing_locations(
                ast.Module(body=[node], type_ignores=[]))
            exec(compile(module, f"<kernel-class:{node.name}>", "exec"), namespace)
            self.env[node.name] = namespace[node.name]
        elif isinstance(node, ast.Return):
            self._return_value = self.eval(node.value) if node.value else None
            self._did_return = True
        else:
            raise InterpError(f"unsupported statement {ast.dump(node)[:80]}")

    def exec_if(self, node: ast.If) -> None:
        cond = _unwrap_typed_scalar(self.eval(node.test))
        if isinstance(cond, SSA):
            # region semantics: assignments inside scf.if are branch-local
            # (no yield machinery); restore the pre-if env afterwards so
            # sibling branches (elif chains) start from the same state
            snapshot = dict(self.env)
            snaps = [(v, v.__snapshot__()) for v in snapshot.values()
                     if hasattr(type(v), "__snapshot__")]
            carried = []
            for name in sorted(self._assigned_names(node.body + node.orelse)):
                if name not in snapshot:
                    continue
                value = self._as_ssa(snapshot[name])
                if value is not None:
                    carried.append((name, value))
            if carried:
                results = self.emitter.open_if_results(
                    cond, [value.type for _, value in carried])
                for s in node.body:
                    self.exec_stmt(s)
                then_values = [
                    self._as_ssa(self.env.get(name, snapshot[name]), template=value)
                    for name, value in carried
                ]
                self.emitter.yield_values(then_values)
                self.emitter.open_else()
                self.env = dict(snapshot)
                for value, snap in snaps:
                    value.__restore__(snap)
                for s in node.orelse:
                    self.exec_stmt(s)
                else_values = [
                    self._as_ssa(self.env.get(name, snapshot[name]), template=value)
                    for name, value in carried
                ]
                self.emitter.yield_values(else_values)
                self.emitter.close_if()
                self.env = dict(snapshot)
                for value, snap in snaps:
                    value.__restore__(snap)
                for (name, _), result in zip(carried, results):
                    self.env[name] = result
                return
            self.emitter.open_if(cond)
            for s in node.body:
                self.exec_stmt(s)
            if node.orelse:
                self.emitter.raw("} else {")
                self.env = dict(snapshot)   # sibling branch: same entry state
                for v, snap in snaps:
                    v.__restore__(snap)
                for s in node.orelse:
                    self.exec_stmt(s)
            self.emitter.close_if()  # closes the else/then block
            self.env = snapshot
            for v, snap in snaps:
                v.__restore__(snap)
        else:
            # constexpr branch: evaluate in Python
            if cond:
                for s in node.body:
                    self.exec_stmt(s)
            else:
                for s in node.orelse:
                    self.exec_stmt(s)

    def exec_while(self, node: ast.While) -> None:
        """Python while: constexpr conditions trace as (multi-)trip inline
        loops; SSA conditions need scf.while (honest boundary for now)."""
        canonical_for = self._while_as_for(node)
        if canonical_for is not None:
            self.exec_for(canonical_for)
            return
        cond = self.eval(node.test)
        if not isinstance(cond, (bool, int)):
            self._exec_dynamic_while(node)
            return
        guard = 0
        while bool(cond):
            for st in node.body:
                self.exec_stmt(st)
            cond = self.eval(node.test)
            guard += 1
            if guard > 10000:
                raise InterpError("while-loop trace guard tripped")

    def _exec_dynamic_while(self, node: ast.While) -> None:
        snapshot = dict(self.env)
        carried = []
        for name in sorted(self._assigned_names(node.body)):
            if name not in snapshot:
                continue
            original = snapshot[name]
            value = self._as_ssa(original)
            if value is not None:
                carried.append((name, None, value))
            elif isinstance(original, (tuple, list)):
                for index, item in enumerate(original):
                    item_ssa = self._as_ssa(item)
                    if item_ssa is not None:
                        carried.append((name, index, item_ssa))
        state_snapshots = {}
        for name, value in snapshot.items():
            if hasattr(value, "stage") and hasattr(value, "phase"):
                stage = self._as_ssa(value.stage)
                phase = self._as_ssa(value.phase)
                if stage is not None and phase is not None:
                    state_snapshots[name] = value.__snapshot__() \
                        if hasattr(value, "__snapshot__") else \
                        (value.stage, value.phase)
                    carried.append((name, "stage", stage))
                    carried.append((name, "phase", phase))
        if not carried:
            assigned = sorted(self._assigned_names(node.body))
            raise NotImplementedError(
                "dynamic while-loop without scalar loop-carried values in "
                f"{getattr(self.fn, '__name__', '<kernel>')}: assigned={assigned}, "
                f"available={[(name, type(snapshot.get(name)).__name__) for name in assigned]}")

        before_args, results = self.emitter.open_while(
            [value for _, _, value in carried])

        def bind(name, field, value):
            if field is None:
                self.env[name] = value
            elif isinstance(field, int):
                original = self.env[name]
                sequence = list(original)
                sequence[field] = value
                self.env[name] = type(original)(sequence)
            else:
                setattr(self.env[name], field, value)

        def read(name, field):
            value = self.env[name]
            if field is None:
                return value
            if isinstance(field, int):
                return value[field]
            return getattr(value, field)

        self.env = dict(snapshot)
        for name, saved in state_snapshots.items():
            self.env[name].__restore__(saved)
        for (name, attr, _), arg in zip(carried, before_args):
            bind(name, attr, arg)
        cond = _unwrap_typed_scalar(self.eval(node.test))
        if not isinstance(cond, SSA) or cond.type != "i1":
            raise InterpError("dynamic while condition did not lower to i1")
        condition_values = [
            self._as_ssa(
                read(name, attr),
                template=value,
            )
            for name, attr, value in carried
        ]
        self.emitter.while_condition(cond, condition_values)

        body_args = self.emitter.open_while_do(
            [value.type for _, _, value in carried])
        self.env = dict(snapshot)
        for name, saved in state_snapshots.items():
            self.env[name].__restore__(saved)
        for (name, attr, _), arg in zip(carried, body_args):
            bind(name, attr, arg)
        for stmt in node.body:
            self.exec_stmt(stmt)
        yielded = [
            self._as_ssa(
                read(name, attr),
                template=value,
            )
            for name, attr, value in carried
        ]
        self.emitter.yield_values(yielded)
        self.emitter.close_while()

        self.env = dict(snapshot)
        for name, saved in state_snapshots.items():
            self.env[name].__restore__(saved)
        for (name, attr, _), result in zip(carried, results):
            bind(name, attr, result)

    @staticmethod
    def _assigned_names(nodes) -> set[str]:
        assigned = set()

        def add_target(target):
            if isinstance(target, ast.Name):
                assigned.add(target.id)
            elif isinstance(target, (ast.Tuple, ast.List)):
                for elt in target.elts:
                    add_target(elt)

        def visit(node):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    add_target(target)
            elif isinstance(node, ast.AugAssign):
                add_target(node.target)
            for child in ast.iter_child_nodes(node):
                visit(child)

        for node in nodes:
            visit(node)
        return assigned

    def _as_ssa(self, value, template=None):
        if hasattr(value, "ssa"):
            if value.ssa is None:
                value.ir_value()
            if isinstance(value.ssa, SSA):
                return value.ssa
        value = _unwrap_typed_scalar(value)
        if isinstance(value, SSA):
            return value
        if template is None:
            return None
        return self._const_like(template, value)

    @staticmethod
    def _while_as_for(node: ast.While):
        """Recognize ``while i < end: ...; i += step`` as an scf.for."""
        if not isinstance(node.test, ast.Compare) or len(node.test.ops) != 1 \
                or not isinstance(node.test.ops[0], ast.Lt) \
                or not isinstance(node.test.left, ast.Name) or not node.body:
            return None
        name = node.test.left.id
        tail = node.body[-1]
        if isinstance(tail, ast.AugAssign) \
                and isinstance(tail.target, ast.Name) \
                and tail.target.id == name \
                and isinstance(tail.op, ast.Add):
            step = tail.value
        elif isinstance(tail, ast.Assign) and len(tail.targets) == 1 \
                and isinstance(tail.targets[0], ast.Name) \
                and tail.targets[0].id == name \
                and isinstance(tail.value, ast.BinOp) \
                and isinstance(tail.value.op, ast.Add) \
                and isinstance(tail.value.left, ast.Name) \
                and tail.value.left.id == name:
            step = tail.value.right
        else:
            return None
        return ast.For(
            target=ast.Name(id=name, ctx=ast.Store()),
            iter=ast.Call(
                func=ast.Name(id="range", ctx=ast.Load()),
                args=[
                    ast.Name(id=name, ctx=ast.Load()),
                    node.test.comparators[0],
                    step,
                ],
                keywords=[],
            ),
            body=node.body[:-1],
            orelse=node.orelse,
            type_comment=None,
        )

    def exec_for(self, node: ast.For) -> None:
        # `for x in range(...)` / cutlass.range_constexpr(...) supported
        fname = ""
        if isinstance(node.iter, ast.Call):
            f = node.iter.func
            fname = getattr(f, "id", None) or getattr(f, "attr", "")                 if isinstance(f, (ast.Name, ast.Attribute)) else ""
        assert isinstance(node.iter, ast.Call) and fname in ("range", "range_constexpr"), \
            f"only range()/range_constexpr() loops supported (got {fname!r})"
        iter_args = [_unwrap_typed_scalar(self.eval(a)) for a in node.iter.args]
        if fname == "range_constexpr":
            iter_args = [int(getattr(a, "value", a)) for a in iter_args]
        import os as _os
        if _os.environ.get("SC_TRACE"):
            print("EXEC_FOR range args:", iter_args, file=__import__("sys").stderr)
        if all(isinstance(a, int) for a in iter_args):
            # constexpr loop: unroll in Python (mirrors official partial eval)
            filled = (list(iter_args) + [0, None, 1])[:3]
            lo = filled[0] if len(iter_args) > 1 else 0
            hi = filled[1] if len(iter_args) > 1 else iter_args[0]
            st = filled[2] if len(iter_args) > 2 else 1
            rng = range(lo, hi, st)
            # rebind the target as constexpr per iteration
            for i in rng:
                self.assign(node.target, i)
                for s in node.body:
                    self.exec_stmt(s)
            return
        # dynamic bounds -> scf.for (normalize 1-arg range to (0, n, 1))
        if len(iter_args) == 1:
            iter_args = [0, iter_args[0]]
        args = iter_args + [1]
        lb = self._to_index(args[0]) if not isinstance(args[0], int) else self._index_const(args[0])
        ub = self._to_index(args[1]) if not isinstance(args[1], int) else self._index_const(args[1])
        st = self._to_index(args[2]) if not isinstance(args[2], int) else self._index_const(args[2])
        tgt = node.target.id if isinstance(node.target, ast.Name) else None
        loop_snapshot = dict(self.env)
        carried = self._loop_carried(node.body, tgt)
        init_vals, init_shapes = [], []
        scalar_templates = {}
        for name in carried:
            v = self.env[name]
            if isinstance(v, SSA):
                init_vals.append(v)
                init_shapes.append((name, None))
                scalar_templates[name] = v
            else:
                init_vals.extend(v)
                init_shapes.append((name, len(v)))

        iv, arg_ssas, res_ssas = self.emitter.open_for(lb, ub, st, init_vals or None)
        self.assign(node.target, iv)
        k = 0
        for name, length in init_shapes:
            if length is None:
                self.env[name] = arg_ssas[k]
                k += 1
            else:
                self.env[name] = list(arg_ssas[k:k + length])
                k += length
        for s in node.body:
            self.exec_stmt(s)
        if init_shapes:
            cur = []
            for name, length in init_shapes:
                v = self.env[name]
                if length is None:
                    cur.append(self._as_ssa(v, template=scalar_templates[name]))
                else:
                    cur.extend(
                        self._as_ssa(item, template=init_vals[0])
                        for item in v
                    )
            self.emitter.yield_for(cur)
        self.emitter.close_for()
        self.env = dict(loop_snapshot)
        k = 0
        for name, length in init_shapes:
            if length is None:
                self.env[name] = res_ssas[k]
                k += 1
            else:
                self.env[name] = list(res_ssas[k:k + length])
                k += length

    def _loop_carried(self, body, target_name=None) -> list[str]:
        """Names rebound in the loop body holding SSA values (any type) or
        lists of SSAs — excluding the loop induction variable."""
        assigned = set()

        def walk(n):
            if isinstance(n, ast.Assign):
                for t in n.targets:
                    if isinstance(t, ast.Name):
                        assigned.add(t.id)
            for ch in ast.iter_child_nodes(n):
                walk(ch)

        for st in body:
            walk(st)
        out = []
        for name in sorted(assigned):
            if name == target_name:
                continue
            v = self.env.get(name)
            if isinstance(v, SSA):
                out.append(name)
            elif isinstance(v, list) and v and all(isinstance(x, SSA) for x in v):
                out.append(name)
        return out

    def _to_index(self, v) -> SSA:
        if isinstance(v, SSA):
            if v.type == "index":
                return v
            return self.emitter.ssa("index", f"arith.index_cast {v.name} : {v.type} to index")
        return self._index_const(int(v))

    def _index_const(self, v: int) -> SSA:
        return self.emitter.ssa("index", f"arith.constant {int(v)} : index")

    def assign(self, target: ast.expr, value) -> None:
        if isinstance(target, ast.Tuple):
            vals = _explode(value, len(target.elts))
            for t, v in zip(target.elts, vals):
                self.assign(t, v)
        elif isinstance(target, ast.Name):
            self.env[target.id] = value
        elif isinstance(target, ast.Subscript):
            base = self.eval(target.value)
            idx = self.eval(target.slice)
            base[idx] = value
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
            # support star-elements: (1, *t) -> flattened tuple
            vals = []
            for e in node.elts:
                if isinstance(e, ast.Starred):
                    v = self.eval(e.value)
                    if hasattr(v, "shape") and not isinstance(v, (list, tuple)):
                        v = tuple(v.shape)
                    vals.extend(list(v))
                else:
                    vals.append(self.eval(e))
            return tuple(vals)
        if isinstance(node, ast.List):
            return [self.eval(e) for e in node.elts]
        if isinstance(node, ast.Dict):
            return {
                self.eval(key): self.eval(value)
                for key, value in zip(node.keys, node.values)
            }
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            lv = self.eval(node.left)
            rv = self.eval(node.right)
            if isinstance(lv, list) and isinstance(rv, list):
                return lv + rv
            return self.binop(node.op, lv, rv)
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
            if isinstance(node.op, ast.USub):
                if isinstance(v, SSA) and v.type == "f32":
                    z = self.emitter.ssa("f32", "arith.constant 0.0 : f32")
                    return self.emitter.ssa(
                        v.type, f"arith.subf {z.name}, {v.name} : f32")
                if isinstance(v, SSA):
                    zero = self.emitter.const_i32(0)
                    return self.emitter.ssa(
                        v.type, f"arith.subi {zero.name}, {v.name} : {v.type}")
                if isinstance(v, (int, float)):
                    return -v
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
        if isinstance(node, ast.Subscript):
            base = self.eval(node.value)
            return self._subscript(base, node.slice)
        if isinstance(node, ast.ListComp):
            # support [expr for x in range(...)] with constexpr bounds
            gen = node.generators[0]
            assert isinstance(gen.iter, ast.Call) and gen.iter.func.id == "range", \
                "listcomp over range() only"
            args = [self.eval(a) for a in gen.iter.args]
            assert all(isinstance(a, int) for a in args), "listcomp bounds must be constexpr"
            rng = range(*(args if len(args) > 1 else (0, args[0])))
            out = []
            for i in rng:
                if isinstance(gen.target, ast.Tuple) and len(gen.target.elts) == 2:
                    self.env[gen.target.elts[0].id] = i
                    self.env[gen.target.elts[1].id] = i - rng.start
                else:
                    self.env[gen.target.id] = i
                out.append(self.eval(node.elt))
            return out
        if isinstance(node, ast.IfExp):
            cond = _unwrap_typed_scalar(self.eval(node.test))
            if isinstance(cond, SSA):
                def select_value(expr):
                    value = self.eval(expr)
                    if hasattr(value, "ssa"):
                        if value.ssa is None:
                            value.ir_value()
                        return value.ssa
                    return value

                true_value = select_value(node.body)
                false_value = select_value(node.orelse)
                true_ssa, false_ssa = self._pair(true_value, false_value)
                return self.emitter.ssa(
                    true_ssa.type,
                    f"arith.select {cond.name}, {true_ssa.name}, "
                    f"{false_ssa.name} : {true_ssa.type}",
                )
            return self.eval(node.body) if cond else self.eval(node.orelse)
        if isinstance(node, ast.JoinedStr):
            parts = []
            for v in node.values:
                if isinstance(v, ast.FormattedValue):
                    parts.append(_fmt(self.eval(v.value)))
                else:
                    parts.append(v.value)
            return "".join(parts)
        raise InterpError(f"unsupported expression {ast.dump(node)[:80]}")

    def _subscript(self, base, slice_node):
        """Dispatch t[...]: tile slicing, partition coords, fragment slots,
        flat pointer access."""
        from .kernel_objects import BlockTile, CoordPair, Fragment, KernelTensor, ThrPartition
        from .meta import TiledCoordinateMeta, TiledTensorMeta

        if isinstance(base, TiledCoordinateMeta):
            idx = self._single_dynamic_index(slice_node)
            if idx is not None:
                return BlockTile(None, base, idx, is_coord=True)
            raise InterpError("coord tensor slice needs a block index")
        if isinstance(base, KernelTensor):
            from .meta import TiledTensorMeta

            meta = base.meta
            is_tiled = isinstance(meta, (TiledTensorMeta, TiledCoordinateMeta))
            idx = self._single_dynamic_index(slice_node)
            if is_tiled and idx is not None:
                # ((None,None), bidx) style hierarchical block slice
                if isinstance(meta, TiledCoordinateMeta):
                    return BlockTile(None, meta, idx, is_coord=True)
                return BlockTile(base.ptr, meta, idx)
            # flat element access (1-D untiled tensors, M2 path)
            i = self.eval(slice_node)
            import os as _os
            if _os.environ.get("SC_TRACE3"):
                import sys as _s
                print("LOAD flat idx SSA:", i.name, i.type, file=_s.stderr)
            elem = getattr(meta, "element_type", None)
            return self._load_elem(base.ptr, i, elem, meta)
        if isinstance(base, ThrPartition):
            j = int(_const_index(self.eval(slice_node)))
            from . import builtins as _b

            return _b._coord_value(self.emitter, base, j)
        from .kernel_objects import _FragSlice
        if isinstance(base, (Fragment, _FragSlice)):
            v = self.eval(slice_node)
            return base[v]
        if isinstance(base, SSA):
            return self._load_elem(base, self.eval(slice_node))
        if isinstance(base, (list, tuple)):
            sl = slice_node
            if isinstance(sl, ast.Slice):
                lo = self.eval(sl.lower) if sl.lower else None
                hi = self.eval(sl.upper) if sl.upper else None
                st = self.eval(sl.step) if sl.step else None
                return base[(lo if lo is None else int(_const_index(lo))):
                            (hi if hi is None else int(_const_index(hi))):
                            (st if st is None else int(_const_index(st)))]
            v = self.eval(sl)
            if isinstance(v, (tuple, list)):        # (None,..,k) coords
                v = next((c for c in v if c is not None), 0)
            return base[int(_const_index(v))]
        from . import builtins as _bb

        if isinstance(base, _bb.SmemArray):
            i = self.eval(slice_node)
            off = getattr(base, "stage_offset", None)
            if off is not None:
                i = self.emitter.idx_binop("arith.addi", i, off)
            _f16 = base.elem.name.lower() in ("f16", "float16")
            p = self.emitter.gep_smem(base.ptr, i,
                                      "f16" if _f16 else "f32")
            ty = "f16" if _f16 else "f32"
            return self.emitter.ssa(ty, f"llvm.load {p.name} : !llvm.ptr<3> -> {ty}")
        # meta-view objects (PartitionedMmaOperand, Tensor views, ...): defer
        # to their own __getitem__ so host algebra stays with the object
        if hasattr(type(base), "__getitem__") and not isinstance(base, type):
            if isinstance(slice_node, ast.Tuple):
                parts = []
                for x in slice_node.elts:
                    if isinstance(x, ast.Starred):
                        v = self.eval(x.value)
                        parts.extend(v if isinstance(v, (tuple, list)) else [v])
                    elif isinstance(x, ast.Constant):
                        parts.append(x.value)
                    else:
                        parts.append(self.eval(x))
                sl = tuple(parts)
            else:
                sl = self.eval(slice_node)
            if isinstance(slice_node, ast.Slice):
                lo = self.eval(slice_node.lower) if slice_node.lower else None
                hi = self.eval(slice_node.upper) if slice_node.upper else None
                sl = slice(lo, hi, self.eval(slice_node.step)
                           if slice_node.step else None)
            return base[sl]
        raise InterpError(f"subscript on {type(base)} unsupported")

    def _single_dynamic_index(self, slice_node):
        """From ((None,None), bidx) extract the single non-None SSA index.

        Handles both literal tuple slices and names bound to tuple values.
        """
        def walk_node(n):
            if isinstance(n, ast.Tuple):
                for e in n.elts:
                    r = walk_node(e)
                    if r is not None:
                        return r
                return None
            if isinstance(n, ast.Constant) and n.value is None:
                return None
            v = self.eval(n)
            return _single_ssa(v)

        def _single_ssa(v):
            if isinstance(v, SSA):
                return v
            if isinstance(v, (tuple, list)):
                found = None
                for x in v:
                    r = _single_ssa(x)
                    if r is not None:
                        if found is not None:
                            return None  # more than one dynamic index
                        found = r
                return found
            return None

        if isinstance(slice_node, ast.Tuple):
            return walk_node(slice_node)
        return _single_ssa(self.eval(slice_node))

    # -- raw pointer element access (M2 tensors-as-pointers ABI) -------------

    def _flat_coord(self, idx, tensor_meta=None):
        """Flatten (r, c[, ...]) SSA/int coordinates through the tensor's
        strides into a single i64/i32 offset SSA (or int)."""
        if not isinstance(idx, (tuple, list)):
            if hasattr(idx, "ssa"):
                if idx.ssa is None:
                    idx.ir_value()
                return idx.ssa
            return idx
        strides = None
        if tensor_meta is not None:
            strides = getattr(tensor_meta, "stride", None)
        if not strides:
            strides = tuple(1 for _ in idx)  # fallback: contiguous
        from .emitter import SSA as _SSA
        e = self.emitter
        total = None
        for c, st in zip(idx, strides):
            v = c
            if hasattr(v, "ssa"):
                if v.ssa is None:
                    v.ir_value()
                v = v.ssa
            if isinstance(v, _SSA):
                if v.type == "index":
                    v = e.ssa("i64", f"arith.index_cast {v.name} : index to i64")
                elif v.type == "i32":
                    v = e.ssa("i64", f"arith.extsi {v.name} : i32 to i64")
                term = e.ssa(
                    "i64",
                    f"arith.muli {v.name}, "
                    f"{e.ssa('i64', f'arith.constant {int(st)} : i64').name} : i64",
                ) if int(st) != 1 else v
            else:
                term = int(v) * int(st)
            if total is None:
                total = term
            elif isinstance(total, int) and isinstance(term, int):
                total += term
            else:
                if isinstance(total, int):
                    total = e.ssa(
                        "i64", f"arith.constant {total} : i64")
                if isinstance(term, int):
                    term = e.ssa(
                        "i64", f"arith.constant {term} : i64")
                total = e.ssa(
                    "i64", f"arith.addi {total.name}, {term.name} : i64")
        return total

    def _load_elem(self, base: SSA, idx, elem=None, tensor_meta=None):
        assert isinstance(base, SSA) and base.type.startswith("!llvm.ptr"), \
            f"subscript load needs a pointer, got {base!r}"
        idx = self._flat_coord(idx, tensor_meta)
        elem_name = getattr(elem, "name", "")
        ety = {
            "f16": "f16", "Float16": "f16", "float16": "f16",
            "bf16": "bf16", "BFloat16": "bf16", "bfloat16": "bf16",
            "f32": "f32", "Float32": "f32", "float32": "f32",
            "i8": "i8", "Uint8": "i8", "Int8": "i8", "uint8": "i8",
            "i32": "i32", "Int32": "i32", "Uint32": "i32",
            "i64": "i64", "Int64": "i64", "Uint64": "i64",
        }.get(elem_name, "f32")
        p = self.emitter.gep(base, idx, ety)
        return self.emitter.ssa(
            ety, f"llvm.load {p.name} : !llvm.ptr<1> -> {ety}")

    def _store_elem(self, base: SSA, idx, val, elem=None, tensor_meta=None):
        assert isinstance(base, SSA) and base.type.startswith("!llvm.ptr"), \
            f"subscript store needs a pointer, got {base!r}"
        idx = self._flat_coord(idx, tensor_meta)
        _n = getattr(elem, "name", "")
        ety = {"f16": "f16", "Float16": "f16", "float16": "f16",
               "bf16": "bf16", "BFloat16": "bf16", "bfloat16": "bf16",
               "f32": "f32", "Float32": "f32", "float32": "f32",
               "i8": "i8", "Uint8": "i8", "Int8": "i8", "uint8": "i8",
               "i32": "i32", "Int32": "i32", "Uint32": "i32",
               "int32": "i32", "i64": "i64", "Int64": "i64"}.get(_n, "f32")
        if hasattr(val, "ssa"):  # bridge TypedScalar
            if val.ssa is None:
                val.ir_value()
            val = val.ssa
        p = self.emitter.gep(base, idx, ety)
        if isinstance(val, (int, float)):
            vv = int(val) if ety.startswith("i") else float(val)
            val = self.emitter.ssa(ety, f"arith.constant {vv} : {ety}")
        elif getattr(val, "type", None) and val.type != ety:
            vt, val_t = val.type, ety
            if vt == "i32" and val_t == "i8":
                val = self.emitter.ssa("i8", f"arith.trunci {val.name} : i32 to i8")
            elif vt == "i8" and val_t == "i32":
                val = self.emitter.ssa("i32", f"arith.extsi {val.name} : i8 to i32")
            elif vt == "i64" and val_t == "i32":
                val = self.emitter.ssa("i32", f"arith.trunci {val.name} : i64 to i32")
            elif vt == "f32" and val_t == "f16":
                val = self.emitter.ssa("f16", f"arith.truncf {val.name} : f32 to f16")
            elif vt == "f16" and val_t == "f32":
                val = self.emitter.ssa("f32", f"llvm.fpext {val.name} : f16 to f32")
            elif vt == "f32" and val_t == "bf16":
                val = self.emitter.ssa(
                    "bf16", f"arith.truncf {val.name} : f32 to bf16")
            elif vt == "bf16" and val_t == "f32":
                val = self.emitter.ssa(
                    "f32", f"arith.extf {val.name} : bf16 to f32")
        if ety in ("f16", "bf16"):
            if p.type.startswith("!llvm.ptr<3"):
                self.emitter.raw(
                    f"llvm.store {val.name}, {p.name} : {ety}, !llvm.ptr<3>")
            else:
                self.emitter.raw(
                    f"llvm.store {val.name}, {p.name} : {ety}, !llvm.ptr<1>")
            return
        if ety == "f32":
            self.emitter.store_f32(val, p)
            return
        self.emitter.raw(f"llvm.store {val.name}, {p.name} : {ety}, {p.type}")

    def binop(self, op, lhs, rhs):
        lhs = _unwrap_typed_scalar(lhs)
        rhs = _unwrap_typed_scalar(rhs)
        py_only = isinstance(lhs, (int, float)) and isinstance(rhs, (int, float))
        arith = {
            ast.Add: "arith.addi", ast.Sub: "arith.subi", ast.Mult: "arith.muli",
            ast.FloorDiv: "arith.divsi", ast.Mod: "arith.remsi",
            ast.Div: "arith.divi", ast.BitAnd: "arith.andi",
            ast.BitOr: "arith.ori", ast.BitXor: "arith.xori",
            ast.LShift: "arith.shli", ast.RShift: "arith.shrsi",
        }.get(type(op))
        if arith is None:
            if isinstance(op, ast.Add) and not py_only:
                arith = "arith.addf"
            else:
                if py_only:
                    return _py_binop(op, lhs, rhs)
                raise InterpError(f"unsupported binop {op}")
        from .kernel_objects import FragmentView as _FV

        _fvops = {ast.Add: "addf", ast.Sub: "subf",
                  ast.Mult: "mulf", ast.Div: "divf"}
        if isinstance(lhs, _FV) or isinstance(rhs, _FV):
            if type(op) not in _fvops:
                raise InterpError(f"unsupported FragmentView binop {op}")
            _o = _fvops[type(op)]
            if isinstance(lhs, _FV) and isinstance(rhs, _FV):
                if lhs.vecs and rhs.vecs:
                    vecs = []
                    for x, y in zip(lhs.vecs, rhs.vecs):
                        vecs.append(self.emitter.ssa(
                            x.type, f"arith.{_o} {x.name}, {y.name} : {x.type}"))
                    return _FV(vecs=vecs)
                vals = []
                for x, y in zip(lhs.values, rhs.values):
                    vals.append(self.emitter.ssa(
                        x.type, f"arith.{_o} {x.name}, {y.name} : {x.type}"))
                return _FV(vals)
            # scalar broadcast
            fv, scal, rev = (lhs, rhs, False) if isinstance(lhs, _FV) else (rhs, lhs, True)
            scal = _unwrap_typed_scalar(scal)
            scal_ssa = scal if isinstance(scal, SSA) else None
            vals = []
            for x in (fv.values or []):
                if scal_ssa is None:
                    c = self._const_like(x, float(scal))
                    y = c
                else:
                    y = self._pair_ssa(x, scal_ssa)
                a, b = (x, y) if not rev else (y, x)
                vals.append(self.emitter.ssa(
                    x.type, f"arith.{_o} {a.name}, {b.name} : {x.type}"))
            return _FV(vals)
        if isinstance(lhs, _FV) and isinstance(rhs, _FV):
            if not isinstance(op, ast.Add):
                raise InterpError("only FragmentView + FragmentView supported")
            if lhs.vecs and rhs.vecs:
                vecs = []
                for x, y in zip(lhs.vecs, rhs.vecs):
                    vecs.append(self.emitter.ssa(
                        x.type, f"arith.addf {x.name}, {y.name} : {x.type}"))
                return _FV(vecs=vecs)
            vals = []
            for x, y in zip(lhs.values, rhs.values):
                vals.append(self.emitter.ssa(x.type, f"arith.addf {x.name}, {y.name} : {x.type}"))
            return _FV(vals)
        if py_only:
            return _py_binop(op, lhs, rhs)
        a, b = self._pair(lhs, rhs)
        fop = arith[:-1] + "f" if a.type.startswith("f") and arith.endswith("i") else arith
        return self.emitter.ssa(a.type, f"{fop} {a.name}, {b.name} : {a.type}")

    def compare(self, node: ast.Compare):
        assert len(node.ops) == 1 and len(node.comparators) == 1, "chained compare unsupported"
        lhs = _unwrap_typed_scalar(self.eval(node.left))
        rhs = _unwrap_typed_scalar(self.eval(node.comparators[0]))
        op = node.ops[0]
        if isinstance(lhs, SSA) or isinstance(rhs, SSA):
            a, b = self._pair(lhs, rhs)
            pred = {
                ast.Eq: "eq", ast.NotEq: "ne", ast.Lt: "slt", ast.LtE: "sle",
                ast.Gt: "sgt", ast.GtE: "sge",
            }[type(op)]
            if a.type.startswith("f"):
                pred = {
                    ast.Eq: "oeq", ast.NotEq: "one", ast.Lt: "olt",
                    ast.LtE: "ole", ast.Gt: "ogt", ast.GtE: "oge",
                }[type(op)]
            return self.emitter.ssa("i1", f"arith.cmpi {pred}, {a.name}, {b.name} : {a.type}"
                                    if not a.type.startswith("f")
                                    else f"arith.cmpf {pred}, {a.name}, {b.name} : {a.type}")
        return _py_compare(op, lhs, rhs)

    def call(self, node: ast.Call):
        func = self.eval(node.func)
        args = [self.eval(a) for a in node.args]
        kwargs = {kw.arg: self.eval(kw.value) for kw in node.keywords if kw.arg}
        return func(*args, **kwargs)

    # --------------------------------------------------------------- helpers
    def to_i1(self, v) -> SSA:
        if isinstance(v, SSA) and v.type == "i1":
            return v
        if isinstance(v, SSA):
            raise InterpError("dynamic value in boolean context needs explicit compare")
        return self.emitter.ssa(
            "i1", f"arith.constant {int(bool(v))} : i1")

    def _pair(self, lhs, rhs) -> tuple[SSA, SSA]:
        if not isinstance(lhs, SSA):
            lhs = self._const_like(rhs, lhs)
        if not isinstance(rhs, SSA):
            rhs = self._const_like(lhs, rhs)
        if lhs.type != rhs.type:
            # integer-domain promotion: index <-> i32/i64 (thread coords
            # vs runtime scalars mix freely in official kernels)
            pair = (lhs.type, rhs.type)
            if "f32" in pair or "f16" in pair:
                raise InterpError(f"type mismatch {lhs.type} vs {rhs.type}")
            if lhs.type == "index":
                rhs = self.emitter.ssa(
                    "index", f"arith.index_cast {rhs.name} : {rhs.type} to index")
            elif rhs.type == "index":
                lhs = self.emitter.ssa(
                    "index", f"arith.index_cast {lhs.name} : {lhs.type} to index")
            else:
                raise InterpError(f"type mismatch {lhs.type} vs {rhs.type}")
        return lhs, rhs

    def _pair_ssa(self, a: SSA, b: SSA) -> SSA:
        if a.type == b.type:
            return b
        if a.type == "f32" and b.type in ("i32", "index"):
            return self.emitter.ssa("f32", f"arith.sitofp {b.name} : {b.type} to f32")
        if b.type == "f32" and a.type in ("i32", "index"):
            return self.emitter.ssa("f32", f"arith.sitofp {a.name} : {a.type} to f32")
        return b

    def _const_like(self, like: SSA, value) -> SSA:
        if like.type == "i32":
            return self.emitter.const_i32(int(value))
        if like.type == "i64":
            return self.emitter.const_i64(int(value))
        if like.type == "index":
            return self.emitter.ssa("index", f"arith.constant {int(value)} : index")
        if like.type == "f32":
            return self.emitter.ssa(
                "f32", "arith.constant " +
                f"{float(value):.10e}".replace("+", "") + " : f32")
        raise InterpError(f"no constant materializer for {like.type}")


class _DelayedCall:
    pass


def _explode(value, n: int) -> list:
    if isinstance(value, (tuple, list)) and len(value) == n:
        return list(value)
    # view-like objects (TmaPartitioned pairs etc.) with iteration protocol
    if hasattr(value, "__len__") and hasattr(value, "__getitem__") \
            and not isinstance(value, (str, bytes)) and len(value) == n:
        return [value[i] for i in range(n)]
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


def _unwrap_typed_scalar(v):
    """Fold cutlass.Int32(0)-style host constants during AST tracing.

    Typed values that wrap SSA objects are runtime casts and remain typed;
    only literal scalar payloads participate in Python constexpr arithmetic.
    Bridge TypedScalars (official cutlass_dsl scalars): SSA-carrying ones
    lower to their SSA handle, literal ones to the Python payload.
    """
    try:
        from cutlass.dtypes import TypedValue
    except ImportError:
        TypedValue = ()
    if TypedValue and isinstance(v, TypedValue) and isinstance(v.value, (bool, int, float)):
        return v.value
    try:
        from cutlass._bridge_helpers import TypedScalar
    except ImportError:
        return v
    if isinstance(v, TypedScalar):
        if v.ssa is not None:
            return v.ssa
        return v.value
    return v


def _py_binop(op, a, b):
    import operator as _op

    return {
        ast.Add: _op.add, ast.Sub: _op.sub, ast.Mult: _op.mul,
        ast.Div: _op.truediv, ast.FloorDiv: _op.floordiv, ast.Mod: _op.mod,
        ast.BitAnd: _op.and_, ast.BitOr: _op.or_, ast.BitXor: _op.xor,
        ast.LShift: _op.lshift, ast.RShift: _op.rshift,
    }[type(op)](a, b)


def _py_compare(op, a, b):
    import operator as _op

    return {
        ast.Eq: _op.eq, ast.NotEq: _op.ne, ast.Lt: _op.lt,
        ast.LtE: _op.le, ast.Gt: _op.gt, ast.GtE: _op.ge,
    }[type(op)](a, b)


def _is_tma_view(v) -> bool:
    from .cute_objects import Tensor as HostTensor

    return isinstance(v, HostTensor) and getattr(v, "is_tma_view", False)


def _is_coord_meta(v) -> bool:
    from .meta import CoordinateTensorMeta, TiledCoordinateMeta

    b = getattr(v, "base", v)
    return isinstance(v, (CoordinateTensorMeta, TiledCoordinateMeta)) or \
        isinstance(b, CoordinateTensorMeta)


def _const_index(v) -> int:
    """constexpr index (python int or Constexpr-wrapped); hierarchical
    tuple coords flatten to their linear index over (4,2)-style grids."""
    if isinstance(v, SSA):
        raise InterpError("value index must be compile-time constant")
    if isinstance(v, (tuple, list)):
        i = 0
        for c in v:
            i = i * 2 + int(getattr(c, "value", c))
        return i
    return int(getattr(v, "value", v))


def _fmt(v) -> str:
    if isinstance(v, SSA):
        return v.name
    return str(v)
