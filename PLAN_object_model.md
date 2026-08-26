# PLAN_object_model —— 系统性动态 layout 代数：复用已构建 cutlass-compiler 中的 cute 方言

> 来自另一个 zcode 会话，2026-08-26 08:25 交接。**用户已认可此方向**，要求：系统理解本方案 → 融入你的后续计划 → 清理不再需要的任务 → 执行。
> 整合并入 DEVLOG 后，本文件可移动或删除。有问题在 `scratch/race.log` 留言 @another-session（那是我在看的通道）。
> 恭喜 M6.5 落地（b57f6c1，50 tests）。你验证过的 8-warp / S2G epilogue / persistent 协议在下述方案里全部复用，不是白做。

## 0. TL;DR

不再手写任何 layout 代数。**我们已在用的 `build-compiler/` cutlass-compiler 二进制里就注册着官方开源的 cute 方言和三个代数 pass**（实测 `--help` 确认：`--cute-fold-static` / `--cute-expand-ops` / `--cute-to-base`）。新前端层把每个代数操作发射成 `cute.*` 文本 MLIR op，语义（静态折叠、动态展开、lowering）全部由这份带 5 万+ 行官方单测的 C++ 完成。Python 侧只有类型序列化 + 薄发射层 + 全静态快路径，永远不出现第二份代数实现。未来任何动态 layout 需求（含 swizzle、ComposedLayout、新 atom）都落在"发 op / 加表"两个扩展点上——这就是"系统性"的兑现方式。

## 1. 已验证的机器事实（全部实测，非推测）

1. `build-compiler/tools/cutlass-compiler/cutlass-compiler --help` 输出：
   `Available Dialects: arith, builtin, cf, cute, func, gpu, llvm, math, nvvm, scf, ub`，且有：
   - `--cute-expand-ops`（动态代数经 dialect conversion 展开）
   - `--cute-fold-static`（纯静态 op 折叠到 cute.static + DCE）
   - `--cute-to-base`（剩余 cute op 降低到 LLVM/arith/scf/ub）
2. 开放方言 op 表：`third_party/cutlass/cutlass_compiler/cute_ir/include/cute_ir/Dialect/Cute/IR/CuteOps.td`，72 个 def，覆盖：IntTuple/Shape/Stride/Coord/Tile 从**运行时动态叶子**构造、Layout(shape,stride)、**ComposedLayout(inner=layout-or-swizzle, offset, outer)**、按 major-ness 紧致化（coalesce 类）、用户指定序紧致化、**scaled-basis 恒等布局**、shape/stride 提取、inner-A 提取等——DEVLOG 标为"深水区"的 ComposedLayout/swizzle/动态维全部在内。
3. **TiledMma/thrfrg 类 op 不在开放方言**（官方经闭源 cute_nvgpu 的 C++ 类型方法实现；cute_nvgpu 全仓库无源码，只存在于官方 wheel 预编译 `_cutlass_ir.so`）。→ partition 在我们 Python 层做，但做成"trait 表 + 通用代数"（见 §2.3），不是逐内核手搓。
4. 官方 Python 层（`.venv-reference/` 的 wheel 与 `scratch/cutlass/python/CuTeDSL/`）是薄门面：`core.py` 6,662 行里 `coalesce` 就是一行 `_cute_ir.coalesce(...)`，对象模型直接子类化 `ir.Value`；真代数在 C++：cutegen 头 25,646 行（layout.hpp 4,474 含 composition L3730 / logical_divide L4044 / zipped_divide L4109；rec_var*.hpp 8,347 含 gcd；composed_layout.hpp 1,419；scaled_basis.hpp 426）+ cute_ir 16,070 行（fold/canonicalize/expand）+ ~52K 行单测。
5. `include/cute` C++ 无运行时动态（类型擦除）代数——**pybind 包 C++ 模板库不可行**；NVIDIA 自己的答案就是 cutegen（与 MLIR 耦合）。运行时 per-call JIT 编译 C++ 也不对：代数结果是关于 kernel 参数的符号表达式，不是宿主数值。MLIR 这一层才是正确 JIT 边界。
6. 我们的 emitter 本来就发文本 MLIR 喂 cutlass-compiler（`python/self_cuedsl/compiler/ptx_pipeline.py`——正确路径 `python/self_cutedsl/compiler/ptx_pipeline.py`），混合发射是现成机制。
7. 目标 API 面：`scratch/cutlass/examples/python/CuTeDSL/cute/hopper/kernel/dense_gemm/dense_gemm.py`（1,615 行，98 个符号；注意 4.7 里 `make_tma_atom` 已更名 `nvgpu.cpasync.make_tiled_tma_atom`）。

## 2. 方案架构

```
self_cutedsl/object_model/            （全部新建文件，不碰现有 WIP）
├─ types.py      Shape/Stride/Layout/Tile/ComposedLayout ↔ !cute<...> 文本类型序列化
│                （参照官方 typing.py 的 __get_mlir_types__ 语义改写为文本版，保留 BSD 版权头）
├─ algebra.py    make_layout/size/shape/stride/get/coalesce/composition/complement/
│                zipped_divide/logical_divide/slice_/group_modes/identity/
│                tile_to_shape/shape_div/inverse/local_tile/local_partition ...
│                · 全静态（叶子全是 Python int）→ 纯 Python 常量折叠，供 trace 期宿主决策
│                · 含动态叶子 → 发射 cute.* op 文本，动态叶子引用 %ssa（kernel 参数/循环变量）
├─ mma_atoms.py  每 atom 一张闭式 trait 表（A/B/C per-thread fragment 布局），
│                partition_A/B/C 统一经 algebra（composition/zipped_divide）发射
│                m16n8k16 的表用你现有已验证内核的手写偏移反向校准
├─ tma.py        make_tiled_tma_atom / tma_partition（host 侧 cuTensorMapEncodeTiled 已有；
│                去掉 builtins.py 里第二坐标硬编码 0；补 prefetch.tensormap；multicast 走 inline_ptx）
└─ pipeline.py   PipelineTmaAsync 最小类（官方 producer/consumer API 面），叠在已验证 mbarrier 内建上

编译流水线：emitter（上游方言 + cute.* 混合文本）
  → cutlass-compiler --cute-fold-static --cute-expand-ops --cute-to-base
                     --one-shot-convert-to-llvm --attach-nvvm-target=chip=sm_120a
                     --emit-gpu-binary=compilation-target=isa → PTX（同一个二进制）
```

设计原则：
- **语义唯一权威 = C++**。Python 静态快路径只做全静态整数代数（~300 行，供 smem 尺寸/stage 数等 trace 期决策），混合静态/动态一律保持符号进 IR。
- **TiledMma = 数据化**。新 atom = 加一张 trait 表，代数层零改动。这是"系统"与"手搓"的分界线。
- **三重 oracle 差分测试**：① cutlass-compiler fold 后文本；② header-only `include/cute` 的 C++ 探针程序（编译一次，纯 CPU，含 cutlass 头算 crd2idx）；③ `.venv-reference/` 官方 wheel 数值。三者一致才过关。composition/complement 的层次 gcd 边角语义由官方实现兜死。
- 每次发射自动跑"发射→cutlass-compiler 解析→fold"roundtrip 验证器，类型文本语法错误当场暴露。

## 3. 分阶段执行（每阶段独立验收，总 ~12–16h）

| 阶段 | 内容 | 估时 | 验收门 |
|---|---|---|---|
| Spike | 手写 ~20 行混合 .mlir（一个带动态叶子的 cute composition/divide + 一个 nvvm op），过 cutlass-compiler 三 cute pass + one-shot，确认出 PTX；同时从 CuteTypes.cpp / cute_ir 测试抄录 `!cute<...>` 文本语法；扫 `scratch/cutlass/cutlass_compiler/test/Integration/` 找现成 .mlir 样例 | ~1h | **GO/NO-GO** |
| S1 | types.py + algebra.py 核心 12 op（动态叶子接 SSA 符号）+ 三重 oracle 差分测试 | ~3h | 纯 CPU property 测试全绿 |
| S2 | dense_gemm 调用面其余 op + mma_atoms.py + partition；拿现有 golden GEMM 内核做**等价替换**回归（协议/mbarrier/ldmatrix 全不动，只换索引派生来源） | ~3h | 现有 golden 数值不变 |
| S3 | pipeline.py + tma.py 泛化（含 prefetch、去硬编码坐标） | ~2h | 流水线 golden 回归 |
| S4 | 全对象模型拼 dense_gemm 形态内核；目标挂"dense_gemm.py 原样跑" | ~3–4h | 新 golden |

NO-GO 退路（仅 Spike 失败才启用）：把 cute 代数放子模块单独过 cutlass-compiler fold/expand，产出的 arith 文本机械拼回主模块（emitter 是文本基础的，拼接直接）——语义仍归 C++。终极退路 pybind 绑定 cutlass_compiler 源码树（cute_nvgpu 闭源注定补不全，仅兜底）。

## 4. 顺带发现的独立 bug（现在就该修）

- `builtins.py` 里 `copy` 定义了两次：`:126`（elementwise 谓词化 tiled-copy 路径）被 `:440`（TmaPartitioned 路径）在模块级**覆盖**——前者已成死代码。
- `prefetch_descriptor` 是 no-op（`cutlass_compat/cutlass/cute/nvgpu/cpasync.py:35`）。
- `cute_objects.py` 的 `KernelTensorView`/`view_select`（M6.2）是无消费者死代码，连 compat 层都不导出。

## 5. 不必再做的任务（替代关系）

1. ~~"dense_gemm 原样跑 = N-atom 平铺 + 8-warp + epilogue + utils helpers 手写实现"~~（DEVLOG 会话总结里的旧计划）→ **替换为本方案 S1–S4**。已验证的手写内核全部保留：回归基线 + mma_atoms trait 表校准源。
2. **停止新增手写内核变体**（M6.6 起不要再手写第 N 个 GEMM 变体）。下一个性能档位等 S4 就绪后在 dense_gemm 形态上调。当前 persistent perf 调优（artifacts/perf/self_persistent_gemm.json）可收尾这轮，但优先级低于 Spike/S1。
3. M6.1 的 facade 骨架（`make_tiled_mma` 丢弃 atom、惰性 partition 标记、`tma_partition` 打包壳）：过渡期保留兼容旧测试，S2 完成后删除。
4. `KernelTensorView` 死代码：删。
5. scratch 里 dbg_*/racecheck 探针：归档；race.log 保留（那是两会话留言板）。
6. M7 blockscaled / M8 release 顺延不变——blockscaled 未来正好吃本方案扩展点（新 atom = 加表）。

建议路线图：M6.6 = Spike+S1，M6.7 = S2，M6.8 = S3+S4（原样跑目标挂这里）。

## 6. 协调

- 本会话暂不动手实现，此计划归你。建议先花 1h 跑 Spike 再全量排期。
- 我不再碰你的 WIP；你 Spike/S1 期间也只新建 `object_model/` 与新测试文件，避免与 perf 收尾冲突。
- 有问题/方案分歧：`scratch/race.log` 留言 @another-session（我轮询它）。
- 参照官方代码（typing 序列化逻辑、trait 数值）保留 BSD 版权头——本项目定位就是开源复刻。

## 附录 A：现状审计结论（2026-08-26，三路并行调查）

| 机制 | 现状 | 证据 |
|---|---|---|
| 内核内动态 layout 代数 | 缺失（host constexpr 元数据；无 coalesce/composition/complement/shape_div/gcd） | layout.py:19-53 扁平乘加；全仓库零命中 |
| TMA atom/partition | 外观壳（copy 第二坐标硬编码 0；无 multicast） | builtins.py:449-451 |
| TiledMma partition | 骨架（make_tiled_mma 丢弃 atom；gemm(None,...)；fragment 偏移手写） | builtins.py:305-316, cute_objects.py:325-343 |
| PipelineTmaAsync | 无库（stage/parity 每测试手算；官方 API 零命中） | test_tma_pipeline_gemm.py:62-70 |
| elementwise TiledCopy（rank-2） | **系统性实现**（官方 elementwise_add.py 原样跑的根基） | tiled.py:55-75, builtins.py:80-95 |
