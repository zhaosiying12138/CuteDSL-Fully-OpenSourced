import os
import subprocess

import lit.formats
import lit.util

config.name = "selfcute"
config.test_format = lit.formats.ShTest(False)  # internal shell (LLVM-23 deprecates external)

config.suffixes = [".mlir"]

config.test_exec_root = os.path.join(config.selfcute_obj_root, "test")
config.test_source_root = os.path.join(config.selfcute_src_root, "test")

config.substitutions.append(("%selfcute_opt", os.path.join(
    config.selfcute_obj_root, "bin", "selfcute-opt")))

_tools = getattr(config, "lit_tools_dir", None) or os.environ.get("LLVM_TOOLS_DIR", "")
config.substitutions.append(("%FileCheck", os.path.join(_tools, "FileCheck")))
