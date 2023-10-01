#  Licensed under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
import platform
import re
import shutil
import subprocess
import tempfile

import lit.formats
import lit.util

from lit.llvm import llvm_config
from lit.llvm.subst import ToolSubst
from lit.llvm.subst import FindTool

# Configuration file for the 'lit' test runner.

# name: The name of this test suite.
config.name = "Pylir"

config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)

# suffixes: A list of file extensions to treat as test files.
config.suffixes = [".mlir", ".ll", ".py", ".td"]

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.pylir_obj_root, "test")

config.substitutions.append(("%PATH%", config.environment["PATH"]))

llvm_config.with_system_environment(
    [
        "HOME",
        "INCLUDE",
        "LIB",
        "TMP",
        "TEMP",
        "TSAN_OPTIONS",
        "ASAN_OPTIONS",
        "UBSAN_OPTIONS",
    ]
)

llvm_config.use_default_substitutions()

for arch in config.llvm_targets_to_build.split(";"):
    config.available_features.add(arch.lower() + "-registered-target")

# excludes: A list of directories to exclude from the testsuite. The 'Inputs'
# subdirectories contain auxiliary inputs for various tests in their parent
# directories.
config.excludes = ["Inputs", "CMakeLists.txt", "pylir-lit.py", "lit.cfg.py"]

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.pylir_obj_root, "test")

# Tweak the PATH to include the tools dir.
llvm_config.with_environment("PATH", config.llvm_tools_dir, append_path=True)

tool_dirs = [config.pylir_tools_dir, config.mlir_tools_dir, config.llvm_tools_dir]

tools = [
    ToolSubst("pylir", extra_args=["%{PYLIR_ADDITIONAL_FLAGS}"]),
    "pylir-opt",
    ToolSubst(
        "pylir-tblgen",
        extra_args=["-I" + x for x in config.mlir_include_dirs.split(";")]
        + ["-I" + config.pylir_src_root + "/src"],
        unresolved="fatal",
    ),
]

llvm_config.add_tool_substitutions(tools, tool_dirs)

# FileCheck -enable-var-scope is enabled by default in MLIR test
# This option avoids to accidentally reuse variable across -LABEL match,
# it can be explicitly opted-in by prefixing the variable name with $
config.environment["FILECHECK_OPTS"] = "-enable-var-scope --allow-unused-prefixes=false"

# Optional additional flags added to all pylir invocations. This can be
# overwritten in subdirectories through their `lit.local.cfg`.
config.substitutions.append(("%{PYLIR_ADDITIONAL_FLAGS}", ""))
