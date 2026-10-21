# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Cold-JIT source isolation, using real compilation without GPU execution."""

import os
import re
import shutil
import subprocess
from importlib import metadata
from pathlib import Path

import pytest
from tokenspeed_kernel.thirdparty.flashinfer.sm90_moe import _source_tree


def test_private_cold_jit_ignores_shared_include_directory_changes(tmp_path):
    try:
        package = metadata.distribution("flashinfer-python")
    except metadata.PackageNotFoundError:
        pytest.skip("Requires FlashInfer sources")
    if package.version != "0.7.2":
        pytest.skip("The private source patch targets FlashInfer 0.7.2")
    cuda_home = os.environ.get("CUDA_HOME")
    nvcc = str(Path(cuda_home) / "bin/nvcc") if cuda_home else shutil.which("nvcc")
    if not nvcc or not Path(nvcc).is_file():
        pytest.skip("Requires the CUDA toolkit, but no GPU")
    version = subprocess.run(
        [nvcc, "--version"], check=True, capture_output=True, text=True
    ).stdout
    release = re.search(r"release (\d+)\.(\d+)", version)
    if release is None or tuple(map(int, release.groups())) < (12, 8):
        pytest.skip("Requires CUDA toolkit 12.8 or newer")
    cuda = Path(nvcc).resolve().parent.parent
    package_root = Path(package.locate_file("flashinfer"))
    directory, _ = _source_tree(
        package_root, package_root / "data/csrc", tmp_path / "source", 4, -1
    )
    csrc = directory / "csrc"
    common = csrc / "nv_internal/cpp/common"
    fixtures = Path(__file__).with_name("sm90_moe_jit")
    executable = tmp_path / "compiler_probe"
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES="",
        TRTLLM_DG_CACHE_DIR=str(tmp_path / "compiled"),
        TRTLLM_DG_NVCC_COMPILER=str(cuda / "bin/nvcc"),
    )
    command = [
        nvcc,
        str(fixtures / "compiler_probe.cu"),
        str(fixtures / "device_stubs.cpp"),
        str(common / "logger.cpp"),
        str(common / "tllmException.cpp"),
        str(common / "stringUtils.cpp"),
        "-o",
        str(executable),
        "-std=c++17",
        "-O2",
        "-arch=sm_90a",
        "--cudart=shared",
        "-DFLASHINFER_SM90_GROUPED_MOE_OPTIMIZATIONS=1",
        f"-I{csrc}/nv_internal/tensorrt_llm",
        f"-I{csrc}/nv_internal",
        f"-I{csrc}/nv_internal/include",
        f"-L{cuda}/lib64",
        "-lnvrtc",
        "-ldl",
        "-Xlinker",
        f"-rpath={cuda}/lib64",
    ]
    built = subprocess.run(
        command, env=env, capture_output=True, text=True, timeout=180
    )
    assert built.returncode == 0, built.stdout + built.stderr
    # Model a later module overwriting FlashInfer's shared include-directory state.
    foreign = tmp_path / "foreign"
    (foreign / "deep_gemm").mkdir(parents=True)
    (foreign / "deep_gemm/nvrtc_cutlass.cuh").write_text(
        "#error Private compilation used another module's include directory\n"
    )
    compiled = subprocess.run(
        [str(executable), str(csrc / "nv_internal/tensorrt_llm"), str(foreign)],
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert compiled.returncode == 0, compiled.stdout + compiled.stderr
