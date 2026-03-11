#!/usr/bin/env python3
################################################################################
# GPU test helpers for subtile-based kernel unit tests.
#
# Provides:
#   - TileConfig dataclass for parameterized tile configurations
#   - Mock/kernel creation helpers (_mock_dtype, _create_kernel, create_writer_for_gpu)
#   - rocIsa initialization (init_rocisa)
#   - Unified kernel asm generator (generate_kernel_asm) with export/integration layouts
#   - Convenience wrapper for single-register-export tests (generate_export_kernel)
#   - Assembly & GPU execution (assemble_kernel, run_on_gpu, build_and_run)
#   - Debug utilities (print_offset_grid)
################################################################################

import ctypes
import os
import re
import sys
import struct
import subprocess
import tempfile

# Add tensilelite to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TENSILE_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
sys.path.insert(0, TENSILE_ROOT)

try:
    from hip import hip, hiprtc  # type: ignore
    HAS_HIP = True
except ImportError:
    HAS_HIP = False

from unittest.mock import MagicMock
from types import SimpleNamespace
from dataclasses import dataclass

from rocisa.register import RegisterPool
from rocisa.enum import RegisterType
from Tensile.Components.SubtileBasedKernel import TileInfo

# ---- Constants ----
GFX_TARGET = "gfx950"
WAVESIZE   = 64
NUM_WAVES  = 4
NUM_THREADS = WAVESIZE * NUM_WAVES  # 256
BPE        = 2      # fp16
LOAD_WIDTH = 16     # dwordx4


@dataclass
class TileConfig:
    """Parameterized tile configuration for testing."""
    mt_a: int       # MacroTileA
    mt_b: int       # MacroTileB
    depth_u: int    # DepthU
    stride_a: int = 0   # StrideA0I (in elements), only needed by GRA tests
    stride_b: int = 0   # StrideB1J (in elements), only needed by GRA tests
    use_swizzling: bool = False  # Whether to enable swizzling, only needed by GRA tests

    @property
    def label(self):
        swz = "_swz" if self.use_swizzling else ""
        stride = ""
        if self.stride_a and self.stride_a != self.depth_u:
            stride = f"_s{self.stride_a}"
        return f"{self.mt_a}x{self.mt_b}x{self.depth_u}{swz}{stride}"


# ---- HIP helpers ----

def hip_check(result):
    """Check HIP call result."""
    if isinstance(result, tuple):
        err = result[0]
        if err != 0:
            raise RuntimeError(f"HIP error {err}")
        return result[1] if len(result) == 2 else result[1:]
    if result != 0:
        raise RuntimeError(f"HIP error {result}")


# ---- Mock / setup helpers ----

def _mock_dtype(num_bytes=2):
    """Create a mock DataType that returns numBytes()."""
    mock = MagicMock()
    mock.numBytes.return_value = num_bytes
    return mock


def _create_kernel(cfg):
    """Create a minimal kernel dict matching the given tile config."""
    dtype = _mock_dtype(BPE)
    if ((cfg.mt_a//16) % 2 == 0) and ((cfg.mt_b//16) % 2 == 0):
        MIWaveGroup = [2,2]
    elif ((cfg.mt_a//16) % 2 != 0) and ((cfg.mt_b//16) % 4 == 0):
        MIWaveGroup = [1,4]
    elif ((cfg.mt_a//16) % 4 == 0) and ((cfg.mt_b//16) % 2 != 0):
        MIWaveGroup = [4,1]
    else:
        raise ValueError(f"Unsupported tile config for wave grouping: mt_a={cfg.mt_a}, mt_b={cfg.mt_b}")

    return {
        "DepthU": cfg.depth_u,
        "MacroTileA": cfg.mt_a,
        "MacroTileB": cfg.mt_b,
        "MacroTile0": cfg.mt_a,
        "MacroTile1": cfg.mt_b,
        "MatrixInstM": 16,
        "MatrixInstK": 32,
        "MIWaveGroup": MIWaveGroup,
        "WavefrontSize": WAVESIZE,
        "ProblemType": {
            "DataTypeA": dtype,
            "DataTypeB": dtype,
            "ComputeDataType": _mock_dtype(4),
        },
    }


def create_writer_for_gpu(cfg):
    """Create a mock writer with register pools laid out for GPU execution.

    Register layout (must match the kernel prologue/epilogue):
      v0           = Serial
      v1+          = allocated by allocOffsetRegisters and the function under test

      s0:s1        = kernarg_segment_ptr
      s2           = workgroup_id_x
      s3           = padding
      s[4:5]       = input_A_ptr (integration) or output_ptr (export kernel)
      s[6:7]       = input_B_ptr (integration) or free (export kernel)
      s[8:9]       = output_ptr (integration) or strideA/B (export kernel)
      s10          = strideA (integration, mapped via .set sgprStrideA0I)
      s11          = strideB (integration, mapped via .set sgprStrideB1J)
      s12+         = sgprPool for temps (sHalfOffset, subtile offsets, etc.)

    Note: export kernel (test_graTileAssignment) uses .set sgprStrideA0I=8,
    sgprStrideB1J=9, which is fine since those sgprs are still reserved.
    """
    writer = SimpleNamespace()

    writer.vgprPool = RegisterPool(0, RegisterType.Vgpr,
                                    defaultPreventOverflow=False, printRP=False)
    writer.sgprPool = RegisterPool(0, RegisterType.Sgpr,
                                    defaultPreventOverflow=False, printRP=False)

    # Reserve v0 for Serial (hardware workitem_id)
    writer.vgprPool.checkOut(1)

    # Reserve s0-s11 for hardware regs + kernarg loads
    # s[0:1]=kernarg, s2=workgroup_id_x, s3=pad,
    # s[4:5]=input_A, s[6:7]=input_B, s[8:9]=output, s10=strideA, s11=strideB
    writer.sgprPool.checkOut(12)

    # Build kernel and TileInfo
    kernel = _create_kernel(cfg)
    
    tileInfoA = TileInfo('A', kernel)
    tileInfoB = TileInfo('B', kernel)

    writer.states = SimpleNamespace(
        a=SimpleNamespace(tileInfo=tileInfoA),
        b=SimpleNamespace(tileInfo=tileInfoB),
        regCaps={"MaxSgpr": 106, "MaxVgpr": 256},
    )

    tileInfoA.allocOffsetRegisters(writer, kernel)
    tileInfoB.allocOffsetRegisters(writer, kernel)

    return writer, kernel, tileInfoA, tileInfoB


def create_writer_for_subtile_test(cfg):
    """Create a mock writer with allocations needed by production GR/LR functions.

    Extends create_writer_for_gpu with:
      - SrdA/SrdB sgprs (4 each, 4-aligned) for buffer_load descriptors
      - LocalWriteBaseAddr/LocalWriteDTLOffset sgprs for DTL init
      - ldsStartOffset{A,B} for emitSubtileBufferLoad m0 calculations
      - vgprTile registers allocated via TileInfo.allocVgprTileRegisters

    Returns:
        (writer, kernel, tileInfoA, tileInfoB, named_sgprs)
        named_sgprs maps symbolic names to allocated sgpr indices for .set directives.
    """
    writer, kernel, tileInfoA, tileInfoB = create_writer_for_gpu(cfg)

    # Allocate SrdA: 4 sgprs, 4-aligned
    srdA = writer.sgprPool.checkOutAligned(4, 4, "SrdA", preventOverflow=False)
    # Allocate SrdB: 4 sgprs, 4-aligned
    srdB = writer.sgprPool.checkOutAligned(4, 4, "SrdB", preventOverflow=False)
    # Allocate LocalWriteBaseAddr and LocalWriteDTLOffset
    lwba = writer.sgprPool.checkOut(1, "LocalWriteBaseAddr", preventOverflow=False)
    lwdtl = writer.sgprPool.checkOut(1, "LocalWriteDTLOffset", preventOverflow=False)

    # Set LDS start offsets used by emitSubtileBufferLoad
    writer.ldsStartOffsetA = 0
    writer.ldsStartOffsetB = cfg.mt_a * cfg.depth_u * BPE

    # Allocate vgprTile registers for local read destinations
    tileInfoA.allocVgprTileRegisters(writer, kernel)
    tileInfoB.allocVgprTileRegisters(writer, kernel)

    named_sgprs = {
        "SrdA": srdA,
        "SrdB": srdB,
        "StrideA0I": 10,
        "StrideB1J": 11,
        "LocalWriteBaseAddr": lwba,
        "LocalWriteDTLOffset": lwdtl,
    }

    return writer, kernel, tileInfoA, tileInfoB, named_sgprs


def init_rocisa():
    """Initialize rocIsa singleton if needed."""
    from rocisa import rocIsa
    ri = rocIsa.getInstance()
    if not ri.isInit():
        import shutil
        asmpath = shutil.which('amdclang++') or '/usr/bin/amdclang++'
        ri.init((9, 5, 0), asmpath)
    ri.setKernel((9, 5, 0), WAVESIZE)


# ---- Kernel assembly generator ----

# Pre-defined prologue / args for the two standard kernel layouts.

EXPORT_PROLOGUE = """\
  s_load_dwordx2 s[4:5], s[0:1], 0x00     // output_ptr
  s_load_dword s[sgprStrideA0I], s[0:1], 0x08   // strideA -> s8
  s_load_dword s[sgprStrideB1J], s[0:1], 0x0c   // strideB -> s9
  s_waitcnt lgkmcnt(0)"""

EXPORT_ARGS = (
    ("output_ptr", 8, "global_buffer", "u32"),
    ("strideA",    4, "by_value",      "u32"),
    ("strideB",    4, "by_value",      "u32"),
)

EXPORT_SET_DIRECTIVES = ".set sgprStrideA0I, 8\n.set sgprStrideB1J, 9"

INTEGRATION_PROLOGUE = """\
  s_load_dwordx4 s[4:7], s[0:1], 0x00       // input_A_ptr (s[4:5]) + input_B_ptr (s[6:7])
  s_load_dwordx4 s[8:11], s[0:1], 0x10      // output_ptr (s[8:9]) + strideA (s10) + strideB (s11)
  s_waitcnt lgkmcnt(0)"""

INTEGRATION_ARGS = (
    ("input_A_ptr", 8, "global_buffer", "f16"),
    ("input_B_ptr", 8, "global_buffer", "f16"),
    ("output_ptr",  8, "global_buffer", "u32"),
    ("strideA",     4, "by_value",      "u32"),
    ("strideB",     4, "by_value",      "u32"),
)


def _generate_args_metadata(args):
    """Generate YAML args metadata and kernarg_segment_size from an args list.

    Each arg is a tuple: (name, size, value_kind, value_type).
    For global_buffer args, address_space is set to "global".

    Returns:
        (args_yaml_str, kernarg_size) tuple.
    """
    lines = []
    offset = 0
    for name, size, value_kind, value_type in args:
        lines.append(f"      - .name:            {name}")
        lines.append(f"        .size:            {size}")
        lines.append(f"        .offset:          {offset}")
        lines.append(f"        .value_kind:      {value_kind}")
        lines.append(f"        .value_type:      {value_type}")
        if value_kind == "global_buffer":
            lines.append(f"        .address_space:   global")
        offset += size
    kernarg_size = (offset + 7) & ~7  # align to 8
    return "\n".join(lines), kernarg_size


def _scan_register_indices(*texts):
    """Extract vgpr and sgpr index sets from assembly text.

    Handles bare references (s10, v3) and range references (s[4:7], v[0:3]).
    Also extracts indices from .set directives (.set sgprFoo, 12).
    """
    vgprs = set()
    sgprs = set()
    for text in texts:
        vgprs.update(int(m) for m in re.findall(r'\bv(\d+)\b', text))
        sgprs.update(int(m) for m in re.findall(r'\bs(\d+)\b', text))
        for start, end in re.findall(r'\bs\[(\d+):(\d+)\]', text):
            sgprs.update(range(int(start), int(end) + 1))
        for start, end in re.findall(r'\bv\[(\d+):(\d+)\]', text):
            vgprs.update(range(int(start), int(end) + 1))
        # .set sgprFoo, N  /  .set vgprBar, N
        sgprs.update(int(m) for m in re.findall(r'\.set\s+sgpr\w+,\s*(\d+)', text))
        vgprs.update(int(m) for m in re.findall(r'\.set\s+vgpr\w+,\s*(\d+)', text))
    return vgprs, sgprs


def generate_kernel_asm(inner_asm, lds_size=0, set_directives="",
                        args=None, prologue=None):
    """Generate a complete AMDHSA kernel wrapping inner_asm.

    Args:
        inner_asm:       Assembly code between prologue and endpgm.
        lds_size:        LDS allocation size in bytes.
        set_directives:  .set lines for register name mappings.
        args:            Kernarg descriptors — sequence of
                         (name, size, value_kind, value_type) tuples.
                         Defaults to INTEGRATION_ARGS.
        prologue:        Prologue asm (kernarg loads). Defaults to
                         INTEGRATION_PROLOGUE.

    Returns:
        Assembly source string.
    """
    if args is None:
        args = INTEGRATION_ARGS
    if prologue is None:
        prologue = INTEGRATION_PROLOGUE

    args_metadata, kernarg_size = _generate_args_metadata(args)

    # Auto-compute register counts from prologue + inner_asm + set directives
    vgpr_indices, sgpr_indices = _scan_register_indices(
        prologue, inner_asm, set_directives)

    max_vgpr = max(vgpr_indices | {0}) + 1
    max_sgpr = max(sgpr_indices | {0}) + 1
    max_vgpr = max(((max_vgpr + 3) // 4) * 4, 4)  # align to 4 for accum_offset

    return f"""\
.amdgcn_target "amdgcn-amd-amdhsa--{GFX_TARGET}"

// Register name mappings
.set vgprSerial, 0
{set_directives}

.text
.protected test_kernel
.globl test_kernel
.p2align 8
.type test_kernel,@function

.section .rodata,#alloc
.p2align 6
.amdhsa_kernel test_kernel
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
  .amdhsa_accum_offset {max_vgpr}
  .amdhsa_next_free_vgpr {max_vgpr}
  .amdhsa_next_free_sgpr {max_sgpr}
  .amdhsa_group_segment_fixed_size {lds_size}
  .amdhsa_private_segment_fixed_size 0
  .amdhsa_system_sgpr_workgroup_id_x 1
  .amdhsa_system_sgpr_workgroup_id_y 0
  .amdhsa_system_sgpr_workgroup_id_z 0
  .amdhsa_system_vgpr_workitem_id 0
  .amdhsa_float_denorm_mode_32 3
  .amdhsa_float_denorm_mode_16_64 3
.end_amdhsa_kernel

.text
test_kernel:
  // ---- Prologue: Load kernel arguments ----
{prologue}

  // ---- Test code ----
{inner_asm}
  s_waitcnt vmcnt(0)
  s_endpgm

.amdgpu_metadata
---
amdhsa.version:
  - 1
  - 1
amdhsa.kernels:
  - .name: test_kernel
    .symbol: 'test_kernel.kd'
    .language: OpenCL C
    .language_version:
      - 2
      - 0
    .args:
{args_metadata}
    .kernarg_segment_size: {kernarg_size}
    .kernarg_segment_align: 8
    .group_segment_fixed_size: {lds_size}
    .private_segment_fixed_size: 0
    .wavefront_size: {WAVESIZE}
    .sgpr_count: {max_sgpr}
    .vgpr_count: {max_vgpr}
    .max_flat_workgroup_size: {NUM_THREADS}
...
.end_amdgpu_metadata
"""


def generate_export_kernel(test_asm, export_reg, is_sgpr=False):
    """Generate a kernel that runs test_asm and exports a single register.

    Args:
        test_asm:    Assembly string from the function under test.
        export_reg: Register index to export (e.g. 3 for v3 or s3).
        is_sgpr:    True to export an sgpr (uniform value, broadcast to all
                    threads), False to export a vgpr (per-thread value).

    Returns:
        Assembly source string.
    """
    # Find highest register indices used by test_asm
    vgpr_indices = set(int(m) for m in re.findall(r'\bv(\d+)\b', test_asm))

    tmp_vgpr = max(vgpr_indices | {0}) + 1      # byte-offset register
    data_vgpr = tmp_vgpr + 1 if is_sgpr else export_reg

    # Build epilogue
    epilogue = f"  v_lshlrev_b32 v{tmp_vgpr}, 2, v0\n"
    if is_sgpr:
        epilogue += f"  v_mov_b32 v{data_vgpr}, s{export_reg}\n"
    epilogue += f"  global_store_dword v{tmp_vgpr}, v{data_vgpr}, s[4:5]\n"

    inner_asm = f"""\
  // ---- Generated test code ----
{test_asm}
  // ---- Epilogue: Export register ----
{epilogue}"""

    return generate_kernel_asm(inner_asm, args=EXPORT_ARGS,
                               prologue=EXPORT_PROLOGUE,
                               set_directives=EXPORT_SET_DIRECTIVES)


def run_integration_on_gpu(co_path, input_data_A, input_data_B, cfg, output_size=None):
    """Launch an integration kernel on GPU and return output buffer.

    Uses the standard integration kernarg layout:
      input_A_ptr (8B), input_B_ptr (8B), output_ptr (8B), strideA (4B), strideB (4B)

    Args:
        co_path:      Path to assembled .co file.
        input_data_A: numpy array of input values for matrix A.
        input_data_B: numpy array of input values for matrix B.
        cfg:          TileConfig with mt_a, mt_b, depth_u, stride_a, stride_b.
        output_size:  Output buffer size in bytes (default: (mt_a+mt_b)*depth_u*BPE).

    Returns:
        bytes: Raw output buffer contents.
    """
    hip_check(hip.hipInit(0))

    module = hip_check(hip.hipModuleLoad(co_path.encode() if isinstance(co_path, str) else co_path))
    kernel = hip_check(hip.hipModuleGetFunction(module, b"test_kernel"))

    input_bytes_A = input_data_A.tobytes()
    input_bytes_B = input_data_B.tobytes()
    input_size_A = len(input_bytes_A)
    input_size_B = len(input_bytes_B)
    if output_size is None:
        output_size = (cfg.mt_a + cfg.mt_b) * cfg.depth_u * BPE

    lds_size = (cfg.mt_a + cfg.mt_b) * cfg.depth_u * BPE

    d_input_A = hip_check(hip.hipMalloc(input_size_A))
    d_input_B = hip_check(hip.hipMalloc(input_size_B))
    d_output = hip_check(hip.hipMalloc(output_size))

    hip_check(hip.hipMemcpyHtoD(d_input_A, input_bytes_A, input_size_A))
    hip_check(hip.hipMemcpyHtoD(d_input_B, input_bytes_B, input_size_B))
    hip_check(hip.hipMemset(d_output, 0, output_size))

    class KernelArgs(ctypes.Structure):
        _fields_ = [
            ("input_A_ptr", ctypes.c_uint64),
            ("input_B_ptr", ctypes.c_uint64),
            ("output_ptr", ctypes.c_uint64),
            ("stride_a", ctypes.c_uint32),
            ("stride_b", ctypes.c_uint32),
        ]

    kargs = KernelArgs(int(d_input_A), int(d_input_B), int(d_output),
                       cfg.stride_a, cfg.stride_b)
    kargs_size = ctypes.c_size_t(ctypes.sizeof(kargs))

    HIP_LAUNCH_PARAM_BUFFER_POINTER = 0x01
    HIP_LAUNCH_PARAM_BUFFER_SIZE    = 0x02
    HIP_LAUNCH_PARAM_END            = 0x03

    extra = (ctypes.c_void_p * 5)(
        ctypes.c_void_p(HIP_LAUNCH_PARAM_BUFFER_POINTER),
        ctypes.c_void_p(ctypes.addressof(kargs)),
        ctypes.c_void_p(HIP_LAUNCH_PARAM_BUFFER_SIZE),
        ctypes.c_void_p(ctypes.addressof(kargs_size)),
        ctypes.c_void_p(HIP_LAUNCH_PARAM_END),
    )

    hip_check(hip.hipModuleLaunchKernel(
        kernel,
        1, 1, 1,
        NUM_THREADS, 1, 1,
        lds_size,
        None,
        None,
        extra
    ))
    hip_check(hip.hipDeviceSynchronize())

    h_output = bytearray(output_size)
    hip_check(hip.hipMemcpyDtoH(h_output, d_output, output_size))

    hip_check(hip.hipFree(d_input_A))
    hip_check(hip.hipFree(d_input_B))
    hip_check(hip.hipFree(d_output))
    hip_check(hip.hipModuleUnload(module))

    return bytes(h_output)


# ---- Assemble / run ----

def assemble_kernel(asm_source, output_path):
    """Assemble .s source to .co code object."""
    with tempfile.NamedTemporaryFile(suffix=".s", mode="w", delete=False) as f:
        f.write(asm_source)
        asm_path = f.name

    obj_path = asm_path.replace(".s", ".o")

    try:
        subprocess.check_call([
            "amdclang++", "-x", "assembler",
            "--target=amdgcn-amd-amdhsa",
            f"-mcpu={GFX_TARGET}",
            "-mwavefrontsize64",
            "-mcode-object-version=5",
            "-o", obj_path,
            asm_path
        ])
        os.rename(obj_path, output_path)
    finally:
        if os.path.exists(asm_path):
            os.unlink(asm_path)
        if os.path.exists(obj_path) and obj_path != output_path:
            os.unlink(obj_path)


def run_on_gpu(co_path, stride_a, stride_b, num_threads):
    """Load code object, launch kernel, read single output buffer."""
    hip_check(hip.hipInit(0))
    device = hip_check(hip.hipGetDevice())

    module = hip_check(hip.hipModuleLoad(co_path.encode() if isinstance(co_path, str) else co_path))
    kernel = hip_check(hip.hipModuleGetFunction(module, b"test_kernel"))

    buf_size = num_threads * 4  # 4 bytes per u32
    d_out = hip_check(hip.hipMalloc(buf_size))
    hip_check(hip.hipMemset(d_out, 0, buf_size))

    class KernelArgs(ctypes.Structure):
        _fields_ = [
            ("ptr_out", ctypes.c_uint64),
            ("stride_a", ctypes.c_uint32),
            ("stride_b", ctypes.c_uint32),
        ]

    kargs = KernelArgs(int(d_out), stride_a, stride_b)
    kargs_size = ctypes.c_size_t(ctypes.sizeof(kargs))

    HIP_LAUNCH_PARAM_BUFFER_POINTER = 0x01
    HIP_LAUNCH_PARAM_BUFFER_SIZE    = 0x02
    HIP_LAUNCH_PARAM_END            = 0x03

    extra = (ctypes.c_void_p * 5)(
        ctypes.c_void_p(HIP_LAUNCH_PARAM_BUFFER_POINTER),
        ctypes.c_void_p(ctypes.addressof(kargs)),
        ctypes.c_void_p(HIP_LAUNCH_PARAM_BUFFER_SIZE),
        ctypes.c_void_p(ctypes.addressof(kargs_size)),
        ctypes.c_void_p(HIP_LAUNCH_PARAM_END),
    )

    hip_check(hip.hipModuleLaunchKernel(
        kernel,
        1, 1, 1,                 # grid
        num_threads, 1, 1,       # block
        0,                       # shared mem
        None,                    # stream
        None,                    # kernel params (unused with extra)
        extra                    # extra params
    ))
    hip_check(hip.hipDeviceSynchronize())

    h_out = bytearray(buf_size)
    hip_check(hip.hipMemcpyDtoH(h_out, d_out, buf_size))

    hip_check(hip.hipFree(d_out))
    hip_check(hip.hipModuleUnload(module))

    return struct.unpack(f"{num_threads}I", h_out)


def build_and_run(test_asm, export_reg, is_sgpr, cfg, tmp_path, label):
    """Generate, assemble, run a single-register export kernel. Returns results tuple."""
    sys.stdout.flush()  # flush before GPU calls to avoid buffering issues with HIP runtime
    asm = generate_export_kernel(test_asm, export_reg, is_sgpr=is_sgpr)
    co_path = str(tmp_path / f"test_{label}.co")
    asm_path = str(tmp_path / f"test_{label}.s")
    with open(asm_path, "w") as f:
        f.write(asm)
    assemble_kernel(asm, co_path)
    return run_on_gpu(co_path, cfg.stride_a, cfg.stride_b, NUM_THREADS)


# ---- Utilities ----

def print_offset_grid(label, results, wavesize, num_waves):
    """Print offsets as a 2D grid: rows = waves, columns = lanes."""
    print(f"\n--- {label} offsets (rows=waves, cols=lanes) ---")
    print(f"{'wave':>6}", end="")
    for lane in range(wavesize):
        print(f" {lane:>6}", end="")
    print()
    print("-" * (7 + 7 * wavesize))
    for w in range(num_waves):
        print(f"{w:>6}", end="")
        for lane in range(wavesize):
            tid = w * wavesize + lane
            print(f" {results[tid]:>6}", end="")
        print()
