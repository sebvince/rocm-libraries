#!/usr/bin/env python3
################################################################################
# End-to-end GPU integration test: GRA offsets write to LDS via buffer_load,
# LRA offsets read from LDS via ds_read, then output is compared to input.
#
# Data flow:
#   Input matrix (fp16, mt_a x depthU, row-major with stride=depthU)
#     -> buffer_load_dwordx4 ... offen lds   (GRA offset = voffset + LDS write addr)
#     -> s_barrier
#     -> ds_read_b128 ... (LRA offset)
#     -> flat_store_dwordx4 to output buffer
#
# With stride == depthU, GRA offsets become LDS-compatible, so the same offset
# serves as both the global voffset and the LDS write address.
#
# Usage:
#   pytest test_graLraIntegration.py -v -s
#   python test_graLraIntegration.py --debug
################################################################################

import ctypes
import os
import re
import sys
import tempfile

import pytest
import numpy as np
from types import SimpleNamespace

from gpu_test_helpers import (
    HAS_HIP,
    TileConfig,
    BPE, LOAD_WIDTH, WAVESIZE, NUM_THREADS, NUM_WAVES,
    GFX_TARGET,
    create_writer_for_gpu,
    init_rocisa,
    assemble_kernel,
    print_offset_grid,
)

if HAS_HIP:
    from hip import hip  # type: ignore

from Tensile.Components.SubtileBasedKernel import graTileAssignment, lraTileAssignment


# ---------------------------------------------------------------------------
# Config: stride == depthU so GRA offset == LDS offset
# ---------------------------------------------------------------------------
INTEGRATION_CFG = TileConfig(
    mt_a=32, mt_b=64, depth_u=64,
    stride_a=64, stride_b=64,
    use_swizzling=True,
)


def hip_check(result):
    """Check HIP call result."""
    if isinstance(result, tuple):
        err = result[0]
        if err != 0:
            raise RuntimeError(f"HIP error {err}")
        return result[1] if len(result) == 2 else result[1:]
    if result != 0:
        raise RuntimeError(f"HIP error {result}")


# ---------------------------------------------------------------------------
# Assembly generation
# ---------------------------------------------------------------------------

def generate_gra_lra_asm(cfg, gra_only=False):
    """Run graTileAssignment (and optionally lraTileAssignment), return asm + metadata.

    When gra_only=True, only GRA offset computation is emitted.  The kernel will
    linearly dump LDS (ds_read at tid*16) to show how GRA laid data into LDS.
    """
    writer, kernel, tileInfoA, tileInfoB = create_writer_for_gpu(cfg)
    init_rocisa()

    gra_module = graTileAssignment(writer, kernel, useSwizzling=cfg.use_swizzling)
    gra_asm = str(gra_module)

    if gra_only:
        combined_asm = gra_asm
    else:
        lra_module = lraTileAssignment(writer, kernel)
        lra_asm = str(lra_module)
        combined_asm = gra_asm + "\n" + lra_asm

    return combined_asm, tileInfoA, tileInfoB, kernel


def generate_integration_kernel(gra_lra_asm, tileInfoA, gra_only=False, wave_id=1):
    """Generate a full kernel that:
    1. Loads kernargs (input_ptr, output_ptr, strideA, strideB)
    2. Runs GRA+LRA offset computation
    3. Sets up SRD for buffer_load
    4. buffer_load_dwordx4 offen lds (GRA offset -> LDS write)
    5. s_barrier
    6. ds_read_b128 × 2 (LRA offsets [0] and [1]) or linear tid*16 dump
    7. flat_store_dwordx4 to output (single subtile: 16 × depth_u)

    Normal mode: selected wave exports a 16×64 subtile using both LR offsets.
      1st store at c = (laneId/16)*8, 2nd store at c+32 → fills all 64 columns.
    gra_only: linear LDS dump (32×64), all threads.
    """
    # Find highest register indices in the GRA/LRA asm
    vgpr_indices = set(int(m) for m in re.findall(r'\bv(\d+)\b', gra_lra_asm))
    sgpr_indices = set(int(m) for m in re.findall(r'\bs(\d+)\b', gra_lra_asm))

    # Reserve registers past what GRA/LRA use
    next_v = max(vgpr_indices | {0}) + 1  # first free vgpr
    next_s = max(sgpr_indices | {0}) + 1  # first free sgpr

    # GRA offset register for matrix A (first/only one)
    gra_offset_reg = tileInfoA.sharedVgprGROffset[0]
    # LRA offset registers for matrix A (two per subtile)
    lra_offset_reg0 = tileInfoA.sharedVgprLROffset[0]
    lra_offset_reg1 = tileInfoA.sharedVgprLROffset[1]

    # ds_read_b128 destination must be 4-aligned
    if next_v % 4 != 0:
        next_v += 4 - (next_v % 4)
    data0 = next_v; next_v += 4  # 4 consecutive dwords

    # flat_store address pair must be 2-aligned
    if next_v % 2 != 0:
        next_v += 1
    addr_lo = next_v; next_v += 1
    addr_hi = next_v; next_v += 1

    # Temp vgpr for byte offset and for moving s[7] into vgpr
    tmp = next_v; next_v += 1
    tmp2 = next_v; next_v += 1

    # SRD for buffer_load: 4 consecutive sgprs, must be 4-aligned
    if next_s % 4 != 0:
        next_s += 4 - (next_s % 4)
    srd = next_s; next_s += 4

    # Exec save pair for wave masking (normal mode)
    exec_save = 0  # unused in gra_only
    if not gra_only:
        if next_s % 2 != 0:
            next_s += 1
        exec_save = next_s; next_s += 2

    max_vgpr = next_v
    max_sgpr = next_s
    # Align vgpr count to 4 for accum_offset
    max_vgpr = max(((max_vgpr + 3) // 4) * 4, 4)

    # LDS size: mt_a * depthU * BPE (only matrix A for this test)
    lds_size = INTEGRATION_CFG.mt_a * INTEGRATION_CFG.depth_u * BPE

    # Build step 5+6 asm: ds_read + output store
    # Output row stride (in bytes): depth_u columns * BPE
    stride_bytes = INTEGRATION_CFG.depth_u * BPE  # 64*2 = 128
    stride_shift = stride_bytes.bit_length() - 1   # 7  (r << 7 = r * 128)
    # Each ds_read_b128 loads 8 fp16 elements; c = (laneId/16)*8
    # c_bytes = (laneId/16) * 8 * BPE = (laneId/16) * 16
    col_bytes = 8 * BPE                            # 8 elements * 2 = 16
    col_shift = col_bytes.bit_length() - 1         # 4  ((laneId>>4) << 4)
    # 2nd LR offset writes 32 elements (columns) further
    col2_offset_bytes = 32 * BPE                   # 32 * 2 = 64

    if gra_only:
        step56_asm = f"""\
  // ---- 5+6. Linear LDS dump (tid*16) ----
  v_lshlrev_b32 v{tmp}, 4, v0               // byte offset = tid * 16
  // gra-only: read LDS linearly at tid*16 to dump full LDS contents
  ds_read_b128 v[{data0}:{data0+3}], v{tmp}
  s_waitcnt lgkmcnt(0)
  // Write 16 bytes to output_ptr + tid * 16
  v_mov_b32 v{tmp2}, s[7]                    // output_ptr hi
  v_add_co_u32 v{addr_lo}, vcc, s[6], v{tmp}
  v_addc_co_u32 v{addr_hi}, vcc, v{tmp2}, 0, vcc
  flat_store_dwordx4 v[{addr_lo}:{addr_hi}], v[{data0}:{data0+3}]"""
    else:
        step56_asm = f"""\
  // ---- 5+6. Subtile export: wave {wave_id}, dual ds_read (LR[0] + LR[1]) ----
  // Select wave {wave_id} only (waveId = threadId / 64)
  v_lshrrev_b32 v{tmp}, 6, v0                    // waveId
  v_cmp_eq_u32 vcc, {wave_id}, v{tmp}
  s_and_saveexec_b64 s[{exec_save}:{exec_save+1}], vcc

  // Compute output base offset for 16×64 subtile
  // laneId = threadId % 64
  // r = laneId % 16,  c = (laneId / 16) * 8
  // byte_offset = r * {stride_bytes} + c * {BPE}
  v_and_b32 v{tmp}, 0x3F, v0                     // laneId = threadId % 64
  v_and_b32 v{tmp2}, 0xF, v{tmp}                 // r = laneId % 16
  v_lshlrev_b32 v{tmp2}, {stride_shift}, v{tmp2} // r * {stride_bytes}
  v_lshrrev_b32 v{tmp}, 4, v{tmp}                // laneId / 16  (0-3)
  v_lshlrev_b32 v{tmp}, {col_shift}, v{tmp}      // c_bytes = (laneId/16) * {col_bytes}
  v_add_u32 v{tmp}, v{tmp}, v{tmp2}              // byte_offset_base

  // 1st read+store: LR offset[0] → columns 0-7, 8-15, 16-23, 24-31
  ds_read_b128 v[{data0}:{data0+3}], v{lra_offset_reg0}
  s_waitcnt lgkmcnt(0)
  v_mov_b32 v{tmp2}, s[7]                        // output_ptr hi
  v_add_co_u32 v{addr_lo}, vcc, s[6], v{tmp}
  v_addc_co_u32 v{addr_hi}, vcc, v{tmp2}, 0, vcc
  flat_store_dwordx4 v[{addr_lo}:{addr_hi}], v[{data0}:{data0+3}]
  s_waitcnt vmcnt(0)

  // 2nd read+store: LR offset[1] → columns +32 (32-39, 40-47, 48-55, 56-63)
  ds_read_b128 v[{data0}:{data0+3}], v{lra_offset_reg1}
  s_waitcnt lgkmcnt(0)
  v_add_u32 v{tmp}, v{tmp}, {col2_offset_bytes}  // shift by 32 elements ({col2_offset_bytes} bytes)
  v_add_co_u32 v{addr_lo}, vcc, s[6], v{tmp}
  v_addc_co_u32 v{addr_hi}, vcc, v{tmp2}, 0, vcc // tmp2 still has output_ptr hi
  flat_store_dwordx4 v[{addr_lo}:{addr_hi}], v[{data0}:{data0+3}]

  s_or_b64 exec, exec, s[{exec_save}:{exec_save+1}]"""

    # Kernarg layout:
    #   offset 0:  input_ptr  (8B)
    #   offset 8:  output_ptr (8B)
    #   offset 16: strideA    (4B)
    #   offset 20: strideB    (4B)
    #
    # Register mapping:
    #   s[4:5] = input_ptr
    #   s[6:7] = output_ptr
    #   s8     = sgprStrideA0I
    #   s9     = sgprStrideB1J

    return f"""\
.amdgcn_target "amdgcn-amd-amdhsa--{GFX_TARGET}"

// Register name mappings
.set vgprSerial, 0
.set sgprStrideA0I, 8
.set sgprStrideB1J, 9

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
  // ---- 1. Prologue: Load kernel arguments ----
  s_load_dwordx4 s[4:7], s[0:1], 0x00       // input_ptr (s[4:5]) + output_ptr (s[6:7])
  s_load_dword s[sgprStrideA0I], s[0:1], 0x10   // strideA -> s8
  s_load_dword s[sgprStrideB1J], s[0:1], 0x14   // strideB -> s9
  s_waitcnt lgkmcnt(0)

  // ---- 2. GRA + LRA offset computation ----
{gra_lra_asm}

  // ---- 3. Set up SRD for matrix A ----
  s_mov_b64 s[{srd}:{srd+1}], s[4:5]        // base = input_ptr
  s_mov_b32 s[{srd+2}], 0xFFFFFFFF          // NumRecords = max
  s_mov_b32 s[{srd+3}], 0x20000             // OOB_SELECT=2 (raw buffer)

  // ---- 4. buffer_load to LDS using GRA offset ----
  v_lshrrev_b32 v{tmp} , 6, v0 
  s_nop 1
  v_readfirstlane_b32 s4, v{tmp}  
  s_mul_i32 s4, s4, 0x400 // Hardcoded to reads size 8x128Bytes per warp. 
  s_mov_b32 m0, s4                            // LDS base offset = 0
  //DEBUG purpose: use tid as input
  // v_lshlrev_b32 v{gra_offset_reg}, 4, v0 

  buffer_load_dwordx4 v{gra_offset_reg}, s[{srd}:{srd+3}], 0 offen lds
  s_waitcnt vmcnt(0)
  s_barrier

{step56_asm}
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
      - .name:            input_ptr
        .size:            8
        .offset:          0
        .value_kind:      global_buffer
        .value_type:      f16
        .address_space:   global
      - .name:            output_ptr
        .size:            8
        .offset:          8
        .value_kind:      global_buffer
        .value_type:      u32
        .address_space:   global
      - .name:            strideA
        .size:            4
        .offset:          16
        .value_kind:      by_value
        .value_type:      u32
      - .name:            strideB
        .size:            4
        .offset:          20
        .value_kind:      by_value
        .value_type:      u32
    .kernarg_segment_size: 24
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


# ---------------------------------------------------------------------------
# GPU execution
# ---------------------------------------------------------------------------

def run_integration_on_gpu(co_path, input_data, cfg, output_size=None):
    """Launch integration kernel and return output buffer.

    Args:
        co_path:     Path to assembled .co file
        input_data:  numpy array of fp16 values (mt_a * depthU elements)
        cfg:         TileConfig
        output_size: output buffer size in bytes (default: mt_a * depth_u * BPE)
    """
    hip_check(hip.hipInit(0))

    module = hip_check(hip.hipModuleLoad(co_path.encode() if isinstance(co_path, str) else co_path))
    kernel = hip_check(hip.hipModuleGetFunction(module, b"test_kernel"))

    input_bytes = input_data.tobytes()
    input_size = len(input_bytes)
    if output_size is None:
        output_size = cfg.mt_a * cfg.depth_u * BPE

    lds_size = cfg.mt_a * cfg.depth_u * BPE

    # Allocate GPU buffers
    d_input = hip_check(hip.hipMalloc(input_size))
    d_output = hip_check(hip.hipMalloc(output_size))

    # Upload input, zero output
    hip_check(hip.hipMemcpyHtoD(d_input, input_bytes, input_size))
    hip_check(hip.hipMemset(d_output, 0, output_size))

    class KernelArgs(ctypes.Structure):
        _fields_ = [
            ("input_ptr", ctypes.c_uint64),
            ("output_ptr", ctypes.c_uint64),
            ("stride_a", ctypes.c_uint32),
            ("stride_b", ctypes.c_uint32),
        ]

    kargs = KernelArgs(int(d_input), int(d_output), cfg.stride_a, cfg.stride_b)
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
        1, 1, 1,             # grid
        NUM_THREADS, 1, 1,   # block
        lds_size,            # shared mem (LDS)
        None,                # stream
        None,                # kernel params
        extra                # extra params
    ))
    hip_check(hip.hipDeviceSynchronize())

    h_output = bytearray(output_size)
    hip_check(hip.hipMemcpyDtoH(h_output, d_output, output_size))

    hip_check(hip.hipFree(d_input))
    hip_check(hip.hipFree(d_output))
    hip_check(hip.hipModuleUnload(module))

    return bytes(h_output)


def print_matrix_fp16(label, data_bytes, rows, cols):
    """Print a byte buffer as a matrix of fp16 values (rows x cols)."""
    arr = np.frombuffer(data_bytes, dtype=np.float16).reshape(rows, cols)
    print(f"\n--- {label} ({rows} x {cols}, fp16) ---")
    # Column header
    hdr = "     " + "".join(f"{c:>7d}" for c in range(cols))
    print(hdr)
    for r in range(rows):
        vals = "".join(f"{arr[r, c]:7.1f}" for c in range(cols))
        print(f"[{r:3d}] {vals}")


def rebuild_output_matrix(output_bytes, rows, cols):
    """Return output as rows x cols bytes (raw matrix in row-major order)."""
    total = rows * cols * BPE
    return bytes(output_bytes[:total])


def wave_to_input_row_start(wave_id, subtile_rows):
    """Map wave ID to the starting input row for its subtile.

    Wave 0, 2 → rows 0  (input rows 0..subtile_rows-1)
    Wave 1, 3 → rows 16 (input rows subtile_rows..2*subtile_rows-1)
    """
    return (wave_id % 2) * subtile_rows


# ---------------------------------------------------------------------------
# Pytest tests
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_HIP, reason="HIP Python bindings not available")
class TestGraLraIntegration:

    @pytest.fixture(params=[0, 1, 2, 3], ids=lambda w: f"wave{w}")
    def integration_env(self, tmp_path, request):
        """Set up the integration test for a single wave's subtile."""
        cfg = INTEGRATION_CFG
        wave_id = request.param

        combined_asm, tileInfoA, tileInfoB, kernel = generate_gra_lra_asm(cfg)
        full_asm = generate_integration_kernel(combined_asm, tileInfoA,
                                               gra_only=False, wave_id=wave_id)

        co_path = str(tmp_path / f"integration_wave{wave_id}.co")
        asm_path = str(tmp_path / f"integration_wave{wave_id}.s")
        with open(asm_path, "w") as f:
            f.write(full_asm)

        assemble_kernel(full_asm, co_path)

        # Input: sequential fp16 values starting at 1 for easy debugging
        # (0 is indistinguishable from uninitialized memory)
        num_elements = cfg.mt_a * cfg.depth_u
        input_data = np.arange(1, num_elements + 1, dtype=np.float16)

        return SimpleNamespace(
            cfg=cfg,
            co_path=co_path,
            asm_path=asm_path,
            full_asm=full_asm,
            input_data=input_data,
            tileInfoA=tileInfoA,
            tileInfoB=tileInfoB,
            wave_id=wave_id,
        )

    def test_gra_to_lds_to_lra_roundtrip(self, integration_env):
        """Verify: input -> GRA buffer_load lds -> LRA ds_read -> output matches input subtile.

        Each wave exports a 16×64 subtile. Wave 0,2 → input rows 0-15;
        wave 1,3 → input rows 16-31. Output should match exactly.
        """
        env = integration_env
        cfg = env.cfg
        input_bytes = env.input_data.tobytes()
        sys.stdout.flush()

        subtile_rows = cfg.mt_a // 2  # 16
        out_rows, out_cols = subtile_rows, cfg.depth_u
        output_size = out_rows * out_cols * BPE
        stride_bytes = cfg.depth_u * BPE

        # Run kernel on GPU
        output_bytes = run_integration_on_gpu(env.co_path, env.input_data, cfg,
                                              output_size=output_size)

        # Expected: input rows for this wave's subtile
        row_start = wave_to_input_row_start(env.wave_id, subtile_rows)
        subtile_size = subtile_rows * stride_bytes
        expected = input_bytes[row_start * stride_bytes :
                               row_start * stride_bytes + subtile_size]
        actual = output_bytes[:subtile_size]

        # Per-row comparison for clear error reporting
        errors = 0
        for r in range(subtile_rows):
            row_off = r * stride_bytes
            exp_row = expected[row_off:row_off + stride_bytes]
            act_row = actual[row_off:row_off + stride_bytes]
            if exp_row != act_row:
                errors += 1
                if errors <= 16:
                    exp_fp16 = np.frombuffer(exp_row, dtype=np.float16)
                    act_fp16 = np.frombuffer(act_row, dtype=np.float16)
                    print(f"  MISMATCH wave {env.wave_id} row {r} "
                          f"(input row {row_start + r}):")
                    print(f"    expected: {exp_fp16}")
                    print(f"    actual:   {act_fp16}")

        assert errors == 0, (f"Wave {env.wave_id}: {errors}/{subtile_rows} rows "
                             f"mismatch (input rows {row_start}..{row_start + subtile_rows - 1})")


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="GRA+LRA integration GPU test")
    parser.add_argument("--debug", action="store_true",
                        help="Print detailed per-thread output")
    parser.add_argument("--show-matrices", action="store_true",
                        help="Display input and output as fp16 matrices (mt_a x depth_u)")
    parser.add_argument("--gra-only", action="store_true",
                        help="Skip LRA; read back from LDS using GRA offset to see how GRA loads data into LDS")
    parser.add_argument("--wave", default="all",
                        help="Which wave exports the subtile: 0-3 or 'all' (default: all)")
    args = parser.parse_args()

    cfg = INTEGRATION_CFG
    print(f"Config: {cfg.label}")
    print(f"  mt_a={cfg.mt_a}, mt_b={cfg.mt_b}, depth_u={cfg.depth_u}")
    print(f"  stride_a={cfg.stride_a}, stride_b={cfg.stride_b}")
    print(f"  LDS size: {cfg.mt_a * cfg.depth_u * BPE} bytes")

    combined_asm, tileInfoA, tileInfoB, kernel = generate_gra_lra_asm(cfg, gra_only=args.gra_only)
    print(f"\n  TileInfoA: GR regs={tileInfoA.sharedVgprGROffset}, "
          f"LR regs={tileInfoA.sharedVgprLROffset}, "
          f"numGR={tileInfoA.numGRPerSubtile}, numLR={tileInfoA.numLRPerSubtile}")
    # Parse --wave: single wave (0-3) or "all"
    if args.wave == "all":
        wave_list = [0, 1, 2, 3]
    else:
        wave_list = [int(args.wave)]

    if args.gra_only:
        print("  Mode: GRA-only (ds_read at tid*16, LRA skipped)")
    else:
        print(f"  Mode: Subtile export (wave {args.wave}, 16×{cfg.depth_u} output)")

    if args.debug:
        print("\n--- Combined GRA+LRA Assembly ---")
        print(combined_asm)
        print("--- End ---\n")

    if not HAS_HIP:
        print("HIP not available - cannot run GPU test")
        sys.exit(1)

    # Create input: 0, 1, 2, ... for easy debugging
    num_elements = cfg.mt_a * cfg.depth_u
    input_data = np.arange(0, num_elements, dtype=np.float16)
    input_bytes = input_data.tobytes()

    subtile_rows = cfg.mt_a // 2  # 16 (subtile = half the macro tile)
    total_errors = 0

    if args.gra_only:
        # GRA-only: single run, linear LDS dump
        with tempfile.TemporaryDirectory() as tmp_dir:
            full_asm = generate_integration_kernel(combined_asm, tileInfoA,
                                                  gra_only=True)
            print(full_asm)
            co_path = os.path.join(tmp_dir, "integration_test.co")
            asm_path = os.path.join(tmp_dir, "integration_test.s")
            with open(asm_path, "w") as f:
                f.write(full_asm)

            if args.debug:
                print("\n--- Full Kernel Assembly ---")
                print(full_asm)
                print("--- End ---\n")

            assemble_kernel(full_asm, co_path)
            print(f"  Assembled: {co_path}")

            out_rows, out_cols = cfg.mt_a, cfg.depth_u
            output_size = out_rows * out_cols * BPE

            sys.stdout.flush()
            output_bytes = run_integration_on_gpu(co_path, input_data, cfg,
                                                  output_size=output_size)

            if args.show_matrices:
                print_matrix_fp16("Input", input_bytes, cfg.mt_a, cfg.depth_u)
                out_matrix_bytes = rebuild_output_matrix(output_bytes, out_rows, out_cols)
                print_matrix_fp16("Output (GRA layout in LDS)", out_matrix_bytes, out_rows, out_cols)

            # Verify: output should match input exactly
            lds_size = cfg.mt_a * cfg.depth_u * BPE
            actual = output_bytes[:lds_size]
            expected = input_bytes[:lds_size]
            if actual != expected:
                for i in range(0, lds_size, LOAD_WIDTH):
                    a = actual[i:i + LOAD_WIDTH]
                    e = expected[i:i + LOAD_WIDTH]
                    if a != e:
                        total_errors += 1
                        if total_errors <= 10:
                            tid = i // LOAD_WIDTH
                            exp_fp16 = np.frombuffer(e, dtype=np.float16)
                            act_fp16 = np.frombuffer(a, dtype=np.float16)
                            print(f"  MISMATCH LDS[{i}..{i+LOAD_WIDTH}] (tid={tid})")
                            print(f"    expected: {exp_fp16}")
                            print(f"    actual:   {act_fp16}")

    else:
        # Normal mode: test each wave
        for wave_id in wave_list:
            row_start = wave_to_input_row_start(wave_id, subtile_rows)
            print(f"\n  === Wave {wave_id} (expected input rows {row_start}..{row_start + subtile_rows - 1}) ===")

            with tempfile.TemporaryDirectory() as tmp_dir:
                full_asm = generate_integration_kernel(combined_asm, tileInfoA,
                                                      gra_only=False, wave_id=wave_id)
                if args.debug:
                    print(full_asm)

                co_path = os.path.join(tmp_dir, "integration_test.co")
                asm_path = os.path.join(tmp_dir, "integration_test.s")
                with open(asm_path, "w") as f:
                    f.write(full_asm)

                assemble_kernel(full_asm, co_path)

                out_rows, out_cols = subtile_rows, cfg.depth_u
                output_size = out_rows * out_cols * BPE

                sys.stdout.flush()
                output_bytes = run_integration_on_gpu(co_path, input_data, cfg,
                                                      output_size=output_size)

                if args.show_matrices:
                    if wave_id == wave_list[0]:
                        print_matrix_fp16("Input", input_bytes, cfg.mt_a, cfg.depth_u)
                    out_matrix_bytes = rebuild_output_matrix(output_bytes, out_rows, out_cols)
                    print_matrix_fp16(f"Output (wave {wave_id} subtile)", out_matrix_bytes, out_rows, out_cols)

                # Verify: output should match input rows [row_start..row_start+15]
                stride_bytes = cfg.depth_u * BPE
                subtile_size = subtile_rows * stride_bytes
                expected = input_bytes[row_start * stride_bytes :
                                       row_start * stride_bytes + subtile_size]
                actual = output_bytes[:subtile_size]

                wave_errors = 0
                if actual == expected:
                    print(f"  PASS: output matches input rows [{row_start}..{row_start + subtile_rows - 1}]")
                else:
                    for r in range(subtile_rows):
                        row_off = r * stride_bytes
                        exp_row = expected[row_off:row_off + stride_bytes]
                        act_row = actual[row_off:row_off + stride_bytes]
                        if exp_row != act_row:
                            wave_errors += 1
                            if wave_errors <= 16:
                                exp_fp16 = np.frombuffer(exp_row, dtype=np.float16)
                                act_fp16 = np.frombuffer(act_row, dtype=np.float16)
                                print(f"  MISMATCH row {r} (input row {row_start + r}):")
                                print(f"    expected: {exp_fp16}")
                                print(f"    actual:   {act_fp16}")
                    total_errors += wave_errors

    print(f"\n  Result: {len(wave_list)} wave(s) tested, {total_errors} errors")
    if total_errors > 0:
        print("  FAILED")
        sys.exit(1)
    else:
        print("  PASSED")
