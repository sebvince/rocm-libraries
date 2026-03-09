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
import struct
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

# Import reference offset computation for LRA reads
from test_lraTileAssignment import compute_expected_lr_offset


# ---------------------------------------------------------------------------
# Config: stride == depthU so GRA offset == LDS offset
# ---------------------------------------------------------------------------
INTEGRATION_CFG = TileConfig(
    mt_a=32, mt_b=64, depth_u=64,
    stride_a=64, stride_b=64,
    use_swizzling=False,
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


def generate_integration_kernel(gra_lra_asm, tileInfoA, gra_only=False):
    """Generate a full kernel that:
    1. Loads kernargs (input_ptr, output_ptr, strideA, strideB)
    2. Runs GRA+LRA offset computation
    3. Sets up SRD for buffer_load
    4. buffer_load_dwordx4 offen lds (GRA offset -> LDS write)
    5. s_barrier
    6. ds_read_b128 (LRA offset or linear tid*16 -> VGPRs)
    7. flat_store_dwordx4 (VGPRs -> output buffer)

    When gra_only=True, ds_read uses tid*16 to linearly dump the entire LDS,
    showing how GRA arranged data. 256 threads × 16B = 4096B = full LDS.
    """
    # Find highest register indices in the GRA/LRA asm
    vgpr_indices = set(int(m) for m in re.findall(r'\bv(\d+)\b', gra_lra_asm))
    sgpr_indices = set(int(m) for m in re.findall(r'\bs(\d+)\b', gra_lra_asm))

    # Reserve registers past what GRA/LRA use
    next_v = max(vgpr_indices | {0}) + 1  # first free vgpr
    next_s = max(sgpr_indices | {0}) + 1  # first free sgpr

    # GRA offset register for matrix A (first/only one)
    gra_offset_reg = tileInfoA.sharedVgprGROffset[0]
    # LRA offset register for matrix A (first of numLRPerSubtile)
    lra_offset_reg = tileInfoA.sharedVgprLROffset[0]

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

    max_vgpr = next_v
    max_sgpr = next_s
    # Align vgpr count to 4 for accum_offset
    max_vgpr = max(((max_vgpr + 3) // 4) * 4, 4)

    # LDS size: mt_a * depthU * BPE (only matrix A for this test)
    lds_size = INTEGRATION_CFG.mt_a * INTEGRATION_CFG.depth_u * BPE

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
  v_lshrrev_b32 v2, 6, v0 
  s_nop 1
  v_readfirstlane_b32 s4, v2  
  s_mul_i32 s4, s4, 0x400 // Hardcoded to reads size 8x128Bytes per warp. 
  s_mov_b32 m0, s4                            // LDS base offset = 0
  //DEBUG purpose: use tid as input
  // v_lshlrev_b32 v{gra_offset_reg}, 4, v0 

  buffer_load_dwordx4 v{gra_offset_reg}, s[{srd}:{srd+3}], 0 offen lds
  s_waitcnt vmcnt(0)
  s_barrier

  // ---- 5+6. {'Linear LDS dump (tid*16)' if gra_only else 'ds_read LRA + store'} ----
  v_lshlrev_b32 v{tmp}, 4, v0               // byte offset = tid * 16
{'  // gra-only: read LDS linearly at tid*16 to dump full LDS contents' if gra_only else '  // normal: read LDS at LRA offset'}
  ds_read_b128 v[{data0}:{data0+3}], v{tmp if gra_only else lra_offset_reg}
  
  s_waitcnt lgkmcnt(0)
  // Write 16 bytes to output_ptr + tid * 16
  v_mov_b32 v{tmp2}, s[7]                    // move output_ptr hi to vgpr
  v_add_co_u32 v{addr_lo}, vcc, s[6], v{tmp}
  v_addc_co_u32 v{addr_hi}, vcc, v{tmp2}, 0, vcc
  flat_store_dwordx4 v[{addr_lo}:{addr_hi}], v[{data0}:{data0+3}]
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

def run_integration_on_gpu(co_path, input_data, cfg):
    """Launch integration kernel and return output buffer.

    Args:
        co_path:    Path to assembled .co file
        input_data: numpy array of fp16 values (mt_a * depthU elements)
        cfg:        TileConfig

    Returns:
        output bytes (NUM_THREADS * 16 bytes)
    """
    hip_check(hip.hipInit(0))

    module = hip_check(hip.hipModuleLoad(co_path.encode() if isinstance(co_path, str) else co_path))
    kernel = hip_check(hip.hipModuleGetFunction(module, b"test_kernel"))

    input_bytes = input_data.tobytes()
    input_size = len(input_bytes)
    output_size = NUM_THREADS * 16  # 16 bytes per thread

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


def rebuild_output_matrix(output_bytes, cfg, tileInfoA, gra_only=False):
    """Reconstruct output as mt_a x depth_u bytes.

    When gra_only=True, the output is already a linear LDS dump (tid*16 layout),
    so the raw bytes are the LDS contents in order — no reordering needed.
    Otherwise, places each thread's chunk back at its LRA offset.
    """
    total = cfg.mt_a * cfg.depth_u * BPE
    if gra_only:
        # Output is a linear dump of LDS, already in byte order
        return bytes(output_bytes[:total])
    buf = bytearray(total)
    for tid in range(NUM_THREADS):
        offset = compute_expected_lr_offset(tid, cfg, tileInfoA)[0]
        chunk = output_bytes[tid * LOAD_WIDTH:(tid + 1) * LOAD_WIDTH]
        buf[offset:offset + LOAD_WIDTH] = chunk
    return bytes(buf)


# ---------------------------------------------------------------------------
# Pytest tests
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_HIP, reason="HIP Python bindings not available")
class TestGraLraIntegration:

    @pytest.fixture
    def integration_env(self, tmp_path):
        """Set up the integration test: generate asm, assemble, prepare input."""
        cfg = INTEGRATION_CFG

        combined_asm, tileInfoA, tileInfoB, kernel = generate_gra_lra_asm(cfg)
        full_asm = generate_integration_kernel(combined_asm, tileInfoA)
        print(full_asm)

        co_path = str(tmp_path / "integration_test.co")
        asm_path = str(tmp_path / "integration_test.s")
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
        )

    def test_gra_to_lds_to_lra_roundtrip(self, integration_env):
        """Verify: input -> GRA buffer_load lds -> LRA ds_read -> output matches input.

        Since stride == depthU, GRA offsets == LDS byte addresses, so LDS is an
        identical copy of the input.  Each thread reads 16 bytes from LDS at its
        LRA offset, so output[tid] should equal input[LRA_offset:LRA_offset+16].
        """
        env = integration_env
        cfg = env.cfg
        input_bytes = env.input_data.tobytes()
        sys.stdout.flush()

        # Run kernel on GPU
        output_bytes = run_integration_on_gpu(env.co_path, env.input_data, cfg)

        # Verify: for each thread, output == input[LRA_offset : LRA_offset + 16]
        errors = 0
        for tid in range(NUM_THREADS):
            lr_offset = compute_expected_lr_offset(tid, cfg, env.tileInfoA)[0]

            expected_bytes = input_bytes[lr_offset:lr_offset + LOAD_WIDTH]
            actual_bytes = output_bytes[tid * LOAD_WIDTH:(tid + 1) * LOAD_WIDTH]

            if actual_bytes != expected_bytes:
                errors += 1
                if errors <= 10:
                    exp_dwords = struct.unpack("4I", expected_bytes)
                    act_dwords = struct.unpack("4I", actual_bytes)
                    print(f"  MISMATCH tid={tid}: LR_offset={lr_offset}")
                    print(f"    expected: {exp_dwords}")
                    print(f"    actual:   {act_dwords}")

        assert errors == 0, f"{errors}/{NUM_THREADS} threads produced wrong output"


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
    if args.gra_only:
        print("  Mode: GRA-only (ds_read uses GRA offset, LRA skipped)")

    if args.debug:
        print("\n--- Combined GRA+LRA Assembly ---")
        print(combined_asm)
        print("--- End ---\n")

    if not HAS_HIP:
        print("HIP not available - cannot run GPU test")
        sys.exit(1)

    with tempfile.TemporaryDirectory() as tmp_dir:
        full_asm = generate_integration_kernel(combined_asm, tileInfoA, gra_only=args.gra_only)
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

        # Create input: 0, 1, 2, ... for easy debugging
        num_elements = cfg.mt_a * cfg.depth_u
        input_data = np.arange(0, num_elements, dtype=np.float16)
        input_bytes = input_data.tobytes()

        # Run
        sys.stdout.flush()
        output_bytes = run_integration_on_gpu(co_path, input_data, cfg)

        if args.show_matrices:
            print_matrix_fp16("Input", input_bytes, cfg.mt_a, cfg.depth_u)
            out_matrix_bytes = rebuild_output_matrix(output_bytes, cfg, tileInfoA, gra_only=args.gra_only)
            label = "Output (GRA layout in LDS)" if args.gra_only else "Output"
            print_matrix_fp16(label, out_matrix_bytes, cfg.mt_a, cfg.depth_u)

        # Verify
        errors = 0
        if args.gra_only:
            # Linear LDS dump: output bytes should match input bytes exactly
            lds_size = cfg.mt_a * cfg.depth_u * BPE
            actual = output_bytes[:lds_size]
            expected = input_bytes[:lds_size]
            if actual != expected:
                # Find first mismatch for debugging
                for i in range(0, lds_size, LOAD_WIDTH):
                    a = actual[i:i + LOAD_WIDTH]
                    e = expected[i:i + LOAD_WIDTH]
                    if a != e:
                        errors += 1
                        if errors <= 10:
                            tid = i // LOAD_WIDTH
                            exp_fp16 = np.frombuffer(e, dtype=np.float16)
                            act_fp16 = np.frombuffer(a, dtype=np.float16)
                            print(f"  MISMATCH LDS[{i}..{i+LOAD_WIDTH}] (tid={tid})")
                            print(f"    expected: {exp_fp16}")
                            print(f"    actual:   {act_fp16}")
        else:
            for tid in range(NUM_THREADS):
                offset = compute_expected_lr_offset(tid, cfg, tileInfoA)[0]

                expected_bytes = input_bytes[offset:offset + LOAD_WIDTH]
                actual_bytes = output_bytes[tid * LOAD_WIDTH:(tid + 1) * LOAD_WIDTH]

                match = actual_bytes == expected_bytes

                if args.debug or not match:
                    exp_dwords = struct.unpack("4I", expected_bytes)
                    act_dwords = struct.unpack("4I", actual_bytes)
                    status = "OK  " if match else "FAIL"
                    print(f"  {status} tid={tid:3d}  LR_off={offset:5d}  "
                          f"exp={exp_dwords}  act={act_dwords}")

                if not match:
                    errors += 1

        print(f"\n  Result: {NUM_THREADS} threads, {errors} errors")
        if errors > 0:
            print("  FAILED")
            sys.exit(1)
        else:
            print("  PASSED")
