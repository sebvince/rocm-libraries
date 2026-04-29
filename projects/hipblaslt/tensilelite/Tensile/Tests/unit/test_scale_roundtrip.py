#!/usr/bin/env python3
################################################################################
# End-to-end GPU roundtrip test for scale tensors (MX FP4).
#
# Verifies GR and LR offset consistency: data loaded from global memory
# via GR offsets, written to LDS, and read back via LR offsets should
# deliver the correct scale bytes.
#
# Flow per matrix (A or B):
#   1. Compute GR + LR offsets (production code)
#   2. flat_load_ubyte from global scale buffer at GR offset
#   3. ds_write_b8 to LDS at position = serial
#   4. s_barrier
#   5. ds_read_u8 from LDS at (LR offset - dataLdsSize)
#   6. Export result
#
# Usage:
#   pytest test_scale_roundtrip.py -v -s
#   python test_scale_roundtrip.py --debug
################################################################################

import os
import struct
import sys
import tempfile

import pytest
import numpy as np

from gpu_test_helpers import (
    HAS_HIP,
    TileConfig,
    WAVESIZE, NUM_WAVES, NUM_THREADS,
    create_writer,
    init_rocisa,
    generate_kernel_asm,
    generate_load_params,
    assemble_and_run,
    print_offset_grid,
)

from test_graTileAssignment import compute_expected_scale_gr_offset
from test_lraTileAssignment import compute_expected_scale_lr_offset

from Tensile.Components.SubtileBasedKernel import (
    graTileAssignmentScaleSwizzled,
    lraTileAssignmentScaleSwizzled,
)

# ---------------------------------------------------------------------------
# Test configurations
# ---------------------------------------------------------------------------
SCALE_ROUNDTRIP_CONFIGS = [
    # 2x2 wave group, dense stride
    TileConfig(mt_a=256, mt_b=256, depth_u=64, stride_a=64,  stride_b=64,  mxblock=32),
    # 1x4 wave group
    TileConfig(mt_a=80,  mt_b=64,  depth_u=64, stride_a=64,  stride_b=64,  mxblock=32),
    # 4x1 wave group
    TileConfig(mt_a=64,  mt_b=80,  depth_u=64, stride_a=64,  stride_b=64,  mxblock=32),
    # Non-trivial stride (stride > depthU, tests stride division by mxBlock)
    TileConfig(mt_a=96,  mt_b=256, depth_u=64, stride_a=128, stride_b=128, mxblock=32),
]


# ---------------------------------------------------------------------------
# LDS size computation (mirrors lraTileAssignmentScaleSwizzled)
# ---------------------------------------------------------------------------
def compute_lds_sizes(cfg, tileInfoA, tileInfoB, kernel):
    """Compute LDS layout sizes matching production code."""
    MT0A = tileInfoA.globalMMATileGrid[0] * tileInfoA.mmaTileShape[0]
    MT0B = tileInfoB.globalMMATileGrid[0] * tileInfoB.mmaTileShape[0]
    dataLdsSize = (MT0A * cfg.depth_u * tileInfoA.bpe) + \
                  (MT0B * cfg.depth_u * tileInfoB.bpe)

    numWaves = kernel["MIWaveGroup"][0] * kernel["MIWaveGroup"][1]

    scaleALdsRaw = MT0A * tileInfoA.scaleDepthU * tileInfoA.scaleBpe if tileInfoA.mxBlock > 0 else 0
    ldsAlignment = WAVESIZE * numWaves * (tileInfoA.scaleLoadWidth if tileInfoA.mxBlock > 0 else 1)
    scaleALdsSize = ((scaleALdsRaw + ldsAlignment - 1) // ldsAlignment) * ldsAlignment if scaleALdsRaw > 0 else 0

    scaleBLdsRaw = MT0B * tileInfoB.scaleDepthU * tileInfoB.scaleBpe if tileInfoB.mxBlock > 0 else 0
    scaleBLdsSize = ((scaleBLdsRaw + ldsAlignment - 1) // ldsAlignment) * ldsAlignment if scaleBLdsRaw > 0 else 0

    return dataLdsSize, scaleALdsSize, scaleBLdsSize


def compute_input_size(cfg, tileInfo):
    """Max GR offset + 1 across all threads."""
    max_off = 0
    for tid in range(NUM_THREADS):
        offsets = compute_expected_scale_gr_offset(tid, cfg, tileInfo)
        max_off = max(max_off, offsets[0])
    return max_off + 1


def generate_input_data(size):
    """Deterministic byte array for scale input."""
    return np.array([(i * 7 + 13) & 0xFF for i in range(size)], dtype=np.uint8)


# ---------------------------------------------------------------------------
# ASM generation using production code paths
# ---------------------------------------------------------------------------
def generate_roundtrip_kernel(cfg, tc):
    """Generate a complete kernel using production scale GR/LR code paths."""
    init_rocisa()
    writer, kernel, tileInfoA, tileInfoB = create_writer(cfg)

    # Reserve s0-s11 for hardware regs + kernarg loads
    writer.sgprPool.checkOut(12)
    writer.sgprs["StrideA0I"] = 10
    writer.sgprs["StrideB1J"] = 11

    # Allocate offset registers (shared GR/LR offset vgprs)
    tileInfoA.allocOffsetRegisters(writer, kernel)
    tileInfoB.allocOffsetRegisters(writer, kernel)

    # Scale GR + LR offset computation
    gra_module = graTileAssignmentScaleSwizzled(writer, kernel)
    lra_module = lraTileAssignmentScaleSwizzled(writer, kernel)

    # Roundtrip logic: which matrix to test
    tileInfo = tileInfoA if tc == 'A' else tileInfoB
    grOffReg = tileInfo.sharedVgprGROffset[0]
    lrOffReg = tileInfo.sharedVgprLROffset[0]
    ptrLo = 4 if tc == 'A' else 6
    ptrHi = 5 if tc == 'A' else 7

    # LDS layout
    dataLdsSize, scaleALdsSize, scaleBLdsSize = compute_lds_sizes(
        cfg, tileInfoA, tileInfoB, kernel)
    scaleBase = dataLdsSize if tc == 'A' else (dataLdsSize + scaleALdsSize)
    lds_bytes = scaleALdsSize + scaleBLdsSize

    # Allocate temp registers for roundtrip
    vAddr = writer.vgprPool.checkOutAligned(2, 2, "addr", preventOverflow=False)
    vData = writer.vgprPool.checkOut(1, "data", preventOverflow=False)
    vLrAdj = writer.vgprPool.checkOut(1, "lr_adj", preventOverflow=False)
    vByteOff = writer.vgprPool.checkOut(1, "byte_off", preventOverflow=False)
    sTmp = writer.sgprPool.checkOut(1, "tmp", preventOverflow=False)

    roundtrip_asm = f"""\
  // ---- Roundtrip for scale {tc} ----
  v_mov_b32 v{vAddr}, s{ptrLo}
  v_mov_b32 v{vAddr+1}, s{ptrHi}
  v_add_co_u32 v{vAddr}, vcc, v{vAddr}, v{grOffReg}
  v_addc_co_u32 v{vAddr+1}, vcc, v{vAddr+1}, 0, vcc
  flat_load_ubyte v{vData}, v[{vAddr}:{vAddr+1}]
  s_waitcnt vmcnt(0) lgkmcnt(0)
  ds_write_b8 v0, v{vData}
  s_waitcnt lgkmcnt(0)
  s_barrier
  s_mov_b32 s{sTmp}, {scaleBase}
  v_sub_u32 v{vLrAdj}, v{lrOffReg}, s{sTmp}
  ds_read_u8 v{vData}, v{vLrAdj}
  s_waitcnt lgkmcnt(0)
  v_lshlrev_b32 v{vByteOff}, 2, v0
  global_store_dword v{vByteOff}, v{vData}, s[8:9]"""

    # Prologue
    prologue = generate_load_params([
        (4, 4, 0x00, "input_A_ptr + input_B_ptr"),
        (8, 4, 0x10, "output_ptr + strideA + strideB"),
    ])

    inner_asm = "\n".join([
        str(prologue),
        str(gra_module),
        str(lra_module),
        roundtrip_asm,
    ])

    args = (
        ("input_scale_A_ptr", 8, "global_buffer", "u8"),
        ("input_scale_B_ptr", 8, "global_buffer", "u8"),
        ("output_ptr",        8, "global_buffer", "u32"),
        ("strideA",           4, "by_value",      "u32"),
        ("strideB",           4, "by_value",      "u32"),
    )

    kernel_asm = generate_kernel_asm(inner_asm, writer, args, lds_bytes)
    return kernel_asm, writer, kernel, tileInfoA, tileInfoB, lds_bytes


# ---------------------------------------------------------------------------
# Python reference
# ---------------------------------------------------------------------------
def compute_expected_roundtrip(cfg, tileInfoA, tileInfoB, input_data, tc, kernel):
    """Compute expected scale byte for each thread after the roundtrip.

    Data path: thread T reads LDS[lrOff(T) - scaleBase]. That position
    was written by thread writer=lrOff(T)-scaleBase, who loaded
    input[grOff(writer)].
    """
    tileInfo = tileInfoA if tc == 'A' else tileInfoB
    otherTileInfo = tileInfoB if tc == 'A' else tileInfoA

    dataLdsSize, scaleALdsSize, _ = compute_lds_sizes(cfg, tileInfoA, tileInfoB, kernel)
    scaleBase = dataLdsSize if tc == 'A' else (dataLdsSize + scaleALdsSize)

    expected = [0] * NUM_THREADS
    for T in range(NUM_THREADS):
        lr_offset = compute_expected_scale_lr_offset(T, cfg, tileInfo, otherTileInfo)[0]
        writer = lr_offset - scaleBase

        assert 0 <= writer < NUM_THREADS, \
            f"Thread {T}: LR offset {lr_offset} - scaleBase {scaleBase} = {writer} out of range"

        gr_offset = compute_expected_scale_gr_offset(writer, cfg, tileInfo)[0]

        assert 0 <= gr_offset < len(input_data), \
            f"Writer thread {writer}: GR offset {gr_offset} >= input size {len(input_data)}"

        expected[T] = int(input_data[gr_offset])

    return expected


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
def build_and_run_roundtrip(cfg, tc, tmp_path, debug=False):
    """Generate, assemble, run roundtrip for one matrix; return (results, expected)."""
    sys.stdout.flush()

    kernel_asm, writer, kernel, tileInfoA, tileInfoB, lds_bytes = \
        generate_roundtrip_kernel(cfg, tc)

    if debug:
        print(f"\n--- Kernel ASM (scale {tc}, {cfg.label}) ---")
        print(kernel_asm)
        print("--- End ---\n")

    tileInfo = tileInfoA if tc == 'A' else tileInfoB
    input_size = compute_input_size(cfg, tileInfo)
    input_data = generate_input_data(input_size)

    other_tileInfo = tileInfoB if tc == 'A' else tileInfoA
    other_input_size = compute_input_size(cfg, other_tileInfo)
    other_input = generate_input_data(other_input_size)

    if tc == 'A':
        input_a, input_b = input_data, other_input
    else:
        input_a, input_b = other_input, input_data

    out_size = NUM_THREADS * 4
    label = f"scale_roundtrip_{tc}_{cfg.label}"
    output_bytes = assemble_and_run(kernel_asm, tmp_path, label, out_size,
                                    inputs=(input_a, input_b),
                                    scalars=(cfg.stride_a, cfg.stride_b),
                                    lds_size=lds_bytes)

    results = struct.unpack(f"{NUM_THREADS}I", output_bytes)
    expected = compute_expected_roundtrip(cfg, tileInfoA, tileInfoB, input_data, tc, kernel)
    return results, expected


# ---------------------------------------------------------------------------
# Pytest tests
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not HAS_HIP, reason="HIP Python bindings not available")
class TestScaleRoundtripGPU:

    @pytest.fixture(params=SCALE_ROUNDTRIP_CONFIGS, ids=lambda c: c.label)
    def cfg(self, request):
        return request.param

    def test_roundtrip_scale_a(self, cfg, tmp_path):
        """Verify GR -> LDS -> LR roundtrip for scale A."""
        results, expected = build_and_run_roundtrip(cfg, 'A', tmp_path)
        errors = 0
        for tid in range(NUM_THREADS):
            if results[tid] != expected[tid]:
                errors += 1
                if errors <= 8:
                    print(f"  MISMATCH scale A tid={tid}: got {results[tid]}, expected {expected[tid]}")
        assert errors == 0, f"Scale A roundtrip {cfg.label}: {errors}/{NUM_THREADS} mismatches"

    def test_roundtrip_scale_b(self, cfg, tmp_path):
        """Verify GR -> LDS -> LR roundtrip for scale B."""
        results, expected = build_and_run_roundtrip(cfg, 'B', tmp_path)
        errors = 0
        for tid in range(NUM_THREADS):
            if results[tid] != expected[tid]:
                errors += 1
                if errors <= 8:
                    print(f"  MISMATCH scale B tid={tid}: got {results[tid]}, expected {expected[tid]}")
        assert errors == 0, f"Scale B roundtrip {cfg.label}: {errors}/{NUM_THREADS} mismatches"


# ---------------------------------------------------------------------------
# Debug helpers
# ---------------------------------------------------------------------------
def print_intermediate_values(cfg, tileInfoA, tileInfoB, tc, kernel):
    """Print GR offset, LR offset, writer thread, and expected value per thread."""
    tileInfo = tileInfoA if tc == 'A' else tileInfoB
    otherTileInfo = tileInfoB if tc == 'A' else tileInfoA

    dataLdsSize, scaleALdsSize, _ = compute_lds_sizes(cfg, tileInfoA, tileInfoB, kernel)
    scaleBase = dataLdsSize if tc == 'A' else (dataLdsSize + scaleALdsSize)

    print(f"\n  Intermediate values for scale {tc}:")
    print(f"  dataLdsSize={dataLdsSize}, scaleALdsSize={scaleALdsSize}, scaleBase={scaleBase}")
    print(f"  scaleBlockSize={tileInfo.scaleBlockSize}, scaleDepthU={tileInfo.scaleDepthU}, "
          f"loadRatioGR={tileInfo.loadRatioGR}")
    print(f"  {'tid':>4} {'GR_off':>7} {'LR_off':>7} {'LR_adj':>7} {'writer':>7}")
    print(f"  {'-'*36}")
    for T in range(min(NUM_THREADS, 64)):  # first wave
        gr_off = compute_expected_scale_gr_offset(T, cfg, tileInfo)[0]
        lr_off = compute_expected_scale_lr_offset(T, cfg, tileInfo, otherTileInfo)[0]
        writer_t = lr_off - scaleBase
        print(f"  {T:>4} {gr_off:>7} {lr_off:>7} {lr_off - scaleBase:>7} {writer_t:>7}")


def print_scale_asm_only(kernel_asm):
    """Print only the scale-related sections of the kernel assembly."""
    in_section = False
    for line in kernel_asm.split('\n'):
        if ('GR Offset Calculation for Scale' in line or
            'LR Offset Calculation for Scale' in line or
            'Roundtrip for scale' in line):
            in_section = True
        if in_section:
            print(f"  {line}")
            if line.strip().startswith('s_endpgm') or line.strip().startswith('s_barrier'):
                in_section = False


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Scale GR-LR roundtrip GPU test")
    parser.add_argument("--debug", action="store_true",
                        help="Print intermediate values and full asm (implies --grid)")
    parser.add_argument("--grid", action="store_true",
                        help="Display actual/expected as 2D matrix grids")
    parser.add_argument("--config", type=int, default=None, help="Config index (default: all)")
    parser.add_argument("--tc", default="AB", help="Matrix to test: A, B, or AB (default)")
    parser.add_argument("--list", action="store_true", help="List available configs and exit")
    args = parser.parse_args()

    if args.list:
        for i, cfg in enumerate(SCALE_ROUNDTRIP_CONFIGS):
            print(f"  {i}: {cfg.label}  (mt_a={cfg.mt_a}, mt_b={cfg.mt_b}, "
                  f"du={cfg.depth_u}, stride_a={cfg.stride_a}, stride_b={cfg.stride_b}, "
                  f"mx={cfg.mxblock})")
        sys.exit(0)

    if args.debug:
        args.grid = True

    if not HAS_HIP:
        print("HIP not available")
        sys.exit(1)

    configs = SCALE_ROUNDTRIP_CONFIGS if args.config is None else [SCALE_ROUNDTRIP_CONFIGS[args.config]]
    tc_list = list(args.tc)
    total_errors = 0

    for cfg in configs:
        for tc in tc_list:
            print(f"\n{'='*60}")
            print(f"  Config: {cfg.label}, matrix: {tc}")
            print(f"{'='*60}")

            kernel_asm, _, kernel, tileInfoA, tileInfoB, lds_bytes = \
                generate_roundtrip_kernel(cfg, tc)

            if args.debug:
                print("\n--- Scale ASM (filtered) ---")
                print_scale_asm_only(kernel_asm)
                print("--- End ---")
                print_intermediate_values(cfg, tileInfoA, tileInfoB, tc, kernel)

            tileInfo = tileInfoA if tc == 'A' else tileInfoB
            input_size = compute_input_size(cfg, tileInfo)
            input_data = generate_input_data(input_size)

            other_tileInfo = tileInfoB if tc == 'A' else tileInfoA
            other_input_size = compute_input_size(cfg, other_tileInfo)
            other_input = generate_input_data(other_input_size)

            if tc == 'A':
                input_a, input_b = input_data, other_input
            else:
                input_a, input_b = other_input, input_data

            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_path = type('P', (), {'__truediv__': lambda s, n: os.path.join(tmp_dir, n)})()
                out_size = NUM_THREADS * 4
                label = f"scale_roundtrip_{tc}_{cfg.label}"
                output_bytes = assemble_and_run(kernel_asm, tmp_path, label, out_size,
                                                inputs=(input_a, input_b),
                                                scalars=(cfg.stride_a, cfg.stride_b),
                                                lds_size=lds_bytes)

            results = struct.unpack(f"{NUM_THREADS}I", output_bytes)
            expected = compute_expected_roundtrip(
                cfg, tileInfoA, tileInfoB, input_data, tc, kernel)

            if args.grid:
                print_offset_grid(f"Scale {tc} GPU result ({cfg.label})",
                                  results, WAVESIZE, NUM_WAVES)
                print_offset_grid(f"Scale {tc} EXPECTED ({cfg.label})",
                                  expected, WAVESIZE, NUM_WAVES)

            errors = 0
            for tid in range(NUM_THREADS):
                if results[tid] != expected[tid]:
                    errors += 1
                    if errors <= 8 or args.debug:
                        print(f"  MISMATCH tid={tid}: got {results[tid]}, "
                              f"expected {expected[tid]}")

            if errors == 0:
                print(f"  PASS")
            else:
                print(f"  FAIL: {errors} mismatches")
                if args.grid:
                    diff = [("." if results[t] == expected[t]
                             else f"{results[t]}!={expected[t]}")
                            for t in range(NUM_THREADS)]
                    print(f"\n  Diff (wave x lane):")
                    for w in range(NUM_WAVES):
                        print(f"  w{w}: ", end="")
                        for lane in range(WAVESIZE):
                            tid = w * WAVESIZE + lane
                            if diff[tid] != ".":
                                print(f" t{tid}:{diff[tid]}", end="")
                        print()
                total_errors += errors

    print(f"\n{'='*60}")
    print(f"{'PASSED' if total_errors == 0 else f'FAILED ({total_errors} errors)'}")
    sys.exit(0 if total_errors == 0 else 1)
