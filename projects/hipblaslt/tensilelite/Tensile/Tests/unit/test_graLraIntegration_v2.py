#!/usr/bin/env python3
################################################################################
# End-to-end GPU integration test (v2): Uses production globalReadDoSubtile /
# localReadDoSubtile code paths instead of hand-written asm.
#
# Tests multiple tile sizes, wave group configs ([2,2], [1,4], [4,1]),
# and stride variants (stride==depthU, stride>depthU).
#
# Usage:
#   pytest test_graLraIntegration_v2.py -v -s
#   python test_graLraIntegration_v2.py --debug --wave all
################################################################################

import os
import sys
import tempfile

import pytest
import numpy as np

from gpu_test_helpers import (
    HAS_HIP,
    TileConfig,
    BPE, WAVESIZE, NUM_THREADS,
    create_writer_for_subtile_test,
    init_rocisa,
    assemble_and_run,
    generate_kernel_asm,
    generate_load_params,
)

from Tensile.Components.SubtileBasedKernel import (
    graTileAssignment,
    lraTileAssignment,
    globalReadDTLInitCommonSgpr,
    globalReadDoSubtile,
    localReadDoSubtile,
)
from rocisa.code import Module
from rocisa.container import sgpr
from rocisa.instruction import SMovB32, SMovB64, SWaitCnt, SBarrier


# ---------------------------------------------------------------------------
# Test configurations
# ---------------------------------------------------------------------------
CONFIGS = [
    # 2x2 configs (both mt_a//16 and mt_b//16 even)
    TileConfig(mt_a=256, mt_b=256, depth_u=64, stride_a=64,  stride_b=64),
    TileConfig(mt_a=96,  mt_b=128, depth_u=64, stride_a=64,  stride_b=64),
    # 1x4 config (mt_a//16 odd, mt_b//16 div by 4)
    TileConfig(mt_a=48,  mt_b=128, depth_u=64, stride_a=64,  stride_b=64),
    # 4x1 config (mt_a//16 div by 4, mt_b//16 odd)
    TileConfig(mt_a=128, mt_b=48,  depth_u=64, stride_a=64,  stride_b=64),
    # Stride > depthU variants
    TileConfig(mt_a=256, mt_b=256, depth_u=64, stride_a=128, stride_b=128),
    TileConfig(mt_a=96,  mt_b=128, depth_u=64, stride_a=128, stride_b=128),
    
]


# ---------------------------------------------------------------------------
# Assembly generation using production code paths
# ---------------------------------------------------------------------------

def generate_srd_setup():
    """Generate SRD buffer descriptor setup for A and B using rocisa instructions."""
    module = Module("SRD setup")
    module.add(SMovB64(dst=sgpr("SrdA+0", 2), src=sgpr(4, 2), comment="SrdA base = input_A_ptr"))
    module.add(SMovB32(dst=sgpr("SrdA+2"), src="0xFFFFFFFF",   comment="SrdA NumRecords = max"))
    module.add(SMovB32(dst=sgpr("SrdA+3"), src="0x20000",      comment="SrdA OOB_SELECT=2"))
    module.add(SMovB64(dst=sgpr("SrdB+0", 2), src=sgpr(6, 2), comment="SrdB base = input_B_ptr"))
    module.add(SMovB32(dst=sgpr("SrdB+2"), src="0xFFFFFFFF",   comment="SrdB NumRecords = max"))
    module.add(SMovB32(dst=sgpr("SrdB+3"), src="0x20000",      comment="SrdB OOB_SELECT=2"))
    return module


def generate_export_asm(wave_id, tileInfoA, tileInfoB):
    """Generate assembly to export vgprTile registers from a selected wave.

    Each vgprTile is 4 consecutive vgprs (one dwordx4 = 16 bytes per lane).
    Output layout: [A tiles][B tiles], each tile = WAVESIZE * 16 bytes.
    """
    lines = []
    lines.append(f"  // ---- Wave-gated export (wave {wave_id}) ----")

    # Find a free vgpr range for address computation (need 3: tmp, addr_lo, addr_hi)
    # Scan all used vgprs to find next free
    all_tile_vgprs = set()
    for t in tileInfoA.vgprTiles:
        for v in t:
            all_tile_vgprs.add(v)
    for t in tileInfoB.vgprTiles:
        for v in t:
            all_tile_vgprs.add(v)

    # Also need to know the GR/LR offset vgprs
    for v in tileInfoA.sharedVgprGROffset:
        all_tile_vgprs.add(v)
    for v in tileInfoB.sharedVgprGROffset:
        all_tile_vgprs.add(v)
    for v in tileInfoA.sharedVgprLROffset:
        all_tile_vgprs.add(v)
    for v in tileInfoB.sharedVgprLROffset:
        all_tile_vgprs.add(v)

    # Find 4 consecutive free vgprs past all used
    next_v = max(all_tile_vgprs | {0}) + 1
    tmp = next_v; next_v += 1
    if next_v % 2 != 0:
        next_v += 1
    addr_lo = next_v; next_v += 1
    addr_hi = next_v; next_v += 1

    # Wave masking
    lines.append(f"  v_lshrrev_b32 v{tmp}, 6, v0                // waveId")
    lines.append(f"  v_cmp_eq_u32 vcc, {wave_id}, v{tmp}")
    lines.append(f"  s_and_saveexec_b64 s[2:3], vcc              // gate to wave {wave_id}")

    # Compute laneId for offset
    lines.append(f"  v_and_b32 v{tmp}, 0x3F, v0                  // laneId")

    tile_index = 0
    all_tiles = list(tileInfoA.vgprTiles) + list(tileInfoB.vgprTiles)

    for tile in all_tiles:
        vgpr_start = tile.regList.regValues[0]
        num_regs = len(tile.regList.regValues)
        assert num_regs == 4, f"Expected 4 regs per tile, got {num_regs}"

        # Output offset = tile_index * WAVESIZE * 16 + laneId * 16
        base_offset = tile_index * WAVESIZE * 16
        lines.append(f"  // Export tile {tile_index}: v[{vgpr_start}:{vgpr_start+3}]")
        lines.append(f"  v_lshlrev_b32 v{addr_lo}, 4, v{tmp}       // laneId * 16")
        if base_offset > 0:
            lines.append(f"  v_add_u32 v{addr_lo}, {base_offset}, v{addr_lo}  // + tile base")
        lines.append(f"  v_mov_b32 v{addr_hi}, s9                   // output_ptr hi")
        lines.append(f"  v_add_co_u32 v{addr_lo}, vcc, s8, v{addr_lo}")
        lines.append(f"  v_addc_co_u32 v{addr_hi}, vcc, v{addr_hi}, 0, vcc")
        lines.append(f"  flat_store_dwordx4 v[{addr_lo}:{addr_hi}], v[{vgpr_start}:{vgpr_start+3}]")
        lines.append(f"  s_waitcnt vmcnt(0)")
        tile_index += 1

    lines.append(f"  s_or_b64 exec, exec, s[2:3]                // restore exec")
    return "\n".join(lines), next_v


def generate_integration_kernel_v2(cfg, wave_id=0):
    """Generate a complete kernel using production GR/LR code paths."""
    init_rocisa()

    writer, kernel, tileInfoA, tileInfoB = create_writer_for_subtile_test(cfg)

    # GRA + LRA offset computation
    gra_module = graTileAssignment(writer, kernel, useSwizzling=True)
    lra_module = lraTileAssignment(writer, kernel)

    # DTL init (computes LocalWriteBaseAddr from waveId)
    dtl_module = globalReadDTLInitCommonSgpr(writer, kernel)

    # Global read
    gr_a_module = globalReadDoSubtile('A', writer, kernel)
    gr_b_module = globalReadDoSubtile('B', writer, kernel)

    # Wait + Barrier
    wait_gr = SWaitCnt(dscnt=-1, vlcnt=0, vscnt=-1)
    barrier = SBarrier()

    # Local read
    lr_a_module = localReadDoSubtile('A', writer, kernel)
    lr_b_module = localReadDoSubtile('B', writer, kernel)

    # Wait for LR
    wait_lr = SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1)

    # Export
    export_asm, _next_v = generate_export_asm(wave_id, tileInfoA, tileInfoB)

    # Build inner_asm: SRD setup + production code + export
    srd_module = generate_srd_setup()
    production_asm = "\n".join([
        str(gra_module),
        str(lra_module),
        str(dtl_module),
        str(gr_a_module),
        str(gr_b_module),
        str(wait_gr),
        str(barrier),
        str(lr_a_module),
        str(lr_b_module),
        str(wait_lr),
    ])

    inner_asm = f"""{srd_module}
  // ---- GRA + LRA offset computation ----
{production_asm}

  // ---- Export ----
{export_asm}
"""

    prologue = generate_load_params([
        (4, 4, 0x00, "input_A_ptr + input_B_ptr"),
        (8, 4, 0x10, "output_ptr + strideA + strideB"),
    ])

    args = (
        ("input_A_ptr", 8, "global_buffer", "f16"),
        ("input_B_ptr", 8, "global_buffer", "f16"),
        ("output_ptr",  8, "global_buffer", "u32"),
        ("strideA",     4, "by_value",      "u32"),
        ("strideB",     4, "by_value",      "u32"),
    )

    lds_size = (cfg.mt_a + cfg.mt_b) * cfg.depth_u * BPE
    kernel_asm = generate_kernel_asm(inner_asm, writer, str(prologue), args, lds_size)

    num_tiles_a = len(tileInfoA.vgprTiles)
    num_tiles_b = len(tileInfoB.vgprTiles)
    total_tiles = num_tiles_a + num_tiles_b
    output_size = total_tiles * WAVESIZE * 16

    return kernel_asm, writer, kernel, tileInfoA, tileInfoB, output_size


# ---------------------------------------------------------------------------
# Host-side expected value computation
# ---------------------------------------------------------------------------

def compute_expected_output(cfg, tileInfoA, tileInfoB, kernel, input_A, input_B, wave_id):
    """Compute expected vgprTile contents for a given wave.

    After the GR->LDS->LR roundtrip (with swizzle/de-swizzle cancelling out),
    each vgprTile holds data in MFMA register format:
      - lane16 (lane % 16) maps to row lane16 within the MMA tile
      - lane16Group (lane // 16) maps to column group (8 fp16 per group)

    Returns a list of numpy arrays, one per vgprTile (A tiles first, then B tiles).
    Each array is WAVESIZE * 8 fp16 values (4 dwords = 8 fp16 per lane).
    """
    mi_wave_group = kernel["MIWaveGroup"]
    results = []

    for tc, tileInfo, input_data in [('A', tileInfoA, input_A), ('B', tileInfoB, input_B)]:
        stride = cfg.stride_a if tc == 'A' else cfg.stride_b
        input_matrix = input_data.reshape(-1, stride)

        # Wave offset in the tile dimension
        if tc == 'A':
            wave_offset_factor = wave_id % mi_wave_group[0]
        else:
            wave_offset_factor = wave_id // mi_wave_group[0]

        wave_row_offset = wave_offset_factor * tileInfo.localMMATileGrid[0] * 16

        # Build reverse map: vgprTile index -> (mmaId0, mmaId1)
        tile_to_mma = {}
        for linearId, subtile in enumerate(tileInfo.localSubtiles):
            for mfmaIdx, tileIdx in enumerate(subtile.localReadMap):
                sId0, sId1 = tileInfo.getLocalSubtileIdFromLinearId(linearId)
                # Decode mfmaIdx into (mfmaR, mfmaC) within subtile shape
                mfmaR = mfmaIdx % tileInfo.subtileShape[0]
                mfmaC = mfmaIdx // tileInfo.subtileShape[0]
                mmaId0 = sId0 * tileInfo.subtileShape[0] + mfmaR
                mmaId1 = sId1 * tileInfo.subtileShape[1] + mfmaC
                tile_to_mma[tileIdx] = (mmaId0, mmaId1)

        for tileIdx in range(len(tileInfo.vgprTiles)):
            if tileIdx not in tile_to_mma:
                results.append(np.zeros(WAVESIZE * 8, dtype=np.float16))
                continue

            mmaId0, mmaId1 = tile_to_mma[tileIdx]
            tile_data = np.zeros(WAVESIZE * 8, dtype=np.float16)

            for lane in range(WAVESIZE):
                lane16 = lane % 16
                lane16Group = lane // 16

                row_in_input = wave_row_offset + mmaId0 * 16 + lane16
                col_in_input = mmaId1 * 32 + lane16Group * 8

                if row_in_input < input_matrix.shape[0] and col_in_input + 8 <= input_matrix.shape[1]:
                    tile_data[lane * 8 : lane * 8 + 8] = input_matrix[row_in_input, col_in_input : col_in_input + 8]

            results.append(tile_data)

    return results


def compare_tiles(actual_bytes, expected_tiles, tileInfoA, tileInfoB, wave_id, debug=False):
    """Compare GPU output against expected tile data. Returns number of errors."""
    errors = 0
    tile_size = WAVESIZE * 16  # bytes per tile

    num_tiles_a = len(tileInfoA.vgprTiles)
    num_tiles_b = len(tileInfoB.vgprTiles)
    total_tiles = num_tiles_a + num_tiles_b

    for tile_idx in range(total_tiles):
        tc = 'A' if tile_idx < num_tiles_a else 'B'
        local_idx = tile_idx if tile_idx < num_tiles_a else tile_idx - num_tiles_a

        offset = tile_idx * tile_size
        actual = np.frombuffer(actual_bytes[offset:offset + tile_size], dtype=np.float16)
        expected = expected_tiles[tile_idx]

        if not np.array_equal(actual, expected):
            errors += 1
            if errors <= 8 or debug:
                # Find first mismatching lane
                for lane in range(WAVESIZE):
                    a_slice = actual[lane * 8 : lane * 8 + 8]
                    e_slice = expected[lane * 8 : lane * 8 + 8]
                    if not np.array_equal(a_slice, e_slice):
                        print(f"  MISMATCH wave {wave_id} {tc} tile {local_idx} lane {lane}:")
                        print(f"    expected: {e_slice}")
                        print(f"    actual:   {a_slice}")
                        if not debug:
                            break

    return errors


# ---------------------------------------------------------------------------
# Pytest tests
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_HIP, reason="HIP Python bindings not available")
class TestGraLraIntegrationV2:

    @pytest.fixture(params=CONFIGS, ids=lambda c: c.label)
    def cfg(self, request):
        return request.param

    @pytest.fixture(params=[0, 1, 2, 3], ids=lambda w: f"wave{w}")
    def wave_id(self, request):
        return request.param

    def test_gra_to_lds_to_lra_roundtrip(self, cfg, wave_id, tmp_path):
        """Verify GR -> LDS -> LR roundtrip using production code paths."""
        sys.stdout.flush()

        kernel_asm, writer, kernel, tileInfoA, tileInfoB, output_size = \
            generate_integration_kernel_v2(cfg, wave_id=wave_id)

        # Create input data
        input_A = np.arange(1, cfg.mt_a * cfg.stride_a + 1, dtype=np.float16)
        input_B = -np.arange(1, cfg.mt_b * cfg.stride_b + 1, dtype=np.float16)

        lds_size = (cfg.mt_a + cfg.mt_b) * cfg.depth_u * BPE
        label = f"v2_{cfg.label}_wave{wave_id}"
        output_bytes = assemble_and_run(kernel_asm, tmp_path, label, output_size,
                                        inputs=(input_A, input_B),
                                        scalars=(cfg.stride_a, cfg.stride_b),
                                        lds_size=lds_size)

        expected_tiles = compute_expected_output(cfg, tileInfoA, tileInfoB, kernel,
                                                 input_A, input_B, wave_id)
        errors = compare_tiles(output_bytes, expected_tiles, tileInfoA, tileInfoB, wave_id)

        assert errors == 0, f"Wave {wave_id}, config {cfg.label}: {errors} tile mismatches"


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="GRA+LRA integration GPU test v2")
    parser.add_argument("--debug", action="store_true",
                        help="Print detailed output and asm")
    parser.add_argument("--wave", default="all",
                        help="Which wave to test: 0-3 or 'all' (default: all)")
    parser.add_argument("--config", type=int, default=None,
                        help="Config index to test (default: all)")
    args = parser.parse_args()

    if not HAS_HIP:
        print("HIP not available - cannot run GPU test")
        sys.exit(1)

    wave_list = [0, 1, 2, 3] if args.wave == "all" else [int(args.wave)]
    config_list = CONFIGS if args.config is None else [CONFIGS[args.config]]

    total_errors = 0
    total_tests = 0

    for cfg_idx, cfg in enumerate(config_list):
        print(f"\n{'='*60}")
        print(f"Config: {cfg.label}")
        print(f"  mt_a={cfg.mt_a}, mt_b={cfg.mt_b}, depth_u={cfg.depth_u}")
        print(f"  stride_a={cfg.stride_a}, stride_b={cfg.stride_b}")

        for wave_id in wave_list:
            total_tests += 1
            print(f"\n  --- Wave {wave_id} ---")

            kernel_asm, writer, kernel, tileInfoA, tileInfoB, output_size = \
                generate_integration_kernel_v2(cfg, wave_id=wave_id)

            num_tiles_a = len(tileInfoA.vgprTiles)
            num_tiles_b = len(tileInfoB.vgprTiles)
            print(f"  A tiles: {num_tiles_a}, B tiles: {num_tiles_b}, output: {output_size} bytes")
            print(f"  MIWaveGroup: {kernel['MIWaveGroup']}")

            if args.debug:
                print(f"\n--- Kernel ASM ---\n{kernel_asm}\n--- End ---\n")

            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_path = type('P', (), {'__truediv__': lambda s, n: os.path.join(tmp_dir, n)})()

                input_A = np.arange(1, cfg.mt_a * cfg.stride_a + 1, dtype=np.float16)
                input_B = -np.arange(1, cfg.mt_b * cfg.stride_b + 1, dtype=np.float16)

                lds_size = (cfg.mt_a + cfg.mt_b) * cfg.depth_u * BPE
                label = f"v2_{cfg.label}_wave{wave_id}"
                output_bytes = assemble_and_run(kernel_asm, tmp_path, label, output_size,
                                                inputs=(input_A, input_B),
                                                scalars=(cfg.stride_a, cfg.stride_b),
                                                lds_size=lds_size)

                expected_tiles = compute_expected_output(cfg, tileInfoA, tileInfoB, kernel,
                                                        input_A, input_B, wave_id)
                errors = compare_tiles(output_bytes, expected_tiles, tileInfoA, tileInfoB,
                                       wave_id, debug=args.debug)

                if errors == 0:
                    print(f"  PASS")
                else:
                    print(f"  FAIL: {errors} tile mismatches")
                    total_errors += errors

    print(f"\n{'='*60}")
    print(f"Result: {total_tests} tests, {total_errors} errors")
    if total_errors > 0:
        print("FAILED")
        sys.exit(1)
    else:
        print("PASSED")
