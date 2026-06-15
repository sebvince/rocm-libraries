#!/usr/bin/env python3
################################################################################
# Codegen tests for gfx1250 (wave32) subtile paths.
#
# Exercises TileInfo, GR/LR emit, and kernel helpers with wave32 kernel dicts.
# No GPU hardware required -- tests run against the Python codegen layer only.
#
# Usage:
#   pytest test_subtile_gfx1250_codegen.py -v
################################################################################

import os
import sys
import shutil

import pytest
from types import SimpleNamespace
from unittest.mock import MagicMock

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TENSILE_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
sys.path.insert(0, TENSILE_ROOT)


GFX1250_ISA = (12, 5, 0)
WAVESIZE_32 = 32


def _init_rocisa_gfx1250():
    from rocisa import rocIsa
    from Tensile.Common.Architectures import gfxToIsa
    ri = rocIsa.getInstance()
    isa = gfxToIsa("gfx1250")
    asmpath = shutil.which('amdclang++') or '/usr/bin/amdclang++'
    ri.init(isa, asmpath)
    ri.setKernel(isa, WAVESIZE_32)


def _mock_dtype(num_bytes=2):
    mock = MagicMock()
    mock.numBytes.return_value = num_bytes
    return mock


def _create_gfx1250_kernel(mt_a, mt_b, mi_wave_group=None, depth_u=64):
    dtype = _mock_dtype(2)
    if mi_wave_group is None:
        mi_wave_group = [1, 1]
    return {
        "DepthU": depth_u,
        "_DepthU": depth_u,
        "_DepthUA": depth_u,
        "_DepthUB": depth_u,
        "MacroTileA": mt_a,
        "MacroTileB": mt_b,
        "MacroTile0": mt_a,
        "MacroTile1": mt_b,
        "MatrixInstM": 16,
        "MatrixInstN": 16,
        "MatrixInstK": 32,
        "MIWaveGroup": mi_wave_group,
        "WavefrontSize": WAVESIZE_32,
        "UseSubtileImpl": True,
        "ISA": GFX1250_ISA,
        "MIArchVgpr": True,
        "NonTemporalA": 0,
        "NonTemporalB": 0,
        "enableTDMA": True,
        "enableTDMB": True,
        "ProblemType": {
            "DataTypeA": dtype,
            "DataTypeB": dtype,
            "ComputeDataType": _mock_dtype(4),
        },
    }


def _create_writer_gfx1250(kernel):
    from rocisa.register import RegisterPool
    from rocisa.enum import RegisterType
    from Tensile.Components.Subtile.Kernel import TileInfo, AB_B16_W32

    writer = SimpleNamespace()
    writer.vgprPool = RegisterPool(0, RegisterType.Vgpr,
                                   defaultPreventOverflow=False, printRP=False)
    writer.sgprPool = RegisterPool(0, RegisterType.Sgpr,
                                   defaultPreventOverflow=False, printRP=False)
    writer.agprPool = RegisterPool(0, RegisterType.Accvgpr,
                                   defaultPreventOverflow=False, printRP=False)
    writer.sgprs = {}
    writer.vgprPool.checkOut(1)  # v0 = Serial

    tiA = TileInfo(AB_B16_W32, 'A', writer, kernel)
    tiB = TileInfo(AB_B16_W32, 'B', writer, kernel)

    writer.states = SimpleNamespace(
        a=SimpleNamespace(tileInfo=tiA),
        b=SimpleNamespace(tileInfo=tiB),
        regCaps={"MaxSgpr": 106, "MaxVgpr": 256, "PhysicalMaxVgpr": 512},
        archCaps={"LDSBankCount": 64, "LDSBankWidth": 4},
        asmCaps={"HasMFMA": False, "HasWMMA_AccImmZero": True},
        subtileLdsSwizzle=False,
    )
    readSize = 2 * tiA.subtileSize
    numASubtiles = tiA.globalSubtileGrid[0] * tiA.globalSubtileGrid[1]
    writer.ldsStartOffsetA = 0
    writer.ldsStartOffsetB = int(((numASubtiles * tiA.subtileSize + readSize - 1) // readSize) * readSize)

    return writer, tiA, tiB


def _setup_sgprs(writer):
    """Reserve base SGPRs and TDM descriptor SGPRs."""
    writer.sgprPool.checkOut(12)
    writer.sgprs["StrideA0I"] = 10
    writer.sgprs["StrideB1J"] = 11
    for tc in ['A', 'B']:
        writer.sgprs["tdm%sGroup0" % tc] = writer.sgprPool.checkOutAligned(4, 4, preventOverflow=False)
        writer.sgprs["tdm%sGroup1" % tc] = writer.sgprPool.checkOutAligned(8, 4, preventOverflow=False)
        writer.sgprs["tdmLdsAddr%s" % tc] = writer.sgprPool.checkOut(1, preventOverflow=False)
        writer.sgprs["tdmLdsSwapMask%s" % tc] = writer.sgprPool.checkOut(1, preventOverflow=False)
        writer.sgprs["Address%s" % tc] = writer.sgprPool.checkOutAligned(2, 2, preventOverflow=False)


CONFIGS_1x1 = [
    (32, 32, [1, 1]),
    (64, 64, [1, 1]),
]

CONFIGS_MULTI_WAVE = [
    (64, 64,  [2, 2]),
    (128, 32, [4, 1]),
    (32, 128, [1, 4]),
]


class TestGfx1250SubtileCodegen:
    """Codegen tests for gfx1250 subtile paths."""

    # -- TDM GR offset skip --

    @pytest.mark.parametrize("mt_a,mt_b,wg", CONFIGS_1x1 + CONFIGS_MULTI_WAVE,
                             ids=[f"{a}x{b}_wg{w[0]}x{w[1]}" for a, b, w in CONFIGS_1x1 + CONFIGS_MULTI_WAVE])
    def test_tdm_skips_gr_offset_alloc(self, mt_a, mt_b, wg):
        """TDM-enabled kernel: GR offset registers should not be allocated."""
        _init_rocisa_gfx1250()
        kernel = _create_gfx1250_kernel(mt_a, mt_b, mi_wave_group=wg)
        writer, tiA, tiB = _create_writer_gfx1250(kernel)
        tiA.allocOffsetRegisters(writer, kernel)
        assert tiA.sharedVgprGROffset == []

    # -- LR tile assignment (covers wave partition, rotation skip) --

    @pytest.mark.parametrize("mt_a,mt_b,wg", CONFIGS_MULTI_WAVE,
                             ids=[f"{a}x{b}_wg{w[0]}x{w[1]}" for a, b, w in CONFIGS_MULTI_WAVE])
    def test_lr_tile_assignment_multi_wave(self, mt_a, mt_b, wg):
        """LR tile assignment with multi-wave TDM produces valid assembly."""
        _init_rocisa_gfx1250()
        from Tensile.Components.Subtile.SubtileLREmit import lraTileAssignment
        kernel = _create_gfx1250_kernel(mt_a, mt_b, mi_wave_group=wg)
        writer, tiA, tiB = _create_writer_gfx1250(kernel)
        _setup_sgprs(writer)
        tiA.allocOffsetRegisters(writer, kernel)
        tiB.allocOffsetRegisters(writer, kernel)
        module = lraTileAssignment(writer, kernel)
        asm = str(module)
        assert "TDM wave partition" in asm
        assert "rotation" not in asm.lower()

    # -- emitSingleDsRead dual load for 8-VGPR tiles --

    @pytest.mark.parametrize("mt_a,mt_b,wg", CONFIGS_1x1,
                             ids=[f"{a}x{b}" for a, b, _ in CONFIGS_1x1])
    def test_ds_read_dual_load(self, mt_a, mt_b, wg):
        """Wave32 8-VGPR tiles emit two DSLoadB128 (lo + hi K-halves)."""
        _init_rocisa_gfx1250()
        from Tensile.Components.Subtile.SubtileLREmit import emitSingleDsRead
        kernel = _create_gfx1250_kernel(mt_a, mt_b)
        writer, tiA, tiB = _create_writer_gfx1250(kernel)
        _setup_sgprs(writer)
        tiA.allocOffsetRegisters(writer, kernel)
        tiB.allocOffsetRegisters(writer, kernel)
        tiA.allocVgprTileRegisters_legacy(writer, kernel)
        tile = tiA.vgprTiles[0]
        assert len(tile.regList.indices) == 8
        result = emitSingleDsRead(tiA, 0, 0, 0, tile)
        asm = str(result)
        assert asm.count("ds_load_b128") == 2
        assert "read=0" in asm
        assert "read=1" in asm

    # -- selectDGeometry wave32 --

    def test_select_d_geometry_wave32(self):
        """selectDGeometry returns CD_F32_W32 for wave32 kernels."""
        from Tensile.Components.Subtile.Kernel import selectDGeometry, CD_F32_W32
        kernel = _create_gfx1250_kernel(64, 64)
        assert selectDGeometry(kernel) is CD_F32_W32

    # -- initVgprTilesToZero scalar fallback --

    def test_zero_tiles_wmma(self):
        """gfx1250 tile zeroing uses v_wmma_f32_16x16x4_f32 with acc2_imm=0."""
        _init_rocisa_gfx1250()
        from Tensile.Components.Subtile.Kernel import initVgprTilesToZero
        kernel = _create_gfx1250_kernel(32, 32)
        writer, tiA, tiB = _create_writer_gfx1250(kernel)
        _setup_sgprs(writer)
        tiA.allocOffsetRegisters(writer, kernel)
        tiA.allocVgprTileRegisters_legacy(writer, kernel)
        module = initVgprTilesToZero(writer, kernel, tiA)
        asm = str(module)
        assert "v_wmma_f32_16x16x4_f32" in asm
        assert ", 0" in asm  # acc2_imm=0

    # -- globalReadLDSBufferSwap TDM path --

    @pytest.mark.parametrize("tc", ['A', 'B'])
    def test_gr_lds_buffer_swap_tdm(self, tc):
        """TDM LDS buffer swap emits XOR on tracking SGPR."""
        _init_rocisa_gfx1250()
        from Tensile.Components.Subtile.SubtileGREmit import globalReadLDSBufferSwap
        kernel = _create_gfx1250_kernel(64, 64, mi_wave_group=[2, 2])
        writer, tiA, tiB = _create_writer_gfx1250(kernel)
        _setup_sgprs(writer)
        module = globalReadLDSBufferSwap(tc, writer, kernel)
        asm = str(module)
        assert "s_xor_b32" in asm
        assert "sync descriptor LDS addr" in asm

    # -- globalReadPtrUpdates TDM path --

    @pytest.mark.parametrize("tc", ['A', 'B'])
    def test_gr_ptr_updates_tdm(self, tc):
        """TDM pointer update increments Address and syncs descriptor."""
        _init_rocisa_gfx1250()
        from Tensile.Components.Subtile.SubtileGREmit import globalReadPtrUpdates
        kernel = _create_gfx1250_kernel(64, 64, mi_wave_group=[2, 2])
        writer, tiA, tiB = _create_writer_gfx1250(kernel)
        _setup_sgprs(writer)
        tiA.allocOffsetRegisters(writer, kernel)
        tiB.allocOffsetRegisters(writer, kernel)
        module = globalReadPtrUpdates(tc, writer, kernel)
        asm = str(module)
        assert "s_add_u64" in asm
        assert "sync descriptor global addr" in asm

    # -- emitSingleBufferLoad TDM path --

    @pytest.mark.parametrize("tc", ['A', 'B'])
    def test_buffer_load_tdm(self, tc):
        """TDM emitSingleBufferLoad emits tensor_load_to_lds."""
        _init_rocisa_gfx1250()
        from Tensile.Components.Subtile.SubtileGREmit import emitSingleBufferLoad
        kernel = _create_gfx1250_kernel(64, 64)
        writer, tiA, tiB = _create_writer_gfx1250(kernel)
        _setup_sgprs(writer)
        tiA.allocOffsetRegisters(writer, kernel)
        tiB.allocOffsetRegisters(writer, kernel)
        ti = tiA if tc == 'A' else tiB
        module = emitSingleBufferLoad(ti, kernel, 0, 0)
        asm = str(module)
        assert "tensor_load_to_lds" in asm


# ---------------------------------------------------------------------------
# Fused A/B global read (single TDM per wave)
# ---------------------------------------------------------------------------

from contextlib import contextmanager


def _add_alloc_tmp_sgpr(writer):
    """Give the mock writer a minimal allocTmpSgpr context manager."""
    @contextmanager
    def _allocTmpSgpr(n, *args, **kwargs):
        base = writer.sgprPool.checkOut(n, preventOverflow=False)
        try:
            yield SimpleNamespace(idx=base)
        finally:
            writer.sgprPool.checkIn(base)
    writer.allocTmpSgpr = _allocTmpSgpr


class TestGfx1250FusedGRAB:
    """Codegen tests for the fused A/B global read path."""

    def test_should_fuse_grab_gating(self):
        """shouldFuseGRAB enables only the supported gfx1250 dense same-dtype case."""
        from Tensile.Components.Subtile.Kernel import shouldFuseGRAB

        # Eligible: TDM + subtile + even wave count + same dtype.
        kernel = _create_gfx1250_kernel(64, 64, mi_wave_group=[2, 2])
        assert shouldFuseGRAB(kernel) is True

        # Single wave (<2): not eligible.
        assert shouldFuseGRAB(_create_gfx1250_kernel(64, 64, mi_wave_group=[1, 1])) is False

        # Odd wave count: not eligible.
        assert shouldFuseGRAB(_create_gfx1250_kernel(64, 64, mi_wave_group=[3, 1])) is False

        # Mixed dtype: not eligible.
        k_mixed = _create_gfx1250_kernel(64, 64, mi_wave_group=[2, 2])
        k_mixed["ProblemType"]["DataTypeB"] = _mock_dtype(1)
        assert shouldFuseGRAB(k_mixed) is False

        # TDMSplit: not eligible (fused implies a single mt/2 load).
        k_split = _create_gfx1250_kernel(64, 64, mi_wave_group=[2, 2])
        k_split["TDMSplit"] = True
        assert shouldFuseGRAB(k_split) is False

        # StreamK: eligible -- tdmApplyStreamKOffsetSubtile is a no-op assert
        # (StreamKLocalStart==0), so it composes with the fused descriptor.
        k_streamk = _create_gfx1250_kernel(64, 64, mi_wave_group=[2, 2])
        k_streamk["StreamK"] = 3
        assert shouldFuseGRAB(k_streamk) is True

        # Non-TDM: not eligible.
        k_notdm = _create_gfx1250_kernel(64, 64, mi_wave_group=[2, 2])
        k_notdm["enableTDMA"] = False
        assert shouldFuseGRAB(k_notdm) is False

    def test_parity_merge_emits_cselect(self):
        """tdmFusedParityMerge parity-selects all descriptor words + LDS state."""
        _init_rocisa_gfx1250()
        from Tensile.Components.Subtile.SubtileGREmit import tdmFusedParityMerge
        kernel = _create_gfx1250_kernel(64, 64, mi_wave_group=[2, 2])
        writer, tiA, tiB = _create_writer_gfx1250(kernel)
        _setup_sgprs(writer)
        _add_alloc_tmp_sgpr(writer)

        asm = str(tdmFusedParityMerge(writer, kernel))
        # 4 group0 + 8 group1 + ldsAddr + swapMask = 14 parity selects.
        assert asm.count("s_cselect_b32") == 14, asm
        assert "parity = waveId & 1" in asm

    def test_parity_merge_asserts_on_odd_waves(self):
        """restrict+assert: parity merge rejects an odd wave count."""
        _init_rocisa_gfx1250()
        from Tensile.Components.Subtile.SubtileGREmit import tdmFusedParityMerge
        kernel = _create_gfx1250_kernel(64, 64, mi_wave_group=[3, 1])
        writer, tiA, tiB = _create_writer_gfx1250(kernel)
        _setup_sgprs(writer)
        _add_alloc_tmp_sgpr(writer)
        with pytest.raises(AssertionError):
            tdmFusedParityMerge(writer, kernel)


class TestGfx1250LdsSegmentSplit:
    """Gate predicate for the A0-B-A1 segment-conflict layout."""

    SEG = 65536  # 64KB gfx1250 LDS segment

    def test_eligible_base_distance_ge_segment(self):
        from Tensile.Components.Subtile.Kernel import shouldSplitLdsSegmentsA
        k = _create_gfx1250_kernel(256, 256, mi_wave_group=[2, 2], depth_u=128)
        # Real kernel: A0 base 0, A1 base 106496 -> delta 104K >= 64K.
        assert shouldSplitLdsSegmentsA(k, 0, 106496) is True

    def test_reject_base_distance_below_segment(self):
        from Tensile.Components.Subtile.Kernel import shouldSplitLdsSegmentsA
        k = _create_gfx1250_kernel(256, 256, mi_wave_group=[2, 2], depth_u=128)
        # Bases only 40K apart -> simultaneous A0/A1 accesses can share a segment.
        assert shouldSplitLdsSegmentsA(k, 0, 40000) is False

    def test_base_distance_boundary(self):
        """delta == SEG is enough (the two streams stay exactly one segment apart)."""
        from Tensile.Components.Subtile.Kernel import shouldSplitLdsSegmentsA
        k = _create_gfx1250_kernel(256, 256, mi_wave_group=[2, 2], depth_u=128)
        assert shouldSplitLdsSegmentsA(k, 0, self.SEG) is True
        assert shouldSplitLdsSegmentsA(k, 0, self.SEG - 1) is False

    def test_static_overlap_ok_if_base_distance_ge_segment(self):
        """A0/A1 byte ranges may statically overlap a segment; only delta matters."""
        from Tensile.Components.Subtile.Kernel import shouldSplitLdsSegmentsA
        k = _create_gfx1250_kernel(256, 256, mi_wave_group=[2, 2], depth_u=128)
        # delta == SEG: even if A0 spans seg0/seg1 and A1 spans seg1/seg2, the
        # simultaneous (delta-apart) accesses never co-occupy a segment.
        assert shouldSplitLdsSegmentsA(k, 0, 65536) is True

    def test_reject_non_2x2(self):
        from Tensile.Components.Subtile.Kernel import shouldSplitLdsSegmentsA
        k = _create_gfx1250_kernel(128, 32, mi_wave_group=[4, 1], depth_u=128)
        assert shouldSplitLdsSegmentsA(k, 0, 106496) is False

    def test_reject_when_fusion_off(self):
        from Tensile.Components.Subtile.Kernel import shouldSplitLdsSegmentsA
        k = _create_gfx1250_kernel(256, 256, mi_wave_group=[2, 2], depth_u=128)
        k["enableTDMA"] = False  # disables fused GR eligibility
        assert shouldSplitLdsSegmentsA(k, 0, 106496) is False
