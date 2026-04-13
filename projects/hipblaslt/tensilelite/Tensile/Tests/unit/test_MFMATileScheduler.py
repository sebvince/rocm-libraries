"""Tests for MFMATileScheduler — validates each step against the design doc examples.

Example Granularities 1 from newDesign.md:
  - LR A, B : 1x1  (k=1, mn=1)
  - LR SA, SB : 2x2 (k=2, mn=2)
  - GR A, B : 1x2 (loadRatioGR=1.0)
  - GR SA, SB : all tiles (2x2)
  - 2 MFMA tiles in M, 2 in N, 2 subIterK
"""

from Tensile.Components.SubtileBasedKernel import TileInfo
from Tensile.Components.MFMATileScheduler import (
    MFMATileScheduler,
    MFMATileSize,
    MFMATileRange,
    ReadGranularity,
    SchedulerConfig,
    EmittedModule,
)
from unittest.mock import MagicMock


def _mock_dtype(num_bytes=2):
    mock = MagicMock()
    mock.numBytes.return_value = num_bytes
    return mock


def create_kernel(MT0=256, MT1=256, fp4=False, depthU=None):
    mxblock = 32 if fp4 else 0
    bpe = 0.5 if fp4 else 2
    matrixInstK = 128 if fp4 else 32
    if depthU is None:
        depthU = 256 if fp4 else 64
    dtype = _mock_dtype(bpe)
    problemType = {
        "DataTypeA": dtype,
        "DataTypeB": dtype,
        "ComputeDataType": _mock_dtype(4),
    }
    if fp4:
        problemType["MXBlockA"] = mxblock
        problemType["MXBlockB"] = mxblock
    kernel = {
        "DepthU": depthU,
        "_DepthUA": depthU,
        "_DepthUB": depthU,
        "MacroTileA": MT0,
        "MacroTileB": MT1,
        "MacroTile0": MT0,
        "MacroTile1": MT1,
        "MatrixInstM": 16,
        "MatrixInstN": 16,
        "MatrixInstK": matrixInstK,
        "MIWaveGroup": [2, 2],
        "WavefrontSize": 64,
        "SourceSwap": False,
        "MIArchVgpr": False,
        "NonTemporalA": 0,
        "NonTemporalB": 0,
        "NonTemporalMXSA": 0,
        "NonTemporalMXSB": 0,
        "ProblemType": problemType,
    }
    if fp4:
        kernel["_DepthUMXSA"] = depthU // mxblock
        kernel["_DepthUMXSB"] = depthU // mxblock
    return kernel


def make_example_granularities_1():
    """Example Granularities 1 from the design doc: LR A,B=1x1, LR SA,SB=2x2."""
    return SchedulerConfig(
        numMFMATilesM=2,
        numMFMATilesN=2,
        numSubIterK=2,
        lrA=ReadGranularity(MFMATileSize(k=1, mn=1)),
        lrB=ReadGranularity(MFMATileSize(k=1, mn=1)),
        lrSA=ReadGranularity(MFMATileSize(k=2, mn=2)),
        lrSB=ReadGranularity(MFMATileSize(k=2, mn=2)),
        grA=ReadGranularity(MFMATileSize(k=2, mn=1)),
        grB=ReadGranularity(MFMATileSize(k=2, mn=1)),
        grSA=ReadGranularity(MFMATileSize(k=2, mn=2)),
        grSB=ReadGranularity(MFMATileSize(k=2, mn=2)),
    )


# ── Step 1: Place LRs ─────────────────────────────────────

def test_step1_place_LRs():
    """Validate Step 1 output matches design doc Example Granularities 1."""
    cfg = make_example_granularities_1()
    sched = MFMATileScheduler(cfg)
    slots = sched.step1_place_LRs()

    output = sched.print_step1()
    print(output)

    # subIterK=0: MFMA + LR A + LR B + LR SA
    assert len(slots) == 2

    # subIterK=0
    s0 = slots[0]
    assert s0.mfma is not None
    assert s0.mfma.subIterK == 0
    # LRs at subIterK=0: A(loads k=1), B(loads k=1), SA(loads k=[0,1])
    lr_tensors_0 = [lr.tensor for lr in s0.lrs]
    assert 'A' in lr_tensors_0
    assert 'B' in lr_tensors_0
    assert 'SA' in lr_tensors_0

    # LR A at subIterK=0 loads subIterK [1] for MT n
    lr_a_0 = [lr for lr in s0.lrs if lr.tensor == 'A'][0]
    assert lr_a_0.tiles.subIterK_start == 1
    assert lr_a_0.tiles.subIterK_end == 2
    assert lr_a_0.mtIteration == "n"

    # LR SA at subIterK=0 loads subIterK [0,1] for MT n+1
    lr_sa = [lr for lr in s0.lrs if lr.tensor == 'SA'][0]
    assert lr_sa.tiles.subIterK_start == 0
    assert lr_sa.tiles.subIterK_end == 2
    assert lr_sa.mtIteration == "n+1"

    # subIterK=1
    s1 = slots[1]
    lr_tensors_1 = [lr.tensor for lr in s1.lrs]
    assert 'A' in lr_tensors_1
    assert 'B' in lr_tensors_1
    assert 'SB' in lr_tensors_1

    # LR A at subIterK=1 loads subIterK [0] for MT n+1 (wrap-around)
    lr_a_1 = [lr for lr in s1.lrs if lr.tensor == 'A'][0]
    assert lr_a_1.tiles.subIterK_start == 0
    assert lr_a_1.tiles.subIterK_end == 1
    assert lr_a_1.mtIteration == "n+1"

    # LR SB at subIterK=1 loads subIterK [0,1] for MT n+1
    lr_sb = [lr for lr in s1.lrs if lr.tensor == 'SB'][0]
    assert lr_sb.tiles.subIterK_start == 0
    assert lr_sb.tiles.subIterK_end == 2
    assert lr_sb.mtIteration == "n+1"

    # Verify print format matches design doc Step 1
    expected_step1 = """\
MAINLOOP:
  Partition 0:
    subIterK=0:
      MFMAs (MT n, subIterK 0  ) A : [0-1] , B : [0-1]
      LR A  (MT n, subIterK [1]) [0-1]
      LR B  (MT n, subIterK [1]) [0-1]
      LR SA (MT n+1, subIterK [0,1]) [0-1]
    subIterK=1:
      MFMAs (MT n, subIterK 1  ) A : [0-1] , B : [0-1]
      LR A  (MT n+1, subIterK [0]) [0-1]
      LR B  (MT n+1, subIterK [0]) [0-1]
      LR SB (MT n+1, subIterK [0,1]) [0-1]
"""
    assert output == expected_step1


# ── Step 2: Assign VGPR sets ──────────────────────────────

def test_step2_assign_vgpr_sets():
    """Validate Step 2: VGPR set assignments match design doc."""
    cfg = make_example_granularities_1()
    sched = MFMATileScheduler(cfg)
    slots = sched.step2_assign_vgpr_sets()

    output = sched.print_step2()
    print(output)

    # subIterK=0: MFMA reads set 0 for all tensors
    s0 = slots[0]
    assert s0.mfma_sets['A'] == 0
    assert s0.mfma_sets['B'] == 0
    assert s0.mfma_sets['SA'] == 0
    assert s0.mfma_sets['SB'] == 0

    # LR A at subIterK=0 writes set 1 (opposite of MFMA read set 0)
    assert s0.lr_sets['A'] == 1
    assert s0.lr_sets['B'] == 1
    assert s0.lr_sets['SA'] == 1

    # subIterK=1: MFMA reads set 1 (what LR wrote at subIterK=0)
    s1 = slots[1]
    assert s1.mfma_sets['A'] == 1
    assert s1.mfma_sets['B'] == 1
    # SA/SB: MFMA still reads set 0 (SA was loaded but consumed later)
    assert s1.mfma_sets['SA'] == 0
    assert s1.mfma_sets['SB'] == 0

    # LR A at subIterK=1 writes set 0
    assert s1.lr_sets['A'] == 0
    assert s1.lr_sets['B'] == 0
    assert s1.lr_sets['SB'] == 1


# ── Step 3: Place GRs ────────────────────────────────────

def test_step3_place_GRs():
    """Validate Step 3: GR placements match design doc."""
    cfg = make_example_granularities_1()
    sched = MFMATileScheduler(cfg)
    slots = sched.step3_place_GRs()

    output = sched.print_step3()
    print(output)

    # subIterK=0: GR SA + GR A
    s0 = slots[0]
    gr_tensors_0 = [gr.tensor for gr in s0.grs]
    assert 'A' in gr_tensors_0
    assert 'SA' in gr_tensors_0

    # subIterK=1: GR SB + GR B
    s1 = slots[1]
    gr_tensors_1 = [gr.tensor for gr in s1.grs]
    assert 'B' in gr_tensors_1
    assert 'SB' in gr_tensors_1

    # All GRs target MT n+2
    for slot in slots:
        for gr in slot.grs:
            assert gr.mtIteration == 'n+2'


# ── Step 5: Group and serialize ──────────────────────────

def test_step5_group():
    """Validate Step 5: grouped output matches design doc."""
    cfg = make_example_granularities_1()
    sched = MFMATileScheduler(cfg)
    grouped = sched.step5_group()

    output = sched.print_step5()
    print(output)

    # subIterK=0: MFMA, LR A, LR B, LR SA, GR SA, GR A
    g0 = grouped[0]
    op_kinds_0 = [op.kind for op in g0.ops]
    assert op_kinds_0[0] == 'MFMA'
    assert op_kinds_0[1] == 'LR'  # A
    assert op_kinds_0[2] == 'LR'  # B
    assert op_kinds_0[3] == 'LR'  # SA

    # MFMA has WaitLROp before
    assert any(dep.kind == 'wait_lr' for dep in g0.ops[0].before)

    # First LR has WaitGROp before
    assert any(dep.kind == 'wait_gr' for dep in g0.ops[1].before)

    # subIterK=1: MFMA, LR A, LR B, LR SB, GR SB, GR B
    g1 = grouped[1]
    lr_tensors = [op.placement.tensor for op in g1.ops if op.kind == 'LR']
    assert lr_tensors == ['A', 'B', 'SB']


# ── Step 6: EmittedModules ──────────────────────────────────

def test_step6_emit():
    """Validate Step 6: EmittedModule list with correct before-links."""
    cfg = make_example_granularities_1()
    sched = MFMATileScheduler(cfg)
    all_emitted = sched.step6_emit()

    output = sched.print_step6()
    print(output)

    assert len(all_emitted) == 2  # 2 subIterKs

    # ── subIterK=0 ──
    e0 = all_emitted[0]

    # Exactly one MFMA
    mfmas = [e for e in e0 if e.opType == 'mfma']
    assert len(mfmas) == 1

    # MFMA's before chain should include wait_lr
    mfma = mfmas[0]
    assert mfma.before is not None
    assert e0[mfma.before].opType == 'wait_lr'

    # LRs: 3 (A, B, SA)
    lrs = [e for e in e0 if e.opType == 'lr']
    assert len(lrs) == 3

    # First LR's chain: wait_gr → lr_inc → lr
    first_lr = lrs[0]
    chain = _walk_before_chain(e0, first_lr.moduleId)
    chain_types = [e0[mid].opType for mid in chain]
    assert 'wait_gr' in chain_types
    assert 'lr_inc' in chain_types

    # Second LR links back to first LR
    second_lr = lrs[1]
    assert second_lr.before is not None
    assert e0[second_lr.before].opType == 'lr'

    # GRs: 2 (SA, A)
    grs = [e for e in e0 if e.opType == 'gr']
    assert len(grs) == 2

    # First GR's chain: ref(LR A) → wait_lr → sync → GR
    first_gr = grs[0]
    chain = _walk_before_chain(e0, first_gr.moduleId)
    chain_types = [e0[mid].opType for mid in chain]
    assert 'sync' in chain_types
    assert 'wait_lr' in chain_types

    # Second GR links to first GR
    assert grs[1].before is not None
    assert e0[grs[1].before].opType == 'gr'

    # ── subIterK=1 ──
    e1 = all_emitted[1]

    mfmas1 = [e for e in e1 if e.opType == 'mfma']
    assert len(mfmas1) == 1

    lrs1 = [e for e in e1 if e.opType == 'lr']
    assert len(lrs1) == 3  # A, B, SB

    grs1 = [e for e in e1 if e.opType == 'gr']
    assert len(grs1) == 2  # SB, B


def _walk_before_chain(emitted, start_id):
    """Walk the before-chain backwards, returning list of moduleIds."""
    chain = []
    cur = emitted[start_id].before
    while cur is not None:
        chain.append(cur)
        cur = emitted[cur].before
    return chain


# ── from_tile_info ──────────────────────────────────────────

def test_from_tile_info_64x64_fp4():
    """Build config from TileInfo for MT=64, fp4 — should match Example Granularities 1."""
    kernel = create_kernel(64, 64, fp4=True)
    tiA = TileInfo('A', kernel)
    tiB = TileInfo('B', kernel)
    scaleTiA = TileInfo('MXSA', kernel)
    scaleTiB = TileInfo('MXSB', kernel)

    cfg = SchedulerConfig.from_tile_info(
        tiA, tiB,
        lrA=ReadGranularity(MFMATileSize(k=1, mn=1)),
        lrB=ReadGranularity(MFMATileSize(k=1, mn=1)),
        grA=ReadGranularity(MFMATileSize(k=2, mn=1)),
        grB=ReadGranularity(MFMATileSize(k=2, mn=1)),
        scaleTileInfoA=scaleTiA, scaleTileInfoB=scaleTiB,
        lrSA=ReadGranularity(MFMATileSize(k=2, mn=2)),
        lrSB=ReadGranularity(MFMATileSize(k=2, mn=2)),
        grSA=ReadGranularity(MFMATileSize(k=2, mn=2)),
        grSB=ReadGranularity(MFMATileSize(k=2, mn=2)),
    )

    # MT=64, MatrixInstM=16, MIWaveGroup=[2,2] → localMMATileGrid[0] = 64/16/2 = 2
    assert cfg.numMFMATilesM == 2
    assert cfg.numMFMATilesN == 2
    # subtileShape[1] = 2
    assert cfg.numSubIterK == 2
    assert cfg.hasScale

    # Should produce same schedule as manual config
    sched = MFMATileScheduler(cfg)
    output = sched.step1_place_LRs()
    print(sched.print_step1())
    assert len(output) == 2
    assert output[0].mfma.tileA.tileId_end == 2  # 2 MFMA tiles in M


def test_from_tile_info_256x256():
    """Build config from TileInfo for MT=256, no scale."""
    kernel = create_kernel(256, 256)
    tiA = TileInfo('A', kernel)
    tiB = TileInfo('B', kernel)

    cfg = SchedulerConfig.from_tile_info(
        tiA, tiB,
        lrA=ReadGranularity(MFMATileSize(k=1, mn=1)),
        lrB=ReadGranularity(MFMATileSize(k=1, mn=1)),
        grA=ReadGranularity(MFMATileSize(k=2, mn=1)),
        grB=ReadGranularity(MFMATileSize(k=2, mn=1)),
    )

    # MT=256, MatrixInstM=16, MIWaveGroup=[2,2] → localMMATileGrid[0] = 256/16/2 = 8
    assert cfg.numMFMATilesM == 8
    assert cfg.numMFMATilesN == 8
    assert cfg.numSubIterK == 2
    assert not cfg.hasScale


# ── Standalone mode ─────────────────────────────────────────

if __name__ == "__main__":
    import sys

    # Build config from TileInfo (MT=64, fp4 = Example Granularities 1)
    kernel = create_kernel(256, 256, fp4=True)
    tiA = TileInfo('A', kernel)
    tiB = TileInfo('B', kernel)
    scaleTiA = TileInfo('MXSA', kernel)
    scaleTiB = TileInfo('MXSB', kernel)

    cfg = SchedulerConfig.from_tile_info(
        tiA, tiB,
        lrA=ReadGranularity(MFMATileSize(k=1, mn=1)),
        lrB=ReadGranularity(MFMATileSize(k=1, mn=1)),
        grA=ReadGranularity(MFMATileSize(k=2, mn=1)),
        grB=ReadGranularity(MFMATileSize(k=2, mn=1)),
        scaleTileInfoA=scaleTiA, scaleTileInfoB=scaleTiB,
        lrSA=ReadGranularity(MFMATileSize(k=2, mn=2)),
        lrSB=ReadGranularity(MFMATileSize(k=2, mn=2)),
        grSA=ReadGranularity(MFMATileSize(k=2, mn=2)),
        grSB=ReadGranularity(MFMATileSize(k=2, mn=2)),
    )

    print(f"Config: numMFMATilesM={cfg.numMFMATilesM}, "
          f"numMFMATilesN={cfg.numMFMATilesN}, "
          f"numSubIterK={cfg.numSubIterK}, "
          f"hasScale={cfg.hasScale}")
    print()

    sched = MFMATileScheduler(cfg)

    steps = [
        ("Step 1: Place LRs",          lambda: (sched.step1_place_LRs(), sched.print_step1())),
        ("Step 2: Assign VGPR sets",    lambda: (sched.step2_assign_vgpr_sets(), sched.print_step2())),
        ("Step 3: Place GRs",           lambda: (sched.step3_place_GRs(), sched.print_step3())),
        ("Step 4: Annotate deps",       lambda: (sched.step4_annotate_deps(), sched.print_step4())),
        ("Step 5: Group and serialize", lambda: (sched.step5_group(), sched.print_step5())),
        ("Step 6: EmittedModules",      lambda: (sched.step6_emit(), sched.print_step6())),
    ]

    interactive = "--interactive" in sys.argv or "-i" in sys.argv

    for i, (title, run) in enumerate(steps):
        _, output = run()
        print(f"{'=' * 60}")
        print(f"  {title}")
        print(f"{'=' * 60}")
        print(output)
        if interactive and i < len(steps) - 1:
            input("Press Enter for next step...")
