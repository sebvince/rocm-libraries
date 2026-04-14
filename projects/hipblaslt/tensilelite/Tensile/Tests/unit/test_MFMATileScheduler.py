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

def make_256x256_fp4():
    """MT=256x256, DU=256, FP4 config built from TileInfo."""
    kernel = create_kernel(256, 256, fp4=True)
    tiA = TileInfo('A', kernel)
    tiB = TileInfo('B', kernel)
    scaleTiA = TileInfo('MXSA', kernel)
    scaleTiB = TileInfo('MXSB', kernel)
    return SchedulerConfig.from_tile_info(
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


def _get_lr(slot, tensor):
    """Get the LR placement for a given tensor in a slot."""
    matches = [lr for lr in slot.lrs if lr.tensor == tensor]
    assert len(matches) == 1, f"Expected 1 LR for {tensor}, got {len(matches)}"
    return matches[0]

#OK
def test_step1_LR_1x1_partition_1x1():
    """Validate Step 1: MT=256x256, DU=256, FP4.

    Config: numMFMATilesM=8, numMFMATilesN=8, numSubIterK=2
    LR A,B: k=1 (one per subIterK), LR SA,SB: k=2 (one per MT, split across subIterKs)
    """
    cfg = make_256x256_fp4()
    assert cfg.numMFMATilesM == 8
    assert cfg.numMFMATilesN == 8
    assert cfg.numSubIterK == 2
    assert cfg.hasScale

    sched = MFMATileScheduler(cfg)
    partitions = sched.step1_place_LRs()
    slots = partitions[0]
    print(sched.print_step1())

    assert len(slots) == 2

    # ── subIterK=0 ──
    s0 = slots[0]

    # MFMA at subIterK=0 consumes all 8 M/N tiles
    assert s0.mfma.subIterK == 0
    assert s0.mfma.tileA.tileId_start == 0
    assert s0.mfma.tileA.tileId_end == 8
    assert s0.mfma.tileB.tileId_end == 8

    # 3 LRs: A, B (k=1), SA (k=2, placed at subIterK=0)
    assert [lr.tensor for lr in s0.lrs] == ['A', 'B', 'SA']

    # LR A loads next subIterK [1], same MT, all 8 tiles
    lr_a0 = _get_lr(s0, 'A')
    assert lr_a0.mtIteration == "n"
    assert lr_a0.tiles.subIterK_start == 1
    assert lr_a0.tiles.subIterK_end == 2
    assert lr_a0.tiles.tileId_end == 8

    # LR B same pattern as A
    lr_b0 = _get_lr(s0, 'B')
    assert lr_b0.mtIteration == "n"
    assert lr_b0.tiles.subIterK_start == 1

    # LR SA loads all subIterKs [0,1] for next MT
    lr_sa = _get_lr(s0, 'SA')
    assert lr_sa.mtIteration == "n+1"
    assert lr_sa.tiles.subIterK_start == 0
    assert lr_sa.tiles.subIterK_end == 2
    assert lr_sa.tiles.tileId_end == 8

    # ── subIterK=1 ──
    s1 = slots[1]

    assert s1.mfma.subIterK == 1

    # 3 LRs: A, B (k=1 wrap-around → MT n+1), SB (k=2, placed at subIterK=1)
    assert [lr.tensor for lr in s1.lrs] == ['A', 'B', 'SB']

    # LR A wraps to subIterK [0] of next MT
    lr_a1 = _get_lr(s1, 'A')
    assert lr_a1.mtIteration == "n+1"
    assert lr_a1.tiles.subIterK_start == 0
    assert lr_a1.tiles.subIterK_end == 1

    # LR SB loads all subIterKs [0,1] for next MT
    lr_sb = _get_lr(s1, 'SB')
    assert lr_sb.mtIteration == "n+1"
    assert lr_sb.tiles.subIterK_start == 0
    assert lr_sb.tiles.subIterK_end == 2
    assert lr_sb.tiles.tileId_end == 8

#OK
def test_step1_LR_1x2_partition_1x1():
    """Validate Step 1: MT=256x256, DU=256, FP4, LR A/B with k=2.

    This matches design doc Example Granularities 2:
      LR A,B: 1x2 (MFMATileSize(k=2, mn=1))
      LR SA,SB: 2x2

    With k=2 == numSubIterK, LR A/B load all subIterKs at once.
    Split: LR A at subIterK=0, LR B at subIterK=1.
    """
    kernel = create_kernel(256, 256, fp4=True)
    tiA = TileInfo('A', kernel)
    tiB = TileInfo('B', kernel)
    scaleTiA = TileInfo('MXSA', kernel)
    scaleTiB = TileInfo('MXSB', kernel)

    cfg = SchedulerConfig.from_tile_info(
        tiA, tiB,
        lrA=ReadGranularity(MFMATileSize(k=2, mn=1)),
        lrB=ReadGranularity(MFMATileSize(k=2, mn=1)),
        grA=ReadGranularity(MFMATileSize(k=2, mn=1)),
        grB=ReadGranularity(MFMATileSize(k=2, mn=1)),
        scaleTileInfoA=scaleTiA, scaleTileInfoB=scaleTiB,
        lrSA=ReadGranularity(MFMATileSize(k=2, mn=2)),
        lrSB=ReadGranularity(MFMATileSize(k=2, mn=2)),
        grSA=ReadGranularity(MFMATileSize(k=2, mn=2)),
        grSB=ReadGranularity(MFMATileSize(k=2, mn=2)),
    )

    assert cfg.numMFMATilesM == 8
    assert cfg.numMFMATilesN == 8
    assert cfg.numSubIterK == 2

    sched = MFMATileScheduler(cfg)
    partitions = sched.step1_place_LRs()
    slots = partitions[0]
    print(sched.print_step1())

    assert len(slots) == 2

    # ── subIterK=0: LR A (loads MT n+1, all subIterKs) + LR SA ──
    s0 = slots[0]
    assert [lr.tensor for lr in s0.lrs] == ['A', 'SA']

    lr_a = _get_lr(s0, 'A')
    assert lr_a.mtIteration == "n+1"
    assert lr_a.tiles.subIterK_start == 0
    assert lr_a.tiles.subIterK_end == 2
    assert lr_a.tiles.tileId_end == 8

    lr_sa = _get_lr(s0, 'SA')
    assert lr_sa.mtIteration == "n+1"
    assert lr_sa.tiles.subIterK_start == 0
    assert lr_sa.tiles.subIterK_end == 2

    # ── subIterK=1: LR B (loads MT n+1, all subIterKs) + LR SB ──
    s1 = slots[1]
    assert [lr.tensor for lr in s1.lrs] == ['B', 'SB']

    lr_b = _get_lr(s1, 'B')
    assert lr_b.mtIteration == "n+1"
    assert lr_b.tiles.subIterK_start == 0
    assert lr_b.tiles.subIterK_end == 2
    assert lr_b.tiles.tileId_end == 8

    lr_sb = _get_lr(s1, 'SB')
    assert lr_sb.mtIteration == "n+1"
    assert lr_sb.tiles.subIterK_start == 0
    assert lr_sb.tiles.subIterK_end == 2

#OK
def test_step1_LR_1x1_partition_1x1_DU512():
    """Validate Step 1: MT=256x256, DU=512, FP4.

    DU=512 gives localMMATileGrid=[8,4] → numSubIterK=4, numMFMATilesM/N=8.
    LR A,B: k=1 → one LR per subIterK.
    LR SA,SB: k=2 → 2 LRs per tensor (chunks of 2 subIterKs), placed consecutively.
      SA chunks at subIterK=0 and 1, SB chunks at subIterK=2 and 3.
    """
    kernel = create_kernel(256, 256, fp4=True, depthU=512)
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

    assert cfg.numMFMATilesM == 8
    assert cfg.numMFMATilesN == 8
    assert cfg.numSubIterK == 4
    assert cfg.hasScale

    sched = MFMATileScheduler(cfg)
    partitions = sched.step1_place_LRs()
    slots = partitions[0]
    print(sched.print_step1())

    assert len(slots) == 4

    # ── subIterK=0: LR A, LR B, LR SA (chunk 0: loads [2,3] of MT n) ──
    s0 = slots[0]
    assert s0.mfma.subIterK == 0
    assert s0.mfma.tileA.tileId_end == 8
    assert [lr.tensor for lr in s0.lrs] == ['A', 'B', 'SA']

    lr_a0 = _get_lr(s0, 'A')
    assert lr_a0.mtIteration == "n"
    assert lr_a0.tiles.subIterK_start == 1
    assert lr_a0.tiles.subIterK_end == 2

    lr_sa0 = _get_lr(s0, 'SA')
    assert lr_sa0.mtIteration == "n"
    assert lr_sa0.tiles.subIterK_start == 2
    assert lr_sa0.tiles.subIterK_end == 4

    # ── subIterK=1: LR A, LR B, LR SB (chunk 0: loads [2,3] of MT n) ──
    s1 = slots[1]
    assert [lr.tensor for lr in s1.lrs] == ['A', 'B', 'SB']

    lr_a1 = _get_lr(s1, 'A')
    assert lr_a1.mtIteration == "n"
    assert lr_a1.tiles.subIterK_start == 2
    assert lr_a1.tiles.subIterK_end == 3

    lr_sb0 = _get_lr(s1, 'SB')
    assert lr_sb0.mtIteration == "n"
    assert lr_sb0.tiles.subIterK_start == 2
    assert lr_sb0.tiles.subIterK_end == 4

    # ── subIterK=2: LR A, LR B, LR SA (chunk 1: loads [0,1] of MT n+1) ──
    s2 = slots[2]
    assert [lr.tensor for lr in s2.lrs] == ['A', 'B', 'SA']

    lr_a2 = _get_lr(s2, 'A')
    assert lr_a2.mtIteration == "n"
    assert lr_a2.tiles.subIterK_start == 3
    assert lr_a2.tiles.subIterK_end == 4

    lr_sa1 = _get_lr(s2, 'SA')
    assert lr_sa1.mtIteration == "n+1"
    assert lr_sa1.tiles.subIterK_start == 0
    assert lr_sa1.tiles.subIterK_end == 2

    # ── subIterK=3: LR A, LR B (wrap to MT n+1), LR SB (chunk 1: loads [0,1] of MT n+1) ──
    s3 = slots[3]
    assert [lr.tensor for lr in s3.lrs] == ['A', 'B', 'SB']

    lr_a3 = _get_lr(s3, 'A')
    assert lr_a3.mtIteration == "n+1"
    assert lr_a3.tiles.subIterK_start == 0
    assert lr_a3.tiles.subIterK_end == 1

    lr_sb1 = _get_lr(s3, 'SB')
    assert lr_sb1.mtIteration == "n+1"
    assert lr_sb1.tiles.subIterK_start == 0
    assert lr_sb1.tiles.subIterK_end == 2

#OK
def test_step1_LR_1x2_partition_1x1_DU512():
    """Validate Step 1: MT=256x256, DU=512, FP4, LR A/B with k=2.

    DU=512 gives numSubIterK=4. All tensors have k=2 granularity.
    2 chunks of 2 subIterKs each. Tensors split within each chunk:
      chunk 0 (subIterK 0,1): A+SA at slot 0, B+SB at slot 1, loads [2,3] of MT n
      chunk 1 (subIterK 2,3): A+SA at slot 2, B+SB at slot 3, loads [0,1] of MT n+1
    """
    kernel = create_kernel(256, 256, fp4=True, depthU=512)
    tiA = TileInfo('A', kernel)
    tiB = TileInfo('B', kernel)
    scaleTiA = TileInfo('MXSA', kernel)
    scaleTiB = TileInfo('MXSB', kernel)

    cfg = SchedulerConfig.from_tile_info(
        tiA, tiB,
        lrA=ReadGranularity(MFMATileSize(k=2, mn=1)),
        lrB=ReadGranularity(MFMATileSize(k=2, mn=1)),
        grA=ReadGranularity(MFMATileSize(k=2, mn=1)),
        grB=ReadGranularity(MFMATileSize(k=2, mn=1)),
        scaleTileInfoA=scaleTiA, scaleTileInfoB=scaleTiB,
        lrSA=ReadGranularity(MFMATileSize(k=2, mn=2)),
        lrSB=ReadGranularity(MFMATileSize(k=2, mn=2)),
        grSA=ReadGranularity(MFMATileSize(k=2, mn=2)),
        grSB=ReadGranularity(MFMATileSize(k=2, mn=2)),
    )

    assert cfg.numMFMATilesM == 8
    assert cfg.numMFMATilesN == 8
    assert cfg.numSubIterK == 4

    sched = MFMATileScheduler(cfg)
    partitions = sched.step1_place_LRs()
    slots = partitions[0]
    print(sched.print_step1())

    assert len(slots) == 4

    # ── subIterK=0: LR A + LR SA (chunk 0, loads [2,3] of MT n) ──
    s0 = slots[0]
    assert [lr.tensor for lr in s0.lrs] == ['A', 'SA']

    lr_a0 = _get_lr(s0, 'A')
    assert lr_a0.mtIteration == "n"
    assert lr_a0.tiles.subIterK_start == 2
    assert lr_a0.tiles.subIterK_end == 4

    lr_sa0 = _get_lr(s0, 'SA')
    assert lr_sa0.mtIteration == "n"
    assert lr_sa0.tiles.subIterK_start == 2
    assert lr_sa0.tiles.subIterK_end == 4

    # ── subIterK=1: LR B + LR SB (chunk 0, loads [2,3] of MT n) ──
    s1 = slots[1]
    assert [lr.tensor for lr in s1.lrs] == ['B', 'SB']

    lr_b0 = _get_lr(s1, 'B')
    assert lr_b0.mtIteration == "n"
    assert lr_b0.tiles.subIterK_start == 2
    assert lr_b0.tiles.subIterK_end == 4

    lr_sb0 = _get_lr(s1, 'SB')
    assert lr_sb0.mtIteration == "n"
    assert lr_sb0.tiles.subIterK_start == 2
    assert lr_sb0.tiles.subIterK_end == 4

    # ── subIterK=2: LR A + LR SA (chunk 1, loads [0,1] of MT n+1) ──
    s2 = slots[2]
    assert [lr.tensor for lr in s2.lrs] == ['A', 'SA']

    lr_a1 = _get_lr(s2, 'A')
    assert lr_a1.mtIteration == "n+1"
    assert lr_a1.tiles.subIterK_start == 0
    assert lr_a1.tiles.subIterK_end == 2

    lr_sa1 = _get_lr(s2, 'SA')
    assert lr_sa1.mtIteration == "n+1"
    assert lr_sa1.tiles.subIterK_start == 0
    assert lr_sa1.tiles.subIterK_end == 2

    # ── subIterK=3: LR B + LR SB (chunk 1, loads [0,1] of MT n+1) ──
    s3 = slots[3]
    assert [lr.tensor for lr in s3.lrs] == ['B', 'SB']

    lr_b1 = _get_lr(s3, 'B')
    assert lr_b1.mtIteration == "n+1"
    assert lr_b1.tiles.subIterK_start == 0
    assert lr_b1.tiles.subIterK_end == 2

    lr_sb1 = _get_lr(s3, 'SB')
    assert lr_sb1.mtIteration == "n+1"
    assert lr_sb1.tiles.subIterK_start == 0
    assert lr_sb1.tiles.subIterK_end == 2


def test_step1_LR_1x1_partition_2x2():
    """Validate Step 1: MT=256x256, DU=256, FP4, LR A/B with k=1, 2x2 partition grid.

    Same partition layout as test_step1_LR_1x2_2x2 but with k=1 LR granularity
    for A and B: each subIterK gets its own LR per tensor.

    Partition layout (column-major):
      P0: A[0-3], B[0-3]   P2: A[0-3], B[4-7]
      P1: A[4-7], B[0-3]   P3: A[4-7], B[4-7]

    Within-partition K-prefetch (non-wrapping) is placed only for LRs not already
    placed by an earlier partition (tracked by a `placed` set across all partitions).
    Wrapping LRs are only placed when the tile range changes for the next partition
    (tracked by `loaded_ranges` dict).

    Loaded ranges tracking (wrapping):
      Start: A={(0,4)}, B={(0,4)}
      P0: load A (nxt A=(4,8) not loaded) → A={(0,4),(4,8)}
      P1: load B (nxt B=(4,8) not loaded) → B={(0,4),(4,8)}
      P2: no wrapping LRs (both A and B already loaded)
      P3: last partition → wrapping LRs for all (MT n+1)

    K-prefetch dedup (placed set):
      P0 places: A[0-3] k[1], B[0-3] k[1]
      P1 skips:  B[0-3] k[1] (already placed by P0), places A[4-7] k[1]
      P2 skips:  A[0-3] k[1] (already placed by P0), places B[4-7] k[1]
      P3 skips:  A[4-7] k[1] (placed by P1), B[4-7] k[1] (placed by P2)
    """
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
        numPartitionsM=2,
        numPartitionsN=2,
    )

    assert cfg.numPartitions == 4
    assert cfg.partitionSizeM == 4
    assert cfg.partitionSizeN == 4

    sched = MFMATileScheduler(cfg)
    partitions = sched.step1_place_LRs()
    print(sched.print_step1())

    assert len(partitions) == 4

    # ── Partition 0: MFMA A[0-3],B[0-3] → LR A wrapping for P1 (A changes [0-3]→[4-7]) ──
    p0 = partitions[0]
    assert len(p0) == 2
    assert p0[0].mfma.tileA.tileId_start == 0
    assert p0[0].mfma.tileA.tileId_end == 4
    assert p0[0].mfma.tileB.tileId_start == 0
    assert p0[0].mfma.tileB.tileId_end == 4

    # subIterK=0: LR A + LR B (k=1, within-partition K-prefetch for subIterK=1, cur tiles)
    #           + LR SA (k=2, next partition tiles [4-7])
    assert [lr.tensor for lr in p0[0].lrs] == ['A', 'B', 'SA']
    lr_a0 = _get_lr(p0[0], 'A')
    assert lr_a0.tiles.tileId_start == 0
    assert lr_a0.tiles.tileId_end == 4
    assert lr_a0.tiles.subIterK_start == 1
    assert lr_a0.tiles.subIterK_end == 2
    assert lr_a0.mtIteration == "n"

    lr_b0 = _get_lr(p0[0], 'B')
    assert lr_b0.tiles.tileId_start == 0
    assert lr_b0.tiles.tileId_end == 4
    assert lr_b0.tiles.subIterK_start == 1
    assert lr_b0.tiles.subIterK_end == 2
    assert lr_b0.mtIteration == "n"

    lr_sa0 = _get_lr(p0[0], 'SA')
    assert lr_sa0.tiles.tileId_start == 4
    assert lr_sa0.tiles.tileId_end == 8
    assert lr_sa0.tiles.subIterK_start == 0
    assert lr_sa0.tiles.subIterK_end == 2
    assert lr_sa0.mtIteration == "n"

    # subIterK=1: LR A only (k=1, wrapping → loads subIterK [0] with next partition tiles [4-7])
    #             LR B skipped (wrapping, B tiles unchanged for P1)
    assert [lr.tensor for lr in p0[1].lrs] == ['A']
    lr_a1 = _get_lr(p0[1], 'A')
    assert lr_a1.tiles.tileId_start == 4
    assert lr_a1.tiles.tileId_end == 8
    assert lr_a1.tiles.subIterK_start == 0
    assert lr_a1.tiles.subIterK_end == 1
    assert lr_a1.mtIteration == "n"

    # ── Partition 1: MFMA A[4-7],B[0-3] → LR B wrapping for P2 (B changes [0-3]→[4-7]) ──
    p1 = partitions[1]
    assert p1[0].mfma.tileA.tileId_start == 4
    assert p1[0].mfma.tileA.tileId_end == 8
    assert p1[0].mfma.tileB.tileId_start == 0
    assert p1[0].mfma.tileB.tileId_end == 4

    # subIterK=0: LR A (k=1, K-prefetch, cur tiles [4-7])
    #           + LR SB (k=2, next partition tiles [4-7])
    #           LR B skipped (B[0-3] k[1] already placed by P0)
    assert [lr.tensor for lr in p1[0].lrs] == ['A', 'SB']
    lr_a1p = _get_lr(p1[0], 'A')
    assert lr_a1p.tiles.tileId_start == 4
    assert lr_a1p.tiles.tileId_end == 8
    assert lr_a1p.tiles.subIterK_start == 1
    assert lr_a1p.tiles.subIterK_end == 2
    assert lr_a1p.mtIteration == "n"

    lr_sb0 = _get_lr(p1[0], 'SB')
    assert lr_sb0.tiles.tileId_start == 4
    assert lr_sb0.tiles.tileId_end == 8
    assert lr_sb0.mtIteration == "n"

    # subIterK=1: LR B only (k=1, wrapping → loads subIterK [0] with next partition tiles [4-7])
    #             LR A skipped (wrapping, A tiles unchanged for P2)
    assert [lr.tensor for lr in p1[1].lrs] == ['B']
    lr_b1 = _get_lr(p1[1], 'B')
    assert lr_b1.tiles.tileId_start == 4
    assert lr_b1.tiles.tileId_end == 8
    assert lr_b1.tiles.subIterK_start == 0
    assert lr_b1.tiles.subIterK_end == 1
    assert lr_b1.mtIteration == "n"

    # ── Partition 2: MFMA A[0-3],B[4-7] → within-partition K-prefetch only (no wrapping LRs) ──
    p2 = partitions[2]
    assert p2[0].mfma.tileA.tileId_start == 0
    assert p2[0].mfma.tileA.tileId_end == 4
    assert p2[0].mfma.tileB.tileId_start == 4
    assert p2[0].mfma.tileB.tileId_end == 8

    # subIterK=0: LR B only (k=1, K-prefetch for B[4-7] k[1], new tiles)
    #           LR A skipped (A[0-3] k[1] already placed by P0)
    assert [lr.tensor for lr in p2[0].lrs] == ['B']
    lr_b2 = _get_lr(p2[0], 'B')
    assert lr_b2.tiles.tileId_start == 4
    assert lr_b2.tiles.tileId_end == 8
    assert lr_b2.tiles.subIterK_start == 1
    assert lr_b2.tiles.subIterK_end == 2
    assert lr_b2.mtIteration == "n"

    # subIterK=1: no LRs (wrapping, but both A and B tiles unchanged for P3)
    assert len(p2[1].lrs) == 0

    # ── Partition 3: MFMA A[4-7],B[4-7] → Load all for MT n+1 ──
    p3 = partitions[3]
    assert p3[0].mfma.tileA.tileId_start == 4
    assert p3[0].mfma.tileA.tileId_end == 8
    assert p3[0].mfma.tileB.tileId_start == 4
    assert p3[0].mfma.tileB.tileId_end == 8

    # subIterK=0: LR SA only (k=2, next partition P0 tiles [0-3])
    #           LR A skipped (A[4-7] k[1] already placed by P1)
    #           LR B skipped (B[4-7] k[1] already placed by P2)
    assert [lr.tensor for lr in p3[0].lrs] == ['SA']
    lr_sa3 = _get_lr(p3[0], 'SA')
    assert lr_sa3.tiles.tileId_start == 0
    assert lr_sa3.tiles.tileId_end == 4
    assert lr_sa3.mtIteration == "n+1"

    # subIterK=1: LR A, LR B (k=1, wrapping → P0 of MT n+1, tiles [0-3])
    #           + LR SB (k=2, P0 tiles [0-3])
    assert [lr.tensor for lr in p3[1].lrs] == ['A', 'B', 'SB']
    lr_a3_s1 = _get_lr(p3[1], 'A')
    assert lr_a3_s1.tiles.tileId_start == 0
    assert lr_a3_s1.tiles.tileId_end == 4
    assert lr_a3_s1.tiles.subIterK_start == 0
    assert lr_a3_s1.tiles.subIterK_end == 1
    assert lr_a3_s1.mtIteration == "n+1"

    lr_sb3 = _get_lr(p3[1], 'SB')
    assert lr_sb3.tiles.tileId_start == 0
    assert lr_sb3.tiles.tileId_end == 4
    assert lr_sb3.mtIteration == "n+1"



def test_step1_LR_1x1_partition_2x2_DU512():
    """Validate Step 1: MT=256x256, DU=512, FP4, LR A/B k=1, 2x2 partition grid.

    DU=512 gives numSubIterK=4. 8x8 tiles split into 4 partitions of 4x4.
    Partition layout (column-major):
      P0: A[0-3], B[0-3]   P2: A[0-3], B[4-7]
      P1: A[4-7], B[0-3]   P3: A[4-7], B[4-7]

    K-prefetch dedup (placed set):
      P0 places: A[0-3] k[1..3], B[0-3] k[1..3]
      P1 skips:  B[0-3] k[1..3] (placed by P0), places A[4-7] k[1..3]
      P2 skips:  A[0-3] k[1..3] (placed by P0), places B[4-7] k[1..3]
      P3 skips:  A[4-7] k[1..3] (placed by P1), B[4-7] k[1..3] (placed by P2)

    Loaded ranges tracking (wrapping):
      Start: A={(0,4)}, B={(0,4)}
      P0: load A (nxt A=(4,8) not loaded) → A={(0,4),(4,8)}
      P1: load B (nxt B=(4,8) not loaded) → B={(0,4),(4,8)}
      P2: no wrapping LRs (both already loaded)
      P3: last partition → wrapping LRs for all (MT n+1)
    """
    kernel = create_kernel(256, 256, fp4=True, depthU=512)
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
        numPartitionsM=2,
        numPartitionsN=2,
    )

    assert cfg.numPartitions == 4
    assert cfg.numSubIterK == 4
    assert cfg.partitionSizeM == 4
    assert cfg.partitionSizeN == 4

    sched = MFMATileScheduler(cfg)
    partitions = sched.step1_place_LRs()
    print(sched.print_step1())

    assert len(partitions) == 4

    # ── Partition 0: MFMA A[0-3],B[0-3], 4 subIterK ──
    p0 = partitions[0]
    assert len(p0) == 4
    for s in p0:
        assert s.mfma.tileA.tileId_start == 0
        assert s.mfma.tileA.tileId_end == 4
        assert s.mfma.tileB.tileId_start == 0
        assert s.mfma.tileB.tileId_end == 4

    # subIterK=0: A k[1], B k[1], SA [2,3] for next partition [4-7]
    assert [lr.tensor for lr in p0[0].lrs] == ['A', 'B', 'SA']
    lr = _get_lr(p0[0], 'A')
    assert (lr.tiles.tileId_start, lr.tiles.tileId_end) == (0, 4)
    assert (lr.tiles.subIterK_start, lr.tiles.subIterK_end) == (1, 2)
    assert lr.mtIteration == "n"
    lr = _get_lr(p0[0], 'B')
    assert (lr.tiles.tileId_start, lr.tiles.tileId_end) == (0, 4)
    assert (lr.tiles.subIterK_start, lr.tiles.subIterK_end) == (1, 2)
    assert lr.mtIteration == "n"
    lr = _get_lr(p0[0], 'SA')
    assert (lr.tiles.tileId_start, lr.tiles.tileId_end) == (0, 4)
    assert (lr.tiles.subIterK_start, lr.tiles.subIterK_end) == (2, 4)
    assert lr.mtIteration == "n"

    # subIterK=1: A k[2], B k[2], SB k[2,3] K-prefetch for cur tiles [0-3]
    assert [lr.tensor for lr in p0[1].lrs] == ['A', 'B', 'SB']
    lr = _get_lr(p0[1], 'A')
    assert (lr.tiles.subIterK_start, lr.tiles.subIterK_end) == (2, 3)
    assert lr.mtIteration == "n"
    lr = _get_lr(p0[1], 'B')
    assert (lr.tiles.subIterK_start, lr.tiles.subIterK_end) == (2, 3)
    assert lr.mtIteration == "n"
    lr = _get_lr(p0[1], 'SB')
    assert (lr.tiles.tileId_start, lr.tiles.tileId_end) == (0, 4)
    assert (lr.tiles.subIterK_start, lr.tiles.subIterK_end) == (2, 4)
    assert lr.mtIteration == "n"

    # subIterK=2: A k[3], B k[3], SA [0,1] wrapping for next partition [4-7]
    assert [lr.tensor for lr in p0[2].lrs] == ['A', 'B', 'SA']
    lr = _get_lr(p0[2], 'A')
    assert (lr.tiles.subIterK_start, lr.tiles.subIterK_end) == (3, 4)
    assert lr.mtIteration == "n"
    lr = _get_lr(p0[2], 'B')
    assert (lr.tiles.subIterK_start, lr.tiles.subIterK_end) == (3, 4)
    assert lr.mtIteration == "n"
    lr = _get_lr(p0[2], 'SA')
    assert (lr.tiles.tileId_start, lr.tiles.tileId_end) == (4, 8)
    assert (lr.tiles.subIterK_start, lr.tiles.subIterK_end) == (0, 2)
    assert lr.mtIteration == "n"

    # subIterK=3: A wrapping k[0] for next partition [4-7]
    #             B wrapping skipped (B unchanged for P1)
    assert [lr.tensor for lr in p0[3].lrs] == ['A']
    lr = _get_lr(p0[3], 'A')
    assert (lr.tiles.tileId_start, lr.tiles.tileId_end) == (4, 8)
    assert (lr.tiles.subIterK_start, lr.tiles.subIterK_end) == (0, 1)
    assert lr.mtIteration == "n"

    # ── Partition 1: MFMA A[4-7],B[0-3] ──
    p1 = partitions[1]
    assert len(p1) == 4
    for s in p1:
        assert s.mfma.tileA.tileId_start == 4
        assert s.mfma.tileA.tileId_end == 8
        assert s.mfma.tileB.tileId_start == 0
        assert s.mfma.tileB.tileId_end == 4

    # subIterK=0: A k[1] (new tiles [4-7]), B skipped (already placed by P0)
    #           + SA k[2,3] K-prefetch for new tiles [4-7]
    #           SB K-prefetch [0-3] skipped (placed by P0)
    assert [lr.tensor for lr in p1[0].lrs] == ['A', 'SA']
    lr = _get_lr(p1[0], 'A')
    assert (lr.tiles.tileId_start, lr.tiles.tileId_end) == (4, 8)
    assert (lr.tiles.subIterK_start, lr.tiles.subIterK_end) == (1, 2)
    assert lr.mtIteration == "n"
    lr = _get_lr(p1[0], 'SA')
    assert (lr.tiles.tileId_start, lr.tiles.tileId_end) == (4, 8)
    assert (lr.tiles.subIterK_start, lr.tiles.subIterK_end) == (2, 4)
    assert lr.mtIteration == "n"

    # subIterK=1: A k[2]
    assert [lr.tensor for lr in p1[1].lrs] == ['A']
    lr = _get_lr(p1[1], 'A')
    assert (lr.tiles.subIterK_start, lr.tiles.subIterK_end) == (2, 3)
    assert lr.mtIteration == "n"

    # subIterK=2: A k[3], SB [0,1] wrapping for next partition [4-7]
    assert [lr.tensor for lr in p1[2].lrs] == ['A', 'SB']
    lr = _get_lr(p1[2], 'A')
    assert (lr.tiles.subIterK_start, lr.tiles.subIterK_end) == (3, 4)
    assert lr.mtIteration == "n"
    lr = _get_lr(p1[2], 'SB')
    assert (lr.tiles.tileId_start, lr.tiles.tileId_end) == (4, 8)
    assert (lr.tiles.subIterK_start, lr.tiles.subIterK_end) == (0, 2)
    assert lr.mtIteration == "n"

    # subIterK=3: B wrapping k[0] for next partition [4-7]
    #             A wrapping skipped (A unchanged for P2)
    assert [lr.tensor for lr in p1[3].lrs] == ['B']
    lr = _get_lr(p1[3], 'B')
    assert (lr.tiles.tileId_start, lr.tiles.tileId_end) == (4, 8)
    assert (lr.tiles.subIterK_start, lr.tiles.subIterK_end) == (0, 1)
    assert lr.mtIteration == "n"

    # ── Partition 2: MFMA A[0-3],B[4-7] — no wrapping, K-prefetch deduped ──
    p2 = partitions[2]
    assert len(p2) == 4
    for s in p2:
        assert s.mfma.tileA.tileId_start == 0
        assert s.mfma.tileA.tileId_end == 4
        assert s.mfma.tileB.tileId_start == 4
        assert s.mfma.tileB.tileId_end == 8

    # subIterK=0: B k[1] (new tiles [4-7]), A skipped (A[0-3] k[1] placed by P0)
    assert [lr.tensor for lr in p2[0].lrs] == ['B']
    lr = _get_lr(p2[0], 'B')
    assert (lr.tiles.tileId_start, lr.tiles.tileId_end) == (4, 8)
    assert (lr.tiles.subIterK_start, lr.tiles.subIterK_end) == (1, 2)
    assert lr.mtIteration == "n"

    # subIterK=1: B k[2], SB k[2,3] K-prefetch for new tiles [4-7]
    assert [lr.tensor for lr in p2[1].lrs] == ['B', 'SB']
    lr = _get_lr(p2[1], 'B')
    assert (lr.tiles.subIterK_start, lr.tiles.subIterK_end) == (2, 3)
    assert lr.mtIteration == "n"
    lr = _get_lr(p2[1], 'SB')
    assert (lr.tiles.tileId_start, lr.tiles.tileId_end) == (4, 8)
    assert (lr.tiles.subIterK_start, lr.tiles.subIterK_end) == (2, 4)
    assert lr.mtIteration == "n"

    # subIterK=2: B k[3]
    assert [lr.tensor for lr in p2[2].lrs] == ['B']
    lr = _get_lr(p2[2], 'B')
    assert (lr.tiles.subIterK_start, lr.tiles.subIterK_end) == (3, 4)
    assert lr.mtIteration == "n"

    # subIterK=3: no LRs (wrapping skipped, both A and B already loaded)
    assert len(p2[3].lrs) == 0

    # ── Partition 3: MFMA A[4-7],B[4-7] — last, loads for MT n+1 ──
    p3 = partitions[3]
    assert len(p3) == 4
    for s in p3:
        assert s.mfma.tileA.tileId_start == 4
        assert s.mfma.tileA.tileId_end == 8
        assert s.mfma.tileB.tileId_start == 4
        assert s.mfma.tileB.tileId_end == 8

    # subIterK=0: empty — SA [4-7] k[2,3] K-prefetch placed by P1,
    #             A/B K-prefetch placed by P1/P2
    assert len(p3[0].lrs) == 0

    # subIterK=1: empty — SB [4-7] k[2,3] K-prefetch placed by P2
    assert len(p3[1].lrs) == 0

    # subIterK=2: SA [0,1] for MT n+1 tiles [0-3]
    assert [lr.tensor for lr in p3[2].lrs] == ['SA']
    lr = _get_lr(p3[2], 'SA')
    assert (lr.tiles.tileId_start, lr.tiles.tileId_end) == (0, 4)
    assert (lr.tiles.subIterK_start, lr.tiles.subIterK_end) == (0, 2)
    assert lr.mtIteration == "n+1"

    # subIterK=3: A wrapping k[0] [0-3], B wrapping k[0] [0-3], SB [0,1] [0-3]
    assert [lr.tensor for lr in p3[3].lrs] == ['A', 'B', 'SB']
    lr = _get_lr(p3[3], 'A')
    assert (lr.tiles.tileId_start, lr.tiles.tileId_end) == (0, 4)
    assert (lr.tiles.subIterK_start, lr.tiles.subIterK_end) == (0, 1)
    assert lr.mtIteration == "n+1"
    lr = _get_lr(p3[3], 'B')
    assert (lr.tiles.tileId_start, lr.tiles.tileId_end) == (0, 4)
    assert (lr.tiles.subIterK_start, lr.tiles.subIterK_end) == (0, 1)
    assert lr.mtIteration == "n+1"
    lr = _get_lr(p3[3], 'SB')
    assert (lr.tiles.tileId_start, lr.tiles.tileId_end) == (0, 4)
    assert (lr.tiles.subIterK_start, lr.tiles.subIterK_end) == (0, 2)
    assert lr.mtIteration == "n+1"


def test_step1_LR_1x2_partition_2x2():
    """Validate Step 1: MT=256x256, DU=256, FP4, LR A/B with k=2, 2x2 partition grid.

    8x8 MFMA tiles split into 4 partitions of 4x4 tiles each.
    Partition layout (column-major):
      P0: A[0-3], B[0-3]   P2: A[0-3], B[4-7]
      P1: A[4-7], B[0-3]   P3: A[4-7], B[4-7]

    Each partition has same subIterK structure as non-partitioned case,
    but with partition-local tile ranges.
    """
    kernel = create_kernel(256, 256, fp4=True)
    tiA = TileInfo('A', kernel)
    tiB = TileInfo('B', kernel)
    scaleTiA = TileInfo('MXSA', kernel)
    scaleTiB = TileInfo('MXSB', kernel)

    cfg = SchedulerConfig.from_tile_info(
        tiA, tiB,
        lrA=ReadGranularity(MFMATileSize(k=2, mn=1)),
        lrB=ReadGranularity(MFMATileSize(k=2, mn=1)),
        grA=ReadGranularity(MFMATileSize(k=2, mn=1)),
        grB=ReadGranularity(MFMATileSize(k=2, mn=1)),
        scaleTileInfoA=scaleTiA, scaleTileInfoB=scaleTiB,
        lrSA=ReadGranularity(MFMATileSize(k=2, mn=2)),
        lrSB=ReadGranularity(MFMATileSize(k=2, mn=2)),
        grSA=ReadGranularity(MFMATileSize(k=2, mn=2)),
        grSB=ReadGranularity(MFMATileSize(k=2, mn=2)),
        numPartitionsM=2,
        numPartitionsN=2,
    )

    assert cfg.numPartitions == 4
    assert cfg.partitionSizeM == 4
    assert cfg.partitionSizeN == 4

    sched = MFMATileScheduler(cfg)
    partitions = sched.step1_place_LRs()
    print(sched.print_step1())

    assert len(partitions) == 4

    # ── Partition 0: MFMA A[0-3],B[0-3] → LR loads A[4-7] for P1 (only A changes) ──
    p0 = partitions[0]
    assert len(p0) == 2
    assert p0[0].mfma.tileA.tileId_start == 0
    assert p0[0].mfma.tileA.tileId_end == 4
    assert p0[0].mfma.tileB.tileId_start == 0
    assert p0[0].mfma.tileB.tileId_end == 4

    # Only A-side LRs (A changes P0→P1: [0-3]→[4-7], B stays [0-3])
    assert [lr.tensor for lr in p0[0].lrs] == ['A', 'SA']
    lr_a = _get_lr(p0[0], 'A')
    assert lr_a.tiles.tileId_start == 4
    assert lr_a.tiles.tileId_end == 8
    assert lr_a.mtIteration == "n"
    # No B or SB LRs
    assert len(p0[1].lrs) == 0

    # ── Partition 1: MFMA A[4-7],B[0-3] → LR loads B[4-7] for P2 ──
    # Only B-side: B[4-7] not yet loaded → load it.
    # A stays [0-3]→[0-3] for P2 so no A wrapping needed.
    p1 = partitions[1]
    assert p1[0].mfma.tileA.tileId_start == 4
    assert p1[0].mfma.tileA.tileId_end == 8
    assert p1[0].mfma.tileB.tileId_start == 0
    assert p1[0].mfma.tileB.tileId_end == 4

    # subIterK=0: no LRs
    assert len(p1[0].lrs) == 0
    # subIterK=1: LR B + SB (wrapping, B changes [0-3]→[4-7] for P2)
    assert [lr.tensor for lr in p1[1].lrs] == ['B', 'SB']
    lr_b1 = _get_lr(p1[1], 'B')
    assert lr_b1.tiles.tileId_start == 4
    assert lr_b1.tiles.tileId_end == 8
    assert lr_b1.mtIteration == "n"

    # ── Partition 2: MFMA A[0-3],B[4-7] → No LRs needed ──
    # A[0-3] in set 0, A[4-7] in set 1. P3 needs A[4-7] → already in set 1.
    # B[0-3] in set 0, B[4-7] in set 1. P3 needs B[4-7] → already in set 1.
    p2 = partitions[2]
    assert p2[0].mfma.tileA.tileId_start == 0
    assert p2[0].mfma.tileA.tileId_end == 4
    assert p2[0].mfma.tileB.tileId_start == 4
    assert p2[0].mfma.tileB.tileId_end == 8

    assert len(p2[0].lrs) == 0
    assert len(p2[1].lrs) == 0

    # ── Partition 3: MFMA A[4-7],B[4-7] → LR loads A[0-3]+B[0-3] for P0 of MT n+1 ──
    p3 = partitions[3]
    assert p3[0].mfma.tileA.tileId_start == 4
    assert p3[0].mfma.tileA.tileId_end == 8
    assert p3[0].mfma.tileB.tileId_start == 4
    assert p3[0].mfma.tileB.tileId_end == 8

    # Both A and B change (wraps to P0 of next MT)
    assert [lr.tensor for lr in p3[0].lrs] == ['A', 'SA']
    assert [lr.tensor for lr in p3[1].lrs] == ['B', 'SB']
    lr_a3 = _get_lr(p3[0], 'A')
    assert lr_a3.tiles.tileId_start == 0
    assert lr_a3.tiles.tileId_end == 4
    assert lr_a3.mtIteration == "n+1"
    lr_b3 = _get_lr(p3[1], 'B')
    assert lr_b3.tiles.tileId_start == 0
    assert lr_b3.tiles.tileId_end == 4
    assert lr_b3.mtIteration == "n+1"



def test_step1_LR_1x1_partition_10x1():
    """Validate Step 1: MT=320x320, BF16, DU=64, LR A/B k=1, 10x1 partition grid.

    numMFMATilesM=10, numMFMATilesN=10, numSubIterK=2, no scale.
    10 partitions along M, each with 1 A-tile and all 10 B-tiles:
      P0: A[0], B[0-9]
      P1: A[1], B[0-9]
      ...
      P9: A[9], B[0-9]

    B never changes across partitions → only A wrapping LRs needed (P0-P8).
    Each partition's A tile is unique → always needs loading.
    B K-prefetch (subIterK=1) placed once in P0, then deduped for P1-P9.
    P9 (last): loads both A and B for MT n+1.
    """
    kernel = create_kernel(320, 320, fp4=False)
    tiA = TileInfo('A', kernel)
    tiB = TileInfo('B', kernel)

    cfg = SchedulerConfig.from_tile_info(
        tiA, tiB,
        lrA=ReadGranularity(MFMATileSize(k=1, mn=1)),
        lrB=ReadGranularity(MFMATileSize(k=1, mn=1)),
        grA=ReadGranularity(MFMATileSize(k=2, mn=1)),
        grB=ReadGranularity(MFMATileSize(k=2, mn=1)),
        numPartitionsM=10,
        numPartitionsN=1,
    )

    assert cfg.numMFMATilesM == 10
    assert cfg.numMFMATilesN == 10
    assert cfg.numSubIterK == 2
    assert not cfg.hasScale
    assert cfg.numPartitions == 10
    assert cfg.partitionSizeM == 1
    assert cfg.partitionSizeN == 10

    sched = MFMATileScheduler(cfg)
    partitions = sched.step1_place_LRs()
    print(sched.print_step1())

    assert len(partitions) == 10

    # ── P0: A + B K-prefetch, then A wrapping ──
    p0 = partitions[0]
    assert p0[0].mfma.tileA.tileId_start == 0
    assert p0[0].mfma.tileA.tileId_end == 1
    assert p0[0].mfma.tileB.tileId_start == 0
    assert p0[0].mfma.tileB.tileId_end == 10

    # subIterK=0: LR A K-prefetch [0] + LR B K-prefetch [0-9] (both first time)
    assert [lr.tensor for lr in p0[0].lrs] == ['A', 'B']
    lr_a0 = _get_lr(p0[0], 'A')
    assert lr_a0.tiles.tileId_start == 0
    assert lr_a0.tiles.tileId_end == 1
    assert lr_a0.tiles.subIterK_start == 1
    assert lr_a0.mtIteration == "n"
    lr_b0 = _get_lr(p0[0], 'B')
    assert lr_b0.tiles.tileId_start == 0
    assert lr_b0.tiles.tileId_end == 10
    assert lr_b0.tiles.subIterK_start == 1
    assert lr_b0.mtIteration == "n"

    # subIterK=1: LR A wrapping → next partition tile [1]
    assert [lr.tensor for lr in p0[1].lrs] == ['A']
    lr_a1 = _get_lr(p0[1], 'A')
    assert lr_a1.tiles.tileId_start == 1
    assert lr_a1.tiles.tileId_end == 2
    assert lr_a1.tiles.subIterK_start == 0
    assert lr_a1.mtIteration == "n"

    # ── P1 through P8: only A LRs (B K-prefetch already placed by P0) ──
    for pi in range(1, 9):
        p = partitions[pi]
        assert p[0].mfma.tileA.tileId_start == pi
        assert p[0].mfma.tileA.tileId_end == pi + 1
        assert p[0].mfma.tileB.tileId_start == 0
        assert p[0].mfma.tileB.tileId_end == 10

        # subIterK=0: LR A K-prefetch (current tile, new each partition)
        assert [lr.tensor for lr in p[0].lrs] == ['A']
        lr_a0 = _get_lr(p[0], 'A')
        assert lr_a0.tiles.tileId_start == pi
        assert lr_a0.tiles.tileId_end == pi + 1
        assert lr_a0.tiles.subIterK_start == 1
        assert lr_a0.mtIteration == "n"

        # subIterK=1: LR A wrapping → next partition tile
        assert [lr.tensor for lr in p[1].lrs] == ['A']
        lr_a1 = _get_lr(p[1], 'A')
        assert lr_a1.tiles.tileId_start == pi + 1
        assert lr_a1.tiles.tileId_end == pi + 2
        assert lr_a1.tiles.subIterK_start == 0
        assert lr_a1.mtIteration == "n"

    # ── P9 (last): loads both A and B for MT n+1 ──
    p9 = partitions[9]
    assert p9[0].mfma.tileA.tileId_start == 9
    assert p9[0].mfma.tileA.tileId_end == 10
    assert p9[0].mfma.tileB.tileId_start == 0
    assert p9[0].mfma.tileB.tileId_end == 10

    # subIterK=0: LR A K-prefetch only (B K-prefetch [0-9] k[1] already placed by P0)
    assert [lr.tensor for lr in p9[0].lrs] == ['A']
    lr_a_last = _get_lr(p9[0], 'A')
    assert lr_a_last.tiles.tileId_start == 9
    assert lr_a_last.tiles.tileId_end == 10
    assert lr_a_last.tiles.subIterK_start == 1
    assert lr_a_last.mtIteration == "n+1"

    # subIterK=1: LR A (wrapping → P0 tile [0]), LR B (wrapping → [0-9])
    assert [lr.tensor for lr in p9[1].lrs] == ['A', 'B']
    lr_a_wrap = _get_lr(p9[1], 'A')
    assert lr_a_wrap.tiles.tileId_start == 0
    assert lr_a_wrap.tiles.tileId_end == 1
    assert lr_a_wrap.tiles.subIterK_start == 0
    assert lr_a_wrap.mtIteration == "n+1"

    lr_b_wrap = _get_lr(p9[1], 'B')
    assert lr_b_wrap.tiles.tileId_start == 0
    assert lr_b_wrap.tiles.tileId_end == 10
    assert lr_b_wrap.tiles.subIterK_start == 0
    assert lr_b_wrap.mtIteration == "n+1"


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


def test_step2_no_scale_k_gran_1():
    """Step 2: no scales, A/B k_gran=1 → sets alternate every subIterK."""
    cfg = SchedulerConfig(
        numMFMATilesM=2,
        numMFMATilesN=2,
        numSubIterK=2,
        lrA=ReadGranularity(MFMATileSize(k=1, mn=1)),
        lrB=ReadGranularity(MFMATileSize(k=1, mn=1)),
        grA=ReadGranularity(MFMATileSize(k=2, mn=1)),
        grB=ReadGranularity(MFMATileSize(k=2, mn=1)),
    )
    assert not cfg.hasScale

    sched = MFMATileScheduler(cfg)
    slots = sched.step2_assign_vgpr_sets()
    print(sched.print_step2())

    # subIterK=0: MFMA reads set 0, LR writes set 1
    s0 = slots[0]
    assert s0.mfma_sets == {'A': 0, 'B': 0}
    assert s0.lr_sets['A'] == 1
    assert s0.lr_sets['B'] == 1

    # subIterK=1: MFMA reads set 1 (flipped), LR writes set 0
    s1 = slots[1]
    assert s1.mfma_sets == {'A': 1, 'B': 1}
    assert s1.lr_sets['A'] == 0
    assert s1.lr_sets['B'] == 0


def test_step2_no_scale_k_gran_numK():
    """Step 2: no scales, A/B k_gran=numSubIterK → sets never advance."""
    cfg = SchedulerConfig(
        numMFMATilesM=2,
        numMFMATilesN=2,
        numSubIterK=2,
        lrA=ReadGranularity(MFMATileSize(k=2, mn=1)),
        lrB=ReadGranularity(MFMATileSize(k=2, mn=1)),
        grA=ReadGranularity(MFMATileSize(k=2, mn=1)),
        grB=ReadGranularity(MFMATileSize(k=2, mn=1)),
    )
    assert not cfg.hasScale

    sched = MFMATileScheduler(cfg)
    slots = sched.step2_assign_vgpr_sets()
    print(sched.print_step2())

    # Both subIterKs: MFMA stays on set 0 (k_gran == numK, no advance)
    assert slots[0].mfma_sets == {'A': 0, 'B': 0}
    assert slots[1].mfma_sets == {'A': 0, 'B': 0}

    # LR A at slot 0 writes set 1 (opposite of MFMA set 0)
    assert slots[0].lr_sets['A'] == 1
    # LR B at slot 1 writes set 1
    assert slots[1].lr_sets['B'] == 1


def test_step2_partition_2x2():
    """Step 2: 2x2 partition, FP4. All 4 partitions get VGPR set assignments.

    A/B k_gran=1 → sets alternate. SA/SB k_gran=2 → sets stay at 0.
    Each partition has different LR placements (from step1), so lr_sets differ.

    Partition LR placements (from step1):
      P0 (A[0-3],B[0-3]): s0: LR A,B,SA   s1: LR A
      P1 (A[4-7],B[0-3]): s0: LR A,SB     s1: LR B
      P2 (A[0-3],B[4-7]): s0: LR B        s1: (none)
      P3 (A[4-7],B[4-7]): s0: LR SA       s1: LR A,B,SB
    """
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
        numPartitionsM=2,
        numPartitionsN=2,
    )
    assert cfg.numPartitions == 4
    assert cfg.hasScale

    sched = MFMATileScheduler(cfg)
    sched.step2_assign_vgpr_sets()
    print(sched.print_step2())
    parts = sched._step2_partitions

    # A/B k_gran=1 (< numK=2): chunk-based, same for all partitions.
    # SA/SB k_gran=2 (>= numK=2): tile-range tracking across partitions.
    #   SA follows A-side tiles: P0/P2=[0-3]→set 0, P1/P3=[4-7]→set 1
    #   SB follows B-side tiles: P0/P1=[0-3]→set 0, P2/P3=[4-7]→set 1

    # ── P0: SA[0-3]=set0, SB[0-3]=set0. LR A,B,SA at s0; LR A at s1. ──
    p0 = parts[0]
    assert p0[0].mfma_sets == {'A': 0, 'B': 0, 'SA': 0, 'SB': 0}
    assert p0[1].mfma_sets == {'A': 1, 'B': 1, 'SA': 0, 'SB': 0}
    assert p0[0].lr_sets == {'A': 1, 'B': 1, 'SA': 1}
    assert p0[1].lr_sets == {'A': 0}

    # ── P1: SA[4-7]=set1 (loaded by P0 LR), SB[0-3]=set0. ──
    p1 = parts[1]
    assert p1[0].mfma_sets == {'A': 0, 'B': 0, 'SA': 1, 'SB': 0}
    assert p1[1].mfma_sets == {'A': 1, 'B': 1, 'SA': 1, 'SB': 0}
    assert p1[0].lr_sets == {'A': 1, 'SB': 1}
    assert p1[1].lr_sets == {'B': 0}

    # ── P2: SA[0-3]=set0, SB[4-7]=set1 (loaded by P1 LR). ──
    p2 = parts[2]
    assert p2[0].mfma_sets == {'A': 0, 'B': 0, 'SA': 0, 'SB': 1}
    assert p2[1].mfma_sets == {'A': 1, 'B': 1, 'SA': 0, 'SB': 1}
    assert p2[0].lr_sets == {'B': 1}
    assert p2[1].lr_sets == {}

    # ── P3: SA[4-7]=set1, SB[4-7]=set1. LR SA at s0; LR A,B,SB at s1. ──
    p3 = parts[3]
    assert p3[0].mfma_sets == {'A': 0, 'B': 0, 'SA': 1, 'SB': 1}
    assert p3[1].mfma_sets == {'A': 1, 'B': 1, 'SA': 1, 'SB': 1}
    assert p3[0].lr_sets == {'SA': 0}
    assert p3[1].lr_sets == {'A': 0, 'B': 0, 'SB': 0}


def test_step2_DU512():
    """Step 2: DU=512, FP4. numSubIterK=4, A/B k_gran=1, SA/SB k_gran=2.

    A/B: sets flip every subIterK → 0,1,0,1
    SA/SB: sets flip every 2 subIterKs → 0,0,1,1
    """
    kernel = create_kernel(256, 256, fp4=True, depthU=512)
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
    assert cfg.numSubIterK == 4
    assert cfg.hasScale

    sched = MFMATileScheduler(cfg)
    slots = sched.step2_assign_vgpr_sets()
    print(sched.print_step2())

    # A/B: k_gran=1 → sets alternate 0,1,0,1
    ab_expected = [0, 1, 0, 1]
    for k in range(4):
        assert slots[k].mfma_sets['A'] == ab_expected[k], f"A mfma set at k={k}"
        assert slots[k].mfma_sets['B'] == ab_expected[k], f"B mfma set at k={k}"

    # SA/SB: k_gran=2 with numK=4 → 2 chunks, ping-pong by chunk index.
    # Chunk 0 (k=0,1): set 0.  Chunk 1 (k=2,3): set 1.
    sa_sb_expected = [0, 0, 1, 1]
    for k in range(4):
        assert slots[k].mfma_sets['SA'] == sa_sb_expected[k], f"SA mfma set at k={k}"
        assert slots[k].mfma_sets['SB'] == sa_sb_expected[k], f"SB mfma set at k={k}"

    # LR sets: each LR writes opposite of its MFMA set
    for k in range(4):
        for t in slots[k].lr_sets:
            assert slots[k].lr_sets[t] == 1 - slots[k].mfma_sets[t], \
                f"LR {t} at k={k} should write opposite of MFMA set"


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
    partitions = sched.step1_place_LRs()
    print(sched.print_step1())
    output = partitions[0]
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
