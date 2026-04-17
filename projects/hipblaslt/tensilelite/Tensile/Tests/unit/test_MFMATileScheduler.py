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
    LRPlacement,
    GRPlacement,
    DepRef,
    WaitGRCounts,
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
        grSA=ReadGranularity(MFMATileSize(k=2, mn=8)),
        grSB=ReadGranularity(MFMATileSize(k=2, mn=8)),
    )


def _get_lr(slot, tensor):
    """Get the LR placement for a given tensor in a slot."""
    matches = [lr for lr in slot.lrs if lr.tensor == tensor]
    assert len(matches) == 1, f"Expected 1 LR for {tensor}, got {len(matches)}"
    return matches[0]

#OK
def test_place_LRs_LR_1x1_partition_1x1():
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
    partitions = sched.place_LRs()
    slots = partitions[0]
    print(sched.print_lr())

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
def test_place_LRs_LR_1x2_partition_1x1():
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
        grSA=ReadGranularity(MFMATileSize(k=2, mn=8)),
        grSB=ReadGranularity(MFMATileSize(k=2, mn=8)),
    )

    assert cfg.numMFMATilesM == 8
    assert cfg.numMFMATilesN == 8
    assert cfg.numSubIterK == 2

    sched = MFMATileScheduler(cfg)
    partitions = sched.place_LRs()
    slots = partitions[0]
    print(sched.print_lr())

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
def test_place_LRs_LR_1x1_partition_1x1_DU512():
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
        grSA=ReadGranularity(MFMATileSize(k=2, mn=8)),
        grSB=ReadGranularity(MFMATileSize(k=2, mn=8)),
    )

    assert cfg.numMFMATilesM == 8
    assert cfg.numMFMATilesN == 8
    assert cfg.numSubIterK == 4
    assert cfg.hasScale

    sched = MFMATileScheduler(cfg)
    partitions = sched.place_LRs()
    slots = partitions[0]
    print(sched.print_lr())

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
def test_place_LRs_LR_1x2_partition_1x1_DU512():
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
        grSA=ReadGranularity(MFMATileSize(k=2, mn=8)),
        grSB=ReadGranularity(MFMATileSize(k=2, mn=8)),
    )

    assert cfg.numMFMATilesM == 8
    assert cfg.numMFMATilesN == 8
    assert cfg.numSubIterK == 4

    sched = MFMATileScheduler(cfg)
    partitions = sched.place_LRs()
    slots = partitions[0]
    print(sched.print_lr())

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


def test_place_LRs_LR_1x1_partition_2x2():
    """Validate Step 1: MT=256x256, DU=256, FP4, LR A/B with k=1, 2x2 partition grid.

    Same partition layout as test_place_LRs_LR_1x2_2x2 but with k=1 LR granularity
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
        grSA=ReadGranularity(MFMATileSize(k=2, mn=8)),
        grSB=ReadGranularity(MFMATileSize(k=2, mn=8)),
        numPartitionsM=2,
        numPartitionsN=2,
    )

    assert cfg.numPartitions == 4
    assert cfg.partitionSizeM == 4
    assert cfg.partitionSizeN == 4

    sched = MFMATileScheduler(cfg)
    partitions = sched.place_LRs()
    print(sched.print_lr())

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



def test_place_LRs_LR_1x1_partition_2x2_DU512():
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
        grSA=ReadGranularity(MFMATileSize(k=2, mn=8)),
        grSB=ReadGranularity(MFMATileSize(k=2, mn=8)),
        numPartitionsM=2,
        numPartitionsN=2,
    )

    assert cfg.numPartitions == 4
    assert cfg.numSubIterK == 4
    assert cfg.partitionSizeM == 4
    assert cfg.partitionSizeN == 4

    sched = MFMATileScheduler(cfg)
    partitions = sched.place_LRs()
    print(sched.print_lr())

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


def test_place_LRs_LR_1x2_partition_2x2():
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
        grSA=ReadGranularity(MFMATileSize(k=2, mn=8)),
        grSB=ReadGranularity(MFMATileSize(k=2, mn=8)),
        numPartitionsM=2,
        numPartitionsN=2,
    )

    assert cfg.numPartitions == 4
    assert cfg.partitionSizeM == 4
    assert cfg.partitionSizeN == 4

    sched = MFMATileScheduler(cfg)
    partitions = sched.place_LRs()
    print(sched.print_lr())

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



def test_place_LRs_LR_1x1_partition_10x1():
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
    partitions = sched.place_LRs()
    print(sched.print_lr())

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


# ── Step 2: Assign VGPR tiles ────────────────────────────

def assert_vgpr_no_conflict_and_unrolling(sched):
    """Generic validation for assign_vgpr_tiles results.

    1. MFMA and LR at same subIterK must not share vgprTileIds (per tensor, all iters).
    2. If needs_unrolling: for each wrapping LR (mtIteration != "n") in the last
       unroll iter, the tiles it writes must match what the first MFMA iteration
       reads — i.e. the mainloop's last LR output feeds the next iteration's first MFMA.
    """
    cfg = sched.config
    parts = sched._partitions
    num_iters = sched.unroll_factor

    # ── Check 1: no MFMA/LR vgprTileId overlap at same subIterK ──
    for pi, slots in enumerate(parts):
        for slot in slots:
            if not slot.mfma:
                continue
            for lr in slot.lrs:
                if lr.tensor not in ('A', 'B') or not lr.vgpr_tile_map:
                    continue
                mfma_map_list = getattr(slot.mfma, f'vgpr_tile_map_{lr.tensor}')
                for ui in range(len(mfma_map_list)):
                    mfma_vids = set(mfma_map_list[ui].values())
                    lr_vids = set(lr.vgpr_tile_map[ui].values())
                    assert mfma_vids.isdisjoint(lr_vids), \
                        f"P{pi} k={slot.subIterK} iter={ui}: MFMA and LR {lr.tensor} share vgprTileIds " \
                        f"(MFMA={mfma_vids}, LR={lr_vids})"

    # ── Check 2: unrolling continuity ──
    if sched.needs_unrolling:
        last_ui = num_iters - 1
        wrapping_writes = {}
        for pi, slots in enumerate(parts):
            for slot in slots:
                for lr in slot.lrs:
                    if lr.mtIteration == "n" or lr.tensor not in ('A', 'B'):
                        continue
                    if not lr.vgpr_tile_map:
                        continue
                    tile_map = lr.vgpr_tile_map[last_ui]
                    for tileId, vid in tile_map.items():
                        for lk in lr.tiles.subIterK_list:
                            wrapping_writes[(lr.tensor, tileId, lk, pi)] = vid

        for pi, slots in enumerate(parts):
            for slot in slots:
                if not slot.mfma:
                    continue
                for tensor, tileRange in [('A', slot.mfma.tileA), ('B', slot.mfma.tileB)]:
                    mfma_map_0 = getattr(slot.mfma, f'vgpr_tile_map_{tensor}')[0]
                    for tileId in tileRange.tileId_list:
                        key = (tensor, tileId, slot.subIterK, pi)
                        if key in wrapping_writes:
                            assert mfma_map_0[tileId] == wrapping_writes[key], \
                                f"P{pi} k={slot.subIterK}: MFMA iter0 {tensor} tile {tileId} " \
                                f"uses vgpr {mfma_map_0[tileId]} but last wrapping LR wrote vgpr " \
                                f"{wrapping_writes[key]}"


def test_assign_vgpr_tiles_basic():
    """Validate Step 2: vgprTile allocation with scale tensors."""
    cfg = make_example_granularities_1()
    sched = MFMATileScheduler(cfg)
    sched.assign_vgpr_tiles()

    output = sched.print_vgpr()
    print(output)

    parts = sched._partitions
    s0 = parts[0][0]
    s1 = parts[0][1]

    # Every MFMA must have tile maps for A, B, SA, SB
    assert len(s0.mfma.vgpr_tile_map_A) > 0
    assert len(s0.mfma.vgpr_tile_map_B) > 0
    assert len(s0.mfma.vgpr_tile_map_SA) > 0
    assert len(s0.mfma.vgpr_tile_map_SB) > 0

    # MFMA at k=0 and k=1 must use different vgprTileIds for A/B (check iter 0)
    # (LR at k=0 writes new tiles for k=1)
    for tensor in ('A', 'B'):
        map_k0 = getattr(s0.mfma, f'vgpr_tile_map_{tensor}')[0]
        map_k1 = getattr(s1.mfma, f'vgpr_tile_map_{tensor}')[0]
        for tileId in map_k0:
            if tileId in map_k1:
                assert map_k0[tileId] != map_k1[tileId], \
                    f"{tensor} tile {tileId}: k=0 and k=1 must use different vgprTileIds"

    # Per-tensor peaks should be set
    assert sched.tile_peaks['A'] > 0
    assert sched.tile_peaks['B'] > 0
    assert sched.tile_peaks['SA'] > 0
    assert sched.tile_peaks['SB'] > 0

    assert sched.needs_unrolling
    assert_vgpr_no_conflict_and_unrolling(sched)


def test_assign_vgpr_tiles_no_scale_k_gran_1():
    """Step 2: no scales, A/B k_gran=1 → tiles alternate every subIterK."""
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
    sched.assign_vgpr_tiles()
    print(sched.print_vgpr())

    parts = sched._partitions
    s0, s1 = parts[0][0], parts[0][1]

    # Both MFMA k=0 and k=1 have tile maps
    for tensor in ('A', 'B'):
        map0 = getattr(s0.mfma, f'vgpr_tile_map_{tensor}')
        map1 = getattr(s1.mfma, f'vgpr_tile_map_{tensor}')
        assert len(map0) > 0
        assert len(map1) > 0

    # No SA/SB peaks
    assert 'SA' not in sched.tile_peaks
    assert 'SB' not in sched.tile_peaks
    assert not sched.needs_unrolling
    assert_vgpr_no_conflict_and_unrolling(sched)


def test_assign_vgpr_tiles_DU512():
    """Step 2: DU=512, FP4. numSubIterK=4, A/B k_gran=1, SA/SB k_gran=2.

    Verifies tile maps are populated and no unrolling is needed.
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
        grSA=ReadGranularity(MFMATileSize(k=2, mn=8)),
        grSB=ReadGranularity(MFMATileSize(k=2, mn=8)),
    )
    assert cfg.numSubIterK == 4
    assert cfg.hasScale

    sched = MFMATileScheduler(cfg)
    sched.assign_vgpr_tiles()
    print(sched.print_vgpr())

    # All MFMAs have tile maps
    for slot in sched._partitions[0]:
        assert len(slot.mfma.vgpr_tile_map_A) > 0
        assert len(slot.mfma.vgpr_tile_map_B) > 0
        assert len(slot.mfma.vgpr_tile_map_SA) > 0
        assert len(slot.mfma.vgpr_tile_map_SB) > 0

    assert not sched.needs_unrolling
    assert sched.tile_peaks['SA'] > 0
    assert sched.tile_peaks['SB'] > 0
    assert_vgpr_no_conflict_and_unrolling(sched)


def test_assign_vgpr_tiles_DU512_partition_2x2():
    """Step 2: DU=512 + 2x2 partition, FP4. numSubIterK=4."""
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
        grSA=ReadGranularity(MFMATileSize(k=2, mn=8)),
        grSB=ReadGranularity(MFMATileSize(k=2, mn=8)),
        numPartitionsM=2,
        numPartitionsN=2,
    )
    assert cfg.numPartitions == 4
    assert cfg.numSubIterK == 4
    assert cfg.hasScale

    sched = MFMATileScheduler(cfg)
    sched.assign_vgpr_tiles()
    print(sched.print_vgpr())
    parts = sched._partitions

    # All partitions' MFMAs have tile maps
    for pi in range(4):
        for slot in parts[pi]:
            assert len(slot.mfma.vgpr_tile_map_A) > 0
            assert len(slot.mfma.vgpr_tile_map_B) > 0

    # Spot-check LR presence per partition (unchanged from place_LRs).
    assert [lr.tensor for lr in parts[0][0].lrs] == ['A', 'B', 'SA']
    assert [lr.tensor for lr in parts[0][1].lrs] == ['A', 'B', 'SB']
    assert [lr.tensor for lr in parts[0][2].lrs] == ['A', 'B', 'SA']
    assert [lr.tensor for lr in parts[0][3].lrs] == ['A']

    assert len(parts[3][0].lrs) == 0
    assert len(parts[3][1].lrs) == 0
    assert [lr.tensor for lr in parts[3][2].lrs] == ['SA']
    assert [lr.tensor for lr in parts[3][3].lrs] == ['A', 'B', 'SB']

    assert not sched.needs_unrolling
    assert_vgpr_no_conflict_and_unrolling(sched)


# ── Step 3: Place GRs ────────────────────────────────────

def _assert_gr(slot, tensor, k_start, k_end, tile_start, tile_end, mt='n+2', idx=0):
    """Assert a GR for tensor exists in slot with expected tile range.

    idx selects which GR for the tensor (0-based) when a tensor appears
    multiple times in one slot (e.g. after a split).
    """
    grs = [gr for gr in slot.grs if gr.tensor == tensor]
    assert len(grs) > idx, \
        f"Expected at least {idx+1} GR(s) for {tensor} in slot {slot.subIterK}, got {len(grs)}"
    gr = grs[idx]
    assert gr.mtIteration == mt, \
        f"GR {tensor}[{idx}] in slot {slot.subIterK}: expected mt={mt}, got {gr.mtIteration}"
    assert gr.tiles.subIterK_start == k_start
    assert gr.tiles.subIterK_end == k_end
    assert gr.tiles.tileId_start == tile_start
    assert gr.tiles.tileId_end == tile_end

#OK
def test_place_GRs_LR_1x1_partition_1x1():
    """Step 3: 256x256, DU256, FP4, k=1.

    1 partition, numK=2. GR order: A, B, SA, SB.
    Scale GRs placed after data GRs to avoid LDS conflicts.
    Load counts: A(8), B(8), SA(1), SB(1) = 18 total, 9 per slot.
    s0: A[0-7](8) + B[0-0](1) = 9
    s1: B[1-7](7) + SA[0-7](1) + SB[0-7](1) = 9
    """
    cfg = make_256x256_fp4()
    sched = MFMATileScheduler(cfg)
    slots = sched.place_GRs()
    print(sched.print_gr())

    # subIterK=0: A[0-7], B[0-0]
    assert [gr.tensor for gr in slots[0].grs] == ['A', 'B']
    _assert_gr(slots[0], 'A', 0, 2, 0, 8)
    _assert_gr(slots[0], 'B', 0, 2, 0, 1)

    # subIterK=1: B[1-7], SA[0-7], SB[0-7]
    assert [gr.tensor for gr in slots[1].grs] == ['B', 'SA', 'SB']
    _assert_gr(slots[1], 'B', 0, 2, 1, 8)
    _assert_gr(slots[1], 'SA', 0, 2, 0, 8)
    _assert_gr(slots[1], 'SB', 0, 2, 0, 8)

def test_place_GRs_LR_1x1_partition_1x1_DU512():
    """Step 3: 256x256, DU512, FP4, k=1.

    1 partition, numK=4. grA/B.k=2, grSA/SB.k=4 (full MT).
    6 GR entries: A/B × 2 k-chunks + SA/SB × 1 k-chunk.
    Phase 1 order: A k[0,2), B k[0,2), SA k[0,4), SB k[0,4),
                   A k[2,4), B k[2,4).
    Loads: A(8+8), B(8+8), SA(1), SB(1) = 34 total, 8/slot.

    LDS conflict (k-range aware):
      LR(MT n): A k[1,2) at s0, A k[2,3) at s1, A k[3,4) at s2
                B same, SA k[2,4) at s0, SB k[2,4) at s1
      GR A/B k[0,2): no overlap with later LRs → no conflict
      GR SA k[0,4): overlaps LR SA k[2,4) at s0, but s0 is not "later
                    than s0" → no conflict at s0. Overlaps nothing later → ok
      GR SB k[0,4): overlaps LR SB k[2,4) at s1 → blocked until s1+
                    (but SB LR at s1 only, no later → ok at s2+? No,
                    s1 is the only LR, "later than s1" = none → ok at s2)
                    Actually check: any lr_slot > cur with overlap.
                    At s0: LR SB k[2,4) at s1 > 0 and overlaps → blocked.
                    At s1: no LR SB after s1 → ok.
      GR A/B k[2,4): overlaps LR A/B k[3,4) at s2 → blocked until s2
    s0: A k[0,2)(8) = 8
    s1: B k[0,2)(8) = 8
    s2: SA k[0,4)(1) + SB k[0,4)(1) + A k[2,4) tiles[0,6)(6) = 8
    s3: A k[2,4) tiles[6,8)(2) + B k[2,4)(8) = 10
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
        grSA=ReadGranularity(MFMATileSize(k=4, mn=8)),
        grSB=ReadGranularity(MFMATileSize(k=4, mn=8)),
    )
    sched = MFMATileScheduler(cfg)
    slots = sched.place_GRs()
    print(sched.print_gr())

    # s0: A k[0,2)
    assert [gr.tensor for gr in slots[0].grs] == ['A']
    _assert_gr(slots[0], 'A', 0, 2, 0, 8)

    # s1: B k[0,2)
    assert [gr.tensor for gr in slots[1].grs] == ['B']
    _assert_gr(slots[1], 'B', 0, 2, 0, 8)

    # s2: SA k[0,4) + SB k[0,4) + A k[2,4) tiles[0,6)
    assert [gr.tensor for gr in slots[2].grs] == ['SA', 'SB', 'A']
    _assert_gr(slots[2], 'SA', 0, 4, 0, 8)
    _assert_gr(slots[2], 'SB', 0, 4, 0, 8)
    _assert_gr(slots[2], 'A', 2, 4, 0, 6)

    # s3: A k[2,4) tiles[6,8) + B k[2,4)
    assert [gr.tensor for gr in slots[3].grs] == ['A', 'B']
    _assert_gr(slots[3], 'A', 2, 4, 6, 8)
    _assert_gr(slots[3], 'B', 2, 4, 0, 8)


def test_place_GRs_LR_1x1_partition_2x2():
    """Step 3: 256x256, DU256, FP4, k=1, 2x2 partition.

    Scale GR mn=8 > partition tiles=4 → range snaps to [0,8), deduped.
    Partition traversal (column-major):
      P0→P1 (n+1): A[4-7](4), B[0-3](4), SA[0-7](1), SB[0-7](1)
      P1→P2 (n+1): A[0-3](4), B[4-7](4), SA dedup, SB dedup
      P2→P3 (n+1): A dedup, B dedup
      P3→P0 (n+2): A[0-3](4), B[0-3](4), SA[0-7](1), SB[0-7](1)
    Cross-MT dedup: n+1 entries matching an n+2 tile range are removed
    (previous iteration's n+2 already loaded the same data into LDS).
      A[0-3] n+1 removed (has n+2 match), B[0-3] n+1 removed (same)
      SA/SB n+1 removed (n+2 covers full range)
    GR list: A[4-7] n+1, B[4-7] n+1, A[0-3] n+2, B[0-3] n+2,
             SA[0-7] n+2, SB[0-7] n+2.
    Total=18, per_slot=9.
    s0: A[4-7] B[4-7] SA[0-7] (n+1/n+2, 9 loads)
    s1: A[0-3] B[0-3] SB[0-7] (n+2, 9 loads)
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
        grSA=ReadGranularity(MFMATileSize(k=2, mn=8)),
        grSB=ReadGranularity(MFMATileSize(k=2, mn=8)),
        numPartitionsM=2,
        numPartitionsN=2,
    )
    sched = MFMATileScheduler(cfg)
    slots = sched.place_GRs()
    print(sched.print_gr())
    parts = sched._partitions

    # GRs are distributed across all 8 slots (4 partitions × 2 subIterKs).
    # 18 atoms / 8 slots = 2 per slot (remainder goes to later slots).
    # LDS conflict is per-partition: P2/P3 have no LR A/B (MT n) so
    # n+2 A/B atoms can go to subIterK=0 there.
    # P0 s0: A n+1 [4-5] (2 loads)
    p = parts[0]
    assert [gr.tensor for gr in p[0].grs] == ['A'], "P0 s0"
    _assert_gr(p[0], 'A', 0, 2, 4, 6, mt='n+1')
    # P0 s1: A n+1 [6-7] (2 loads)
    assert [gr.tensor for gr in p[1].grs] == ['A'], "P0 s1"
    _assert_gr(p[1], 'A', 0, 2, 6, 8, mt='n+1')

    # P1 s0: B n+1 [4-5] (2 loads)
    p = parts[1]
    assert [gr.tensor for gr in p[0].grs] == ['B'], "P1 s0"
    _assert_gr(p[0], 'B', 0, 2, 4, 6, mt='n+1')
    # P1 s1: B n+1 [6-7] (2 loads)
    assert [gr.tensor for gr in p[1].grs] == ['B'], "P1 s1"
    _assert_gr(p[1], 'B', 0, 2, 6, 8, mt='n+1')

    # P2 s0: A n+2 [0-1] (2 loads) — no LR A (MT n) in P2, no conflict
    p = parts[2]
    assert [gr.tensor for gr in p[0].grs] == ['A'], "P2 s0"
    _assert_gr(p[0], 'A', 0, 2, 0, 2, mt='n+2')
    # P2 s1: A n+2 [2-3] (2 loads)
    assert [gr.tensor for gr in p[1].grs] == ['A'], "P2 s1"
    _assert_gr(p[1], 'A', 0, 2, 2, 4, mt='n+2')

    # P3 s0: B n+2 [0-1] (2 loads) — no LR B (MT n) in P3, no conflict
    p = parts[3]
    assert [gr.tensor for gr in p[0].grs] == ['B'], "P3 s0"
    _assert_gr(p[0], 'B', 0, 2, 0, 2, mt='n+2')
    # P3 s1: B n+2 [2-3], SA n+2, SB n+2 (4 loads)
    assert [gr.tensor for gr in p[1].grs] == ['B', 'SA', 'SB'], "P3 s1"
    _assert_gr(p[1], 'B', 0, 2, 2, 4, mt='n+2')
    _assert_gr(p[1], 'SA', 0, 2, 0, 8, mt='n+2')
    _assert_gr(p[1], 'SB', 0, 2, 0, 8, mt='n+2')


def test_place_GRs_LR_1x1_partition_2x2_DU512():
    """Step 3: 256x256, DU512, FP4, k=1, 2x2 partition.

    DU512 → numK=4. GR k-gran=2 → two k-chunks: k[0,1] and k[2,3].
    Cross-MT dedup removes n+1 A/B[0-3] and SA/SB (n+2 covers same tiles).
    GR list: A[4-7] n+1 k[0,1](4), A[4-7] n+1 k[2,3](4),
             B[4-7] n+1 k[0,1](4), B[4-7] n+1 k[2,3](4),
             A[0-3] n+2 k[0,1](4), B[0-3] n+2 k[0,1](4),
             SA[0-7] n+2 k[0,1](1), SB[0-7] n+2 k[0,1](1),
             A[0-3] n+2 k[2,3](4), B[0-3] n+2 k[2,3](4),
             SA[0-7] n+2 k[2,3](1), SB[0-7] n+2 k[2,3](1).
    Total=36 atoms, 16 slots, 2 per slot.
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
        grSA=ReadGranularity(MFMATileSize(k=4, mn=8)),
        grSB=ReadGranularity(MFMATileSize(k=4, mn=8)),
        numPartitionsM=2,
        numPartitionsN=2,
    )
    sched = MFMATileScheduler(cfg)
    slots = sched.place_GRs()
    print(sched.print_gr())
    parts = sched._partitions

    # 10 GR entries, 34 atoms, 16 slots, 2 per slot.
    # SA/SB k=4 → one entry each covering k[0,3] (deduped across k-chunks).

    # P0: A n+1 — k[0,1] in s0/s1, k[2,3] in s2/s3
    p = parts[0]
    assert [gr.tensor for gr in p[0].grs] == ['A'], "P0 s0"
    _assert_gr(p[0], 'A', 0, 2, 4, 6, mt='n+1')
    assert [gr.tensor for gr in p[1].grs] == ['A'], "P0 s1"
    _assert_gr(p[1], 'A', 0, 2, 6, 8, mt='n+1')
    assert [gr.tensor for gr in p[2].grs] == ['A'], "P0 s2"
    _assert_gr(p[2], 'A', 2, 4, 4, 6, mt='n+1')
    assert [gr.tensor for gr in p[3].grs] == ['A'], "P0 s3"
    _assert_gr(p[3], 'A', 2, 4, 6, 8, mt='n+1')

    # P1: B n+1 — k[0,1] in s0/s1, k[2,3] in s2/s3
    p = parts[1]
    assert [gr.tensor for gr in p[0].grs] == ['B'], "P1 s0"
    _assert_gr(p[0], 'B', 0, 2, 4, 6, mt='n+1')
    assert [gr.tensor for gr in p[1].grs] == ['B'], "P1 s1"
    _assert_gr(p[1], 'B', 0, 2, 6, 8, mt='n+1')
    assert [gr.tensor for gr in p[2].grs] == ['B'], "P1 s2"
    _assert_gr(p[2], 'B', 2, 4, 4, 6, mt='n+1')
    assert [gr.tensor for gr in p[3].grs] == ['B'], "P1 s3"
    _assert_gr(p[3], 'B', 2, 4, 6, 8, mt='n+1')

    # P2: A/B n+2 k[0,1]
    p = parts[2]
    assert [gr.tensor for gr in p[0].grs] == ['A'], "P2 s0"
    _assert_gr(p[0], 'A', 0, 2, 0, 2, mt='n+2')
    assert [gr.tensor for gr in p[1].grs] == ['A'], "P2 s1"
    _assert_gr(p[1], 'A', 0, 2, 2, 4, mt='n+2')
    assert [gr.tensor for gr in p[2].grs] == ['B'], "P2 s2"
    _assert_gr(p[2], 'B', 0, 2, 0, 2, mt='n+2')
    assert [gr.tensor for gr in p[3].grs] == ['B'], "P2 s3"
    _assert_gr(p[3], 'B', 0, 2, 2, 4, mt='n+2')

    # P3 s0: SA+SB n+2 k[0,3] (2 loads, full k range)
    p = parts[3]
    assert [gr.tensor for gr in p[0].grs] == ['SA', 'SB'], "P3 s0"
    _assert_gr(p[0], 'SA', 0, 4, 0, 8, mt='n+2')
    _assert_gr(p[0], 'SB', 0, 4, 0, 8, mt='n+2')
    # P3 s1: A n+2 k[2,3] [0-1]
    assert [gr.tensor for gr in p[1].grs] == ['A'], "P3 s1"
    _assert_gr(p[1], 'A', 2, 4, 0, 2, mt='n+2')
    # P3 s2: A n+2 k[2,3] [2-3]
    assert [gr.tensor for gr in p[2].grs] == ['A'], "P3 s2"
    _assert_gr(p[2], 'A', 2, 4, 2, 4, mt='n+2')
    # P3 s3: B n+2 k[2,3] [0-3] (4 loads — overflow)
    assert [gr.tensor for gr in p[3].grs] == ['B'], "P3 s3"
    _assert_gr(p[3], 'B', 2, 4, 0, 4, mt='n+2')


def test_place_GRs_LR_1x1_partition_10x1():
    """Step 3: 320x320, BF16, k=1, 10x1 partition. No scales.

    10 M-partitions, 1 N-partition → B range [0-9] is the same for all.
    Partition traversal: P0→P1, ..., P8→P9 (n+1), P9→P0 (n+2, wraps).
    Cross-MT dedup: A[0] n+1 removed (has n+2), B[0-9] n+1 removed (has n+2).
    GR list: A[1]..A[9] n+1 (9), A[0] n+2 (1), B[0-9] n+2 (10) = 20 atoms.
    20 atoms / 20 slots = 1 per slot.
    P0..P4: A atoms (n+1 then n+2 for A[0])
    P5..P9: B n+2 atoms (1 per slot, remerge not possible)
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
    sched = MFMATileScheduler(cfg)
    slots = sched.place_GRs()
    print(sched.print_gr())
    parts = sched._partitions

    # 1 atom per slot across 20 slots (10 partitions × 2 subIterKs).
    # P0..P3: each slot gets one A n+1 tile
    for pi in range(4):
        p = parts[pi]
        assert [gr.tensor for gr in p[0].grs] == ['A'], f"P{pi} s0"
        _assert_gr(p[0], 'A', 0, 2, pi * 2 + 1, pi * 2 + 2, mt='n+1')
        assert [gr.tensor for gr in p[1].grs] == ['A'], f"P{pi} s1"
        _assert_gr(p[1], 'A', 0, 2, pi * 2 + 2, pi * 2 + 3, mt='n+1')

    # P4 s0: A n+1 [9], s1: A n+2 [0]
    p = parts[4]
    assert [gr.tensor for gr in p[0].grs] == ['A'], "P4 s0"
    _assert_gr(p[0], 'A', 0, 2, 9, 10, mt='n+1')
    assert [gr.tensor for gr in p[1].grs] == ['A'], "P4 s1"
    _assert_gr(p[1], 'A', 0, 2, 0, 1, mt='n+2')

    # P5..P9: each slot gets one B n+2 tile
    for pi in range(5, 10):
        p = parts[pi]
        b_idx = (pi - 5) * 2
        assert [gr.tensor for gr in p[0].grs] == ['B'], f"P{pi} s0"
        _assert_gr(p[0], 'B', 0, 2, b_idx, b_idx + 1, mt='n+2')
        assert [gr.tensor for gr in p[1].grs] == ['B'], f"P{pi} s1"
        _assert_gr(p[1], 'B', 0, 2, b_idx + 1, b_idx + 2, mt='n+2')


# ── Step 4: Annotate deps ────────────────────────────────────

def _dep_refs(placement):
    """Return list of (type, tensor, partition, subIterK_slot, mt_offset) for a placement's deps."""
    result = []
    for dep in placement.deps:
        p = dep.ref
        kind = 'LR' if isinstance(p, LRPlacement) else 'GR'
        result.append((kind, p.tensor, p.partition, p.subIterK_slot, dep.mt_offset))
    return result

#OK
def test_annotate_deps_1x1_partition_DU256():
    """Step 4: 256x256, DU256, FP4, 1 partition, 2 subIterKs.

    Deps follow three rules:
    - MFMA(k) depends on all LRs that loaded subIterK=k data
    - LR depends on GR for same tensor
    - GR depends on collision LR at same subIterK slot

    cross_mt: True when dep is on previous iteration's execution.
    Within one iteration, slot order is 0→1, and within a slot: MFMA→LR→GR.
    """
    cfg = make_256x256_fp4()
    sched = MFMATileScheduler(cfg)
    sched.annotate_deps()
    parts = sched._partitions
    print(sched.print_deps())

    s0 = parts[0][0]  # P0, subIterK=0
    s1 = parts[0][1]  # P0, subIterK=1

    # ── subIterK=0 ──

    # MFMA(k=0) at s0: all LR deps are at s0 or s1 (>= s0) → MT-1
    mfma0_deps = _dep_refs(s0.mfma)
    assert ('LR', 'A',  0, 1, -1) in mfma0_deps
    assert ('LR', 'B',  0, 1, -1) in mfma0_deps
    assert ('LR', 'SA', 0, 0, -1) in mfma0_deps
    assert ('LR', 'SB', 0, 1, -1) in mfma0_deps
    assert len(mfma0_deps) == 4

    # LR A @s0: depends on GR A @s0 (mt="n" vs GR mt="n+2" → MT-2)
    lr_a0 = _get_lr(s0, 'A')
    assert _dep_refs(lr_a0) == [('GR', 'A', 0, 0, -2)]

    # LR B @s0: deduped to last GR B (s1 > s0, same tensor)
    lr_b0 = _get_lr(s0, 'B')
    assert _dep_refs(lr_b0) == [('GR', 'B', 0, 1, -2)]

    # LR SA @s0: depends on GR SA @s1 (mt="n+1" vs GR mt="n+2" → MT-1)
    lr_sa0 = _get_lr(s0, 'SA')
    assert _dep_refs(lr_sa0) == [('GR', 'SA', 0, 1, -1)]

    # GR A @s0: collision on LR A (mt="n") @s0 — same iteration (MT 0)
    gr_a0 = [gr for gr in s0.grs if gr.tensor == 'A'][0]
    assert _dep_refs(gr_a0) == [('LR', 'A', 0, 0, 0)]

    # GR B @s0: collision on LR B (mt="n") @s0 — same iteration (MT 0)
    gr_b0 = [gr for gr in s0.grs if gr.tensor == 'B'][0]
    assert _dep_refs(gr_b0) == [('LR', 'B', 0, 0, 0)]

    # ── subIterK=1 ──

    # MFMA(k=1) at s1: LR A/B (mt="n") at s0 → same MT; LR SA/SB (mt="n+1") → MT-1
    mfma1_deps = _dep_refs(s1.mfma)
    assert ('LR', 'A',  0, 0, 0) in mfma1_deps
    assert ('LR', 'B',  0, 0, 0) in mfma1_deps
    assert ('LR', 'SA', 0, 0, -1) in mfma1_deps
    assert ('LR', 'SB', 0, 1, -1) in mfma1_deps
    assert len(mfma1_deps) == 4

    # LR A @s1: depends on GR A @s0 (mt="n+1" vs GR mt="n+2" → MT-1)
    lr_a1 = _get_lr(s1, 'A')
    assert _dep_refs(lr_a1) == [('GR', 'A', 0, 0, -1)]

    # LR B @s1: deduped to last GR B (s1 > s0, same tensor)
    lr_b1 = _get_lr(s1, 'B')
    assert _dep_refs(lr_b1) == [('GR', 'B', 0, 1, -1)]

    # LR SB @s1: depends on GR SB @s1 (mt="n+1" vs GR mt="n+2" → MT-1)
    lr_sb1 = _get_lr(s1, 'SB')
    assert _dep_refs(lr_sb1) == [('GR', 'SB', 0, 1, -1)]

    # GR B @s1: collision on LR B (mt="n") at s0 — same iteration (MT 0)
    gr_b1 = [gr for gr in s1.grs if gr.tensor == 'B'][0]
    assert _dep_refs(gr_b1) == [('LR', 'B', 0, 0, 0)]

    # GR SA @s1: LR SA (mt="n+1") @s0, prev iter handled data "n" → MT-1
    gr_sa1 = [gr for gr in s1.grs if gr.tensor == 'SA'][0]
    assert _dep_refs(gr_sa1) == [('LR', 'SA', 0, 0, -1)]

    # GR SB @s1: LR SB (mt="n+1") @s1, prev iter handled data "n" → MT-1
    gr_sb1 = [gr for gr in s1.grs if gr.tensor == 'SB'][0]
    assert _dep_refs(gr_sb1) == [('LR', 'SB', 0, 1, -1)]


def test_annotate_deps_2x2_partition_DU512():
    """Step 4: 256x256, DU512, FP4, 2x2 partition, 4 subIterKs.

    With partitions, deps are per-partition. Each partition only sees
    its own LRs and GRs. cross_mt tested on key deps.
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
        grSA=ReadGranularity(MFMATileSize(k=4, mn=8)),
        grSB=ReadGranularity(MFMATileSize(k=4, mn=8)),
        numPartitionsM=2,
        numPartitionsN=2,
    )
    sched = MFMATileScheduler(cfg)
    sched.annotate_deps()
    parts = sched._partitions
    print(sched.print_deps())

    assert len(parts) == 4
    assert len(parts[0]) == 4  # 4 subIterKs per partition

    # ── P0: has LR A, LR B, LR SA, LR SB and GR A only ──

    p0 = parts[0]

    # MFMA(k=0) @P0 A[0-3],B[0-3]: deps on LRs with matching tiles — all from P3
    mfma_p0_s0 = _dep_refs(p0[0].mfma)
    assert ('LR', 'A',  3, 3, -1) in mfma_p0_s0
    assert ('LR', 'B',  3, 3, -1) in mfma_p0_s0
    assert ('LR', 'SA', 3, 2, -1) in mfma_p0_s0
    assert ('LR', 'SB', 3, 3, -1) in mfma_p0_s0
    assert len(mfma_p0_s0) == 4

    # LR A @P0:s0 subIterK[1] [0-3]: GR A @P2:s1 loads subIterK[0,1] ids[2-3] — overlaps both dims
    lr_a_p0_s0 = _get_lr(p0[0], 'A')
    assert _dep_refs(lr_a_p0_s0) == [('GR', 'A', 2, 1, -2)]

    # LR B @P0:s0 subIterK[1] [0-3]: GR B @P2:s3 loads subIterK[0,1] ids[2-3] — overlaps both dims
    lr_b_p0_s0 = _get_lr(p0[0], 'B')
    assert _dep_refs(lr_b_p0_s0) == [('GR', 'B', 2, 3, -2)]

    # GR A(n+1) @P0:s0 k[0,1) ids[4,6): collision on LR A(n) @P1:s0 k[1,2) ids[4,8) — MT-1
    gr_a_p0_s0 = [gr for gr in p0[0].grs if gr.tensor == 'A'][0]
    assert _dep_refs(gr_a_p0_s0) == [('LR', 'A', 1, 0, -1)]

    # ── P3: has LR A, LR B, LR SA, LR SB and GR SA, GR SB, GR A, GR B ──

    p3 = parts[3]

    # MFMA(k=0) @P3 A[4-7],B[4-7]: deps on LRs with matching tiles — from P0 and P1
    mfma_p3_s0 = _dep_refs(p3[0].mfma)
    assert ('LR', 'A',  0, 3, -1) in mfma_p3_s0
    assert ('LR', 'B',  1, 3, -1) in mfma_p3_s0
    assert ('LR', 'SA', 0, 2, -1) in mfma_p3_s0
    assert ('LR', 'SB', 1, 2, -1) in mfma_p3_s0
    assert len(mfma_p3_s0) == 4

    # GR SA(n+2) @P3:s0 k[0,4) ids[0,8): collision on LR SA(n) @P1:s0 k[2,4) ids[4,8) — MT 0
    gr_sa_p3 = [gr for gr in p3[0].grs if gr.tensor == 'SA'][0]
    assert _dep_refs(gr_sa_p3) == [('LR', 'SA', 1, 0, 0)]

    # LR SA @P3:s2: depends on GR SA @P3:s0 (mt="n+1" vs GR mt="n+2" → MT-1)
    lr_sa_p3_s2 = _get_lr(p3[2], 'SA')
    assert _dep_refs(lr_sa_p3_s2) == [('GR', 'SA', 3, 0, -1)]

    # GR B(n+2) @P3:s3 k[2,4) ids[0,4): collision on LR B(n) @P0:s2 k[3,4) ids[0,4) — MT 0
    gr_b_p3_s3 = [gr for gr in p3[3].grs if gr.tensor == 'B'][0]
    assert _dep_refs(gr_b_p3_s3) == [('LR', 'B', 0, 2, 0)]

    # LR B @P3:s3 subIterK[0] [0-3]: GR B @P2:s3 loads subIterK[0,1] ids[2-3] — overlaps both dims
    lr_b_p3_s3 = _get_lr(p3[3], 'B')
    assert _dep_refs(lr_b_p3_s3) == [('GR', 'B', 2, 3, -1)]


# ── Step 4b: Remove cross-subIterK deps ─────────────────────

def _preop_kinds(placement):
    """Return list of (kind, wait_gr_counts_dict_or_None) for a placement's preOps."""
    result = []
    for op in placement.preOps:
        if op.wait_gr_counts:
            result.append((op.kind, {'A': op.wait_gr_counts.A, 'B': op.wait_gr_counts.B,
                                      'SA': op.wait_gr_counts.SA, 'SB': op.wait_gr_counts.SB}))
        else:
            result.append((op.kind, None))
    return result


def test_remove_cross_deps_1x1_partition_DU256():
    """Step 4b: 256x256, DU256, FP4, 1 partition, 2 subIterKs.

    After remove_cross_deps:
    - All cross-subIterK deps are removed from .deps
    - preOps are generated:
      MFMA → wait_lr
      LR → wait_gr(counts)
      GR → wait_lr_sync
    - Same-subIterK deps (MFMA(k=1) → LR A/B at s0) are preserved.
    """
    cfg = make_256x256_fp4()
    sched = MFMATileScheduler(cfg)
    sched.remove_cross_deps()
    parts = sched._partitions
    print(sched.print_remove_deps())

    s0 = parts[0][0]
    s1 = parts[0][1]

    # ── subIterK=0 ──

    # MFMA(k=0): all deps were cross-subIterK (MT-1) → removed, wait_lr preOp
    assert _preop_kinds(s0.mfma) == [('wait_lr', None)]
    assert len(s0.mfma.deps) == 0

    # LR A @s0: dep on GR A @s0 (MT-2) → cross, wait_gr with A=16
    lr_a0 = _get_lr(s0, 'A')
    assert _preop_kinds(lr_a0) == [('wait_gr', {'A': 16, 'B': 0, 'SA': 0, 'SB': 0})]
    assert len(lr_a0.deps) == 0

    # LR B @s0: dep on GR B @s1 (MT-2) → cross, wait_gr with B=16
    lr_b0 = _get_lr(s0, 'B')
    assert _preop_kinds(lr_b0) == [('wait_gr', {'A': 0, 'B': 16, 'SA': 0, 'SB': 0})]
    assert len(lr_b0.deps) == 0

    # LR SA @s0: dep on GR SA @s1 (MT-1) → cross, wait_gr with SA=1
    lr_sa0 = _get_lr(s0, 'SA')
    assert _preop_kinds(lr_sa0) == [('wait_gr', {'A': 0, 'B': 0, 'SA': 1, 'SB': 0})]
    assert len(lr_sa0.deps) == 0

    # GR A @s0: dep on LR A @s0 (MT 0, same slot) → same-subIterK, stays in deps
    gr_a0 = [gr for gr in s0.grs if gr.tensor == 'A'][0]
    assert _preop_kinds(gr_a0) == []
    assert len(gr_a0.deps) == 1

    # GR B @s0: dep on LR B @s0 (MT 0, same slot) → same-subIterK, stays in deps
    gr_b0 = [gr for gr in s0.grs if gr.tensor == 'B'][0]
    assert _preop_kinds(gr_b0) == []
    assert len(gr_b0.deps) == 1

    # ── subIterK=1 ──

    # MFMA(k=1): LR A/B @s0 have mt_offset=0 and are in same partition
    # but different subIterK slot (s0 != s1) → cross.
    # LR SA/SB are MT-1 → cross. All cross → wait_lr preOp, no remaining deps.
    assert _preop_kinds(s1.mfma) == [('wait_lr', None)]
    assert len(s1.mfma.deps) == 0

    # LR A @s1: dep on GR A @s0 (MT-1) → cross, wait_gr with A=8
    lr_a1 = _get_lr(s1, 'A')
    assert _preop_kinds(lr_a1) == [('wait_gr', {'A': 8, 'B': 0, 'SA': 0, 'SB': 0})]
    assert len(lr_a1.deps) == 0

    # LR B @s1: dep on GR B @s1 (MT-1) → cross, wait_gr with B=1
    lr_b1 = _get_lr(s1, 'B')
    assert _preop_kinds(lr_b1) == [('wait_gr', {'A': 0, 'B': 1, 'SA': 0, 'SB': 0})]
    assert len(lr_b1.deps) == 0

    # LR SB @s1: dep on GR SB @s1 (MT-1) → cross, wait_gr with SB=0
    lr_sb1 = _get_lr(s1, 'SB')
    assert _preop_kinds(lr_sb1) == [('wait_gr', {'A': 0, 'B': 0, 'SA': 0, 'SB': 0})]
    assert len(lr_sb1.deps) == 0

    # GR B @s1: dep on LR B @s0 (MT-2) → cross, wait_lr_sync
    gr_b1 = [gr for gr in s1.grs if gr.tensor == 'B'][0]
    assert _preop_kinds(gr_b1) == [('wait_lr_sync', None)]
    assert len(gr_b1.deps) == 0


def test_remove_cross_deps_2x2_partition_DU512():
    """Step 4b: 256x256, DU512, FP4, 2x2 partition, 4 subIterKs.

    Spot-checks key placements across partitions.
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
        grSA=ReadGranularity(MFMATileSize(k=4, mn=8)),
        grSB=ReadGranularity(MFMATileSize(k=4, mn=8)),
        numPartitionsM=2,
        numPartitionsN=2,
    )
    sched = MFMATileScheduler(cfg)
    sched.remove_cross_deps()
    parts = sched._partitions
    print(sched.print_remove_deps())

    # ── P0 spot checks ──

    p0 = parts[0]

    # MFMA(k=0) @P0: all deps cross → wait_lr, no remaining deps
    assert _preop_kinds(p0[0].mfma) == [('wait_lr', None)]
    assert len(p0[0].mfma.deps) == 0

    # LR A @P0:s0: dep on GR A @P2:s1 (MT-2) → wait_gr A=36
    lr_a_p0_s0 = _get_lr(p0[0], 'A')
    assert lr_a_p0_s0.preOps[0].wait_gr_counts.A == 36
    assert len(lr_a_p0_s0.deps) == 0

    # LR SA @P0:s0: dep on GR SA → wait_gr SA=2
    lr_sa_p0_s0 = _get_lr(p0[0], 'SA')
    assert lr_sa_p0_s0.preOps[0].wait_gr_counts.SA == 2
    assert len(lr_sa_p0_s0.deps) == 0

    # GR A @P0:s0: dep on LR A @P0:s3 (MT-1) → wait_lr_sync
    gr_a_p0_s0 = [gr for gr in p0[0].grs if gr.tensor == 'A'][0]
    assert _preop_kinds(gr_a_p0_s0) == [('wait_lr_sync', None)]

    # ── P2 spot checks ──

    p2 = parts[2]

    # GR A(n+2) @P2:s0: dep on LR A(n) @P0:s0 (MT 0, cross-partition) → wait_lr_sync
    gr_a_p2_s0 = [gr for gr in p2[0].grs if gr.tensor == 'A'][0]
    assert _preop_kinds(gr_a_p2_s0) == [('wait_lr_sync', None)]
    assert len(gr_a_p2_s0.deps) == 0

    # GR B @P2:s2: dep on LR B @P2:s2 (MT-2) → cross, wait_lr_sync
    gr_b_p2_s2 = [gr for gr in p2[2].grs if gr.tensor == 'B'][0]
    assert _preop_kinds(gr_b_p2_s2) == [('wait_lr_sync', None)]

    # ── P3 spot checks ──

    p3 = parts[3]

    # All MFMAs across P3 should have wait_lr
    for slot in p3:
        assert _preop_kinds(slot.mfma) == [('wait_lr', None)]

    # LR A @P3:s3: dep on GR A → wait_gr A=20
    lr_a_p3_s3 = _get_lr(p3[3], 'A')
    assert lr_a_p3_s3.preOps[0].wait_gr_counts.A == 20

    # LR SA @P3:s2: dep on GR SA @P3:s0 (MT-1) → wait_gr SA=1
    lr_sa_p3_s2 = _get_lr(p3[2], 'SA')
    assert lr_sa_p3_s2.preOps[0].wait_gr_counts.SA == 1


def _preop_inc_tensors(placement, kind):
    """Return list of tensor names for preOps of given kind on a placement."""
    return [op.tensor for op in placement.preOps if op.kind == kind]


def test_insert_gr_lr_inc_1x1_partition_DU256():
    """Step 5: 256x256, DU256, FP4, 1 partition, 2 subIterKs.

    After insert_gr_lr_inc, lr_inc/gr_inc preOps are inserted at MT transitions.
    Per-tensor tracking in global execution order (LRs then GRs per slot):

      s0: LR A(n), LR B(n), LR SA(n+1), GR A(n+2), GR B(n+2)
      s1: LR A(n+1), LR B(n+1), LR SB(n+1), GR B(n+2), GR SA(n+2), GR SB(n+2)

    MT transitions per tensor:
      A:  LR A s0 (n) → GR A s0 (n+2)  → LR A s1 (n+1)
      B:  LR B s0 (n) → GR B s0 (n+2)  → LR B s1 (n+1) → GR B s1 (n+2)
      SA: LR SA s0 (n+1) → GR SA s1 (n+2)
      SB: LR SB s1 (n+1) → GR SB s1 (n+2)
    """
    cfg = make_256x256_fp4()
    sched = MFMATileScheduler(cfg)
    sched.insert_gr_lr_inc()
    parts = sched._partitions
    print(sched.print_remove_deps())

    s0 = parts[0][0]
    s1 = parts[0][1]

    # ── subIterK=0 ──

    # MFMA: no inc preOps (MFMAs not checked)
    assert _preop_inc_tensors(s0.mfma, 'lr_inc') == []
    assert _preop_inc_tensors(s0.mfma, 'gr_inc') == []

    # LR A @s0: mt=n, first seen → no inc
    assert _preop_inc_tensors(_get_lr(s0, 'A'), 'lr_inc') == []

    # LR B @s0: mt=n, first seen → no inc
    assert _preop_inc_tensors(_get_lr(s0, 'B'), 'lr_inc') == []

    # LR SA @s0: mt=n+1, first seen → no inc
    assert _preop_inc_tensors(_get_lr(s0, 'SA'), 'lr_inc') == []

    # GR A @s0: mt=n+2, A was n → switch → gr_inc(A)
    gr_a0 = [gr for gr in s0.grs if gr.tensor == 'A'][0]
    assert _preop_inc_tensors(gr_a0, 'gr_inc') == ['A']

    # GR B @s0: mt=n+2, B was n → switch → gr_inc(B)
    gr_b0 = [gr for gr in s0.grs if gr.tensor == 'B'][0]
    assert _preop_inc_tensors(gr_b0, 'gr_inc') == ['B']

    # ── subIterK=1 ──

    # LR A @s1: mt=n+1, A was n+2 → switch → lr_inc(A)
    assert _preop_inc_tensors(_get_lr(s1, 'A'), 'lr_inc') == ['A']
    # Also has wait_gr from remove_cross_deps — lr_inc is appended after
    lr_a1 = _get_lr(s1, 'A')
    assert lr_a1.preOps[0].kind == 'wait_gr'
    assert lr_a1.preOps[1].kind == 'lr_inc'

    # LR B @s1: mt=n+1, B was n+2 → switch → lr_inc(B)
    assert _preop_inc_tensors(_get_lr(s1, 'B'), 'lr_inc') == ['B']

    # LR SB @s1: mt=n+1, first seen → no inc
    assert _preop_inc_tensors(_get_lr(s1, 'SB'), 'lr_inc') == []

    # GR B @s1: mt=n+2, B was n+1 → switch → gr_inc(B)
    gr_b1 = [gr for gr in s1.grs if gr.tensor == 'B'][0]
    assert _preop_inc_tensors(gr_b1, 'gr_inc') == ['B']

    # GR SA @s1: mt=n+2, SA was n+1 → switch → gr_inc(SA)
    gr_sa1 = [gr for gr in s1.grs if gr.tensor == 'SA'][0]
    assert _preop_inc_tensors(gr_sa1, 'gr_inc') == ['SA']

    # GR SB @s1: mt=n+2, SB was n+1 → switch → gr_inc(SB)
    gr_sb1 = [gr for gr in s1.grs if gr.tensor == 'SB'][0]
    assert _preop_inc_tensors(gr_sb1, 'gr_inc') == ['SB']


def test_compute_inflight_loads():
    """Unit test for _compute_inflight_loads.

    Uses the DU256 1x1 config where the GR layout is well-known:
      s0: GR A tiles[0-7] k[0-1]  → 8 atomic loads (grA mn=1, k=2)
      s0: GR B tiles[0-0] k[0-1]  → 1 atomic load
      s1: GR B tiles[1-7] k[0-1]  → 7 atomic loads
      s1: GR SA tiles[0-7] k[0-1] → 1 atomic load (grSA mn=8, k=2)
      s1: GR SB tiles[0-7] k[0-1] → 1 atomic load
    """
    cfg = make_256x256_fp4()
    sched = MFMATileScheduler(cfg)
    sched.annotate_deps()

    s0 = sched._partitions[0][0]
    s1 = sched._partitions[0][1]

    # LR A @s0 depends on GR A @s0 (mt_offset=-2)
    # Walk back from (P0, s0): wraps_needed=2
    # Walk: s1 (no GR A), s0 (GR A = dep but wraps=1<2, count += 8),
    #        s1 (no GR A), s0 (GR A = dep, wraps=2>=2) → return 8
    # But we also count the GR A at s0 on the first pass: wraps_completed=1 < 2 → count +=8
    # Second pass: wraps_completed=2 >= 2, it's the dep → return 8
    # So 1 full loop of GR A (8 loads) was encountered before reaching dep on second wrap
    lr_a0 = _get_lr(s0, 'A')
    dep_a0 = lr_a0.deps[0] if lr_a0.deps else lr_a0.preOps  # deps already moved to preOps
    # Use raw annotate_deps state — re-run to get fresh deps
    sched2 = MFMATileScheduler(cfg)
    sched2.annotate_deps()
    lr_a0_fresh = _get_lr(sched2._partitions[0][0], 'A')
    dep_a0 = lr_a0_fresh.deps[0]
    count_a = sched2._compute_inflight_loads(0, 0, 'A', dep_a0)
    # GR A is at s0 with 8 loads. mt_offset=-2 means 2 wraps needed.
    # Walk: s1→s0 (wrap1, count GR A=8 since it's dep but wraps<2), s1→s0 (wrap2, dep found) → 8
    assert count_a == 16  # Wait, let me verify with actual output

    # Instead of manual calculation, verify against the actual remove_cross_deps output
    sched3 = MFMATileScheduler(cfg)
    sched3.remove_cross_deps()
    # LR A @s0 had wait_gr A=16 in the dump
    lr_a0_final = _get_lr(sched3._partitions[0][0], 'A')
    assert lr_a0_final.preOps[0].wait_gr_counts.A == 16

    # LR B @s1 had wait_gr B=1
    lr_b1_final = _get_lr(sched3._partitions[0][1], 'B')
    assert lr_b1_final.preOps[0].wait_gr_counts.B == 1

    # LR SA @s0 had wait_gr SA=1
    lr_sa0_final = _get_lr(sched3._partitions[0][0], 'SA')
    assert lr_sa0_final.preOps[0].wait_gr_counts.SA == 1

    # LR A @s1 had wait_gr A=8
    lr_a1_final = _get_lr(sched3._partitions[0][1], 'A')
    assert lr_a1_final.preOps[0].wait_gr_counts.A == 8


# ── Step 5/6: Group and emit (commented out — will be reworked) ──

# def test_group():
#     """Validate Step 5: grouped output matches design doc."""
#     cfg = make_example_granularities_1()
#     sched = MFMATileScheduler(cfg)
#     grouped = sched.group()
#
#     output = sched.print_group()
#     print(output)
#
#     g0 = grouped[0]
#     op_kinds_0 = [op.kind for op in g0.ops]
#     assert op_kinds_0[0] == 'MFMA'
#     assert op_kinds_0[1] == 'LR'  # A
#     assert op_kinds_0[2] == 'LR'  # B
#     assert op_kinds_0[3] == 'LR'  # SA
#
#     assert any(dep.kind == 'wait_lr' for dep in g0.ops[0].before)
#     assert any(dep.kind == 'wait_gr' for dep in g0.ops[1].before)
#
#     g1 = grouped[1]
#     lr_tensors = [op.placement.tensor for op in g1.ops if op.kind == 'LR']
#     assert lr_tensors == ['A', 'B', 'SB']
#
#
# def test_emit():
#     """Validate Step 6: EmittedModule list with correct before-links."""
#     cfg = make_example_granularities_1()
#     sched = MFMATileScheduler(cfg)
#     all_emitted = sched.emit()
#
#     output = sched.print_emit()
#     print(output)
#
#     assert len(all_emitted) == 2
#
#     e0 = all_emitted[0]
#     mfmas = [e for e in e0 if e.opType == 'mfma']
#     assert len(mfmas) == 1
#     mfma = mfmas[0]
#     assert mfma.before is not None
#     assert e0[mfma.before].opType == 'wait_lr'
#
#     lrs = [e for e in e0 if e.opType == 'lr']
#     assert len(lrs) == 3
#     first_lr = lrs[0]
#     chain = _walk_before_chain(e0, first_lr.moduleId)
#     chain_types = [e0[mid].opType for mid in chain]
#     assert 'wait_gr' in chain_types
#     assert 'lr_inc' in chain_types
#
#     second_lr = lrs[1]
#     assert second_lr.before is not None
#     assert e0[second_lr.before].opType == 'lr'
#
#     grs = [e for e in e0 if e.opType == 'gr']
#     assert len(grs) == 2
#     first_gr = grs[0]
#     chain = _walk_before_chain(e0, first_gr.moduleId)
#     chain_types = [e0[mid].opType for mid in chain]
#     assert 'sync' in chain_types
#     assert 'wait_lr' in chain_types
#     assert grs[1].before is not None
#     assert e0[grs[1].before].opType == 'gr'
#
#     e1 = all_emitted[1]
#     mfmas1 = [e for e in e1 if e.opType == 'mfma']
#     assert len(mfmas1) == 1
#     lrs1 = [e for e in e1 if e.opType == 'lr']
#     assert len(lrs1) == 3
#     grs1 = [e for e in e1 if e.opType == 'gr']
#     assert len(grs1) == 3
#
#
# def _walk_before_chain(emitted, start_id):
#     """Walk the before-chain backwards, returning list of moduleIds."""
#     chain = []
#     cur = emitted[start_id].before
#     while cur is not None:
#         chain.append(cur)
#         cur = emitted[cur].before
#     return chain


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
        grSA=ReadGranularity(MFMATileSize(k=2, mn=8)),
        grSB=ReadGranularity(MFMATileSize(k=2, mn=8)),
    )

    # MT=64, MatrixInstM=16, MIWaveGroup=[2,2] → localMMATileGrid[0] = 64/16/2 = 2
    assert cfg.numMFMATilesM == 2
    assert cfg.numMFMATilesN == 2
    # subtileShape[1] = 2
    assert cfg.numSubIterK == 2
    assert cfg.hasScale

    # Should produce same schedule as manual config
    sched = MFMATileScheduler(cfg)
    partitions = sched.place_LRs()
    print(sched.print_lr())
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


# ── Step 6: group_lr_gr ────────────────────────────────────

def test_group_lr_gr_1x1_partition_DU256():
    """Step 6: 256x256, DU256, FP4, 1 partition, 2 subIterKs.

    After group_lr_gr:
      subIterK=0:
        LRs: A, B, SA → chain A←B←SA, merged preOps on A
        GRs: A, B → chain A←B, first GR dep→last LR (SA), merged preOps on A

      subIterK=1:
        LRs: A, B, SB → chain A←B←SB, merged preOps on A
        GRs: B, SA, SB → chain B←SA←SB, no deps (none originally), merged preOps on B
    """
    cfg = make_256x256_fp4()
    sched = MFMATileScheduler(cfg)
    sched.group_lr_gr()
    parts = sched._partitions
    print(sched.print_group_lr_gr())

    s0 = parts[0][0]
    s1 = parts[0][1]

    # ── subIterK=0: LR chain ──

    lr_a0 = _get_lr(s0, 'A')
    lr_b0 = _get_lr(s0, 'B')
    lr_sa0 = _get_lr(s0, 'SA')

    # First LR (A) has merged wait_gr
    assert len(lr_a0.preOps) == 1
    assert lr_a0.preOps[0].kind == 'wait_gr'
    assert lr_a0.preOps[0].wait_gr_counts.A == 16
    assert lr_a0.preOps[0].wait_gr_counts.B == 16
    assert lr_a0.preOps[0].wait_gr_counts.SA == 1
    assert lr_a0.deps == []

    # LR B chains to LR A
    assert lr_b0.preOps == []
    assert len(lr_b0.deps) == 1
    assert lr_b0.deps[0].ref is lr_a0
    assert lr_b0.deps[0].mt_offset == 0

    # LR SA chains to LR B
    assert lr_sa0.preOps == []
    assert len(lr_sa0.deps) == 1
    assert lr_sa0.deps[0].ref is lr_b0
    assert lr_sa0.deps[0].mt_offset == 0

    # ── subIterK=0: GR chain ──

    gr_a0 = [gr for gr in s0.grs if gr.tensor == 'A'][0]
    gr_b0 = [gr for gr in s0.grs if gr.tensor == 'B'][0]

    # First GR (A) has merged preOps: gr_inc(A), gr_inc(B)
    assert len(gr_a0.preOps) == 2
    assert gr_a0.preOps[0].kind == 'gr_inc' and gr_a0.preOps[0].tensor == 'A'
    assert gr_a0.preOps[1].kind == 'gr_inc' and gr_a0.preOps[1].tensor == 'B'
    # First GR dep points to last LR (SA)
    assert len(gr_a0.deps) == 1
    assert gr_a0.deps[0].ref is lr_sa0

    # GR B chains to GR A
    assert gr_b0.preOps == []
    assert len(gr_b0.deps) == 1
    assert gr_b0.deps[0].ref is gr_a0

    # ── subIterK=1: LR chain ──

    lr_a1 = _get_lr(s1, 'A')
    lr_b1 = _get_lr(s1, 'B')
    lr_sb1 = _get_lr(s1, 'SB')

    # First LR (A) has merged wait_gr + lr_inc ops
    assert lr_a1.preOps[0].kind == 'wait_gr'
    assert lr_a1.preOps[0].wait_gr_counts.A == 8
    assert lr_a1.preOps[0].wait_gr_counts.B == 1
    assert lr_a1.preOps[1].kind == 'lr_inc' and lr_a1.preOps[1].tensor == 'A'
    assert lr_a1.preOps[2].kind == 'lr_inc' and lr_a1.preOps[2].tensor == 'B'
    assert len(lr_a1.preOps) == 3

    # LR B chains to LR A
    assert lr_b1.preOps == []
    assert len(lr_b1.deps) == 1
    assert lr_b1.deps[0].ref is lr_a1

    # LR SB chains to LR B
    assert lr_sb1.preOps == []
    assert len(lr_sb1.deps) == 1
    assert lr_sb1.deps[0].ref is lr_b1

    # ── subIterK=1: GR chain ──

    gr_b1 = [gr for gr in s1.grs if gr.tensor == 'B'][0]
    gr_sa1 = [gr for gr in s1.grs if gr.tensor == 'SA'][0]
    gr_sb1 = [gr for gr in s1.grs if gr.tensor == 'SB'][0]

    # First GR (B) has merged preOps: wait_lr_sync (deduped) + gr_inc(B,SA,SB)
    assert gr_b1.preOps[0].kind == 'wait_lr_sync'
    assert gr_b1.preOps[1].kind == 'gr_inc' and gr_b1.preOps[1].tensor == 'B'
    assert gr_b1.preOps[2].kind == 'gr_inc' and gr_b1.preOps[2].tensor == 'SA'
    assert gr_b1.preOps[3].kind == 'gr_inc' and gr_b1.preOps[3].tensor == 'SB'
    assert len(gr_b1.preOps) == 4
    # No deps (none originally)
    assert gr_b1.deps == []

    # GR SA chains to GR B
    assert gr_sa1.preOps == []
    assert len(gr_sa1.deps) == 1
    assert gr_sa1.deps[0].ref is gr_b1

    # GR SB chains to GR SA
    assert gr_sb1.preOps == []
    assert len(gr_sb1.deps) == 1
    assert gr_sb1.deps[0].ref is gr_sa1


def test_emit_1x1_partition_DU256():
    """Step 7: emit pass — 256x256, DU256, FP4, 1 partition, 2 subIterKs.

    Visualize EmittedModule chains produced from group_lr_gr output.
    """
    cfg = make_256x256_fp4()
    sched = MFMATileScheduler(cfg)
    result = sched.emit()
    print(sched.print_emit())

    assert len(result) == 1       # 1 partition
    assert len(result[0]) == 2    # 2 subIterKs

    # ── subIterK=0 ──
    em0 = result[0][0]
    types0 = [(e.moduleId, e.opType, e.before) for e in em0]
    print("subIterK=0 modules:", types0)

    # Primary modules: mfma + 3 LRs (A, B, SA) + 2 GRs (A, B)
    primary0 = [e for e in em0 if e.opType in ('mfma', 'lr', 'gr')]
    assert len(primary0) == 6
    assert primary0[0].opType == 'mfma'

    # MFMA has a before-link (wait_lr)
    assert em0[0].before is not None
    wait_lr_0 = em0[em0[0].before]
    assert wait_lr_0.opType == 'wait_lr'

    # wait_gr is standalone (no incoming before)
    wait_grs_0 = [e for e in em0 if e.opType == 'wait_gr']
    assert len(wait_grs_0) == 1
    assert wait_grs_0[0].before is None

    # ── subIterK=1 ──
    em1 = result[0][1]
    types1 = [(e.moduleId, e.opType, e.before) for e in em1]
    print("subIterK=1 modules:", types1)

    primary1 = [e for e in em1 if e.opType in ('mfma', 'lr', 'gr')]
    assert len(primary1) == 7   # mfma + 3 LRs (A, B, SB) + 3 GRs (B, SA, SB)
    assert primary1[0].opType == 'mfma'

    # wait_gr standalone
    wait_grs_1 = [e for e in em1 if e.opType == 'wait_gr']
    assert len(wait_grs_1) == 1
    assert wait_grs_1[0].before is None

    # No self-loops
    for em_list in [em0, em1]:
        for e in em_list:
            if e.before is not None:
                assert e.before != e.moduleId, \
                    f"module {e.moduleId} has self-loop"


# ── Integration test: populate_instructions ───────────────

def test_populate_instructions_256x256_fp4():
    """Integration test: populate_instructions with real writer/kernel/VGPR state.

    Sets up the full kernel infrastructure (TileInfo, VGPR allocation, writer)
    and runs MFMATileScheduler through emit → populate_instructions → instructionSchedule.
    Uses the scheduler's own allocVgprTiles for self-contained VGPR management.
    """
    from types import SimpleNamespace
    from rocisa import rocIsa
    from rocisa.register import RegisterPool
    from rocisa.enum import RegisterType
    from Tensile.Components.SubtileBasedScheduler import SubtileBasedScheduler

    # Initialize rocIsa
    ri = rocIsa.getInstance()
    if not ri.isInit():
        import shutil
        asmpath = shutil.which('amdclang++') or '/usr/bin/amdclang++'
        ri.init((9, 5, 0), asmpath)
    ri.setKernel((9, 5, 0), 64)

    # Create kernel + TileInfo
    kernel = create_kernel(256, 256, fp4=True)
    tiA = TileInfo('A', kernel)
    tiB = TileInfo('B', kernel)
    scaleTiA = TileInfo('MXSA', kernel)
    scaleTiB = TileInfo('MXSB', kernel)

    # Create writer with register pools
    writer = SimpleNamespace()
    writer.vgprPool = RegisterPool(0, RegisterType.Vgpr, False)
    writer.agprPool = RegisterPool(0, RegisterType.Accvgpr, False)
    writer.sgprPool = RegisterPool(0, RegisterType.Sgpr, False)
    writer.states = SimpleNamespace(
        regCaps={"MaxSgpr": 106, "MaxVgpr": 256, "PhysicalMaxVgpr": 512},
    )
    dTileInfo = TileInfo('D', kernel)
    dTileInfo.allocVgprTileRegisters(writer, kernel)
    writer.states.d = SimpleNamespace(tileInfo=dTileInfo)
    writer.states.a = SimpleNamespace(tileInfo=tiA)
    writer.states.b = SimpleNamespace(tileInfo=tiB)
    writer.states.mxsa = SimpleNamespace(tileInfo=scaleTiA)
    writer.states.mxsb = SimpleNamespace(tileInfo=scaleTiB)
    tiA.allocOffsetRegisters(writer, kernel)
    tiB.allocOffsetRegisters(writer, kernel)
    scaleTiA.allocOffsetRegisters(writer, kernel)
    scaleTiB.allocOffsetRegisters(writer, kernel)

    # Build MFMATileScheduler logical schedule
    cfg = make_256x256_fp4()
    sched = MFMATileScheduler(cfg)
    sched.emit()

    # Allocate VGPR tiles using scheduler's own method
    sched.allocVgprTiles(writer, tiA, tiB,
                         scaleTileInfoA=scaleTiA, scaleTileInfoB=scaleTiB)

    try:
        # Populate instructions
        sched.populate_instructions(
            writer, kernel,
            tileInfoA=tiA, tileInfoB=tiB,
            dtileInfo=dTileInfo,
            scaleTileInfoA=scaleTiA, scaleTileInfoB=scaleTiB,
        )

        # Verify instructions were populated
        for pi, partition_emitted in enumerate(sched._emitted):
            for k, emitted in enumerate(partition_emitted):
                for em in emitted:
                    assert len(em.instructions) > 0, \
                        f"P{pi} subIterK={k} [{em.moduleId}] {em.opType}: no instructions"

        # Verify: MFMA and LR at same subIterK use different vgprTileIds
        for pi, slots in enumerate(sched._partitions):
            for slot in slots:
                if slot.mfma and slot.lrs:
                    for lr in slot.lrs:
                        if lr.vgpr_tile_map and lr.tensor in ('A', 'B'):
                            mfma_map_list = getattr(slot.mfma, f'vgpr_tile_map_{lr.tensor}')
                            if mfma_map_list:
                                mfma_vids = set(mfma_map_list[0].values())
                                lr_vids = set(lr.vgpr_tile_map[0].values())
                                assert mfma_vids.isdisjoint(lr_vids), \
                                    f"P{pi} subIterK={slot.subIterK}: MFMA and LR {lr.tensor} " \
                                    f"share vgprTileIds"

        # Call instructionSchedule on each subIterK and verify no crash
        for pi, partition_emitted in enumerate(sched._emitted):
            for k, emitted in enumerate(partition_emitted):
                scheduled = SubtileBasedScheduler.instructionSchedule(emitted)
                insts = list(scheduled.flatitems())
                assert len(insts) > 0, \
                    f"P{pi} subIterK={k}: instructionSchedule returned empty"

        # Print for visualization
        for pi, partition_emitted in enumerate(sched._emitted):
            print(f"Partition {pi}:")
            for k, emitted in enumerate(partition_emitted):
                scheduled = SubtileBasedScheduler.instructionSchedule(emitted)
                insts = list(scheduled.flatitems())
                print(f"  subIterK={k}: {len(insts)} instructions")
                for inst in insts:
                    print(f"    {str(inst)}")

    finally:
        sched.deallocVgprTiles(writer)


# ── Standalone mode ─────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import io

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
        grSA=ReadGranularity(MFMATileSize(k=2, mn=8)),
        grSB=ReadGranularity(MFMATileSize(k=2, mn=8)),
    )

    print(f"Config: numMFMATilesM={cfg.numMFMATilesM}, "
          f"numMFMATilesN={cfg.numMFMATilesN}, "
          f"numSubIterK={cfg.numSubIterK}, "
          f"hasScale={cfg.hasScale}")
    print()

    sched = MFMATileScheduler(cfg)

    steps = [
        ("Step 1: Place LRs",          lambda: (sched.place_LRs(), sched.print_lr())),
        ("Step 2: Assign VGPR tiles",   lambda: (sched.assign_vgpr_tiles(), sched.print_vgpr())),
        ("Step 3: Place GRs",           lambda: (sched.place_GRs(), sched.print_gr())),
        ("Step 4: Annotate deps",       lambda: (sched.annotate_deps(), sched.print_deps())),
        ("Step 5: Remove cross deps",  lambda: (sched.remove_cross_deps(), sched.print_remove_deps())),
        ("Step 6: Insert gr/lr inc",   lambda: (sched.insert_gr_lr_inc(), sched.print_group_lr_gr())),
        ("Step 7: Group LR/GR",        lambda: (sched.group_lr_gr(), sched.print_group_lr_gr())),
        ("Step 8: Emit",               lambda: (sched.emit(), sched.print_emit())),
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

    # ── populate_instructions: fill GPU instructions ──
    from types import SimpleNamespace
    from rocisa import rocIsa
    from rocisa.register import RegisterPool
    from rocisa.enum import RegisterType
    from Tensile.Components.SubtileBasedScheduler import SubtileBasedScheduler

    ri = rocIsa.getInstance()
    if not ri.isInit():
        import shutil
        asmpath = shutil.which('amdclang++') or '/usr/bin/amdclang++'
        ri.init((9, 5, 0), asmpath)
    ri.setKernel((9, 5, 0), 64)

    writer = SimpleNamespace()
    writer.vgprPool = RegisterPool(0, RegisterType.Vgpr, False)
    writer.agprPool = RegisterPool(0, RegisterType.Accvgpr, False)
    writer.sgprPool = RegisterPool(0, RegisterType.Sgpr, False)
    writer.states = SimpleNamespace(
        regCaps={"MaxSgpr": 106, "MaxVgpr": 256, "PhysicalMaxVgpr": 512},
    )
    dTileInfo = TileInfo('D', kernel)
    dTileInfo.allocVgprTileRegisters(writer, kernel)
    writer.states.d = SimpleNamespace(tileInfo=dTileInfo)
    writer.states.a = SimpleNamespace(tileInfo=tiA)
    writer.states.b = SimpleNamespace(tileInfo=tiB)
    writer.states.mxsa = SimpleNamespace(tileInfo=scaleTiA)
    writer.states.mxsb = SimpleNamespace(tileInfo=scaleTiB)
    tiA.allocOffsetRegisters(writer, kernel)
    tiB.allocOffsetRegisters(writer, kernel)
    scaleTiA.allocOffsetRegisters(writer, kernel)
    scaleTiB.allocOffsetRegisters(writer, kernel)

    sched.allocVgprTiles(writer, tiA, tiB,
                         scaleTileInfoA=scaleTiA, scaleTileInfoB=scaleTiB)

    sched.populate_instructions(
        writer, kernel,
        tileInfoA=tiA, tileInfoB=tiB,
        dtileInfo=dTileInfo,
        scaleTileInfoA=scaleTiA, scaleTileInfoB=scaleTiB,
    )

    # ── instructionSchedule: extract paths and display instructions ──
    all_emitted = sched._emitted
    buf = io.StringIO()
    buf.write("MAINLOOP (instructionSchedule):\n")
    for pi, partition_emitted in enumerate(all_emitted):
        buf.write(f"  Partition {pi}:\n")
        for k, emitted in enumerate(partition_emitted):
            buf.write(f"    subIterK={k}:\n")

            scheduled = SubtileBasedScheduler.instructionSchedule(emitted)

            # Display paths that instructionSchedule extracts
            mfmaIdx, paths, preMfmaPaths = SubtileBasedScheduler._extractPathsFromBeforeDeps(emitted)
            mfma_em = emitted[mfmaIdx]
            buf.write(f"      MFMA [{mfma_em.moduleId}]: {mfma_em.label}\n")
            for pi2, path in enumerate(preMfmaPaths):
                labels = [f"[{emitted[mid].moduleId}] {emitted[mid].opType}" for mid in path]
                buf.write(f"      PreMFMA {pi2}: {' -> '.join(labels)}\n")
            for pi2, path in enumerate(paths):
                labels = [f"[{emitted[mid].moduleId}] {emitted[mid].opType}" for mid in path]
                buf.write(f"      Path {pi2}: {' -> '.join(labels)}\n")

            # Display flattened instructions from instructionSchedule
            insts = list(scheduled.flatitems())
            if insts:
                buf.write(f"      Instructions:\n")
                for inst in insts:
                    buf.write(f"        {str(inst).rstrip()}\n")
            else:
                buf.write(f"      Instructions: (empty — logical level, no GPU instructions)\n")

    print(f"{'=' * 60}")
    print(f"  Step 9: instructionSchedule")
    print(f"{'=' * 60}")
    print(buf.getvalue())

    sched.deallocVgprTiles(writer)
