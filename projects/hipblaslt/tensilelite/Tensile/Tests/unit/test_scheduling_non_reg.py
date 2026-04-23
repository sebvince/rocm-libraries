"""Non-regression tests for MFMATileScheduler — exact output checks.

These tests capture the expected scheduling output at specific pipeline steps
to detect unintended regressions. They are intended to be temporary and will
be removed once the refactoring stabilizes.
"""

from Tensile.Components.SubtileBasedKernel import TileInfo
from Tensile.Components.MFMATileScheduler import (
    MFMATileScheduler,
    MFMATileSize,
    ReadGranularity,
    SchedulerConfig,
)


def create_kernel(MT0=256, MT1=256, fp4=False, depthU=None):
    from unittest.mock import MagicMock

    mxblock = 32 if fp4 else 0
    bpe = 0.5 if fp4 else 2
    matrixInstK = 128 if fp4 else 32
    if depthU is None:
        depthU = 256 if fp4 else 64
    dtype = MagicMock()
    dtype.numBytes.return_value = bpe
    problemType = {
        "DataTypeA": dtype,
        "DataTypeB": dtype,
        "ComputeDataType": MagicMock(**{"numBytes.return_value": 4}),
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


def make_256x256_bf16():
    kernel = create_kernel(256, 256, fp4=False, depthU=64)
    tiA = TileInfo('A', kernel)
    tiB = TileInfo('B', kernel)
    return SchedulerConfig.from_tile_info(
        tiA, tiB,
        lrA=ReadGranularity(MFMATileSize(k=1, mn=1)),
        lrB=ReadGranularity(MFMATileSize(k=1, mn=1)),
        grA=ReadGranularity(MFMATileSize(k=2, mn=1)),
        grB=ReadGranularity(MFMATileSize(k=2, mn=1)),
        numPartitionsM=1,
        numPartitionsN=1,
    )


EXPECTED_EMIT_DEP_ORDER_256x256_BF16_1x1 = """\
MAINLOOP (dependency paths):
  Partition 0:
    subIterK=0:
      MFMA: [ 0] MFMAs (MT n, subIterK 0  ) A : [0-7] , B : [0-7] <- [4]
      preMFMA path 0:
        [ 4] wait_lr    wait_lr
      path 0:
        [ 1] lr         LR A  (MT n, subIterK [1]) [0-7]
        [ 2] lr         LR B  (MT n, subIterK [1]) [0-7]
        [ 5] wait_lr    wait_lr
        [ 6] sync       sync
        [ 7] gr_inc     gr_inc(A)
        [ 3] gr         GR A (MT n+2, subIterK [0,1]) ids [0-7]
    subIterK=1:
      MFMA: [ 0] MFMAs (MT n, subIterK 1  ) A : [0-7] , B : [0-7] <- [4]
      preMFMA path 0:
        [ 4] wait_lr    wait_lr
      path 0:
        [ 5] wait_gr    wait_gr_sync(A=8)
        [ 6] sync       sync
        [ 7] lr_inc     lr_inc(A)
        [ 8] lr_inc     lr_inc(B)
        [ 1] lr         LR A  (MT n+1, subIterK [0]) [0-7]
        [ 2] lr         LR B  (MT n+1, subIterK [0]) [0-7]
      path 1:
        [ 9] gr_inc     gr_inc(B)
        [ 3] gr         GR B (MT n+2, subIterK [0,1]) ids [0-7]
"""


def make_384x256_bf16():
    kernel = create_kernel(384, 256, fp4=False, depthU=64)
    tiA = TileInfo('A', kernel)
    tiB = TileInfo('B', kernel)
    return SchedulerConfig.from_tile_info(
        tiA, tiB,
        lrA=ReadGranularity(MFMATileSize(k=1, mn=1)),
        lrB=ReadGranularity(MFMATileSize(k=1, mn=1)),
        grA=ReadGranularity(MFMATileSize(k=2, mn=1)),
        grB=ReadGranularity(MFMATileSize(k=2, mn=1)),
        numPartitionsM=2,
        numPartitionsN=1,
    )


EXPECTED_EMIT_DEP_ORDER_384x256_BF16_2x1 = """\
MAINLOOP (dependency paths):
  Partition 0:
    subIterK=0:
      MFMA: [ 0] MFMAs (MT n, subIterK 0  ) A : [0-5] , B : [0-7] <- [4]
      preMFMA path 0:
        [ 4] wait_lr    wait_lr
      path 0:
        [ 1] lr         LR A  (MT n, subIterK [1]) [0-5]
        [ 2] lr         LR B  (MT n, subIterK [1]) [0-7]
      path 1:
        [ 3] gr         GR A (MT n+1, subIterK [0,1]) ids [6-10]
    subIterK=1:
      MFMA: [ 0] MFMAs (MT n, subIterK 1  ) A : [0-5] , B : [0-7] <- [4]
      preMFMA path 0:
        [ 4] wait_lr    wait_lr
      path 0:
        [ 5] wait_gr    wait_gr_sync(A=7,B=8)
        [ 6] sync       sync
        [ 1] lr         LR A  (MT n, subIterK [0]) [6-11]
      path 1:
        [ 2] gr         GR A (MT n+1, subIterK [0,1]) ids [11-11]
        [ 7] sync       sync
        [ 8] gr_inc     gr_inc(A)
        [ 3] gr         GR A (MT n+2, subIterK [0,1]) ids [0-3]
  Partition 1:
    subIterK=0:
      MFMA: [ 0] MFMAs (MT n, subIterK 0  ) A : [6-11] , B : [0-7] <- [4]
      preMFMA path 0:
        [ 4] wait_lr    wait_lr
      path 0:
        [ 1] lr         LR A  (MT n, subIterK [1]) [6-11]
      path 1:
        [ 2] gr         GR A (MT n+2, subIterK [0,1]) ids [4-5]
        [ 5] sync       sync
        [ 6] gr_inc     gr_inc(B)
        [ 3] gr         GR B (MT n+2, subIterK [0,1]) ids [0-2]
    subIterK=1:
      MFMA: [ 0] MFMAs (MT n, subIterK 1  ) A : [6-11] , B : [0-7] <- [4]
      preMFMA path 0:
        [ 4] wait_lr    wait_lr
      path 0:
        [ 5] wait_gr    wait_gr_sync(A=12,B=3)
        [ 6] sync       sync
        [ 7] lr_inc     lr_inc(A)
        [ 8] lr_inc     lr_inc(B)
        [ 1] lr         LR A  (MT n+1, subIterK [0]) [0-5]
        [ 2] lr         LR B  (MT n+1, subIterK [0]) [0-7]
      path 1:
        [ 3] gr         GR B (MT n+2, subIterK [0,1]) ids [3-7]
"""


def test_384x256_bf16_partition_2x1():
    """Exact check of step 11b (emit dependency order) for 384x256 BF16, 2x1 partition."""
    cfg = make_384x256_bf16()
    sched = MFMATileScheduler(cfg)
    sched.emit()
    actual = sched.print_emit_dep_order()
    assert actual == EXPECTED_EMIT_DEP_ORDER_384x256_BF16_2x1, (
        f"Step 11b output mismatch.\n"
        f"--- Expected ---\n{EXPECTED_EMIT_DEP_ORDER_384x256_BF16_2x1}\n"
        f"--- Actual ---\n{actual}"
    )


def test_256x256_bf16_partition_1x1():
    """Exact check of step 11b (emit dependency order) for 256x256 BF16, 1x1 partition."""
    cfg = make_256x256_bf16()
    sched = MFMATileScheduler(cfg)
    sched.emit()
    actual = sched.print_emit_dep_order()
    assert actual == EXPECTED_EMIT_DEP_ORDER_256x256_BF16_1x1, (
        f"Step 11b output mismatch.\n"
        f"--- Expected ---\n{EXPECTED_EMIT_DEP_ORDER_256x256_BF16_1x1}\n"
        f"--- Actual ---\n{actual}"
    )
