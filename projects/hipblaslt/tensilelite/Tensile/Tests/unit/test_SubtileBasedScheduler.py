from types import SimpleNamespace
from unittest.mock import MagicMock
from Tensile.Components.SubtileBasedKernel import TileInfo
from Tensile.Components.SubtileBasedScheduler import SubtileBasedScheduler, SchedulerConfig, PrefetchMode, VGPRTileReUseStrategy, SubgroupOrdering
from rocisa import rocIsa
from rocisa.register import RegisterPool
from rocisa.enum import RegisterType

# Initialize rocIsa for gfx950
ri = rocIsa.getInstance()
if not ri.isInit():
    import shutil
    asmpath = shutil.which('amdclang++') or '/usr/bin/amdclang++'
    ri.init((9, 5, 0), asmpath)
ri.setKernel((9, 5, 0), 64)

def _mock_dtype(num_bytes=2):
    mock = MagicMock()
    mock.numBytes.return_value = num_bytes
    return mock

def create_kernel():
    dtype = _mock_dtype(2)
    problemType = {
        "DataTypeA": dtype,
        "DataTypeB": dtype,
        "ComputeDataType": _mock_dtype(4),
    }
    return {
        "DepthU": 64,
        "MacroTileA": 64,
        "MacroTileB": 64,
        "MacroTile0": 64,
        "MacroTile1": 64,
        "MatrixInstM": 16,
        "MatrixInstN": 16,
        "MatrixInstK": 32,
        "MIWaveGroup": [2, 2],
        "WavefrontSize": 64,
        "SourceSwap": False,
        "MIArchVgpr": False,
        "ProblemType": problemType,
    }

def create_mock_writer(kernel):
    writer = SimpleNamespace()
    writer.vgprPool = RegisterPool(0, RegisterType.Vgpr, False)
    writer.agprPool = RegisterPool(0, RegisterType.Accvgpr, False)
    writer.sgprPool = RegisterPool(0, RegisterType.Sgpr, False)
    writer.states = SimpleNamespace(
        regCaps={"MaxSgpr": 106, "MaxVgpr": 256, "PhysicalMaxVgpr": 512},
    )
    # Allocate D tileInfo (same as KernelWriter line 3843)
    dTileInfo = TileInfo('D', kernel)
    dTileInfo.allocVgprTileRegisters(writer, kernel)
    writer.states.d = SimpleNamespace(tileInfo=dTileInfo)
    return writer

def create_writer_with_tiles(kernel, tiA, tiB):
    writer = create_mock_writer(kernel)
    writer.states.a = SimpleNamespace(tileInfo=tiA)
    writer.states.b = SimpleNamespace(tileInfo=tiB)
    tiA.allocOffsetRegisters(writer, kernel)
    tiB.allocOffsetRegisters(writer, kernel)
    return writer


def test_half_prefetch_across_subgroup_column_major():
    kernel = create_kernel()
    tiA = TileInfo('A', kernel)
    tiB = TileInfo('B', kernel)
    lsgA = tiA.localSubtileGrid[0]
    lsgB = tiB.localSubtileGrid[0]

    cfg = SchedulerConfig(lsgA, lsgB, PrefetchMode.HALF_PREFETCH,
                          VGPRTileReUseStrategy.ACROSS_SUBGROUP,
                          SubgroupOrdering.COLUMN_MAJOR)
    s = SubtileBasedScheduler(tiA, tiB, cfg)

    assert len(s.preloopSteps)  > 0
    assert len(s.mainloopSteps) > 0
    assert len(s.ngllSteps)     > 0
    assert len(s.nllSteps)      > 0

    writer = create_writer_with_tiles(kernel, tiA, tiB)
    s.generateCode(writer, kernel)


if __name__ == "__main__":
    kernel = create_kernel()
    tiA = TileInfo('A', kernel)
    tiB = TileInfo('B', kernel)
    lsgA = tiA.localSubtileGrid[0]
    lsgB = tiB.localSubtileGrid[0]

    configs = [
        # (f"lsg {lsgA}x{lsgB}, group {lsgA}x{lsgB}, HALF_PREFETCH, ACROSS_SUBGROUP, COLUMN_MAJOR",
        #     SchedulerConfig(lsgA, lsgB, PrefetchMode.HALF_PREFETCH, VGPRTileReUseStrategy.ACROSS_SUBGROUP, SubgroupOrdering.COLUMN_MAJOR)),
        (f"lsg {lsgA}x{lsgB}, group {lsgA//2}x{lsgB//2}, HALF_PREFETCH, ACROSS_SUBGROUP, COLUMN_MAJOR",
            SchedulerConfig(lsgA//2, lsgB//2, PrefetchMode.HALF_PREFETCH, VGPRTileReUseStrategy.ACROSS_SUBGROUP, SubgroupOrdering.COLUMN_MAJOR)),
    ]

    for name, cfg in configs:
        print(f"=== {name} ===")
        s = SubtileBasedScheduler(tiA, tiB, cfg)
        s.printSchedule()
        writer = create_writer_with_tiles(kernel, tiA, tiB)
        # s.generateCode(writer, kernel)
        # print()
