import dataclasses
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Set, Optional, Union
from Tensile.Components.SubtileBasedKernel import TileInfo
from Tensile.Components.SubtileBasedKernel import emitMfmaInstruction
from Tensile.Components.SubtileBasedKernel import emitSingleDsRead
from Tensile.Components.SubtileBasedKernel import emitSingleBufferLoad
from Tensile.Components.SubtileBasedKernel import globalReadPtrUpdates, globalReadLDSBufferSwap, localReadLDSBufferSwap
from rocisa.code import Module, Label
from rocisa.instruction import SWaitCnt, SBarrier, SCmpEQU32, SCmpLeU32, SCBranchSCC1, MFMAInstruction, \
    GlobalReadInstruction, LocalReadInstruction
from rocisa.container import sgpr

class PrefetchMode(Enum):
    NO = auto()
    HALF_PREFETCH = auto()
    FULL_PREFETCH = auto()


class VGPRTileReUseStrategy(Enum):
    NONE = auto()
    ACROSS_SUBGROUP = auto()
    WITHIN_SUBGROUP = auto()


class SubgroupOrdering(Enum):
    COLUMN_MAJOR = auto()
    SNAKE_COLUMN_MAJOR = auto()


@dataclass
class SchedulerConfig:
    partitionSizeA: int
    partitionSizeB: int
    prefetchMode: PrefetchMode
    reuseStrategy: VGPRTileReUseStrategy
    ordering: SubgroupOrdering = SubgroupOrdering.COLUMN_MAJOR


@dataclass
class Partition:
    """A rectangle (sizeA x sizeB) of subtiles processed together."""
    partitionId: int
    sizeA: int
    sizeB: int
    subtiles: List[Tuple[int, int]] = field(default_factory=list)

    @property
    def tileAIndices(self) -> List[int]:
        return sorted(set(t[0] for t in self.subtiles))

    @property
    def tileBIndices(self) -> List[int]:
        return sorted(set(t[1] for t in self.subtiles))



# Key type for allocator: (subtileIdx, subIterK)
AllocKey = Tuple[int, int]


class VGPRTileAllocator:
    """Maps (subtileIdx, subIterK) to shared integer VGPR tile IDs, with free-list reuse."""

    def __init__(self):
        self._nextId: int = 0
        self._peak: int = 0
        self._freeList: List[int] = []
        self._allocMapA: Dict[AllocKey, int] = {}
        self._allocMapB: Dict[AllocKey, int] = {}

    def _allocMap(self, tc: str) -> Dict[AllocKey, int]:
        return self._allocMapA if tc == 'A' else self._allocMapB

    def _updatePeak(self):
        current = len(self._allocMapA) + len(self._allocMapB)
        self._peak = max(self._peak, current)

    def allocate(self, tc: str, subtileIdx: int, subIterK: int) -> int:
        key = (subtileIdx, subIterK)
        if self._freeList:
            vid = self._freeList.pop(0)
        else:
            vid = self._nextId
            self._nextId += 1
        self._allocMap(tc)[key] = vid
        self._updatePeak()
        return vid

    def release(self, tc: str, subtileIdx: int, subIterK: int) -> None:
        key = (subtileIdx, subIterK)
        vid = self._allocMap(tc).pop(key)
        self._freeList.append(vid)

    def isAllocated(self, tc: str, subtileIdx: int, subIterK: int) -> bool:
        return (subtileIdx, subIterK) in self._allocMap(tc)

    def getVGPRTileId(self, tc: str, subtileIdx: int, subIterK: int) -> int:
        return self._allocMap(tc)[(subtileIdx, subIterK)]

    def releaseAllForTile(self, tc: str, subtileIdx: int) -> None:
        """Release all subIterK allocations for a given subtile index."""
        allocMap = self._allocMap(tc)
        keys = [k for k in allocMap if k[0] == subtileIdx]
        for k in keys:
            vid = allocMap.pop(k)
            self._freeList.append(vid)

    @property
    def totalVGPRTiles(self) -> int:
        return self._peak



@dataclass
class MFMAOp:
    mtIteration: str  # e.g. "n"
    subIterK: int
    subtiles: List[Tuple[int, int]]
    vgprTileMapA: Dict[int, int]
    vgprTileMapB: Dict[int, int]


@dataclass
class GROp:
    mtIteration: str  # e.g. "n+1", "n+2", "0", "1"
    subtileA: List[int]
    subtileB: List[int]
    lastForMT: bool = False  # True = last partition's GR for this MT → emit ptrUpdate+swap


@dataclass
class WaitGROp:
    mtIteration: str  # e.g. "n", "n+1"
    subtileA: List[int]
    subtileB: List[int]
    inflightLoadsA: Optional[int] = None
    inflightLoadsB: Optional[int] = None


@dataclass
class WaitLROp:
    pass


@dataclass
class SyncOp:
    comment: str = ""


@dataclass
class LROp:
    mtIteration: str  # e.g. "n", "n+1", "0"
    subIterK: int
    lrLoadA: Dict[int, int]
    lrLoadB: Dict[int, int]


@dataclass
class SkipOp:
    compare: str  # "EQ", "LE"
    value: int    # LoopCounter compared against this
    target: str   # "NLL", "NGLL"


@dataclass
class GR_INCOp:
    """Global Read increment: pointer updates + LDS buffer swap for GR."""
    pass


@dataclass
class LR_INCOp:
    """Local Read increment: LDS buffer swap for LR."""
    pass


ScheduleOp = Union[MFMAOp, GROp, WaitGROp, WaitLROp, SyncOp, LROp, SkipOp, GR_INCOp, LR_INCOp]


@dataclass
class PartitionGR:
    """Describes the GR (Global Read) issued by a partition during the mainloop."""
    mtIteration: str       # "n+1" or "n+2"
    targetPartitionId: int # which partition's subtiles we're loading for
    subtileA: Set[int]     # actual subtiles to load (after dedup)
    subtileB: Set[int]


@dataclass
class SubIterKSchedule:
    """Ops for one subIterK iteration within a partition."""
    subIterK: int
    ops: List[ScheduleOp] = field(default_factory=list)
    conflict: Set[int] = field(default_factory=set)


@dataclass
class PartitionSchedule:
    """Schedule for one subtile partition — contains all subIterK iterations."""
    partitionId: int
    subIterKSteps: List[SubIterKSchedule] = field(default_factory=list)


class SubtileBasedScheduler:
    def __init__(self, tileInfoA, tileInfoB, config: SchedulerConfig):
        self.tileInfoA = tileInfoA
        self.tileInfoB = tileInfoB
        self.config = config

        self.MTA = tileInfoA.localSubtileGrid[0]
        self.MTB = tileInfoB.localSubtileGrid[0]

        assert self.MTA % config.partitionSizeA == 0, \
            f"MTA ({self.MTA}) must be divisible by partitionSizeA ({config.partitionSizeA})"
        assert self.MTB % config.partitionSizeB == 0, \
            f"MTB ({self.MTB}) must be divisible by partitionSizeB ({config.partitionSizeB})"

        self.numPartitionsA = self.MTA // config.partitionSizeA
        self.numPartitionsB = self.MTB // config.partitionSizeB

        self.numSubIterK = tileInfoA.subtileShape[1]
        assert self.numSubIterK == tileInfoB.subtileShape[1], \
            "A and B must have same subtileShape[1]"
        assert tileInfoA.localSubtileGrid[1] == 1, \
            f"Scheduler requires localSubtileGrid[1]==1 for A, got {tileInfoA.localSubtileGrid[1]}"
        assert tileInfoB.localSubtileGrid[1] == 1, \
            f"Scheduler requires localSubtileGrid[1]==1 for B, got {tileInfoB.localSubtileGrid[1]}"

        self.partitions: List[Partition] = self._buildPartitions()
        self.allocator = VGPRTileAllocator()
        self.hasDuplicatedReads: bool = False
        self.needsUnrolling: bool = False

        self._runSchedule()

    # ── Outputs ──────────────────────────────────────────────

    @property
    def totalVGPRTiles(self) -> int:
        return self.allocator.totalVGPRTiles

    # ── Partition construction ─────────────────────────────────

    def _generateOrder(self) -> List[Tuple[int, int]]:
        order = []
        if self.config.ordering == SubgroupOrdering.COLUMN_MAJOR:
            for col in range(self.numPartitionsB):
                for row in range(self.numPartitionsA):
                    order.append((row, col))
        elif self.config.ordering == SubgroupOrdering.SNAKE_COLUMN_MAJOR:
            for col in range(self.numPartitionsB):
                if col % 2 == 0:
                    for row in range(self.numPartitionsA):
                        order.append((row, col))
                else:
                    for row in range(self.numPartitionsA - 1, -1, -1):
                        order.append((row, col))
        return order

    def _buildPartitions(self) -> List[Partition]:
        order = self._generateOrder()
        sA = self.config.partitionSizeA
        sB = self.config.partitionSizeB
        partitions = []
        for partitionId, (pA, pB) in enumerate(order):
            subtiles = []
            for a in range(pA * sA, (pA + 1) * sA):
                for b in range(pB * sB, (pB + 1) * sB):
                    subtiles.append((a, b))
            partitions.append(Partition(partitionId=partitionId, sizeA=sA, sizeB=sB, subtiles=subtiles))
        return partitions

    # ── Scheduling core ──────────────────────────────────────

    def _computePartitionGRs(self, preloadedMTn1_A: Set[int], preloadedMTn1_B: Set[int]) -> Dict[int, PartitionGR]:
        """Compute each partition's GR target (mtIteration, targetPartition, subtiles).

        Args:
            preloadedMTn1_A/B: MT n+1 subtiles already loaded by the preloop's GR(MT 1).
                These are excluded from mainloop MT n+1 GRs (dedup).
        """
        numPartitions = len(self.partitions)
        partitionGRs = {}
        loadedA = set(preloadedMTn1_A)
        loadedB = set(preloadedMTn1_B)
        for pi in range(numPartitions):
            targetPi = (pi + 1) % numPartitions
            targetPartition = self.partitions[targetPi]
            if pi == numPartitions - 1:
                # Last partition wraps to next macrotile iteration
                bufA = set(targetPartition.tileAIndices)
                bufB = set(targetPartition.tileBIndices)
                mtIter = "n+2"
            else:
                needA = set(targetPartition.tileAIndices)
                needB = set(targetPartition.tileBIndices)
                bufA = needA - loadedA
                bufB = needB - loadedB
                loadedA |= needA
                loadedB |= needB
                mtIter = "n+1"
            partitionGRs[pi] = PartitionGR(mtIteration=mtIter, targetPartitionId=targetPi,
                                            subtileA=bufA, subtileB=bufB)
        return partitionGRs

    def _buildPreloop(self) -> Tuple[Set[int], Set[int]]:
        """Allocate VGPRTile IDs for the first partition and build preloop GR ops.

        Preloop loads:
          - GR(MT 0): all subtiles
          - GR(MT 1): first partition's subtiles (1 partition worth of MT 1 data)

        Returns (preloadedMT1_A, preloadedMT1_B): what was preloaded for MT 1,
        so _computePartitionGRs can use it as the initial loaded state.
        """
        first = self.partitions[0]
        allA = list(range(self.MTA))
        allB = list(range(self.MTB))
        preloadMT1_A = list(first.tileAIndices)
        preloadMT1_B = list(first.tileBIndices)

        # Number of subIterK to preload LR for
        if self.config.prefetchMode == PrefetchMode.HALF_PREFETCH:
            numPreloadSubIterKs = 1
        elif self.config.prefetchMode == PrefetchMode.FULL_PREFETCH:
            numPreloadSubIterKs = self.numSubIterK

        # Allocate VGPRs for first group and build LR maps
        lrOps = []
        for sik in range(numPreloadSubIterKs):
            lrLoadA = {}
            lrLoadB = {}
            for tA in first.tileAIndices:
                lrLoadA[tA] = self.allocator.allocate('A', tA, sik)
            for tB in first.tileBIndices:
                lrLoadB[tB] = self.allocator.allocate('B', tB, sik)
            lrOps.append(LROp(mtIteration="0", subIterK=sik,
                              lrLoadA=lrLoadA, lrLoadB=lrLoadB))

        # Build preloop steps: GR(MT0) split by partition, WAIT, LR(MT0), SKIP guards, GR(MT1)
        preloopOps: List[ScheduleOp] = []
        # Split MT 0 GR by partition with dedup (same order as mainloop)
        loadedA: Set[int] = set()
        loadedB: Set[int] = set()
        for partition in self.partitions:
            grA = sorted(set(partition.tileAIndices) - loadedA)
            grB = sorted(set(partition.tileBIndices) - loadedB)
            loadedA.update(partition.tileAIndices)
            loadedB.update(partition.tileBIndices)
            if grA or grB:
                preloopOps.append(GROp(mtIteration="0",
                                       subtileA=grA, subtileB=grB,
                                       lastForMT=False))
        # Mark the last MT 0 GR as lastForMT
        for i in range(len(preloopOps) - 1, -1, -1):
            if isinstance(preloopOps[i], GROp):
                preloopOps[i] = dataclasses.replace(preloopOps[i], lastForMT=True)
                break
        preloopOps.append(GR_INCOp())
        preloopOps.append(WaitGROp(mtIteration="0",
                                 subtileA=allA, subtileB=allB,
                                 inflightLoadsA=0, inflightLoadsB=0))
        preloopOps.append(SyncOp(comment="Barrier: wait for GR data before LR"))
        preloopOps.extend(lrOps)
        preloopOps.append(WaitLROp())
        preloopOps.append(SkipOp(compare="LE", value=1, target="NLL"))
        mt1Complete = (set(preloadMT1_A) == set(allA) and set(preloadMT1_B) == set(allB))
        preloopOps.append(GROp(mtIteration="1",
                               subtileA=preloadMT1_A, subtileB=preloadMT1_B,
                               lastForMT=mt1Complete))
        if mt1Complete:
            preloopOps.append(GR_INCOp())
        preloopOps.append(SkipOp(compare="LE", value=2, target="NGLL"))
        preloopSik = SubIterKSchedule(subIterK=0)
        preloopSik.ops = preloopOps
        self.preloopSteps: List[PartitionSchedule] = [
            PartitionSchedule(partitionId=0, subIterKSteps=[preloopSik])]

        return set(preloadMT1_A), set(preloadMT1_B)

    def _runSchedule(self):
        if self.config.prefetchMode == PrefetchMode.NO:
            raise NotImplementedError("PrefetchMode.NO is not yet supported")

        preloadedMT1_A, preloadedMT1_B = self._buildPreloop()
        self.partitionGRs = self._computePartitionGRs(preloadedMT1_A, preloadedMT1_B)

        numPartitions = len(self.partitions)
        self.mainloopSteps: List[PartitionSchedule] = []

        for pi, partition in enumerate(self.partitions):
            pss = PartitionSchedule(partitionId=partition.partitionId)
            gr = self.partitionGRs[pi]
            subIterK0LoadAKeys: Set[int] = set()
            subIterK0LoadBKeys: Set[int] = set()

            for sik in range(self.numSubIterK):
                # USE: current group's tiles at current subIterK
                # MFMA: map subtile indices to VGPR tile IDs
                vgprTileMapA = {}
                vgprTileMapB = {}
                for tA in partition.tileAIndices:
                    vgprTileMapA[tA] = self.allocator.getVGPRTileId('A', tA, sik)
                for tB in partition.tileBIndices:
                    vgprTileMapB[tB] = self.allocator.getVGPRTileId('B', tB, sik)

                # LOAD: determined by prefetch mode
                loadATiles, loadBTiles, loadSubIterK = self._getLoadTargets(pi, sik, numPartitions)
                isWrapAround = self._isWrapAroundLoad(pi, sik, numPartitions)
                curA = set(partition.tileAIndices)
                curB = set(partition.tileBIndices)
                if sik == 0:
                    self._pendingRemap = []

                lrLoadA = {}
                lrLoadB = {}
                if loadATiles is not None:
                    for tA in loadATiles:
                        vid = self._loadTile('A', tA, loadSubIterK, isWrapAround, curA)
                        if vid is not None:
                            lrLoadA[tA] = vid

                if loadBTiles is not None:
                    for tB in loadBTiles:
                        vid = self._loadTile('B', tB, loadSubIterK, isWrapAround, curB)
                        if vid is not None:
                            lrLoadB[tB] = vid

                # Check MFMA and LOAD VGPRTile IDs don't overlap
                mfmaIds = set(vgprTileMapA.values()) | set(vgprTileMapB.values())
                loadIds = set(lrLoadA.values()) | set(lrLoadB.values())
                overlap = mfmaIds & loadIds
                conflict = set()
                if overlap:
                    conflict = overlap
                    self.needsUnrolling = True

                # Build SubIterKSchedule with MFMA and LR ops
                siks = SubIterKSchedule(subIterK=sik)
                mfmas = [(a, b) for a in sorted(vgprTileMapA.keys()) for b in sorted(vgprTileMapB.keys())]
                mtLoad = "n+1" if isWrapAround else "n"
                siks.ops.append(MFMAOp(mtIteration="n", subIterK=sik,
                                      subtiles=mfmas,
                                      vgprTileMapA=vgprTileMapA, vgprTileMapB=vgprTileMapB))
                siks.ops.append(LROp(mtIteration=mtLoad, subIterK=loadSubIterK,
                                    lrLoadA=lrLoadA, lrLoadB=lrLoadB))
                siks.conflict = conflict
                pss.subIterKSteps.append(siks)

                # save subtiles for subIterK=0 to check where to insert GR(n+2)
                if sik == 0:
                    subIterK0LoadAKeys = set(lrLoadA.keys())
                    subIterK0LoadBKeys = set(lrLoadB.keys())

                # WITHIN_SUBGROUP: release current subIterK's MFMA tiles for K-dim reuse
                if self.config.reuseStrategy == VGPRTileReUseStrategy.WITHIN_SUBGROUP:
                    for tA in vgprTileMapA:
                        if self.allocator.isAllocated('A', tA, sik):
                            self.allocator.release('A', tA, sik)
                    for tB in vgprTileMapB:
                        if self.allocator.isAllocated('B', tB, sik):
                            self.allocator.release('B', tB, sik)

            # Insert GROps split across subIterK=0 and subIterK=1
            if gr.subtileA or gr.subtileB:
                totalGR_A = sorted(gr.subtileA)
                totalGR_B = sorted(gr.subtileB)
                splitA = (len(totalGR_A) + 1) // 2
                splitB = (len(totalGR_B) + 1) // 2
                gr0_A, gr1_A = totalGR_A[:splitA], totalGR_A[splitA:]
                gr0_B, gr1_B = totalGR_B[:splitB], totalGR_B[splitB:]

                # lastForMT: true for the last GR that completes a full MT load
                # within this loop iteration. One GR_INC per loop iteration.
                # - For n+1 GRs: true when no more n+1 GRs follow.
                # - For n+2 GRs: true only when there are no n+1 GRs at all
                #   (1 partition case where n+2 loads all subtiles in one shot).
                #   Otherwise n+2 is partial and continues in the next iteration.
                isLastForThisMT = False
                if gr.mtIteration == "n+1":
                    isLastForThisMT = True
                    for fpi in range(pi + 1, numPartitions):
                        fgr = self.partitionGRs[fpi]
                        if (fgr.subtileA or fgr.subtileB) and fgr.mtIteration == "n+1":
                            isLastForThisMT = False
                            break
                elif gr.mtIteration == "n+2":
                    # n+2 gets GR_INC only if no n+1 GRs exist (single partition)
                    hasN1 = any((self.partitionGRs[p].subtileA or self.partitionGRs[p].subtileB)
                                and self.partitionGRs[p].mtIteration == "n+1"
                                for p in range(numPartitions))
                    if not hasN1:
                        # Check this is the last n+2 GR
                        isLastForThisMT = True
                        for fpi in range(pi + 1, numPartitions):
                            fgr = self.partitionGRs[fpi]
                            if (fgr.subtileA or fgr.subtileB) and fgr.mtIteration == "n+2":
                                isLastForThisMT = False
                                break

                if gr0_A or gr0_B:
                    pss.subIterKSteps[0].ops.append(GROp(
                        mtIteration=gr.mtIteration,
                        subtileA=gr0_A, subtileB=gr0_B,
                        lastForMT=False))
                if gr1_A or gr1_B:
                    pss.subIterKSteps[1].ops.append(GROp(
                        mtIteration=gr.mtIteration,
                        subtileA=gr1_A, subtileB=gr1_B,
                        lastForMT=isLastForThisMT))
                elif isLastForThisMT:
                    # All GRs fit in subIterK=0, mark that one as last
                    pss.subIterKSteps[0].ops[-1] = dataclasses.replace(
                        pss.subIterKSteps[0].ops[-1], lastForMT=True)

            self.mainloopSteps.append(pss)

            # Release after partition based on strategy
            if self.config.reuseStrategy == VGPRTileReUseStrategy.WITHIN_SUBGROUP:
                for tc, tileIdx, subIterK, shadowKey in self._pendingRemap:
                    vid = self.allocator._allocMap(tc).pop((shadowKey, subIterK))
                    self.allocator._allocMap(tc)[(tileIdx, subIterK)] = vid
                self._pendingRemap = []
            elif self.config.reuseStrategy == VGPRTileReUseStrategy.ACROSS_SUBGROUP:
                self._releaseUnusedAfterPartition(pi)

        self._insertWaitsAndSync(numPartitions)
        self.ngllSteps = self._buildNGLL()
        self.nllSteps = self._buildNLL()
        self._checkDuplicatedReads()

    def _loadTile(self, tc: str, tileIdx: int, loadSubIterK: int,
                  isWrapAround: bool,
                  currentPartitionTiles: Set[int]) -> Optional[int]:
        """Determine the VGPRTile ID for a load. Returns None if no load needed."""
        allocated = self.allocator.isAllocated(tc, tileIdx, loadSubIterK)

        if isWrapAround and allocated:
            # Wrap-around: reuse partition 0's existing VGPRTile IDs
            return self.allocator.getVGPRTileId(tc, tileIdx, loadSubIterK)

        if not allocated:
            # Fresh allocation
            return self.allocator.allocate(tc, tileIdx, loadSubIterK)

        if self.config.reuseStrategy == VGPRTileReUseStrategy.WITHIN_SUBGROUP \
                and tileIdx in currentPartitionTiles:
            # Tile is allocated by the current partition but will be released after it.
            # Must allocate a new VGPR for the next partition's data.
            # Use a shadow key to avoid overwriting the current allocation.
            shadowKey = -(tileIdx + 1)  # negative to avoid collision
            vid = self.allocator.allocate(tc, shadowKey, loadSubIterK)
            # Store the real tileIdx mapping for later fixup
            self._pendingRemap.append((tc, tileIdx, loadSubIterK, shadowKey))
            return vid

        # NONE / ACROSS_SUBGROUP: tile stays alive, reuse in place
        return None

    def _isWrapAroundLoad(self, partitionIdx: int, subIterK: int, numPartitions: int) -> bool:
        """True when this step's load targets partition 0 for the next macrotile iteration."""
        if self.config.prefetchMode == PrefetchMode.HALF_PREFETCH:
            return partitionIdx == numPartitions - 1 and subIterK == self.numSubIterK - 1
        elif self.config.prefetchMode == PrefetchMode.FULL_PREFETCH:
            return partitionIdx == numPartitions - 1
        return False

    # ── Prefetch modes ───────────────────────────────────────

    def _getLoadTargets(self, partitionIdx: int, subIterK: int,
                        numPartitions: int) -> Tuple[Optional[List[int]], Optional[List[int]], int]:
        """Returns (loadATiles, loadBTiles, targetSubIterK)."""
        if self.config.prefetchMode == PrefetchMode.HALF_PREFETCH:
            return self._loadTargetsHalfPrefetch(partitionIdx, subIterK, numPartitions)
        elif self.config.prefetchMode == PrefetchMode.FULL_PREFETCH:
            return self._loadTargetsFullPrefetch(partitionIdx, subIterK, numPartitions)
        return (None, None, 0)

    def _loadTargetsHalfPrefetch(self, partitionIdx, subIterK, numPartitions):
        """HALF: subIterK=0 loads same-partition subIterK=1, subIterK=last loads next-partition subIterK=0.
        Last partition wraps around to partition 0 (next iteration)."""
        currentPartition = self.partitions[partitionIdx]
        if subIterK < self.numSubIterK - 1:
            targetSubIterK = subIterK + 1
            return (currentPartition.tileAIndices, currentPartition.tileBIndices, targetSubIterK)
        else:
            nextPartition = self.partitions[(partitionIdx + 1) % numPartitions]
            return (nextPartition.tileAIndices, nextPartition.tileBIndices, 0)

    def _loadTargetsFullPrefetch(self, partitionIdx, subIterK, numPartitions):
        """FULL: subIterK=0 loads next-partition subIterK=0, subIterK=1 loads next-partition subIterK=1.
        Last partition wraps around to partition 0 (next iteration)."""
        nextPartition = self.partitions[(partitionIdx + 1) % numPartitions]
        return (nextPartition.tileAIndices, nextPartition.tileBIndices, subIterK)

    # ── Reuse strategies ─────────────────────────────────────

    def _releaseUnusedAfterPartition(self, partitionIdx: int):
        """ACROSS_SUBGROUP: release tiles not appearing in any future partition.
        Partition 0's tiles are always considered "future" because the wrap-around
        LR at the end of the loop loads back into partition 0's vgprTile IDs."""
        currentPartition = self.partitions[partitionIdx]

        # Include partition 0 in the future set — the wrap-around LR needs
        # those tiles to still be allocated so it reuses the same vgprTile IDs.
        futureA: Set[int] = set(self.partitions[0].tileAIndices)
        futureB: Set[int] = set(self.partitions[0].tileBIndices)
        for pi in range(partitionIdx + 1, len(self.partitions)):
            futureA.update(self.partitions[pi].tileAIndices)
            futureB.update(self.partitions[pi].tileBIndices)

        for tA in currentPartition.tileAIndices:
            if tA not in futureA:
                self.allocator.releaseAllForTile('A', tA)
        for tB in currentPartition.tileBIndices:
            if tB not in futureB:
                self.allocator.releaseAllForTile('B', tB)

    def _releasePartitionTiles(self, partition: Partition):
        """WITHIN_SUBGROUP: release all tiles (all subIterK) of this partition."""
        for tA in partition.tileAIndices:
            self.allocator.releaseAllForTile('A', tA)
        for tB in partition.tileBIndices:
            self.allocator.releaseAllForTile('B', tB)

    def _insertWaitsAndSync(self, numPartitions: int):
        """Pass 2: Insert WAIT_LR, WAIT_GR, SyncOp and reorder ops.

        After pass 1, each subIterK has: [MFMAOp, LROp, GROp?]

        This pass produces the final ordering per subIterK:
          subIterK=0 (LR for MT n, GR n+2 collides):
            MFMAs → LR → WAIT_LR → SyncOp → GR(n+2)
          subIterK=1 (LR for MT n+1, WAIT_GR needed):
            MFMAs → GR(n+2) → WAIT_GR → SyncOp → LR(n+1) → WAIT_LR
        """
        # ── Pass 3 prep: build GR events for inflight counting ──
        # Each entry: (opIndex, mtIteration, subtileA_set, subtileB_set)
        # Also record the opIndex of each WAIT_GR candidate (subIterK==0 LR ops).
        grEvents = []
        opIdx = 0
        for pss in self.mainloopSteps:
            for dus in pss.subIterKSteps:
                for op in dus.ops:
                    if isinstance(op, GROp):
                        grEvents.append((opIdx, op.mtIteration,
                                         set(op.subtileA), set(op.subtileB)))
                    opIdx += 1

        def _parseMTOffset(mt: str) -> Optional[int]:
            if mt == "n":
                return 0
            if mt.startswith("n+"):
                return int(mt[2:])
            return None

        def _countInflightSubtileLoads(waitOpIndex, waitMT, waitSubtileA, waitSubtileB):
            """Count inflight subtile loads by walking backwards through
            the mainloop from the WAIT_GR position until we find the GR
            that originally issued the waited subtiles.

            Walk upward from the WAIT_GR. Every GR encountered increments
            the inflight count. When we find a GR whose (shifted) MT and
            subtile sets match the WAIT target, we stop (without counting it).

            When wrapping from the start of the mainloop to the end
            (previous iteration), MT iterations shift down by 1
            (e.g. n+2 becomes n+1, n+1 becomes n).
            """
            waitOffset = _parseMTOffset(waitMT)
            if waitOffset is None:
                return 0, 0

            targetA = set(waitSubtileA)
            targetB = set(waitSubtileB)

            # Split grEvents into before-wait and after-wait (for wrap)
            before = [(mt, a, b) for (idx, mt, a, b) in grEvents if idx < waitOpIndex]
            after  = [(mt, a, b) for (idx, mt, a, b) in grEvents if idx >= waitOpIndex]

            totalA = 0
            totalB = 0

            # Walk backwards through events before the WAIT (no MT shift)
            for (grMT, grA, grB) in reversed(before):
                grOffset = _parseMTOffset(grMT)
                if grOffset is None:
                    continue
                if grOffset == waitOffset and grA == targetA and grB == targetB:
                    return totalA, totalB
                totalA += len(grA)
                totalB += len(grB)

            # Wrap: walk backwards from end of mainloop (shift MT by -1)
            for (grMT, grA, grB) in reversed(after):
                grOffset = _parseMTOffset(grMT)
                if grOffset is None:
                    continue
                shiftedOffset = grOffset - 1
                if shiftedOffset == waitOffset and grA == targetA and grB == targetB:
                    return totalA, totalB
                totalA += len(grA)
                totalB += len(grB)

            return totalA, totalB

        # ── Insert WAIT_LR, WAIT_GR, SyncOp, GR_INC, LR_INC and reorder ──
        pendingA = set()
        pendingB = set()
        lastLRmt = None
        opIdx = 0
        for pss in self.mainloopSteps:
            gr = self.partitionGRs[pss.partitionId]
            pendingA |= gr.subtileA
            pendingB |= gr.subtileB

            for dus in pss.subIterKSteps:
                numOrigOps = len(dus.ops)
                # Extract ops by type from pass 1
                mfmaOps = [op for op in dus.ops if isinstance(op, MFMAOp)]
                lrOps = [op for op in dus.ops if isinstance(op, LROp)]
                grOps = [op for op in dus.ops if isinstance(op, GROp)]
                otherOps = [op for op in dus.ops
                            if not isinstance(op, (MFMAOp, LROp, GROp))]

                lrOp = lrOps[0] if lrOps else None
                hasGRn2 = any(g.mtIteration == "n+2" for g in grOps)

                # Determine if WAIT_GR is needed before this LR
                waitGROp = None
                if lrOp and lrOp.subIterK == 0:
                    waitA = set(lrOp.lrLoadA.keys()) & pendingA
                    waitB = set(lrOp.lrLoadB.keys()) & pendingB
                    if waitA or waitB:
                        # WAIT_GR position: after all original ops in this subIterK step
                        waitOpIdx = opIdx + numOrigOps
                        inflightCountA, inflightCountB = _countInflightSubtileLoads(
                            waitOpIdx, lrOp.mtIteration, sorted(waitA), sorted(waitB))
                        waitGROp = WaitGROp(
                            mtIteration=lrOp.mtIteration,
                            subtileA=sorted(waitA), subtileB=sorted(waitB),
                            inflightLoadsA=inflightCountA, inflightLoadsB=inflightCountB)
                        pendingA -= waitA
                        pendingB -= waitB

                # Rebuild ops in correct order
                newOps = []
                newOps.extend(mfmaOps)
                newOps.extend(otherOps)

                if waitGROp:
                    # subIterK=1 pattern: MFMAs → GR(n+2) → GR_INC? → WAIT_GR → SyncOp → LR_INC? → LR → WAIT_LR
                    newOps.extend(grOps)
                    if any(g.lastForMT for g in grOps):
                        newOps.append(GR_INCOp())
                    newOps.append(waitGROp)
                    newOps.append(SyncOp(comment="Barrier: wait for GR data"))
                    if lrOp:
                        if lastLRmt is not None and lrOp.mtIteration != lastLRmt:
                            newOps.append(LR_INCOp())
                        lastLRmt = lrOp.mtIteration
                        newOps.append(lrOp)
                        newOps.append(WaitLROp())
                elif hasGRn2 and lrOp:
                    # subIterK=0 pattern: MFMAs → LR_INC? → LR → WAIT_LR → SyncOp → GR(n+2) → GR_INC?
                    if lastLRmt is not None and lrOp.mtIteration != lastLRmt:
                        newOps.append(LR_INCOp())
                    lastLRmt = lrOp.mtIteration
                    newOps.append(lrOp)
                    newOps.append(WaitLROp())
                    newOps.append(SyncOp(comment="Barrier: all waves done with LR before GR(n+2) writes"))
                    newOps.extend(grOps)
                    if any(g.lastForMT for g in grOps):
                        newOps.append(GR_INCOp())
                else:
                    # No special dependency: MFMAs → LR_INC? → LR → GR → GR_INC? → WAIT_LR
                    if lrOp:
                        if lastLRmt is not None and lrOp.mtIteration != lastLRmt:
                            newOps.append(LR_INCOp())
                        lastLRmt = lrOp.mtIteration
                        newOps.append(lrOp)
                    newOps.extend(grOps)
                    if grOps and any(g.lastForMT for g in grOps):
                        newOps.append(GR_INCOp())
                    if lrOp:
                        newOps.append(WaitLROp())

                opIdx += numOrigOps
                dus.ops = newOps


    def _buildNGLL(self) -> List[PartitionSchedule]:
        """NGLL (Non Global Load Loop): mainloop without GR(n+2) and GR_INC."""
        ngll = []
        for pss in self.mainloopSteps:
            newPss = PartitionSchedule(partitionId=pss.partitionId)
            for dus in pss.subIterKSteps:
                newDus = SubIterKSchedule(subIterK=dus.subIterK, conflict=dus.conflict)
                for op in dus.ops:
                    if isinstance(op, GROp) and op.mtIteration == "n+2":
                        continue
                    if isinstance(op, GR_INCOp):
                        continue
                    if isinstance(op, WaitGROp):
                        op = WaitGROp(mtIteration=op.mtIteration,
                                    subtileA=op.subtileA, subtileB=op.subtileB,
                                    inflightLoadsA=0, inflightLoadsB=0)
                    newDus.ops.append(op)
                newPss.subIterKSteps.append(newDus)
            ngll.append(newPss)
        return ngll

    def _buildNLL(self) -> List[PartitionSchedule]:
        """NLL (No Load Loop): mainloop without GR, GR_INC, LR_INC, LR(n+1),
        WaitGR(n+1) and their associated SyncOps. Keeps WaitGR(n) and its SYNC."""
        nll = []
        for pss in self.mainloopSteps:
            newPss = PartitionSchedule(partitionId=pss.partitionId)
            for dus in pss.subIterKSteps:
                newDus = SubIterKSchedule(subIterK=dus.subIterK, conflict=dus.conflict)
                ops = dus.ops
                for i, op in enumerate(ops):
                    if isinstance(op, (GROp, GR_INCOp, LR_INCOp)):
                        continue
                    if isinstance(op, LROp) and op.mtIteration == "n+1":
                        continue
                    if isinstance(op, WaitGROp):
                        if op.mtIteration == "n+1":
                            continue
                        op = WaitGROp(mtIteration=op.mtIteration,
                                    subtileA=op.subtileA, subtileB=op.subtileB,
                                    inflightLoadsA=0, inflightLoadsB=0)
                    if isinstance(op, SyncOp):
                        # Skip SyncOps associated with removed ops:
                        # - SyncOp followed by a GROp (barrier before GR writes)
                        # - SyncOp preceded by a WaitGROp(n+1) (barrier after GR wait)
                        nextOp = ops[i + 1] if i + 1 < len(ops) else None
                        prevOp = ops[i - 1] if i > 0 else None
                        if isinstance(nextOp, GROp):
                            continue
                        if isinstance(prevOp, WaitGROp) and prevOp.mtIteration == "n+1":
                            continue
                    newDus.ops.append(op)
                # Remove orphaned WAIT_LR when no LR remains in this subIterK
                hasLR = any(isinstance(op, LROp) for op in newDus.ops)
                if not hasLR:
                    newDus.ops = [op for op in newDus.ops if not isinstance(op, WaitLROp)]
                newPss.subIterKSteps.append(newDus)
            nll.append(newPss)
        return nll

    def _checkDuplicatedReads(self):
        """Detect if any (subtile, subIterK) pair is loaded more than once."""
        seenA: Dict[AllocKey, int] = {}
        seenB: Dict[AllocKey, int] = {}
        # Count preloop LR loads
        for pss in self.preloopSteps:
            for dus in pss.subIterKSteps:
                for op in dus.ops:
                    if isinstance(op, LROp):
                        for tA in op.lrLoadA:
                            key = (tA, op.subIterK)
                            seenA[key] = seenA.get(key, 0) + 1
                        for tB in op.lrLoadB:
                            key = (tB, op.subIterK)
                            seenB[key] = seenB.get(key, 0) + 1
        # Count mainloop LR loads (skip wrap-around which reuses existing allocations)
        for pss in self.mainloopSteps:
            for dus in pss.subIterKSteps:
                for op in dus.ops:
                    if isinstance(op, LROp) and op.mtIteration != "n+1":
                        for tA in op.lrLoadA:
                            key = (tA, op.subIterK)
                            seenA[key] = seenA.get(key, 0) + 1
                        for tB in op.lrLoadB:
                            key = (tB, op.subIterK)
                            seenB[key] = seenB.get(key, 0) + 1
        self.hasDuplicatedReads = (
            any(c > 1 for c in seenA.values()) or
            any(c > 1 for c in seenB.values())
        )

    # ── Debug ────────────────────────────────────────────────

    @staticmethod
    def _printOp(op: ScheduleOp, indent: str = ""):
        if isinstance(op, MFMAOp):
            print(f"{indent}MFMAs (MT {op.mtIteration}, subIterK {op.subIterK}):")
            print(f"{indent}  - {op.subtiles}")
            print(f"{indent}  - USING  A: {op.vgprTileMapA}  B: {op.vgprTileMapB}")
        elif isinstance(op, GROp):
            print(f"{indent}GR (MT {op.mtIteration}):  A: {op.subtileA}  B: {op.subtileB}")
        elif isinstance(op, WaitGROp):
            inflight = f" — inflight SubtileLoads A={op.inflightLoadsA} B={op.inflightLoadsB}" if op.inflightLoadsA is not None else ""
            print(f"{indent}WAIT_GR (MT {op.mtIteration}) A: {op.subtileA}  B: {op.subtileB}{inflight}")
        elif isinstance(op, WaitLROp):
            print(f"{indent}WAIT_LR")
        elif isinstance(op, SyncOp):
            print(f"{indent}SYNC")
        elif isinstance(op, LROp):
            sikLabel = f", subIterK {op.subIterK}" if op.subIterK >= 0 else ""
            print(f"{indent}LR (MT {op.mtIteration}{sikLabel}) A: {op.lrLoadA}  B: {op.lrLoadB}")
        elif isinstance(op, SkipOp):
            print(f"{indent}SKIP_IF_{op.compare}({op.value}, {op.target})")
        elif isinstance(op, GR_INCOp):
            print(f"{indent}GR_INC")
        elif isinstance(op, LR_INCOp):
            print(f"{indent}LR_INC")

    def printSchedule(self):
        print(f"SubtileGridA={self.MTA}, SubtileGridB={self.MTB}")
        print(f"Partition grid: {self.numPartitionsA} x {self.numPartitionsB}")
        print(f"Partition size: {self.config.partitionSizeA} x {self.config.partitionSizeB}")
        print(f"Prefetch: {self.config.prefetchMode.name}")
        print(f"Reuse: {self.config.reuseStrategy.name}")
        print(f"hasDuplicatedReads: {self.hasDuplicatedReads}")
        print(f"needsUnrolling: {self.needsUnrolling}")
        print(f"totalVGPRTiles: {self.totalVGPRTiles} ({self.totalVGPRTiles * 4} VGPRs)")
        print()

        grid = [[None] * self.numPartitionsB for _ in range(self.numPartitionsA)]
        sA = self.config.partitionSizeA
        sB = self.config.partitionSizeB
        for partition in self.partitions:
            pA = partition.subtiles[0][0] // sA
            pB = partition.subtiles[0][1] // sB
            grid[pA][pB] = partition.partitionId
        print(f"Ordering grid ({self.config.ordering.name}):")
        for row in grid:
            print("  " + "  ".join(f"{v:2d}" if v is not None else "  " for v in row))
        print()

        print("PRELOOP:")
        for pss in self.preloopSteps:
            for dus in pss.subIterKSteps:
                for op in dus.ops:
                    self._printOp(op, indent="  ")
        print()

        print("MAINLOOP:")
        for partition in self.mainloopSteps:
            print(f"  Partition {partition.partitionId}:")
            for dus in partition.subIterKSteps:
                print(f"    subIterK={dus.subIterK}:")
                for op in dus.ops:
                    self._printOp(op, indent="      ")
                if dus.conflict:
                    print(f"      *** CONFLICT: USE/LOAD share VGPRTile IDs {dus.conflict} — needs unrolling ***")

        print()
        print("NGLL (No Global Load Loop):")
        for partition in self.ngllSteps:
            print(f"  Partition {partition.partitionId}:")
            for dus in partition.subIterKSteps:
                print(f"    subIterK={dus.subIterK}:")
                for op in dus.ops:
                    self._printOp(op, indent="      ")

        print()
        print("NLL (No Load Loop):")
        for partition in self.nllSteps:
            print(f"  Partition {partition.partitionId}:")
            for dus in partition.subIterKSteps:
                print(f"    subIterK={dus.subIterK}:")
                for op in dus.ops:
                    self._printOp(op, indent="      ")

    # Allocate totalVGPRTiles vpgrTile
    def allocVgprTiles(self, writer):
        """Allocate a shared VGPR tile array for A and B, indexed by the scheduler's vgprTileId."""
        
        self.vgprTiles = []
        mmaTileRegCount = self.tileInfoA.mmaTileRegCount
        for _ in range(self.totalVGPRTiles):
            tile = TileInfo.RegisterTileInfo(writer.vgprPool)
            for j in range(0, mmaTileRegCount, 4):
                vstart = writer.vgprPool.checkOutAligned(4, 4)
                for k in range(4):
                    tile.append(vstart + k)
            self.vgprTiles.append(tile)

    def deallocVgprTiles(self, writer):
        """Deallocate VGPR tiles allocated by allocVgprTiles."""
        for tile in self.vgprTiles:
            pool = tile.regList.regPool
            for val in tile:
                if tile.index(val) % 4 == 0:
                    pool.checkIn(val)
        self.vgprTiles = []

    def emitMFMAs(self, writer, kernel, steps, dtileInfo):
        """Emit MFMA instructions for a list of PartitionSchedules."""

        module = Module()
        for pss in steps:
            for dus in pss.subIterKSteps:
                for op in dus.ops:
                    if not isinstance(op, MFMAOp):
                        continue
                    for (a, b) in op.subtiles:
                        aTile = self.vgprTiles[op.vgprTileMapA[a]]
                        bTile = self.vgprTiles[op.vgprTileMapB[b]]
                        dTile = dtileInfo.vgprTiles[a + b * dtileInfo.localMMATileGrid[0]]
                        module.add(emitMfmaInstruction(
                            writer, kernel, aTile, bTile, dTile, dTile,
                            f"MFMA C[{a},{b}] += A[{a},subIterK{op.subIterK}] * B[{b},subIterK{op.subIterK}]"))
        return module

    def emitLRs(self, writer, kernel, steps):
        """Emit LR (Local Read) ds_load instructions for a list of PartitionSchedules."""
        module = Module()
        for pss in steps:
            for dus in pss.subIterKSteps:
                for op in dus.ops:
                    if not isinstance(op, LROp):
                        continue
                    for tA, vgprTileId in op.lrLoadA.items():
                        dstTile = self.vgprTiles[vgprTileId]
                        module.add(emitSingleDsRead(
                            self.tileInfoA, tA, op.subIterK, dstTile))
                    for tB, vgprTileId in op.lrLoadB.items():
                        dstTile = self.vgprTiles[vgprTileId]
                        module.add(emitSingleDsRead(
                            self.tileInfoB, tB, op.subIterK, dstTile))
        return module

    def emitWaitGR(self, inflightLoadsA, inflightLoadsB):
        """Emit SWaitCnt for GR (buffer_load) based on inflight GR counts.

        Args:
            inflightLoadsA: Number of A GR loads still inflight.
            inflightLoadsB: Number of B GR loads still inflight.
        """
        module = Module()
        grCnt = int(inflightLoadsA / self.tileInfoA.loadRatioGR) + \
                int(inflightLoadsB / self.tileInfoB.loadRatioGR)
        module.add(SWaitCnt(dscnt=-1, vlcnt=grCnt, vscnt=-1,
                            comment=f"Wait GR: A={inflightLoadsA} B={inflightLoadsB} => vlcnt={grCnt}"))
        return module

    def emitGRs(self, writer, kernel, steps):
        """Emit GR (Global Read) buffer_load instructions for a list of PartitionSchedules.

        For each GROp, expands sId0 indices to all sId1 (K-dimension) values,
        with deduplication via globalReadMap tracking.
        """
        module = Module()
        grTrackerA = set()
        grTrackerB = set()
        for pss in steps:
            for siks in pss.subIterKSteps:
                for op in siks.ops:
                    if not isinstance(op, GROp):
                        continue
                    for subtileList, tileInfo, grTracker in [(op.subtileA, self.tileInfoA, grTrackerA),
                                                            (op.subtileB, self.tileInfoB, grTrackerB)]:
                        for sId0 in subtileList:
                            grIds = tileInfo.localSubtiles[tileInfo.getLocalSubtileLinearId(sId0, 0)].globalReadMap
                            if not set(grIds).issubset(grTracker):
                                grTracker.update(grIds)
                                module.add(emitSingleBufferLoad(tileInfo, sId0, 0))
        return module

    def _emitSubIterK(self, writer, kernel, pss, dus):
        """Emit a single subIterK step into a Module."""
        dtileInfo = writer.states.d.tileInfo
        module = Module()
        module.addComment0(f"Partition {pss.partitionId}: subIterK={dus.subIterK}")
        for op in dus.ops:
            if isinstance(op, GROp):
                oneStep = [PartitionSchedule(
                    partitionId=pss.partitionId,
                    subIterKSteps=[SubIterKSchedule(subIterK=dus.subIterK, ops=[op])])]
                module.add(self.emitGRs(writer, kernel, oneStep))
            elif isinstance(op, GR_INCOp):
                module.add(globalReadPtrUpdates('A', writer, kernel))
                module.add(globalReadPtrUpdates('B', writer, kernel))
                module.add(globalReadLDSBufferSwap('A', writer, kernel))
                module.add(globalReadLDSBufferSwap('B', writer, kernel))
            elif isinstance(op, MFMAOp):
                oneStep = [PartitionSchedule(
                    partitionId=pss.partitionId,
                    subIterKSteps=[SubIterKSchedule(subIterK=dus.subIterK, ops=[op])])]
                module.add(self.emitMFMAs(writer, kernel, oneStep, dtileInfo))
            elif isinstance(op, WaitGROp):
                module.add(self.emitWaitGR(op.inflightLoadsA, op.inflightLoadsB))
            elif isinstance(op, WaitLROp):
                module.add(SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for LR to complete"))
            elif isinstance(op, SyncOp):
                module.add(SBarrier(comment=op.comment))
            elif isinstance(op, LR_INCOp):
                module.add(localReadLDSBufferSwap('A', writer, kernel))
                module.add(localReadLDSBufferSwap('B', writer, kernel))
            elif isinstance(op, LROp):
                oneStep = [PartitionSchedule(
                    partitionId=pss.partitionId,
                    subIterKSteps=[SubIterKSchedule(subIterK=dus.subIterK, ops=[op])])]
                module.add(self.emitLRs(writer, kernel, oneStep))
            elif isinstance(op, SkipOp):
                skipLabel = Label(f"SkipTo{op.target}", "")
                cmpMap = {"EQ": SCmpEQU32, "LE": SCmpLeU32}
                module.add(cmpMap[op.compare](
                    src0=sgpr("LoopCounterL"), src1=op.value,
                    comment=f"LoopCounter {op.compare} {op.value}?"))
                module.add(SCBranchSCC1(
                    labelName=skipLabel.getLabelName(),
                    comment=f"skip to {op.target}"))
        return module

    @staticmethod
    def instructionSchedule(module):
        """Schedule MFMAs among other instructions within a subIterK module.

        Rules (invariants preserved by this pass):
          - MFMA instruction order is preserved
          - Non-MFMA instruction order is preserved
          - Insert 1 MFMA between each LR (ds_read) instruction
          - Insert 4 MFMAs between the last LR and the WAIT_LR (SWaitCnt dscnt)
          - Insert 1 MFMA between WAIT_LR and SYNC (SBarrier)
          - No MFMAs between an m0 update and its buffer_load (they are a pair)
          - Remaining MFMAs are spread evenly between buffer_load pairs
        """
        items = module.flatitems()
        if not items:
            return module

        mfmas = [x for x in items if isinstance(x, MFMAInstruction)]
        others = [x for x in items if not isinstance(x, MFMAInstruction)]

        if not mfmas or not others:
            return module

        # Group others into slots: each slot is a list of instructions that
        # must stay together (e.g. m0 update + buffer_load pair).
        # We'll insert MFMAs BETWEEN slots.
        slots = []
        i = 0
        while i < len(others):
            inst = others[i]
            # Pair m0 update with its following buffer_load
            if i + 1 < len(others) and isinstance(others[i + 1], GlobalReadInstruction):
                slots.append(others[i:i+2])
                i += 2
            else:
                slots.append([inst])
                i += 1

        # Classify each slot for MFMA insertion rules
        LR_SLOT = 0       # ds_read (LocalReadInstruction)
        WAITLR_SLOT = 1   # SWaitCnt with dscnt (WAIT_LR)
        SYNC_SLOT = 2     # SBarrier (SYNC)
        GR_SLOT = 3       # m0 + buffer_load pair or standalone buffer_load
        OTHER_SLOT = 4    # everything else

        def classify(slot):
            first = slot[0]
            if isinstance(first, LocalReadInstruction):
                return LR_SLOT
            if isinstance(first, SWaitCnt):
                return WAITLR_SLOT
            if isinstance(first, SBarrier):
                return SYNC_SLOT
            if isinstance(first, GlobalReadInstruction) or \
               (len(slot) > 1 and isinstance(slot[-1], GlobalReadInstruction)):
                return GR_SLOT
            return OTHER_SLOT

        slotTypes = [classify(s) for s in slots]

        # Build MFMA budget: how many MFMAs to insert AFTER each slot.
        # (mfmasAfter[i] = number of MFMAs inserted after slots[i])
        numSlots = len(slots)
        mfmasAfter = [0] * numSlots

        mi = 0  # next MFMA to assign

        # Pass 1: assign fixed MFMAs per rules
        for si in range(numSlots):
            if mi >= len(mfmas):
                break
            st = slotTypes[si]
            nextSt = slotTypes[si + 1] if si + 1 < numSlots else None

            if st == LR_SLOT and nextSt == LR_SLOT:
                # 1 MFMA between each LR
                mfmasAfter[si] = min(1, len(mfmas) - mi)
                mi += mfmasAfter[si]
            elif st == LR_SLOT and nextSt == WAITLR_SLOT:
                # 4 MFMAs between last LR and WAIT_LR
                mfmasAfter[si] = min(4, len(mfmas) - mi)
                mi += mfmasAfter[si]
            elif st == LR_SLOT and nextSt != LR_SLOT and nextSt != WAITLR_SLOT:
                # Last LR but no WAIT_LR follows — still insert 1
                mfmasAfter[si] = min(1, len(mfmas) - mi)
                mi += mfmasAfter[si]
            elif st == WAITLR_SLOT and nextSt == SYNC_SLOT:
                # 1 MFMA between WAIT_LR and SYNC
                mfmasAfter[si] = min(1, len(mfmas) - mi)
                mi += mfmasAfter[si]

        # Pass 2: spread remaining MFMAs evenly between GR slots
        grIndices = [si for si in range(numSlots) if slotTypes[si] == GR_SLOT]
        remaining = len(mfmas) - mi
        if remaining > 0 and grIndices:
            base = remaining // len(grIndices)
            extra = remaining % len(grIndices)
            for gi, si in enumerate(grIndices):
                count = base + (1 if gi < extra else 0)
                mfmasAfter[si] += count
                mi += count

        # Assemble final output
        result = Module()
        mfmaIdx = 0

        # Leading MFMAs: any unassigned MFMAs go before the first slot
        leadingMfmas = len(mfmas) - mi
        for _ in range(leadingMfmas):
            result.add(mfmas[mfmaIdx])
            mfmaIdx += 1

        for si in range(numSlots):
            for inst in slots[si]:
                result.add(inst)
            for _ in range(mfmasAfter[si]):
                if mfmaIdx < len(mfmas):
                    result.add(mfmas[mfmaIdx])
                    mfmaIdx += 1

        # Any remaining MFMAs at the end
        while mfmaIdx < len(mfmas):
            result.add(mfmas[mfmaIdx])
            mfmaIdx += 1

        return result

    def _emitLoop(self, writer, kernel, label, steps):
        """Emit a loop module (mainloop, NGLL, or NLL).

        Emits each subIterK step as a separate module, applies instruction
        interleaving, then combines into the final loop module.
        All waits (WAIT_LR, WAIT_GR, SyncOp) are explicit schedule ops.
        """
        module = Module(label)
        for pss in steps:
            for dus in pss.subIterKSteps:
                subModule = self._emitSubIterK(writer, kernel, pss, dus)
                subModule = self.instructionSchedule(subModule)
                module.add(subModule)
        return module

    def generateCode(self, writer, kernel):
        self.allocVgprTiles(writer)

        preloop  = self._emitLoop(writer, kernel, "PRELOOP", self.preloopSteps)
        mainloop = self._emitLoop(writer, kernel, "MAINLOOP", self.mainloopSteps)

        ngll = Module("NGLL")
        ngll.add(Label("SkipToNGLL", ""))
        ngll.add(self._emitLoop(writer, kernel, "NGLL", self.ngllSteps))

        nll = Module("NLL")
        nll.add(Label("SkipToNLL", ""))
        nll.add(self._emitLoop(writer, kernel, "NLL", self.nllSteps))

        for label, module in [("PRELOOP", preloop), ("MAINLOOP", mainloop),
                              ("NGLL", ngll), ("NLL", nll)]:
            print(f"\n{label}:")
            print(module)
