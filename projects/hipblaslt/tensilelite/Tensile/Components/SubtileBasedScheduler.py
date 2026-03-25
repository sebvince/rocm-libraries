from enum import Enum, auto
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Set, Optional, Union
from Tensile.Components.SubtileBasedKernel import TileInfo
from Tensile.Components.SubtileBasedKernel import emitMfmaInstruction
from Tensile.Components.SubtileBasedKernel import emitSingleDsRead
from Tensile.Components.SubtileBasedKernel import emitSingleBufferLoad
from rocisa.code import Module
from rocisa.instruction import SWaitCnt, SBarrier

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


@dataclass
class WaitOp:
    mtIteration: str  # e.g. "n", "n+1"
    subtileA: List[int]
    subtileB: List[int]
    inflightLoadsA: Optional[int] = None
    inflightLoadsB: Optional[int] = None


@dataclass
class LROp:
    mtIteration: str  # e.g. "n", "n+1", "0"
    subIterK: int
    lrLoadA: Dict[int, int]
    lrLoadB: Dict[int, int]


ScheduleOp = Union[MFMAOp, GROp, WaitOp, LROp]


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


class MFMAScheduler:
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

        # Build preloop steps: GR(MT 0), WAIT(MT 0), LR(MT 0), GR(MT 1)
        preloopOps: List[ScheduleOp] = []
        preloopOps.append(GROp(mtIteration="0",
                               subtileA=allA, subtileB=allB))
        preloopOps.append(WaitOp(mtIteration="0",
                                 subtileA=allA, subtileB=allB,
                                 inflightLoadsA=0, inflightLoadsB=0))
        preloopOps.extend(lrOps)
        preloopOps.append(GROp(mtIteration="1",
                               subtileA=preloadMT1_A, subtileB=preloadMT1_B))
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

            # Insert GROp at the correct subIterK
            if gr.subtileA or gr.subtileB:
                if gr.mtIteration == "n+2":
                    bfSubIterK = 1
                else:
                    hasConflict = bool((gr.subtileA & subIterK0LoadAKeys) or (gr.subtileB & subIterK0LoadBKeys))
                    bfSubIterK = 1 if hasConflict else 0
                pss.subIterKSteps[bfSubIterK].ops.insert(1, GROp(
                    mtIteration=gr.mtIteration,
                    subtileA=sorted(gr.subtileA),
                    subtileB=sorted(gr.subtileB)))

            self.mainloopSteps.append(pss)

            # Release after partition based on strategy
            if self.config.reuseStrategy == VGPRTileReUseStrategy.WITHIN_SUBGROUP:
                for tc, tileIdx, subIterK, shadowKey in self._pendingRemap:
                    vid = self.allocator._allocMap(tc).pop((shadowKey, subIterK))
                    self.allocator._allocMap(tc)[(tileIdx, subIterK)] = vid
                self._pendingRemap = []
            elif self.config.reuseStrategy == VGPRTileReUseStrategy.ACROSS_SUBGROUP:
                self._releaseUnusedAfterPartition(pi)

        self._insertWaitsAndDeps(numPartitions)
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
        """ACROSS_SUBGROUP: release tiles not appearing in any future partition."""
        currentPartition = self.partitions[partitionIdx]

        futureA: Set[int] = set()
        futureB: Set[int] = set()
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

    def _insertWaitsAndDeps(self, numPartitions: int):
        """Insert WAIT ops into mainloop ops."""
        # Build ordered GR events for inflight counting
        grEvents = []
        si = 0
        for pss in self.mainloopSteps:
            for dus in pss.subIterKSteps:
                grOp = next((op for op in dus.ops if isinstance(op, GROp)), None)
                if grOp:
                    grEvents.append((si, pss.partitionId, dus.subIterK,
                                     len(grOp.subtileA) + len(grOp.subtileB),
                                     set(grOp.subtileA), set(grOp.subtileB)))
                si += 1

        def _countInflightGR(waitStepIndex, waitA, waitB):
            sourceGRIdx = None
            for evi, (si, gi, sik, cnt, grA, grB) in enumerate(grEvents):
                if (waitA and waitA <= grA) or (waitB and waitB <= grB):
                    sourceGRIdx = evi
                    break
            if sourceGRIdx is None:
                return None, None
            totalA = 0
            totalB = 0
            sourceStepIdx = grEvents[sourceGRIdx][0]
            for (si, gi, sik, cnt, grA, grB) in grEvents:
                if si >= sourceStepIdx or si < waitStepIndex:
                    totalA += len(grA)
                    totalB += len(grB)
            return totalA, totalB

        # Insert WAIT ops
        pendingA = set()
        pendingB = set()
        si = 0
        for pss in self.mainloopSteps:
            gr = self.partitionGRs[pss.partitionId]
            pendingA |= gr.subtileA
            pendingB |= gr.subtileB

            for dus in pss.subIterKSteps:
                lrOp = dus.ops[-1]
                assert isinstance(lrOp, LROp)

                waitA = set()
                waitB = set()
                if lrOp.subIterK == 0:
                    waitA = set(lrOp.lrLoadA.keys()) & pendingA
                    waitB = set(lrOp.lrLoadB.keys()) & pendingB
                if waitA or waitB:
                    inflightCountA, inflightCountB = _countInflightGR(si, waitA, waitB)
                    dus.ops.insert(-1, WaitOp(
                        mtIteration=lrOp.mtIteration,
                        subtileA=sorted(waitA), subtileB=sorted(waitB),
                        inflightLoadsA=inflightCountA, inflightLoadsB=inflightCountB))
                    pendingA -= waitA
                    pendingB -= waitB

                si += 1


    def _buildNGLL(self) -> List[PartitionSchedule]:
        """NGLL (Non Global Load Loop): mainloop without GR ops."""
        ngll = []
        for pss in self.mainloopSteps:
            newPss = PartitionSchedule(partitionId=pss.partitionId)
            for dus in pss.subIterKSteps:
                newDus = SubIterKSchedule(subIterK=dus.subIterK, conflict=dus.conflict)
                newDus.ops = [op for op in dus.ops if not isinstance(op, GROp)]
                newPss.subIterKSteps.append(newDus)
            ngll.append(newPss)
        return ngll

    def _buildNLL(self) -> List[PartitionSchedule]:
        """NLL (Non Load Loop): mainloop without GR, LR(n+1), and their associated WAITs."""
        nll = []
        for pss in self.mainloopSteps:
            newPss = PartitionSchedule(partitionId=pss.partitionId)
            for dus in pss.subIterKSteps:
                newDus = SubIterKSchedule(subIterK=dus.subIterK, conflict=dus.conflict)
                newDus.ops = [op for op in dus.ops
                              if not isinstance(op, GROp)
                              and not (isinstance(op, LROp) and op.mtIteration == "n+1")
                              and not (isinstance(op, WaitOp) and op.mtIteration == "n+1")]
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
        elif isinstance(op, WaitOp):
            inflight = f" — inflight GRs A={op.inflightLoadsA} B={op.inflightLoadsB}" if op.inflightLoadsA is not None else ""
            print(f"{indent}WAIT (MT {op.mtIteration}) A: {op.subtileA}  B: {op.subtileB}{inflight}")
        elif isinstance(op, LROp):
            sikLabel = f", subIterK {op.subIterK}" if op.subIterK >= 0 else ""
            print(f"{indent}LR (MT {op.mtIteration}{sikLabel}) A: {op.lrLoadA}  B: {op.lrLoadB}")

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

    def emitWaitLR(self, inflightLoadsA, inflightLoadsB):
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
        module.add(SBarrier(comment=""))
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

    def _emitLoop(self, writer, kernel, label, steps):
        """Emit a loop module (mainloop, NGLL, or NLL).

        Emits ops in the order they appear in each SubIterKSchedule,
        preserving the schedule's intended instruction ordering.
        """
        dtileInfo = writer.states.d.tileInfo
        module = Module(label)
        for pss in steps:
            for dus in pss.subIterKSteps:
                module.addComment0(f"Partition {pss.partitionId}: subIterK={dus.subIterK}")
                hasLR = False
                for op in dus.ops:
                    if isinstance(op, GROp):
                        oneStep = [PartitionSchedule(
                            partitionId=pss.partitionId,
                            subIterKSteps=[SubIterKSchedule(subIterK=dus.subIterK, ops=[op])])]
                        module.add(self.emitGRs(writer, kernel, oneStep))
                    elif isinstance(op, MFMAOp):
                        oneStep = [PartitionSchedule(
                            partitionId=pss.partitionId,
                            subIterKSteps=[SubIterKSchedule(subIterK=dus.subIterK, ops=[op])])]
                        module.add(self.emitMFMAs(writer, kernel, oneStep, dtileInfo))
                    elif isinstance(op, WaitOp):
                        module.add(self.emitWaitLR(op.inflightLoadsA, op.inflightLoadsB))
                    elif isinstance(op, LROp):
                        oneStep = [PartitionSchedule(
                            partitionId=pss.partitionId,
                            subIterKSteps=[SubIterKSchedule(subIterK=dus.subIterK, ops=[op])])]
                        module.add(self.emitLRs(writer, kernel, oneStep))
                        module.add(SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for all subtile LRs to complete"))
                        hasLR = True
                if not hasLR:
                    module.add(SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1, comment="Wait for all subtile LRs to complete"))
        return module

    def generateCode(self, writer, kernel):
        self.allocVgprTiles(writer)

        preloop  = self._emitLoop(writer, kernel, "PRELOOP", self.preloopSteps)
        mainloop = self._emitLoop(writer, kernel, "MAINLOOP", self.mainloopSteps)
        ngll     = self._emitLoop(writer, kernel, "NGLL", self.ngllSteps)
        nll      = self._emitLoop(writer, kernel, "NLL", self.nllSteps)

        for label, module in [("PRELOOP", preloop), ("MAINLOOP", mainloop),
                              ("NGLL", ngll), ("NLL", nll)]:
            print(f"\n{label}:")
            print(module)


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    from types import SimpleNamespace
    from unittest.mock import MagicMock
    from Tensile.Components.SubtileBasedKernel import TileInfo
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

    dtype = _mock_dtype(2)
    problemType = {
        "DataTypeA": dtype,
        "DataTypeB": dtype,
        "ComputeDataType": _mock_dtype(4),
    }
    kernel = {
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

    tiA = TileInfo('A', kernel)
    tiB = TileInfo('B', kernel)
    lsgA = tiA.localSubtileGrid[0]
    lsgB = tiB.localSubtileGrid[0]

    configs = [
        (f"lsg {lsgA}x{lsgB}, group {lsgA}x{lsgB}, HALF_PREFETCH, ACROSS_SUBGROUP, COLUMN_MAJOR",
            SchedulerConfig(lsgA, lsgB, PrefetchMode.HALF_PREFETCH, VGPRTileReUseStrategy.ACROSS_SUBGROUP, SubgroupOrdering.COLUMN_MAJOR)),

        #  (f"lsg {lsgA}x{lsgB}, group 4x4, HALF_PREFETCH, ACROSS_SUBGROUP, COLUMN_MAJOR",
        #  SchedulerConfig(4, 4, PrefetchMode.HALF_PREFETCH, VGPRTileReUseStrategy.ACROSS_SUBGROUP)),

    ]

    for name, cfg in configs:
        print(f"=== {name} ===")
        s = MFMAScheduler(tiA, tiB, cfg)
        s.printSchedule()
        writer = create_mock_writer(kernel)
        tiA.allocOffsetRegisters(writer, kernel)
        tiB.allocOffsetRegisters(writer, kernel)
        s.generateCode(writer, kernel)
        print()
