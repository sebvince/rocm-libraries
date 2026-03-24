from enum import Enum, auto
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Set, Optional, Union


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



# Key type for allocator: (subtileIdx, duIdx)
AllocKey = Tuple[int, int]


class VGPRTileAllocator:
    """Maps (subtileIdx, duIdx) to shared integer VGPR tile IDs, with free-list reuse."""

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

    def allocate(self, tc: str, subtileIdx: int, duIdx: int) -> int:
        key = (subtileIdx, duIdx)
        if self._freeList:
            vid = self._freeList.pop(0)
        else:
            vid = self._nextId
            self._nextId += 1
        self._allocMap(tc)[key] = vid
        self._updatePeak()
        return vid

    def release(self, tc: str, subtileIdx: int, duIdx: int) -> None:
        key = (subtileIdx, duIdx)
        vid = self._allocMap(tc).pop(key)
        self._freeList.append(vid)

    def isAllocated(self, tc: str, subtileIdx: int, duIdx: int) -> bool:
        return (subtileIdx, duIdx) in self._allocMap(tc)

    def getVGPRTileId(self, tc: str, subtileIdx: int, duIdx: int) -> int:
        return self._allocMap(tc)[(subtileIdx, duIdx)]

    def releaseAllForTile(self, tc: str, subtileIdx: int) -> None:
        """Release all DU allocations for a given subtile index."""
        allocMap = self._allocMap(tc)
        keys = [k for k in allocMap if k[0] == subtileIdx]
        for k in keys:
            vid = allocMap.pop(k)
            self._freeList.append(vid)

    @property
    def totalVGPRs(self) -> int:
        return self._peak



@dataclass
class MFMAOp:
    mtIteration: str  # e.g. "n"
    duIndex: int
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
    inflightGRCount: Optional[int] = None


@dataclass
class LROp:
    mtIteration: str  # e.g. "n", "n+1", "0"
    duIndex: int
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
class DUSchedule:
    """Ops for one DU iteration within a partition."""
    duIndex: int
    ops: List[ScheduleOp] = field(default_factory=list)
    conflict: Set[int] = field(default_factory=set)


@dataclass
class PartitionSchedule:
    """Schedule for one subtile partition — contains all DU iterations."""
    partitionId: int
    duSteps: List[DUSchedule] = field(default_factory=list)


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

        self.numDU = tileInfoA.subtileShape[1]
        assert self.numDU == tileInfoB.subtileShape[1], \
            "A and B must have same subtileShape[1]"

        self.partitions: List[Partition] = self._buildPartitions()
        self.allocator = VGPRTileAllocator()
        self.hasDuplicatedReads: bool = False
        self.needsUnrolling: bool = False

        self._runSchedule()

    # ── Outputs ──────────────────────────────────────────────

    @property
    def totalVGPRs(self) -> int:
        return self.allocator.totalVGPRs

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

        # Number of DUs to preload LR for
        if self.config.prefetchMode == PrefetchMode.HALF_PREFETCH:
            numPreloadDUs = 1
        elif self.config.prefetchMode == PrefetchMode.FULL_PREFETCH:
            numPreloadDUs = self.numDU

        # Allocate VGPRs for first group and build LR maps
        lrOps = []
        for du in range(numPreloadDUs):
            lrLoadA = {}
            lrLoadB = {}
            for tA in first.tileAIndices:
                lrLoadA[tA] = self.allocator.allocate('A', tA, du)
            for tB in first.tileBIndices:
                lrLoadB[tB] = self.allocator.allocate('B', tB, du)
            lrOps.append(LROp(mtIteration="0", duIndex=du,
                              lrLoadA=lrLoadA, lrLoadB=lrLoadB))

        # Build preloop ops: GR(MT 0), GR(MT 1), then LR(MT 0) per DU
        self.preloopOps: List[ScheduleOp] = []
        self.preloopOps.append(GROp(mtIteration="0",
                                    subtileA=allA, subtileB=allB))
        self.preloopOps.append(GROp(mtIteration="1",
                                    subtileA=preloadMT1_A, subtileB=preloadMT1_B))
        self.preloopOps.extend(lrOps)

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
            du0LoadAKeys: Set[int] = set()
            du0LoadBKeys: Set[int] = set()

            for du in range(self.numDU):
                # USE: current group's tiles at current DU
                # MFMA: map subtile indices to VGPR tile IDs
                vgprTileMapA = {}
                vgprTileMapB = {}
                for tA in partition.tileAIndices:
                    vgprTileMapA[tA] = self.allocator.getVGPRTileId('A', tA, du)
                for tB in partition.tileBIndices:
                    vgprTileMapB[tB] = self.allocator.getVGPRTileId('B', tB, du)

                # LOAD: determined by prefetch mode
                loadATiles, loadBTiles, loadDU = self._getLoadTargets(pi, du, numPartitions)
                isWrapAround = self._isWrapAroundLoad(pi, du, numPartitions)
                curA = set(partition.tileAIndices)
                curB = set(partition.tileBIndices)
                if du == 0:
                    self._pendingRemap = []

                lrLoadA = {}
                lrLoadB = {}
                if loadATiles is not None:
                    for tA in loadATiles:
                        vid = self._loadTile('A', tA, loadDU, isWrapAround, curA)
                        if vid is not None:
                            lrLoadA[tA] = vid

                if loadBTiles is not None:
                    for tB in loadBTiles:
                        vid = self._loadTile('B', tB, loadDU, isWrapAround, curB)
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

                # Build DUSchedule with MFMA and LR ops
                dus = DUSchedule(duIndex=du)
                mfmas = [(a, b) for a in sorted(vgprTileMapA.keys()) for b in sorted(vgprTileMapB.keys())]
                mtLoad = "n+1" if isWrapAround else "n"
                dus.ops.append(MFMAOp(mtIteration="n", duIndex=du,
                                      subtiles=mfmas,
                                      vgprTileMapA=vgprTileMapA, vgprTileMapB=vgprTileMapB))
                dus.ops.append(LROp(mtIteration=mtLoad, duIndex=loadDU,
                                    lrLoadA=lrLoadA, lrLoadB=lrLoadB))
                dus.conflict = conflict
                pss.duSteps.append(dus)

                # save subtiles for DU=0 to check where to insert GR(n+2)
                if du == 0:
                    du0LoadAKeys = set(lrLoadA.keys())
                    du0LoadBKeys = set(lrLoadB.keys())

                # WITHIN_SUBGROUP: release current DU's MFMA tiles for K-dim reuse
                if self.config.reuseStrategy == VGPRTileReUseStrategy.WITHIN_SUBGROUP:
                    for tA in vgprTileMapA:
                        if self.allocator.isAllocated('A', tA, du):
                            self.allocator.release('A', tA, du)
                    for tB in vgprTileMapB:
                        if self.allocator.isAllocated('B', tB, du):
                            self.allocator.release('B', tB, du)

            # Insert GROp at the correct DU
            if gr.subtileA or gr.subtileB:
                if gr.mtIteration == "n+2":
                    bfDU = 1
                else:
                    hasConflict = bool((gr.subtileA & du0LoadAKeys) or (gr.subtileB & du0LoadBKeys))
                    bfDU = 1 if hasConflict else 0
                pss.duSteps[bfDU].ops.insert(1, GROp(
                    mtIteration=gr.mtIteration,
                    subtileA=sorted(gr.subtileA),
                    subtileB=sorted(gr.subtileB)))

            self.mainloopSteps.append(pss)

            # Release after partition based on strategy
            if self.config.reuseStrategy == VGPRTileReUseStrategy.WITHIN_SUBGROUP:
                for tc, tileIdx, duIdx, shadowKey in self._pendingRemap:
                    vid = self.allocator._allocMap(tc).pop((shadowKey, duIdx))
                    self.allocator._allocMap(tc)[(tileIdx, duIdx)] = vid
                self._pendingRemap = []
            elif self.config.reuseStrategy == VGPRTileReUseStrategy.ACROSS_SUBGROUP:
                self._releaseUnusedAfterPartition(pi)

        self._insertWaitsAndDeps(numPartitions)
        self.ngllSteps = self._buildNGLL()
        self.nllSteps = self._buildNLL()
        self._checkDuplicatedReads()

    def _loadTile(self, tc: str, tileIdx: int, loadDU: int,
                  isWrapAround: bool,
                  currentPartitionTiles: Set[int]) -> Optional[int]:
        """Determine the VGPRTile ID for a load. Returns None if no load needed."""
        allocated = self.allocator.isAllocated(tc, tileIdx, loadDU)

        if isWrapAround and allocated:
            # Wrap-around: reuse partition 0's existing VGPRTile IDs
            return self.allocator.getVGPRTileId(tc, tileIdx, loadDU)

        if not allocated:
            # Fresh allocation
            return self.allocator.allocate(tc, tileIdx, loadDU)

        if self.config.reuseStrategy == VGPRTileReUseStrategy.WITHIN_SUBGROUP \
                and tileIdx in currentPartitionTiles:
            # Tile is allocated by the current partition but will be released after it.
            # Must allocate a new VGPR for the next partition's data.
            # Use a shadow key to avoid overwriting the current allocation.
            shadowKey = -(tileIdx + 1)  # negative to avoid collision
            vid = self.allocator.allocate(tc, shadowKey, loadDU)
            # Store the real tileIdx mapping for later fixup
            self._pendingRemap.append((tc, tileIdx, loadDU, shadowKey))
            return vid

        # NONE / ACROSS_SUBGROUP: tile stays alive, reuse in place
        return None

    def _isWrapAroundLoad(self, partitionIdx: int, duIdx: int, numPartitions: int) -> bool:
        """True when this step's load targets partition 0 for the next macrotile iteration."""
        if self.config.prefetchMode == PrefetchMode.HALF_PREFETCH:
            return partitionIdx == numPartitions - 1 and duIdx == self.numDU - 1
        elif self.config.prefetchMode == PrefetchMode.FULL_PREFETCH:
            return partitionIdx == numPartitions - 1
        return False

    # ── Prefetch modes ───────────────────────────────────────

    def _getLoadTargets(self, partitionIdx: int, duIdx: int,
                        numPartitions: int) -> Tuple[Optional[List[int]], Optional[List[int]], int]:
        """Returns (loadATiles, loadBTiles, targetDU)."""
        if self.config.prefetchMode == PrefetchMode.HALF_PREFETCH:
            return self._loadTargetsHalfPrefetch(partitionIdx, duIdx, numPartitions)
        elif self.config.prefetchMode == PrefetchMode.FULL_PREFETCH:
            return self._loadTargetsFullPrefetch(partitionIdx, duIdx, numPartitions)
        return (None, None, 0)

    def _loadTargetsHalfPrefetch(self, partitionIdx, duIdx, numPartitions):
        """HALF: DU=0 loads same-partition DU=1, DU=last loads next-partition DU=0.
        Last partition wraps around to partition 0 (next iteration)."""
        currentPartition = self.partitions[partitionIdx]
        if duIdx < self.numDU - 1:
            targetDU = duIdx + 1
            return (currentPartition.tileAIndices, currentPartition.tileBIndices, targetDU)
        else:
            nextPartition = self.partitions[(partitionIdx + 1) % numPartitions]
            return (nextPartition.tileAIndices, nextPartition.tileBIndices, 0)

    def _loadTargetsFullPrefetch(self, partitionIdx, duIdx, numPartitions):
        """FULL: DU=0 loads next-partition DU=0, DU=1 loads next-partition DU=1.
        Last partition wraps around to partition 0 (next iteration)."""
        nextPartition = self.partitions[(partitionIdx + 1) % numPartitions]
        return (nextPartition.tileAIndices, nextPartition.tileBIndices, duIdx)

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
        """WITHIN_SUBGROUP: release all tiles (all DUs) of this partition."""
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
            for dus in pss.duSteps:
                grOp = next((op for op in dus.ops if isinstance(op, GROp)), None)
                if grOp:
                    grEvents.append((si, pss.partitionId, dus.duIndex,
                                     len(grOp.subtileA) + len(grOp.subtileB),
                                     set(grOp.subtileA), set(grOp.subtileB)))
                si += 1

        def _countInflightGR(waitStepIndex, waitA, waitB):
            sourceGRIdx = None
            for evi, (si, gi, du, cnt, grA, grB) in enumerate(grEvents):
                if (waitA and waitA <= grA) or (waitB and waitB <= grB):
                    sourceGRIdx = evi
                    break
            if sourceGRIdx is None:
                return None
            total = 0
            sourceStepIdx = grEvents[sourceGRIdx][0]
            for (si, gi, du, cnt, grA, grB) in grEvents:
                if si >= sourceStepIdx or si < waitStepIndex:
                    total += cnt
            return total

        # Insert WAIT ops
        pendingA = set()
        pendingB = set()
        si = 0
        for pss in self.mainloopSteps:
            gr = self.partitionGRs[pss.partitionId]
            pendingA |= gr.subtileA
            pendingB |= gr.subtileB

            for dus in pss.duSteps:
                lrOp = dus.ops[-1]
                assert isinstance(lrOp, LROp)

                waitA = set()
                waitB = set()
                if lrOp.duIndex == 0:
                    waitA = set(lrOp.lrLoadA.keys()) & pendingA
                    waitB = set(lrOp.lrLoadB.keys()) & pendingB
                if waitA or waitB:
                    inflightCount = _countInflightGR(si, waitA, waitB)
                    dus.ops.insert(-1, WaitOp(
                        mtIteration=lrOp.mtIteration,
                        subtileA=sorted(waitA), subtileB=sorted(waitB),
                        inflightGRCount=inflightCount))
                    pendingA -= waitA
                    pendingB -= waitB

                si += 1


    def _buildNGLL(self) -> List[PartitionSchedule]:
        """NGLL (Non Global Load Loop): mainloop without GR ops."""
        ngll = []
        for pss in self.mainloopSteps:
            newPss = PartitionSchedule(partitionId=pss.partitionId)
            for dus in pss.duSteps:
                newDus = DUSchedule(duIndex=dus.duIndex, conflict=dus.conflict)
                newDus.ops = [op for op in dus.ops if not isinstance(op, GROp)]
                newPss.duSteps.append(newDus)
            ngll.append(newPss)
        return ngll

    def _buildNLL(self) -> List[PartitionSchedule]:
        """NLL (Non Load Loop): mainloop without GR, LR(n+1), and their associated WAITs."""
        nll = []
        for pss in self.mainloopSteps:
            newPss = PartitionSchedule(partitionId=pss.partitionId)
            for dus in pss.duSteps:
                newDus = DUSchedule(duIndex=dus.duIndex, conflict=dus.conflict)
                newDus.ops = [op for op in dus.ops
                              if not isinstance(op, GROp)
                              and not (isinstance(op, LROp) and op.mtIteration == "n+1")
                              and not (isinstance(op, WaitOp) and op.mtIteration == "n+1")]
                newPss.duSteps.append(newDus)
            nll.append(newPss)
        return nll

    def _checkDuplicatedReads(self):
        """Detect if any (subtile, DU) pair is loaded more than once."""
        seenA: Dict[AllocKey, int] = {}
        seenB: Dict[AllocKey, int] = {}
        # Count preloop LR loads
        for op in self.preloopOps:
            if isinstance(op, LROp):
                for tA in op.lrLoadA:
                    key = (tA, op.duIndex)
                    seenA[key] = seenA.get(key, 0) + 1
                for tB in op.lrLoadB:
                    key = (tB, op.duIndex)
                    seenB[key] = seenB.get(key, 0) + 1
        # Count mainloop LR loads (skip wrap-around which reuses existing allocations)
        for pss in self.mainloopSteps:
            for dus in pss.duSteps:
                for op in dus.ops:
                    if isinstance(op, LROp) and op.mtIteration != "n+1":
                        for tA in op.lrLoadA:
                            key = (tA, op.duIndex)
                            seenA[key] = seenA.get(key, 0) + 1
                        for tB in op.lrLoadB:
                            key = (tB, op.duIndex)
                            seenB[key] = seenB.get(key, 0) + 1
        self.hasDuplicatedReads = (
            any(c > 1 for c in seenA.values()) or
            any(c > 1 for c in seenB.values())
        )

    # ── Debug ────────────────────────────────────────────────

    @staticmethod
    def _printOp(op: ScheduleOp, indent: str = ""):
        if isinstance(op, MFMAOp):
            print(f"{indent}MFMAs (MT {op.mtIteration}, DU {op.duIndex}):")
            print(f"{indent}  - {op.subtiles}")
            print(f"{indent}  - USING  A: {op.vgprTileMapA}  B: {op.vgprTileMapB}")
        elif isinstance(op, GROp):
            print(f"{indent}GR (MT {op.mtIteration}):  A: {op.subtileA}  B: {op.subtileB}")
        elif isinstance(op, WaitOp):
            inflight = f" — {op.inflightGRCount} inflight GRs" if op.inflightGRCount is not None else ""
            print(f"{indent}WAIT (MT {op.mtIteration}) A: {op.subtileA}  B: {op.subtileB}{inflight}")
        elif isinstance(op, LROp):
            duLabel = f", DU {op.duIndex}" if op.duIndex >= 0 else ""
            print(f"{indent}LR (MT {op.mtIteration}{duLabel}) A: {op.lrLoadA}  B: {op.lrLoadB}")

    def printSchedule(self):
        print(f"SubtileGridA={self.MTA}, SubtileGridB={self.MTB}")
        print(f"Partition grid: {self.numPartitionsA} x {self.numPartitionsB}")
        print(f"Partition size: {self.config.partitionSizeA} x {self.config.partitionSizeB}")
        print(f"Prefetch: {self.config.prefetchMode.name}")
        print(f"Reuse: {self.config.reuseStrategy.name}")
        print(f"hasDuplicatedReads: {self.hasDuplicatedReads}")
        print(f"needsUnrolling: {self.needsUnrolling}")
        print(f"totalVGPRTiles: {self.totalVGPRs} ({self.totalVGPRs * 4} VGPRs)")
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
        for op in self.preloopOps:
            self._printOp(op, indent="  ")
        print()

        print("MAINLOOP:")
        for partition in self.mainloopSteps:
            print(f"  Partition {partition.partitionId}:")
            for dus in partition.duSteps:
                print(f"    DU={dus.duIndex}:")
                for op in dus.ops:
                    self._printOp(op, indent="      ")
                if dus.conflict:
                    print(f"      *** CONFLICT: USE/LOAD share VGPRTile IDs {dus.conflict} — needs unrolling ***")

        print()
        print("NGLL (No Global Load Loop):")
        for partition in self.ngllSteps:
            print(f"  Partition {partition.partitionId}:")
            for dus in partition.duSteps:
                print(f"    DU={dus.duIndex}:")
                for op in dus.ops:
                    self._printOp(op, indent="      ")

        print()
        print("NLL (No Load Loop):")
        for partition in self.nllSteps:
            print(f"  Partition {partition.partitionId}:")
            for dus in partition.duSteps:
                print(f"    DU={dus.duIndex}:")
                for op in dus.ops:
                    self._printOp(op, indent="      ")


if __name__ == "__main__":
    class MockTileInfo:
        def __init__(self, localSubtileGrid, subtileShape):
            self.localSubtileGrid = localSubtileGrid
            self.subtileShape = subtileShape

    # MTA=256, MTB=256 -> localSubtileGrid[0] = 8 for each
    lsgA, lsgB = 8, 8

    configs = [
         (f"lsg {lsgA}x{lsgB}, group 4x4, HALF_PREFETCH, ACROSS_SUBGROUP, COLUMN_MAJOR",
         MockTileInfo([lsgA, 1], [1, 2]), MockTileInfo([lsgB, 1], [1, 2]),
         SchedulerConfig(4, 4, PrefetchMode.HALF_PREFETCH, VGPRTileReUseStrategy.ACROSS_SUBGROUP)),

        (f"lsg {lsgA}x{lsgB}, group 4x4, HALF_PREFETCH, WITHIN_SUBGROUP, COLUMN_MAJOR",
            MockTileInfo([lsgA, 1], [1, 2]), MockTileInfo([lsgB, 1], [1, 2]),
            SchedulerConfig(8, 8, PrefetchMode.HALF_PREFETCH, VGPRTileReUseStrategy.ACROSS_SUBGROUP, SubgroupOrdering.COLUMN_MAJOR)),

        #  (f"lsg {lsgA}x{lsgB}, group 4x4, HALF_PREFETCH, WITHIN_SUBGROUP, COLUMN_MAJOR",
        #  MockTileInfo([lsgA, 1], [1, 2]), MockTileInfo([lsgB, 1], [1, 2]),
        #  SchedulerConfig(4, 4, PrefetchMode.HALF_PREFETCH, VGPRTileReUseStrategy.WITHIN_SUBGROUP, SubgroupOrdering.COLUMN_MAJOR)),


        # Needs unrolling.
        #  (f"lsg {lsgA}x{lsgB}, group 4x4, FULL_PREFETCH, NONE, COLUMN_MAJOR",
        #  MockTileInfo([lsgA, 1], [1, 2]), MockTileInfo([lsgB, 1], [1, 2]),
        #  SchedulerConfig(8, 8, PrefetchMode.FULL_PREFETCH, VGPRTileReUseStrategy.NONE, SubgroupOrdering.COLUMN_MAJOR)),

    #       (f"lsg {lsgA}x{lsgB}, group 4x4, HALF_PREFETCH, NONE, COLUMN_MAJOR",
    #      MockTileInfo([lsgA, 1], [1, 2]), MockTileInfo([lsgB, 1], [1, 2]),
    #      SchedulerConfig(8, 8, PrefetchMode.HALF_PREFETCH, VGPRTileReUseStrategy.NONE, SubgroupOrdering.COLUMN_MAJOR)),

    #     (f"lsg {lsgA}x{lsgB}, group 4x4, HALF_PREFETCH, ACROSS_SUBGROUP, COLUMN_MAJOR",
    #         MockTileInfo([lsgA, 1], [1, 2]), MockTileInfo([lsgB, 1], [1, 2]),
    #         SchedulerConfig(8, 8, PrefetchMode.HALF_PREFETCH, VGPRTileReUseStrategy.ACROSS_SUBGROUP, SubgroupOrdering.COLUMN_MAJOR)),
        
    #     (f"lsg {lsgA}x{lsgB}, group 4x4, HALF_PREFETCH, WITHIN_SUBGROUP, COLUMN_MAJOR",
    #         MockTileInfo([lsgA, 1], [1, 2]), MockTileInfo([lsgB, 1], [1, 2]),
    #         SchedulerConfig(8, 8, PrefetchMode.HALF_PREFETCH, VGPRTileReUseStrategy.WITHIN_SUBGROUP, SubgroupOrdering.COLUMN_MAJOR)),


    #     (f"lsg {lsgA}x{lsgB}, group 4x4, HALF_PREFETCH, WITHIN_SUBGROUP, COLUMN_MAJOR",
    #         MockTileInfo([10, 1], [1, 2]), MockTileInfo([10, 1], [1, 2]),
    #         SchedulerConfig(2, 10, PrefetchMode.HALF_PREFETCH, VGPRTileReUseStrategy.WITHIN_SUBGROUP, SubgroupOrdering.COLUMN_MAJOR)),


    ]

    for name, tiA, tiB, cfg in configs:
        print(f"=== {name} ===")
        s = MFMAScheduler(tiA, tiB, cfg)
        s.printSchedule()
        print()
