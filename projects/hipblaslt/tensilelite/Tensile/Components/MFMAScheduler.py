from enum import Enum, auto
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Set, Optional


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
    subtileGroupSizeA: int
    subtileGroupSizeB: int
    prefetchMode: PrefetchMode
    reuseStrategy: VGPRTileReUseStrategy
    ordering: SubgroupOrdering = SubgroupOrdering.SNAKE_COLUMN_MAJOR


@dataclass
class SubtileGroup:
    """A rectangle (sizeA x sizeB) of subtiles processed together."""
    groupId: int
    sizeA: int
    sizeB: int
    subtiles: List[Tuple[int, int]] = field(default_factory=list)

    @property
    def tileAIndices(self) -> List[int]:
        return sorted(set(t[0] for t in self.subtiles))

    @property
    def tileBIndices(self) -> List[int]:
        return sorted(set(t[1] for t in self.subtiles))



# Key type for allocator: (tileIdx, duIdx)
AllocKey = Tuple[int, int]


class VGPRAllocator:
    """Maps (tileIdx, duIdx) to shared integer VGPR tile IDs, with free-list reuse."""

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

    def allocate(self, tc: str, tileIdx: int, duIdx: int) -> int:
        key = (tileIdx, duIdx)
        if self._freeList:
            vid = self._freeList.pop(0)
        else:
            vid = self._nextId
            self._nextId += 1
        self._allocMap(tc)[key] = vid
        self._updatePeak()
        return vid

    def release(self, tc: str, tileIdx: int, duIdx: int) -> None:
        key = (tileIdx, duIdx)
        vid = self._allocMap(tc).pop(key)
        self._freeList.append(vid)

    def isAllocated(self, tc: str, tileIdx: int, duIdx: int) -> bool:
        return (tileIdx, duIdx) in self._allocMap(tc)

    def getVGPRTileId(self, tc: str, tileIdx: int, duIdx: int) -> int:
        return self._allocMap(tc)[(tileIdx, duIdx)]

    def releaseAllForTile(self, tc: str, tileIdx: int) -> None:
        """Release all DU allocations for a given tile index."""
        allocMap = self._allocMap(tc)
        keys = [k for k in allocMap if k[0] == tileIdx]
        for k in keys:
            vid = allocMap.pop(k)
            self._freeList.append(vid)

    @property
    def totalVGPRs(self) -> int:
        return self._peak



@dataclass
class ScheduleStep:
    """One DU iteration within one group."""
    groupId: int
    duIndex: int
    useA: Dict[int, int] = field(default_factory=dict)
    useB: Dict[int, int] = field(default_factory=dict)
    loadA: Dict[int, int] = field(default_factory=dict)
    loadB: Dict[int, int] = field(default_factory=dict)
    conflict: Set[int] = field(default_factory=set)
    isWrapLoad: bool = False
    loadDU: int = -1


class MFMAScheduler:
    def __init__(self, tileInfoA, tileInfoB, config: SchedulerConfig):
        self.tileInfoA = tileInfoA
        self.tileInfoB = tileInfoB
        self.config = config

        self.MTA = tileInfoA.localSubtileGrid[0]
        self.MTB = tileInfoB.localSubtileGrid[0]

        assert self.MTA % config.subtileGroupSizeA == 0, \
            f"MTA ({self.MTA}) must be divisible by subtileGroupSizeA ({config.subtileGroupSizeA})"
        assert self.MTB % config.subtileGroupSizeB == 0, \
            f"MTB ({self.MTB}) must be divisible by subtileGroupSizeB ({config.subtileGroupSizeB})"

        self.numGroupsA = self.MTA // config.subtileGroupSizeA
        self.numGroupsB = self.MTB // config.subtileGroupSizeB

        self.numDU = tileInfoA.subtileShape[1]
        assert self.numDU == tileInfoB.subtileShape[1], \
            "A and B must have same subtileShape[1]"

        self.groups: List[SubtileGroup] = self._buildGroups()
        self.allocator = VGPRAllocator()
        self._schedule: List[ScheduleStep] = []
        self.hasDuplicatedReads: bool = False
        self.needsUnrolling: bool = False

        self._runSchedule()

    # ── Outputs ──────────────────────────────────────────────

    @property
    def totalVGPRs(self) -> int:
        return self.allocator.totalVGPRs

    @property
    def schedule(self) -> List[ScheduleStep]:
        return self._schedule

    # ── Group construction ───────────────────────────────────

    def _generateOrder(self) -> List[Tuple[int, int]]:
        order = []
        if self.config.ordering == SubgroupOrdering.COLUMN_MAJOR:
            for col in range(self.numGroupsB):
                for row in range(self.numGroupsA):
                    order.append((row, col))
        elif self.config.ordering == SubgroupOrdering.SNAKE_COLUMN_MAJOR:
            for col in range(self.numGroupsB):
                if col % 2 == 0:
                    for row in range(self.numGroupsA):
                        order.append((row, col))
                else:
                    for row in range(self.numGroupsA - 1, -1, -1):
                        order.append((row, col))
        return order

    def _buildGroups(self) -> List[SubtileGroup]:
        order = self._generateOrder()
        sA = self.config.subtileGroupSizeA
        sB = self.config.subtileGroupSizeB
        groups = []
        for groupId, (gA, gB) in enumerate(order):
            subtiles = []
            for a in range(gA * sA, (gA + 1) * sA):
                for b in range(gB * sB, (gB + 1) * sB):
                    subtiles.append((a, b))
            groups.append(SubtileGroup(groupId=groupId, sizeA=sA, sizeB=sB, subtiles=subtiles))
        return groups

    # ── Scheduling core ──────────────────────────────────────

    def _bootstrapFirstGroup(self):
        """Allocate VGPRTile IDs for the first group before the main loop."""
        first = self.groups[0]
        if self.config.prefetchMode == PrefetchMode.HALF_PREFETCH:
            for tA in first.tileAIndices:
                self.allocator.allocate('A', tA, 0)
            for tB in first.tileBIndices:
                self.allocator.allocate('B', tB, 0)
        elif self.config.prefetchMode == PrefetchMode.FULL_PREFETCH:
            for du in range(self.numDU):
                for tA in first.tileAIndices:
                    self.allocator.allocate('A', tA, du)
                for tB in first.tileBIndices:
                    self.allocator.allocate('B', tB, du)

    def _runSchedule(self):
        if self.config.prefetchMode == PrefetchMode.NO:
            raise NotImplementedError("PrefetchMode.NO is not yet supported")

        self._bootstrapFirstGroup()

        loadCountA: Dict[AllocKey, int] = {}
        loadCountB: Dict[AllocKey, int] = {}
        numGroups = len(self.groups)

        for gi, group in enumerate(self.groups):
            for du in range(self.numDU):
                step = ScheduleStep(groupId=group.groupId, duIndex=du)

                # USE: current group's tiles at current DU
                for tA in group.tileAIndices:
                    step.useA[tA] = self.allocator.getVGPRTileId('A', tA, du)
                for tB in group.tileBIndices:
                    step.useB[tB] = self.allocator.getVGPRTileId('B', tB, du)

                # LOAD: determined by prefetch mode
                loadATiles, loadBTiles, loadDU = self._getLoadTargets(gi, du, numGroups)
                isWrapAround = self._isWrapAroundLoad(gi, du, numGroups)
                step.isWrapLoad = isWrapAround
                step.loadDU = loadDU
                curA = set(group.tileAIndices)
                curB = set(group.tileBIndices)
                if du == 0:
                    self._pendingRemap = []

                if loadATiles is not None:
                    for tA in loadATiles:
                        vid = self._loadTile('A', tA, loadDU, isWrapAround, loadCountA, curA)
                        if vid is not None:
                            step.loadA[tA] = vid

                if loadBTiles is not None:
                    for tB in loadBTiles:
                        vid = self._loadTile('B', tB, loadDU, isWrapAround, loadCountB, curB)
                        if vid is not None:
                            step.loadB[tB] = vid

                # Check USE and LOAD VGPRTile IDs don't overlap within a step
                useIds = set(step.useA.values()) | set(step.useB.values())
                loadIds = set(step.loadA.values()) | set(step.loadB.values())
                overlap = useIds & loadIds
                if overlap:
                    step.conflict = overlap
                    self.needsUnrolling = True

                self._schedule.append(step)

                # WITHIN_SUBGROUP: release current DU's USE tiles for K-dim reuse
                # Only release tiles that were actually USEd (not tiles loaded for next group)
                if self.config.reuseStrategy == VGPRTileReUseStrategy.WITHIN_SUBGROUP:
                    for tA, vid in step.useA.items():
                        if self.allocator.isAllocated('A', tA, du):
                            self.allocator.release('A', tA, du)
                    for tB, vid in step.useB.items():
                        if self.allocator.isAllocated('B', tB, du):
                            self.allocator.release('B', tB, du)

            # Release after group based on strategy
            if self.config.reuseStrategy == VGPRTileReUseStrategy.WITHIN_SUBGROUP:
                # Per-DU release already freed USE tiles; just fixup shadow keys
                for tc, tileIdx, duIdx, shadowKey in self._pendingRemap:
                    vid = self.allocator._allocMap(tc).pop((shadowKey, duIdx))
                    self.allocator._allocMap(tc)[(tileIdx, duIdx)] = vid
                self._pendingRemap = []
            elif self.config.reuseStrategy == VGPRTileReUseStrategy.ACROSS_SUBGROUP:
                self._releaseUnusedAfterGroup(gi)

        self.hasDuplicatedReads = (
            any(c > 1 for c in loadCountA.values()) or
            any(c > 1 for c in loadCountB.values())
        )

    def _loadTile(self, tc: str, tileIdx: int, loadDU: int,
                  isWrapAround: bool, loadCount: Dict,
                  currentGroupTiles: Set[int]) -> Optional[int]:
        """Determine the VGPRTile ID for a load. Returns None if no load needed."""
        allocated = self.allocator.isAllocated(tc, tileIdx, loadDU)

        if isWrapAround and allocated:
            # Wrap-around: reuse group 0's existing VGPRTile IDs
            return self.allocator.getVGPRTileId(tc, tileIdx, loadDU)

        if not allocated:
            # Fresh allocation
            vid = self.allocator.allocate(tc, tileIdx, loadDU)
            key = (tileIdx, loadDU)
            loadCount[key] = loadCount.get(key, 0) + 1
            return vid

        if self.config.reuseStrategy == VGPRTileReUseStrategy.WITHIN_SUBGROUP \
                and tileIdx in currentGroupTiles:
            # Tile is allocated by the current group but will be released after it.
            # Must allocate a new VGPR for the next group's data.
            # Use a shadow key to avoid overwriting the current allocation.
            shadowKey = -(tileIdx + 1)  # negative to avoid collision
            vid = self.allocator.allocate(tc, shadowKey, loadDU)
            # Store the real tileIdx mapping for later fixup
            self._pendingRemap.append((tc, tileIdx, loadDU, shadowKey))
            key = (tileIdx, loadDU)
            loadCount[key] = loadCount.get(key, 0) + 1
            return vid

        # NONE / ACROSS_SUBGROUP: tile stays alive, reuse in place
        return None

    def _isWrapAroundLoad(self, groupIdx: int, duIdx: int, numGroups: int) -> bool:
        """True when this step's load targets group 0 for the next macrotile iteration."""
        if self.config.prefetchMode == PrefetchMode.HALF_PREFETCH:
            return groupIdx == numGroups - 1 and duIdx == self.numDU - 1
        elif self.config.prefetchMode == PrefetchMode.FULL_PREFETCH:
            return groupIdx == numGroups - 1
        return False

    # ── Prefetch modes ───────────────────────────────────────

    def _getLoadTargets(self, groupIdx: int, duIdx: int,
                        numGroups: int) -> Tuple[Optional[List[int]], Optional[List[int]], int]:
        """Returns (loadATiles, loadBTiles, targetDU)."""
        if self.config.prefetchMode == PrefetchMode.HALF_PREFETCH:
            return self._loadTargetsHalfPrefetch(groupIdx, duIdx, numGroups)
        elif self.config.prefetchMode == PrefetchMode.FULL_PREFETCH:
            return self._loadTargetsFullPrefetch(groupIdx, duIdx, numGroups)
        return (None, None, 0)

    def _loadTargetsHalfPrefetch(self, groupIdx, duIdx, numGroups):
        """HALF: DU=0 loads same-group DU=1, DU=last loads next-group DU=0.
        Last group wraps around to group 0 (next iteration)."""
        currentGroup = self.groups[groupIdx]
        if duIdx < self.numDU - 1:
            targetDU = duIdx + 1
            return (currentGroup.tileAIndices, currentGroup.tileBIndices, targetDU)
        else:
            nextGroup = self.groups[(groupIdx + 1) % numGroups]
            return (nextGroup.tileAIndices, nextGroup.tileBIndices, 0)

    def _loadTargetsFullPrefetch(self, groupIdx, duIdx, numGroups):
        """FULL: DU=0 loads next-group DU=0, DU=1 loads next-group DU=1.
        Last group wraps around to group 0 (next iteration)."""
        nextGroup = self.groups[(groupIdx + 1) % numGroups]
        return (nextGroup.tileAIndices, nextGroup.tileBIndices, duIdx)

    # ── Reuse strategies ─────────────────────────────────────

    def _releaseUnusedAfterGroup(self, groupIdx: int):
        """ACROSS_SUBGROUP: release tiles not appearing in any future group."""
        currentGroup = self.groups[groupIdx]

        futureA: Set[int] = set()
        futureB: Set[int] = set()
        for gi in range(groupIdx + 1, len(self.groups)):
            futureA.update(self.groups[gi].tileAIndices)
            futureB.update(self.groups[gi].tileBIndices)

        for tA in currentGroup.tileAIndices:
            if tA not in futureA:
                self.allocator.releaseAllForTile('A', tA)
        for tB in currentGroup.tileBIndices:
            if tB not in futureB:
                self.allocator.releaseAllForTile('B', tB)

    def _releaseGroupTiles(self, group: SubtileGroup):
        """WITHIN_SUBGROUP: release all tiles (all DUs) of this group."""
        for tA in group.tileAIndices:
            self.allocator.releaseAllForTile('A', tA)
        for tB in group.tileBIndices:
            self.allocator.releaseAllForTile('B', tB)

    # ── Debug ────────────────────────────────────────────────

    def printSchedule(self):
        print(f"SubtileGridA={self.MTA}, SubtileGridB={self.MTB}")
        print(f"Group grid: {self.numGroupsA} x {self.numGroupsB}")
        print(f"Group size: {self.config.subtileGroupSizeA} x {self.config.subtileGroupSizeB}")
        print(f"Prefetch: {self.config.prefetchMode.name}")
        print(f"Reuse: {self.config.reuseStrategy.name}")
        print(f"hasDuplicatedReads: {self.hasDuplicatedReads}")
        print(f"needsUnrolling: {self.needsUnrolling}")
        print(f"totalVGPRTiles: {self.totalVGPRs} ({self.totalVGPRs * 4} VGPRs)")
        print()

        grid = [[None] * self.numGroupsB for _ in range(self.numGroupsA)]
        sA = self.config.subtileGroupSizeA
        sB = self.config.subtileGroupSizeB
        for group in self.groups:
            gA = group.subtiles[0][0] // sA
            gB = group.subtiles[0][1] // sB
            grid[gA][gB] = group.groupId
        print(f"Ordering grid ({self.config.ordering.name}):")
        for row in grid:
            print("  " + "  ".join(f"{v:2d}" if v is not None else "  " for v in row))
        print()

        # Compute buffer_load sets per group
        numGroups = len(self.groups)
        bufferLoadsPerGroup = {}  # gi -> (set of A subtiles, set of B subtiles)
        loadedA = set(self.groups[0].tileAIndices)
        loadedB = set(self.groups[0].tileBIndices)
        for gi in range(numGroups):
            targetGi = (gi + 1) % numGroups
            targetGroup = self.groups[targetGi]
            if gi == numGroups - 1:
                bufA = set(targetGroup.tileAIndices)
                bufB = set(targetGroup.tileBIndices)
            else:
                needA = set(targetGroup.tileAIndices)
                needB = set(targetGroup.tileBIndices)
                bufA = needA - loadedA
                bufB = needB - loadedB
                loadedA |= needA
                loadedB |= needB
            bufferLoadsPerGroup[gi] = (bufA, bufB)

        # Print steps with WAIT tracking
        pendingA = set()
        pendingB = set()
        currentGroup = -1
        for step in self._schedule:
            if step.groupId != currentGroup:
                currentGroup = step.groupId
                bufA, bufB = bufferLoadsPerGroup[currentGroup]
                pendingA |= bufA
                pendingB |= bufB

            print(f"  Group {step.groupId}, DU={step.duIndex}:")
            mfmas = [(a, b) for a in sorted(step.useA.keys()) for b in sorted(step.useB.keys())]
            print(f"    MFMAs: {mfmas}")
            mtLoad = "n+1" if step.isWrapLoad else "n"
            duLabel = f", DU {step.loadDU}" if step.loadDU >= 0 else ""
            print(f"    USE  A: {step.useA}  B: {step.useB}")

            # Check if LOAD needs a WAIT for pending buffer_loads
            # buffer_load writes DU 0 before DU 1, so only DU 0 reads need a WAIT
            waitA = set()
            waitB = set()
            if step.loadDU == 0:
                waitA = set(step.loadA.keys()) & pendingA
                waitB = set(step.loadB.keys()) & pendingB
            if waitA or waitB:
                print(f"    WAIT A: {sorted(waitA)}  B: {sorted(waitB)}")
                pendingA -= waitA
                pendingB -= waitB

            print(f"    LOAD (MT {mtLoad}{duLabel}) A: {step.loadA}  B: {step.loadB}")
            if step.conflict:
                print(f"    *** CONFLICT: USE/LOAD share VGPRTile IDs {step.conflict} — needs unrolling ***")

        # Buffer loads summary per group
        print()
        print("Buffer loads per group (subtile IDs -> LDS):")
        for gi in range(numGroups):
            targetGi = (gi + 1) % numGroups
            bufA, bufB = bufferLoadsPerGroup[gi]
            mtLabel = "n+2" if gi == numGroups - 1 else "n+1"
            print(f"  Group {gi} -> buffer_load for Group {targetGi} (MT {mtLabel}):  A: {sorted(bufA)}  B: {sorted(bufB)}")

        # Double-buffer LDS dependency: MT n+2 buffer_loads vs MT n ds_reads
        # Find last ds_read (LOAD step) for each subtile that the last group buffer_loads
        if numGroups > 1:
            wrapGroup = self.groups[0]
            wrapSubtilesA = set(wrapGroup.tileAIndices)
            wrapSubtilesB = set(wrapGroup.tileBIndices)
            # lastRead[tc][tileIdx] = (groupId, duIndex) of last LOAD referencing it
            lastReadA = {t: ("bootstrap", -1) for t in wrapSubtilesA}
            lastReadB = {t: ("bootstrap", -1) for t in wrapSubtilesB}
            for step in self._schedule:
                if step.isWrapLoad:
                    continue  # wrap LOADs are MT n+1, not MT n reads
                for tA in step.loadA:
                    if tA in wrapSubtilesA:
                        lastReadA[tA] = (step.groupId, step.duIndex)
                for tB in step.loadB:
                    if tB in wrapSubtilesB:
                        lastReadB[tB] = (step.groupId, step.duIndex)

            lastGroupId = self.groups[-1].groupId
            print()
            print(f"Double-buffer LDS dependency (MT n+2 writes vs MT n reads):")
            print(f"  buffer_load at Group {lastGroupId} must wait for last ds_read of same subtile:")
            for tA in sorted(wrapSubtilesA):
                g, d = lastReadA[tA]
                print(f"    A[{tA}]: last read at {f'Group {g}, DU={d}' if g != 'bootstrap' else 'bootstrap'}")
            for tB in sorted(wrapSubtilesB):
                g, d = lastReadB[tB]
                print(f"    B[{tB}]: last read at {f'Group {g}, DU={d}' if g != 'bootstrap' else 'bootstrap'}")


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
         SchedulerConfig(4, 4, PrefetchMode.HALF_PREFETCH, VGPRTileReUseStrategy.ACROSS_SUBGROUP, SubgroupOrdering.COLUMN_MAJOR)),

         (f"lsg {lsgA}x{lsgB}, group 4x4, HALF_PREFETCH, WITHIN_SUBGROUP, COLUMN_MAJOR",
         MockTileInfo([lsgA, 1], [1, 2]), MockTileInfo([lsgB, 1], [1, 2]),
         SchedulerConfig(4, 4, PrefetchMode.HALF_PREFETCH, VGPRTileReUseStrategy.WITHIN_SUBGROUP, SubgroupOrdering.COLUMN_MAJOR)),


        # Needs unrolling.
        #  (f"lsg {lsgA}x{lsgB}, group 4x4, FULL_PREFETCH, NONE, COLUMN_MAJOR",
        #  MockTileInfo([lsgA, 1], [1, 2]), MockTileInfo([lsgB, 1], [1, 2]),
        #  SchedulerConfig(8, 8, PrefetchMode.FULL_PREFETCH, VGPRTileReUseStrategy.NONE, SubgroupOrdering.COLUMN_MAJOR)),

          (f"lsg {lsgA}x{lsgB}, group 4x4, HALF_PREFETCH, NONE, COLUMN_MAJOR",
         MockTileInfo([lsgA, 1], [1, 2]), MockTileInfo([lsgB, 1], [1, 2]),
         SchedulerConfig(8, 8, PrefetchMode.HALF_PREFETCH, VGPRTileReUseStrategy.NONE, SubgroupOrdering.COLUMN_MAJOR)),

        (f"lsg {lsgA}x{lsgB}, group 4x4, HALF_PREFETCH, ACROSS_SUBGROUP, COLUMN_MAJOR",
            MockTileInfo([lsgA, 1], [1, 2]), MockTileInfo([lsgB, 1], [1, 2]),
            SchedulerConfig(8, 8, PrefetchMode.HALF_PREFETCH, VGPRTileReUseStrategy.ACROSS_SUBGROUP, SubgroupOrdering.COLUMN_MAJOR)),
        
        (f"lsg {lsgA}x{lsgB}, group 4x4, HALF_PREFETCH, WITHIN_SUBGROUP, COLUMN_MAJOR",
            MockTileInfo([lsgA, 1], [1, 2]), MockTileInfo([lsgB, 1], [1, 2]),
            SchedulerConfig(8, 8, PrefetchMode.HALF_PREFETCH, VGPRTileReUseStrategy.WITHIN_SUBGROUP, SubgroupOrdering.COLUMN_MAJOR)),


        (f"lsg {lsgA}x{lsgB}, group 4x4, HALF_PREFETCH, WITHIN_SUBGROUP, COLUMN_MAJOR",
            MockTileInfo([10, 1], [1, 2]), MockTileInfo([10, 1], [1, 2]),
            SchedulerConfig(2, 10, PrefetchMode.HALF_PREFETCH, VGPRTileReUseStrategy.WITHIN_SUBGROUP, SubgroupOrdering.COLUMN_MAJOR)),

        (f"lsg {lsgA}x{lsgB}, group 4x4, HALF_PREFETCH, WITHIN_SUBGROUP, COLUMN_MAJOR",
            MockTileInfo([10, 1], [1, 2]), MockTileInfo([10, 1], [1, 2]),
            SchedulerConfig(2, 10, PrefetchMode.HALF_PREFETCH, VGPRTileReUseStrategy.ACROSS_SUBGROUP, SubgroupOrdering.COLUMN_MAJOR)),
    ]

    for name, tiA, tiB, cfg in configs:
        print(f"=== {name} ===")
        s = MFMAScheduler(tiA, tiB, cfg)
        s.printSchedule()
        print()
