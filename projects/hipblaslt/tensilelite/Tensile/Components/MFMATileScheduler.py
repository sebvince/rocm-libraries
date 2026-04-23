"""MFMATile-based logical scheduler.

Builds a logical schedule using MFMA tile indices as the core primitive,
with explicit per-operation load granularity for GR/LR on A, B, SA, SB.

The schedule is built in 8 passes:
  place_LRs                — place LRs based on their granularities
  assign_vgpr_tiles        — assign physical vgprTileIds with per-tensor free-lists
  place_GRs                — place GRs
  annotate_deps            — annotate raw per-op dependencies
  remove_unnecessary_gr_deps — remove redundant LR→GR deps
  remove_unnecessary_lr_deps — remove redundant GR→LR deps covered by MFMA syncs
  remove_cross_deps        — replace cross-subIterK deps with wait preOps
  insert_gr_lr_inc         — insert lr_inc/gr_inc preOps at MT transitions
  group                    — serialize and group (produce paths for instructionSchedule)
  remove_wait_lr_sync      — remove redundant wait_lr_sync after grouping
  emit                     — produce List[EmittedModule] with before-link chains
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Tuple
import copy
import io
import math


# ── Core primitives ─────────────────────────────────────────

@dataclass
class MFMATileSize:
    """Granularity of a single read operation, measured in MFMA tiles.

    k:  how many subIterK steps one read covers
    mn: how many MFMA tiles in the M (for A/SA) or N (for B/SB) dimension
    """
    k: int
    mn: int


@dataclass
class MFMATileRange:
    """A rectangular range of MFMA tile coordinates for one read."""
    subIterK_start: int
    subIterK_end: int          # exclusive
    tileId_start: int
    tileId_end: int            # exclusive

    @property
    def subIterK_list(self) -> List[int]:
        return list(range(self.subIterK_start, self.subIterK_end))

    @property
    def tileId_list(self) -> List[int]:
        return list(range(self.tileId_start, self.tileId_end))

    def fmt_k(self) -> str:
        ids = self.subIterK_list
        if len(ids) == 1:
            return f"[{ids[0]}]"
        return f"[{ids[0]},{ids[-1]}]"

    def fmt_tiles(self) -> str:
        return f"[{self.tileId_start}-{self.tileId_end - 1}]"


# ── Config ──────────────────────────────────────────────────

@dataclass
class ReadGranularity:
    """Load granularity for one operation on one tensor."""
    size: MFMATileSize


@dataclass
class SchedulerConfig:
    """Configuration for the MFMATile-based scheduler."""
    numMFMATilesM: int    # MFMA tiles in M dimension (for A)
    numMFMATilesN: int    # MFMA tiles in N dimension (for B)
    numSubIterK: int      # subIterK steps within the macrotile
    lrA: ReadGranularity
    lrB: ReadGranularity
    grA: ReadGranularity
    grB: ReadGranularity
    lrSA: Optional[ReadGranularity] = None
    lrSB: Optional[ReadGranularity] = None
    grSA: Optional[ReadGranularity] = None
    grSB: Optional[ReadGranularity] = None
    numPartitionsM: int = 1   # partition grid in M dimension
    numPartitionsN: int = 1   # partition grid in N dimension

    @property
    def hasScale(self) -> bool:
        return self.lrSA is not None and self.lrSB is not None

    @property
    def numPartitions(self) -> int:
        return self.numPartitionsM * self.numPartitionsN

    @property
    def partitionSizeM(self) -> int:
        assert self.numMFMATilesM % self.numPartitionsM == 0
        return self.numMFMATilesM // self.numPartitionsM

    @property
    def partitionSizeN(self) -> int:
        assert self.numMFMATilesN % self.numPartitionsN == 0
        return self.numMFMATilesN // self.numPartitionsN

    @staticmethod
    def get_partition_candidates(tileInfoA, tileInfoB) -> list:
        """Return partition candidates as [(numPartitionsM, numPartitionsN), ...].

        Enumerates all divisors of MAX(M, N) in ascending order and
        partitions the larger dimension. Starts with (1, 1).
        """
        M = tileInfoA.localMMATileGrid[0]
        N = tileInfoB.localMMATileGrid[0]
        maxDim = max(M, N)

        divisors = sorted(d for d in range(1, maxDim + 1) if maxDim % d == 0)

        candidates = []
        for d in divisors:
            if N >= M:
                candidates.append((1, d))
            else:
                candidates.append((d, 1))

        return candidates

    @classmethod
    def from_tile_info(cls, tileInfoA, tileInfoB,
                       lrA: ReadGranularity, lrB: ReadGranularity,
                       grA: ReadGranularity, grB: ReadGranularity,
                       scaleTileInfoA=None, scaleTileInfoB=None,
                       lrSA: Optional[ReadGranularity] = None,
                       lrSB: Optional[ReadGranularity] = None,
                       grSA: Optional[ReadGranularity] = None,
                       grSB: Optional[ReadGranularity] = None,
                       numPartitionsM: int = 1,
                       numPartitionsN: int = 1):
        """Build config from TileInfo objects.

        Derives numMFMATilesM/N/K from the tile info:
        - numMFMATilesM = tileInfoA.localMMATileGrid[0]
        - numMFMATilesN = tileInfoB.localMMATileGrid[0]
        - numSubIterK   = tileInfoA.localMMATileGrid[1]
        """
        numMFMATilesM = tileInfoA.localMMATileGrid[0]
        numMFMATilesN = tileInfoB.localMMATileGrid[0]
        numSubIterK = tileInfoA.localMMATileGrid[1]

        assert tileInfoA.localMMATileGrid[1] == tileInfoB.localMMATileGrid[1], \
            "A and B must have same localMMATileGrid[1]"

        return cls(
            numMFMATilesM=numMFMATilesM,
            numMFMATilesN=numMFMATilesN,
            numSubIterK=numSubIterK,
            lrA=lrA, lrB=lrB,
            grA=grA, grB=grB,
            numPartitionsM=numPartitionsM,
            numPartitionsN=numPartitionsN,
            lrSA=lrSA, lrSB=lrSB,
            grSA=grSA, grSB=grSB,
        )


# ── Schedule operation types ────────────────────────────────

class OpKind(Enum):
    MFMA = auto()
    LR = auto()
    GR = auto()
    WAIT_GR = auto()
    WAIT_LR = auto()
    SYNC = auto()
    GR_INC = auto()
    GR_SCALE = auto()
    LR_INC = auto()


@dataclass
class MFMAPlacement:
    """MFMA operation consuming data for one subIterK."""
    subIterK: int
    tileA: MFMATileRange       # A tiles consumed
    tileB: MFMATileRange       # B tiles consumed
    deps: List['DepRef'] = field(default_factory=list)      # populated by annotate_deps()
    preOps: List['DepOp'] = field(default_factory=list)     # populated by remove_cross_deps()
    vgpr_tile_map_A: List[dict] = field(default_factory=list)   # [{tileId: vgprTileId}] per unroll iter
    vgpr_tile_map_B: List[dict] = field(default_factory=list)   # [{tileId: vgprTileId}] per unroll iter
    vgpr_tile_map_SA: List[dict] = field(default_factory=list)  # [{scaleGroupIdx: vgprTileId}] per unroll iter
    vgpr_tile_map_SB: List[dict] = field(default_factory=list)  # [{scaleGroupIdx: vgprTileId}] per unroll iter


@dataclass
class LRPlacement:
    """Local Read placement for one tensor in one subIterK slot."""
    tensor: str                # 'A', 'B', 'SA', 'SB'
    mtIteration: str           # 'n', 'n+1'
    tiles: MFMATileRange
    subIterK_slot: int         # which subIterK this LR is placed in
    partition: int = 0         # which partition this LR belongs to
    deps: List['DepRef'] = field(default_factory=list)      # populated by annotate_deps()
    preOps: List['DepOp'] = field(default_factory=list)     # populated by remove_cross_deps()
    vgpr_tile_map: List[dict] = field(default_factory=list)  # [{tileId: vgprTileId}] per unroll iter


@dataclass
class GRPlacement:
    """Global Read placement for one tensor in one subIterK slot."""
    tensor: str                # 'A', 'B', 'SA', 'SB'
    mtIteration: str           # 'n+2'
    tiles: MFMATileRange
    subIterK_slot: int         # which subIterK this GR is placed in
    partition: int = 0         # which partition this GR belongs to
    deps: List['DepRef'] = field(default_factory=list)      # populated by annotate_deps()
    preOps: List['DepOp'] = field(default_factory=list)     # populated by remove_cross_deps()


# ── Per-subIterK container ──────────────────────────────────

@dataclass
class SubIterKSlot:
    """All operations placed in one subIterK step."""
    subIterK: int
    mfma: Optional[MFMAPlacement] = None
    lrs: List[LRPlacement] = field(default_factory=list)
    grs: List[GRPlacement] = field(default_factory=list)


# ── Dependency types ────────────────────────────────────────

@dataclass
class WaitGRCounts:
    """Per-tensor inflight load counts for wait_gr preOp."""
    A: int = 0
    B: int = 0
    SA: int = 0
    SB: int = 0

    def __str__(self):
        parts = []
        for t in ('A', 'B', 'SA', 'SB'):
            v = getattr(self, t)
            if v:
                parts.append(f"{t}={v}")
        return ",".join(parts) if parts else "0"


@dataclass
class DepOp:
    """A typed dependency in a before-chain.

    Kinds:
      'wait_gr'      — wait for global reads (no sync)
      'wait_gr_sync' — wait for global reads + sync barrier
      'wait_lr'      — wait for local reads (no sync, used before MFMAs)
      'wait_lr_sync' — wait for local reads + sync barrier (used before GRs,
                        to ensure LR finished reading LDS before GR overwrites it)
      'lr_inc'       — LDS buffer swap for LR (needs tensor)
      'gr_inc'       — pointer update + LDS swap for GR (needs tensor)
      'ref'          — reference to another AnnotatedOp in same subIterK (group)
      'lr_ref'       — dependency on a specific LR placement (annotate_deps)
      'gr_ref'       — dependency on a specific GR placement (annotate_deps)
    """
    kind: str
    tensor: str = ""
    ref: Optional[object] = None  # placement (annotate_deps) or AnnotatedOp (group)
    wait_gr_counts: Optional[WaitGRCounts] = None  # only for kind='wait_gr'

    def __str__(self):
        if self.kind in ('wait_gr', 'wait_gr_sync') and self.wait_gr_counts:
            return f"{self.kind}({self.wait_gr_counts})"
        if self.tensor:
            return f"{self.kind}({self.tensor})"
        return self.kind


@dataclass
class DepRef:
    """Dependency on another placement (annotate_deps output)."""
    ref: object     # LRPlacement or GRPlacement
    mt_offset: int = 0  # 0 = same MT, -1 = prev MT, -2 = two MTs back, ...



@dataclass
class AnnotatedOp:
    """An operation with its before-dependencies."""
    kind: str        # 'MFMA', 'LR', 'GR', etc.
    before: List[DepOp] = field(default_factory=list)
    # Original placement reference
    placement: object = None


# ── Grouped output ─────────────────────────────────────────

@dataclass
class GroupedSubIterK:
    """Serialized ops within one subIterK (output of group/annotate_deps)."""
    subIterK: int
    partition: int = 0
    ops: List[AnnotatedOp] = field(default_factory=list)


# ── Emitted output ─────────────────────────────────────────

@dataclass
class EmittedModule:
    """One emitted module with before-link for instruction scheduling.

    Compatible with SubtileBasedScheduler.instructionSchedule().
    Instructions are left empty at the logical level — filled during emission.
    """
    moduleId: int = -1
    instructions: list = field(default_factory=list)
    before: Optional[int] = None   # moduleId that must complete before this module
    opType: str = ""
    label: str = ""                # human-readable label for debugging
    source: object = None          # original placement or DepOp, for populate_instructions


# ── Main scheduler class ───────────────────────────────────

class MFMATileScheduler:
    """MFMATile-based logical scheduler.

    Builds the schedule in 6 passes, each producing testable intermediate output.
    Each pass auto-runs its prerequisites if needed (tracked via self._completed).
    """

    def __init__(self, config: SchedulerConfig):
        self.config = config
        self.tensors: List[str] = ['A', 'B'] + (['SA', 'SB'] if config.hasScale else [])
        self._completed: set = set()   # tracks which passes have run: {'lr', 'vgpr_tiles', 'gr', 'deps', 'group', 'emit'}
        self._partitions: Optional[List[List[SubIterKSlot]]] = None  # shared mutable state across passes
        self._grouped: Optional[List[GroupedSubIterK]] = None
        self._emitted: Optional[List[List[EmittedModule]]] = None
        self._preloop_emitted: Optional[List[List[List[EmittedModule]]]] = None
        self._ngll_emitted: Optional[List[List[List[EmittedModule]]]] = None
        self._nll_emitted: Optional[List[List[List[EmittedModule]]]] = None

    # ── Place LRs ─────────────────────────────────────────

    def _partition_tile_range(self, pi: int) -> dict:
        """Return {'A': (start, end), 'B': (start, end)} for partition pi.

        Uses COLUMN_MAJOR ordering: M (A) varies fastest, N (B) varies slowest.
        """
        cfg = self.config
        # COLUMN_MAJOR: M is inner (pi % M), N is outer (pi // M)
        piM = pi % cfg.numPartitionsM
        piN = pi // cfg.numPartitionsM
        a0 = piM * cfg.partitionSizeM
        b0 = piN * cfg.partitionSizeN
        return {'A': (a0, a0 + cfg.partitionSizeM),
                'B': (b0, b0 + cfg.partitionSizeN)}

    def place_LRs(self) -> List[List[SubIterKSlot]]:
        """Place MFMAs and LRs based on read granularities.

        Returns a list of partitions, each containing a list of SubIterKSlots.

        Each LR prefetches data for the next subIterK group. Within-partition
        prefetches use current partition tiles; cross-partition prefetches
        (wrapping) use next partition tiles.

        Two tracking mechanisms:
        - loaded_ranges: tracks tile ranges in VGPR per side. Wrapping LRs
          are only placed when the next partition's tiles aren't already loaded.
        - placed: tracks (tensor, k-range, tile-range) of non-wrapping LRs
          placed so far across partitions. Skips redundant K-prefetch when
          the same data was already loaded by an earlier partition.
        """
        cfg = self.config
        numP = cfg.numPartitions
        part_ranges = [self._partition_tile_range(pi) for pi in range(numP)]

        # Track which tile ranges are currently loaded in VGPR (for wrapping decisions).
        loaded_ranges = {'A': {part_ranges[0]['A']},
                         'B': {part_ranges[0]['B']}}

        # Track placed K-prefetch LRs across partitions (for dedup).
        placed = set()

        partitions = []
        for pi in range(numP):
            cur, nxt = part_ranges[pi], part_ranges[(pi + 1) % numP]
            is_last = (pi == numP - 1)

            load = {}
            for side in ('A', 'B'):
                load[side] = is_last or nxt[side] not in loaded_ranges[side]

            slots = self._place_LRs_for_partition(cur, nxt, is_last, load, placed)
            for slot in slots:
                for lr in slot.lrs:
                    lr.partition = pi
            partitions.append(slots)

            for side in ('A', 'B'):
                if load[side]:
                    loaded_ranges[side] = {cur[side], nxt[side]}

        self._partitions = partitions
        self._completed.add('lr')
        return partitions

    def _place_LRs_for_partition(self, cur: tuple, nxt: tuple,
                                  is_last: bool,
                                  load: dict,
                                  placed: set) -> List[SubIterKSlot]:
        """Place MFMAs and LRs for one partition."""
        cfg = self.config
        numK = cfg.numSubIterK
        multi_part = cfg.numPartitions > 1

        slots = [SubIterKSlot(subIterK=k) for k in range(numK)]
        slot_mt = {}  # slot_k → lr_mt string, for MT-homogeneity enforcement

        # MFMAs
        for k in range(numK):
            slots[k].mfma = MFMAPlacement(
                subIterK=k,
                tileA=MFMATileRange(k, k + 1, cur['A'][0], cur['A'][1]),
                tileB=MFMATileRange(k, k + 1, cur['B'][0], cur['B'][1]),
            )

        # All tensors that can participate.
        all_tensors = [('A', cfg.lrA), ('B', cfg.lrB)]
        if cfg.hasScale:
            all_tensors.append(('SA', cfg.lrSA))
            all_tensors.append(('SB', cfg.lrSB))

        # Place LRs grouped by k_gran.
        # - Non-wrapping (K-prefetch): all tensors, deduped by placed set.
        # - Wrapping (cross-partition): only tensors whose side needs loading.
        for k_gran in sorted(set(g.size.k for _, g in all_tensors)):
            group_all = [(t, g) for t, g in all_tensors if g.size.k == k_gran]
            num_chunks = numK // k_gran
            for chunk_idx in range(num_chunks):
                next_chunk = (chunk_idx + 1) % num_chunks
                is_wrap = (next_chunk == 0)
                lr_mt = "n+1" if is_last and is_wrap else "n"
                lr_k_start = next_chunk * k_gran
                lr_k_end = lr_k_start + k_gran
                base_slot = chunk_idx * k_gran

                # For wrapping chunks, only include tensors whose side is
                # loading so that slot assignment reflects active tensors.
                # A and B always participate (their wrapping is gated inside
                # the loop) to keep slot indices stable for their k_gran group.
                if is_wrap and multi_part:
                    group = [(t, g) for t, g in group_all
                             if t in ('A', 'B') or load['A' if t in ('A', 'SA') else 'B']]
                else:
                    group = group_all

                # Group by side (A/SA together, B/SB together) for slot assignment
                sides = [[(t, g) for t, g in group if t in ('A', 'SA')],
                         [(t, g) for t, g in group if t in ('B', 'SB')]]
                sides = [s for s in sides if s]

                for side_idx, side in enumerate(sides):
                    slot_k = base_slot + (side_idx % k_gran)
                    # Redirect LRs away from slots committed to a different MT,
                    # keeping each slot MT-homogeneous.
                    # This reduce the number of wait_gr_sync needed as all LRs 
                    # in the same subIterK wait for the same MT iterration.
                    committed = slot_mt.get(slot_k)
                    if committed is not None and committed != lr_mt:
                        slot_k = numK - 1

                    for tensor, gran in side:
                        tile_range = nxt if (is_wrap or not multi_part) else cur
                        side_key = 'A' if tensor in ('A', 'SA') else 'B'
                        ts, te = tile_range[side_key]

                        # Wrapping: use load dict. Non-wrapping: use placed set.
                        if is_wrap and multi_part:
                            if not load[side_key]:
                                continue
                        else:
                            lr_key = (tensor, lr_k_start, lr_k_end, ts, te)
                            if lr_key in placed:
                                continue
                            placed.add(lr_key)

                        lr = LRPlacement(
                            tensor=tensor,
                            mtIteration=lr_mt,
                            tiles=MFMATileRange(lr_k_start, lr_k_end, ts, te),
                            subIterK_slot=slot_k,
                        )
                        slots[slot_k].lrs.append(lr)
                        slot_mt[slot_k] = lr_mt

        return slots

    # ── Assign VGPR tile IDs (free-list allocation) ──────

    def assign_vgpr_tiles(self):
        """Assign physical vgprTileIds to all placements (A, B, SA, SB).

        Free-list allocator with per-tensor FIFO queues, iterated until
        convergence (or max 4 unroll iterations).

        Three phases:
          1. Scan all MFMAs to find last read position for each
             (tensor, tileId, k_data_group) key.
          2. Walk execution order in a loop: each iteration feeds the
             previous next_iter as the starting active state.  Appends
             one tile-map dict per iteration to each placement's list.
             Stops when next_iter matches the seeded state (convergence).
          3. Record unroll_factor, needs_unrolling, and max tile_peaks.

        Keys:
          A/B:   (tensor, tileId, subIterK)
          SA/SB: (tensor, scaleGroupIdx, k_chunk_start)

        Sets self.tile_peaks (per-tensor max across unrolls),
        self.needs_unrolling, self.unroll_factor.
        """
        if 'lr' not in self._completed:
            self.place_LRs()

        cfg = self.config
        numK = cfg.numSubIterK
        MAX_UNROLL = 8

        # Build k_gran lookup for scale tensors
        scale_k_gran = {}
        if cfg.hasScale:
            scale_k_gran['SA'] = cfg.lrSA.size.k
            scale_k_gran['SB'] = cfg.lrSB.size.k

        # ── Phase 1: find last MFMA read for each key ──
        last_read = {}  # key -> flat position
        for pi, slots in enumerate(self._partitions):
            for slot in slots:
                if not slot.mfma:
                    continue
                pos = pi * numK + slot.subIterK
                k = slot.subIterK
                # A/B: key = (tensor, tileId, subIterK)
                for tensor, tileRange in [('A', slot.mfma.tileA),
                                           ('B', slot.mfma.tileB)]:
                    for t in tileRange.tileId_list:
                        last_read[(tensor, t, k)] = pos
                # SA/SB: key = (tensor, scaleGroupIdx, k_chunk_start)
                if cfg.hasScale:
                    for stensor, tileRange in [('SA', slot.mfma.tileA),
                                                ('SB', slot.mfma.tileB)]:
                        sk_gran = scale_k_gran[stensor]
                        k_chunk = (k // sk_gran) * sk_gran
                        for t in tileRange.tileId_list:
                            sg = t // 2
                            last_read[(stensor, sg, k_chunk)] = pos

        # ── Phase 2: iterate until convergence ──
        from collections import deque

        class _FreeList:
            __slots__ = ('free', 'next_id', 'active_count', 'peak')
            def __init__(self):
                self.free = deque()
                self.next_id = 0
                self.active_count = 0
                self.peak = 0
            def alloc(self):
                if self.free:
                    vid = self.free.popleft()  # FIFO for convergence
                else:
                    vid = self.next_id
                    self.next_id += 1
                self.active_count += 1
                self.peak = max(self.peak, self.active_count)
                return vid
            def release(self, vid):
                self.free.append(vid)
                self.active_count -= 1

        tensor_names = ['A', 'B']
        if cfg.hasScale:
            tensor_names += ['SA', 'SB']

        max_peaks = {t: 0 for t in tensor_names}
        carry_active = {}
        all_next_iters = []     # next_iter from each iteration, for cycle detection

        pools = {t: _FreeList() for t in tensor_names}

        for unroll_iter in range(MAX_UNROLL):
            if unroll_iter == 0:
                active = {}
            else:
                active = dict(carry_active)
                # Reset active_count to match carry_active (tiles that survived
                # as live from the previous iteration's wrapping LRs).
                for t in tensor_names:
                    pools[t].active_count = sum(
                        1 for key in active if key[0] == t)

            next_iter = {}

            for pi, slots in enumerate(self._partitions):
                for slot in slots:
                    pos = pi * numK + slot.subIterK
                    k = slot.subIterK

                    # ── MFMA reads: look up or seed ──
                    if slot.mfma:
                        map_A, map_B = {}, {}
                        for tensor, tileRange, tile_map in [
                                ('A', slot.mfma.tileA, map_A),
                                ('B', slot.mfma.tileB, map_B)]:
                            for t in tileRange.tileId_list:
                                key = (tensor, t, k)
                                if key not in active:
                                    active[key] = pools[tensor].alloc()
                                tile_map[t] = active[key]
                        slot.mfma.vgpr_tile_map_A.append(map_A)
                        slot.mfma.vgpr_tile_map_B.append(map_B)

                        # SA/SB reads
                        if cfg.hasScale:
                            map_SA, map_SB = {}, {}
                            for stensor, tileRange, tile_map in [
                                    ('SA', slot.mfma.tileA, map_SA),
                                    ('SB', slot.mfma.tileB, map_SB)]:
                                sk_gran = scale_k_gran[stensor]
                                k_chunk = (k // sk_gran) * sk_gran
                                for t in tileRange.tileId_list:
                                    sg = t // 2
                                    key = (stensor, sg, k_chunk)
                                    if key not in active:
                                        active[key] = pools[stensor].alloc()
                                    tile_map[sg] = active[key]
                            slot.mfma.vgpr_tile_map_SA.append(map_SA)
                            slot.mfma.vgpr_tile_map_SB.append(map_SB)

                    # ── LR writes: allocate new tiles ──
                    for lr in slot.lrs:
                        tensor = lr.tensor
                        is_wrapping = lr.mtIteration != "n"
                        target = next_iter if is_wrapping else active

                        if tensor in ('A', 'B'):
                            tile_map = {}
                            for t in lr.tiles.tileId_list:
                                for lk in lr.tiles.subIterK_list:
                                    key = (tensor, t, lk)
                                    if key in target:
                                        pools[tensor].release(target[key])
                                    vid = pools[tensor].alloc()
                                    target[key] = vid
                                    tile_map[t] = vid
                            lr.vgpr_tile_map.append(tile_map)
                        elif tensor in ('SA', 'SB') and cfg.hasScale:
                            tile_map = {}
                            sk_gran = scale_k_gran[tensor]
                            seen_keys = set()
                            for t in lr.tiles.tileId_list:
                                sg = t // 2
                                for lk in lr.tiles.subIterK_list:
                                    k_chunk = (lk // sk_gran) * sk_gran
                                    key = (tensor, sg, k_chunk)
                                    if key in seen_keys:
                                        continue
                                    seen_keys.add(key)
                                    if key in target:
                                        pools[tensor].release(target[key])
                                    vid = pools[tensor].alloc()
                                    target[key] = vid
                                    tile_map[sg] = vid
                            lr.vgpr_tile_map.append(tile_map)

                    # ── Release tiles whose last read was at this position ──
                    to_release = [key for key, lr_pos in last_read.items()
                                  if lr_pos == pos and key in active]
                    for key in to_release:
                        pools[key[0]].release(active[key])
                        del active[key]

            # Track max peaks across iterations
            for t in tensor_names:
                max_peaks[t] = max(max_peaks[t], pools[t].peak)

            # Check convergence: if this iteration's next_iter matches
            # any previous iteration's next_iter, we found a cycle.
            # The cycle period is (current_iter - matching_iter).
            # All iterations from matching_iter to current_iter-1 form
            # the repeating pattern; iterations before that are prologue.
            converged = False
            for prev_idx, prev_ni in enumerate(all_next_iters):
                if next_iter == prev_ni:
                    # Strip tile maps from the redundant convergence iteration.
                    for pi2, slots2 in enumerate(self._partitions):
                        for slot2 in slots2:
                            if slot2.mfma:
                                slot2.mfma.vgpr_tile_map_A.pop()
                                slot2.mfma.vgpr_tile_map_B.pop()
                                if cfg.hasScale:
                                    slot2.mfma.vgpr_tile_map_SA.pop()
                                    slot2.mfma.vgpr_tile_map_SB.pop()
                            for lr2 in slot2.lrs:
                                lr2.vgpr_tile_map.pop()
                    converged = True
                    break
            if converged:
                break

            # Carry next_iter forward as active for next iteration
            all_next_iters.append(next_iter)
            carry_active = next_iter
        else:
            assert False, (f"assign_vgpr_tiles did not converge after "
                           f"{MAX_UNROLL} unroll iterations")

        # ── Phase 3: record results ──
        # unroll_factor = number of unique iterations (convergence iteration excluded)
        self.unroll_factor = unroll_iter
        self.needs_unrolling = self.unroll_factor > 1
        self.tile_peaks = max_peaks

        self._completed.add('vgpr_tiles')

    # ── Place GRs ─────────────────────────────────────────

    def _build_gr_list(self, part_ranges, offsetMT, offsetPartition,
                             debug=False):
        """Phase 1: Build ordered GR list from placed MFMAs.

        For each partition × subIterK, derive target partition/MT from
        the MFMA and offsets. Add GRs (A, B, SA, SB) with tile and K
        ranges snapped to GR granularity. Dedup within same MT level,
        then remove n+1 entries that also appear at n+2 (cross-MT dedup).
        
        For each subIterK, we apply offsetMT on MT and offsetPartition on partition.

        Returns list of (tensor, mt_str, tile_start, tile_end,
                         k_start, k_end, gr_gran).
        """
        cfg = self.config
        numP = cfg.numPartitions

        seen = set()
        gr_list = []

        for pi in range(numP):
            partition_slots = self._partitions[pi]

            target_pi = (pi + offsetPartition) % numP
            wraps = (pi + offsetPartition) >= numP
            mt_offset = offsetMT + (1 if wraps else 0)
            mt_str = f"n+{mt_offset}"

            target_range = part_ranges[target_pi]

            for slot in partition_slots:
                k = slot.mfma.subIterK

                items = [('A', target_range['A'], cfg.grA),
                         ('B', target_range['B'], cfg.grB)]
                if cfg.hasScale:
                    items.append(('SA', target_range['A'], cfg.grSA))
                    items.append(('SB', target_range['B'], cfg.grSB))

                for tensor, (t_start, t_end), gr_gran in items:
                    mn = gr_gran.size.mn
                    k_gran = gr_gran.size.k

                    gr_tile_start = (t_start // mn) * mn
                    gr_tile_end = ((t_end + mn - 1) // mn) * mn

                    gr_k_start = (k // k_gran) * k_gran
                    gr_k_end = gr_k_start + k_gran

                    key = (tensor, mt_str, gr_tile_start, gr_tile_end,
                           gr_k_start, gr_k_end)
                    if key in seen:
                        continue
                    seen.add(key)
                    gr_list.append((tensor, mt_str, gr_tile_start,
                                    gr_tile_end, gr_k_start, gr_k_end,
                                    gr_gran))

        # Cross-MT dedup: if a tile/k range appears at both n+1 and n+2,
        # the n+1 load is redundant — the previous iteration's n+2 already
        # wrote the same data into LDS.  Remove the n+1 duplicate.
        base_mt = f"n+{offsetMT}"
        n2_keys = {(t, ts, te, ks, ke)
                   for t, mt, ts, te, ks, ke, _ in gr_list
                   if mt != base_mt}
        gr_list = [entry for entry in gr_list
                   if entry[1] != base_mt or
                   (entry[0], entry[2], entry[3], entry[4], entry[5])
                   not in n2_keys]

        if debug:
            print(f"Phase 1: {len(gr_list)} GR entries")
            for i, (t, mt, ts, te, ks, ke, g) in enumerate(gr_list):
                loads = ((te - ts) // g.size.mn) * ((ke - ks) // g.size.k)
                print(f"  [{i}] {t:2s} {mt} tiles[{ts},{te - 1}] k[{ks},{ke - 1}] "
                      f"gr_gran(mn={g.size.mn},k={g.size.k}) loads={loads}")

        return gr_list

    def _build_lr_conflict_map(self):
        """Build per-partition LR(MT n) info for LDS conflict checking.

        Returns dict: (partition_idx, tensor) -> list of
                      (subIterK_slot, k_start, k_end).
        """
        lr_mt_n_info = {}
        for pi, partition_slots in enumerate(self._partitions):
            for slot in partition_slots:
                for lr in slot.lrs:
                    if lr.mtIteration == "n":
                        lr_mt_n_info.setdefault((pi, lr.tensor), []).append(
                            (slot.subIterK,
                             lr.tiles.subIterK_start,
                             lr.tiles.subIterK_end))
        return lr_mt_n_info

    @staticmethod
    def _has_lr_conflict(lr_mt_n_info, tensor, mt_str, pi, subIterK,
                         gr_k_start, gr_k_end):
        """Return True if placing GR(mt_str) at (pi, subIterK) conflicts.

        GR(MT n+2) writes the same LDS buffer as MT n, so it conflicts
        only if a later LR(MT n) in the same partition accesses an
        overlapping subIterK range.
        """
        if "n+2" not in mt_str:
            return False
        for lr_slot, lr_ks, lr_ke in lr_mt_n_info.get((pi, tensor), []):
            if lr_slot > subIterK and gr_k_start < lr_ke and lr_ks < gr_k_end:
                return True
        return False

    def _distribute_grs(self, gr_list, lr_mt_n_info, debug=False):
        """Phase 2: Distribute GR atoms across partition × subIterK slots.

        Explodes GR entries into atomic loads, distributes them into flat
        buckets respecting LDS conflict constraints and load balance,
        then remerges consecutive atoms and places them into partitions.
        """
        cfg = self.config
        numK = cfg.numSubIterK
        numP = cfg.numPartitions
        numSlots = numP * numK

        # 2a. Explode GR entries into atomic loads (1 load each)
        atoms = []
        for tensor, mt_str, t_start, t_end, k_start, k_end, gr_gran in gr_list:
            mn = gr_gran.size.mn
            for pos in range(t_start, t_end, mn):
                atoms.append((tensor, mt_str, pos, pos + mn, k_start, k_end))

        loads_per_slot = len(atoms) // numSlots

        # 2b. Distribute atoms into flat buckets [0..numSlots),
        #     each bucket maps to (partition=flat//numK, subIterK=flat%numK)
        buckets = [[] for _ in range(numSlots)]
        for atom in atoms:
            tensor, mt_str, _, _, ks, ke = atom
            cur = 0
            while cur < numSlots - 1:
                pi = cur // numK
                subK = cur % numK
                if (not self._has_lr_conflict(lr_mt_n_info, tensor, mt_str,
                                              pi, subK, ks, ke) and
                        len(buckets[cur]) < loads_per_slot):
                    break
                cur += 1
            buckets[cur].append(atom)

        if debug:
            print(f"Phase 2b: {len(atoms)} atoms, {numSlots} slots, "
                  f"{loads_per_slot} per slot")
            for flat, bucket in enumerate(buckets):
                pi = flat // numK
                si = flat % numK
                if bucket:
                    items = ", ".join(
                        f"{t} {mt} tile[{ts},{te-1}] k[{ks},{ke-1}]"
                        for t, mt, ts, te, ks, ke in bucket)
                    print(f"  P{pi} s{si}: {len(bucket)} atoms — {items}")
                else:
                    print(f"  P{pi} s{si}: empty")

        # 2c. Remerge consecutive atoms and place into partitions
        for flat, bucket in enumerate(buckets):
            pi = flat // numK
            si = flat % numK
            target_slot = self._partitions[pi][si]
            for atom in bucket:
                tensor, mt_str, ts, te, ks, ke = atom
                if target_slot.grs:
                    prev = target_slot.grs[-1]
                    if (prev.tensor == tensor and
                            prev.mtIteration == mt_str and
                            prev.tiles.subIterK_start == ks and
                            prev.tiles.subIterK_end == ke and
                            prev.tiles.tileId_end == ts):
                        prev.tiles = MFMATileRange(ks, ke, prev.tiles.tileId_start, te)
                        continue
                target_slot.grs.append(GRPlacement(
                    tensor=tensor, mtIteration=mt_str,
                    tiles=MFMATileRange(ks, ke, ts, te),
                    subIterK_slot=si,
                    partition=pi))

    def place_GRs(self) -> List[SubIterKSlot]:
        """Place Global Reads by iterating MFMAs across partitions.

        Phase 1: Build ordered GR list from partition traversal respecting gr granularities.
        Phase 2: Distribute evenly GR atoms across all (partition, subIterK) slots. GR atoms being the smallest load granularity for a specific tensor.

        This should give a sheduling respecting the following rules:
         - GR are in the order we expect them from the LR pov
         - we respect the GR granularities (can change the above rule a bit)
         - Overall loads are spread accross all subIterKs of all partitions.

         TODO:
          - handle 1x4 and 4x1 GR granularities and test them
          - support swapping A and B
        """
        if 'lr' not in self._completed:
            self.place_LRs()

        part_ranges = [self._partition_tile_range(pi)
                       for pi in range(self.config.numPartitions)]

        # TODO: cover PGR3 (offsetMT and offsetPartition may differ)
        offsetMT = 1
        offsetPartition = 1
        # Build ordered list of GRs to place for the entire MT based on the partitioning ordering and the GR granularities.
        gr_list = self._build_gr_list(part_ranges, offsetMT, offsetPartition)
        # Map to keep track of LR(MT n) for each partiion and tensor, used for LDS double buffer conflict checking when placing GRs.
        lr_mt_n_info = self._build_lr_conflict_map()
        # Distribute GRs accross partition.
        self._distribute_grs(gr_list, lr_mt_n_info)

        self._completed.add('gr')
        return self._partitions[0]

    # ── Annotate dependencies ─────────────────────────────

    def annotate_deps(self):
        """Annotate each placement with its raw before-dependencies.

        Populates the `before` field on MFMAPlacement, LRPlacement, and
        GRPlacement objects in self._partitions. Each lr_ref/gr_ref DepOp
        is resolved to point at the specific placement it depends on.

        Iterates all partitions. Two-pass per partition:
        - Pass 1: build lookups from existing placements
        - Pass 2: populate .before on each placement

        Rules:
        - MFMA(subIterK=k) depends on all LRs that loaded subIterK=k data
          (cross-partition: LRs for a tensor may be in any partition)
        - LR depends on GR for same tensor (data must be in LDS)
        - GR depends on collision LR for same tensor (LDS double-buffer)
        """
        if 'gr' not in self._completed:
            self.place_GRs()
        cfg = self.config
        numK = cfg.numSubIterK

        # Build global lr_by_data across all partitions (MFMA deps are cross-partition)
        # lr_by_data[data_k][tensor] → list of LRPlacements loading subIterK=data_k
        lr_by_data = [{} for _ in range(numK)]
        # gr_by_tensor[tensor] → list of all GRPlacements (LR→GR deps are cross-partition)
        gr_by_tensor = {}
        # lr_by_tensor[tensor] → list of all LRPlacements (GR→LR collision is cross-partition)
        lr_by_tensor = {}
        for slots in self._partitions:
            for slot in slots:
                for lr in slot.lrs:
                    for data_k in lr.tiles.subIterK_list:
                        lr_by_data[data_k].setdefault(lr.tensor, []).append(lr)
                    lr_by_tensor.setdefault(lr.tensor, []).append(lr)
                for gr in slot.grs:
                    gr_by_tensor.setdefault(gr.tensor, []).append(gr)

        for pi, slots in enumerate(self._partitions):
            self._annotate_deps_partition(pi, slots, cfg, lr_by_data,
                                          gr_by_tensor, lr_by_tensor)

        self._completed.add('deps')

    def _annotate_deps_partition(self, pi: int, slots: List[SubIterKSlot],
                                 cfg: SchedulerConfig, lr_by_data: list,
                                 gr_by_tensor: dict, lr_by_tensor: dict):
        """Annotate deps for a single partition (in-place on placements)."""
        numK = len(slots)

        # Clear any previous annotations (idempotent re-runs)
        for slot in slots:
            if slot.mfma:
                slot.mfma.deps.clear()
            for lr in slot.lrs:
                lr.deps.clear()
            for gr in slot.grs:
                gr.deps.clear()

        # ── Pass 1: build per-partition lookups ──
        # lr_by_slot[k][tensor] → LRPlacement at subIterK=k
        # gr_by_slot[k][tensor] → GRPlacement at subIterK=k
        # (lr_by_data, gr_by_tensor, lr_by_tensor are built globally in annotate_deps)
        lr_by_slot = [{} for _ in range(numK)]
        gr_by_slot = [{} for _ in range(numK)]

        for k, slot in enumerate(slots):
            for lr in slot.lrs:
                lr_by_slot[k][lr.tensor] = lr

            for gr in slot.grs:
                gr_by_slot[k][gr.tensor] = gr

        # ── Pass 2: populate deps on each placement ──
        # mt_offset: 0 = same MT, -1 = prev MT, -2 = two MTs back, etc.
        # Within one iteration, execution order per slot is MFMA → LR → GR,
        # and slots run in order 0, 1, 2, ...
        _order = {'MFMA': 0, 'LR': 1, 'GR': 2}

        def _parse_mt(mt_str):
            """'n' → 0, 'n+1' → 1, 'n+2' → 2."""
            return 0 if mt_str == "n" else int(mt_str.split('+')[1])

        def _slot_offset(consumer_partition, consumer_slot, consumer_type, producer):
            """Offset from partition+slot ordering: 0 if producer ran first, -1 otherwise."""
            prod_partition = producer.partition
            if prod_partition < consumer_partition:
                return 0
            if prod_partition > consumer_partition:
                return -1
            prod_slot = producer.subIterK_slot
            if prod_slot < consumer_slot:
                return 0
            if prod_slot > consumer_slot:
                return -1
            prod_type = 'LR' if isinstance(producer, LRPlacement) else 'GR'
            return -1 if _order[prod_type] >= _order[consumer_type] else 0

        def _mt_offset(consumer_partition, consumer_slot, consumer_type, producer, consumer=None):
            # MFMA→LR: MFMA always consumes mt="n" (offset 0).
            if consumer_type == 'MFMA' and isinstance(producer, LRPlacement):
                mt_off = _parse_mt(producer.mtIteration)
                if mt_off > 0:
                    return -mt_off
            # LR→GR: mt difference determines how many iterations back.
            if consumer_type == 'LR' and isinstance(producer, GRPlacement) and consumer:
                diff = _parse_mt(producer.mtIteration) - _parse_mt(consumer.mtIteration)
                if diff != 0:
                    return -diff
            # Same effective mt: partition+slot ordering decides.
            return _slot_offset(consumer_partition, consumer_slot, consumer_type, producer)

        def _tiles_overlap(mfma, lr_tensor, lr_tiles):
            """Check if LR tile range overlaps with MFMA's tile range for that tensor."""
            # SA/SB follow A/B tile ranges respectively
            if lr_tensor in ('A', 'SA'):
                mfma_range = mfma.tileA
            else:
                mfma_range = mfma.tileB
            return (lr_tiles.tileId_start < mfma_range.tileId_end and
                    lr_tiles.tileId_end > mfma_range.tileId_start)

        def _range_overlaps(a: MFMATileRange, b: MFMATileRange) -> bool:
            """Check if two tile ranges overlap on both tile ids and subIterK."""
            return (a.tileId_start < b.tileId_end and
                    a.tileId_end > b.tileId_start and
                    a.subIterK_start < b.subIterK_end and
                    a.subIterK_end > b.subIterK_start)

        def _dedup_deps(deps):
            if len(deps) <= 1:
                return deps
            def _exec_order(dep):
                return (dep.mt_offset, dep.ref.partition, dep.ref.subIterK_slot)
            return [max(deps, key=_exec_order)]

        for k, slot in enumerate(slots):
            # MFMA: depends on the most recent LR per tensor (tile-overlapping).
            # Uses lr_by_tensor (all LRs across partitions) so that a more recent
            # LR loading a different subIterK still subsumes older data deps.
            if slot.mfma:
                tensor_names = ['A', 'B']
                if cfg.hasScale:
                    tensor_names += ['SA', 'SB']
                for t in tensor_names:
                    deps_for_t = []
                    for lr in lr_by_tensor.get(t, []):
                        if _tiles_overlap(slot.mfma, t, lr.tiles):
                            deps_for_t.append(DepRef(
                                ref=lr, mt_offset=_mt_offset(pi, k, 'MFMA', lr)))
                    slot.mfma.deps.extend(_dedup_deps(deps_for_t))

            # LR: depends on GR (data must be in LDS before reading)
            # Cross-partition: the GR that loaded the matching tiles may be
            # in a different partition. Filter by tile overlap.
            for lr in slot.lrs:
                for gr in gr_by_tensor.get(lr.tensor, []):
                    if _range_overlaps(lr.tiles, gr.tiles):
                        lr.deps.append(DepRef(
                            ref=gr, mt_offset=_mt_offset(pi, k, 'LR', gr, consumer=lr)))

            # GR: depends on collision LR (LDS double-buffer)
            # GR(n+x) collides with LR(n+x-2) — same buffer, period 2.
            # target_data = parse_mt(gr.mt) - 2. For each LR of same tensor,
            # mt_offset = target_data - parse_mt(lr.mt). Dedup keeps latest.
            #   GR(n+2)→LR(n):   mt_offset = 0   (same iteration)
            #   GR(n+2)→LR(n+1): mt_offset = -1  (prev iter LR(n+1) handled n)
            #   GR(n+1)→LR(n):   mt_offset = -1  (prev iter LR(n) handled n-1)
            for gr in slot.grs:
                target_data = _parse_mt(gr.mtIteration) - 2
                for lr in lr_by_tensor.get(gr.tensor, []):
                    if _range_overlaps(lr.tiles, gr.tiles):
                        mt_off = target_data - _parse_mt(lr.mtIteration)
                        gr.deps.append(DepRef(ref=lr, mt_offset=mt_off))
                if not gr.deps:
                    raise ValueError(
                        f"GR {gr.tensor} mt={gr.mtIteration} at slot {k} "
                        f"has no overlapping LR(n) dependency")

        for slot in slots:
            for lr in slot.lrs:
                lr.deps = _dedup_deps(lr.deps)
            for gr in slot.grs:
                gr.deps = _dedup_deps(gr.deps)

    # ── Remove unnecessary GR deps ────────────────────────

    def remove_unnecessary_gr_deps(self):
        """Remove GR deps on LRs that are already guaranteed by an earlier LR's wait.

        Per tensor, walks LR placements in execution order. If an earlier LR
        already waits for a GR with equal or higher exec_order, the later LR's
        dep is redundant and removed.

        Wraps around: the first LR's dep is compared against the last from the
        previous MT iteration (max dep exec_order shifted by mt_offset -1).
        """
        if 'deps' not in self._completed:
            self.annotate_deps()

        def _dep_exec_order(dep):
            return (dep.mt_offset, dep.ref.partition, dep.ref.subIterK_slot)

        tensors = ['A', 'B']
        if self.config.hasScale:
            tensors += ['SA', 'SB']

        for tensor in tensors:
            lr_with_gr_deps = []
            for pi, slots in enumerate(self._partitions):
                for slot in slots:
                    for lr in slot.lrs:
                        if lr.tensor == tensor and lr.deps:
                            dep = lr.deps[0]
                            if isinstance(dep.ref, GRPlacement):
                                lr_with_gr_deps.append((lr, dep))

            if len(lr_with_gr_deps) <= 1:
                continue

            max_eo = max(_dep_exec_order(dep) for _, dep in lr_with_gr_deps)
            max_guaranteed = (max_eo[0] - 1, max_eo[1], max_eo[2])

            for lr, dep in lr_with_gr_deps:
                eo = _dep_exec_order(dep)
                if eo <= max_guaranteed:
                    lr.deps.clear()
                else:
                    max_guaranteed = eo

        self._completed.add('remove_gr_deps')

    # ── Remove unnecessary LR deps ────────────────────────

    def remove_unnecessary_lr_deps(self):
        """Remove GR→LR deps that are already covered by earlier sync points.

        Two redundancy checks:
        1. MFMA-based: if the MFMA at the previous GR's position depends on a
           same-tensor LR with a later exec order, the dep is redundant.
        2. GR-based: if a previous GR (in execution order) already had the same
           LR dep (same tensor, same or later exec order), the current GR's dep
           is redundant because the previous GR already created that sync point.
        """
        if 'remove_gr_deps' not in self._completed:
            self.remove_unnecessary_gr_deps()

        def _dep_exec_order(dep):
            return (dep.mt_offset, dep.ref.partition, dep.ref.subIterK_slot)

        gr_with_lr_deps = []
        for pi, slots in enumerate(self._partitions):
            for slot in slots:
                for gr in slot.grs:
                    if gr.deps:
                        dep = gr.deps[0]
                        if isinstance(dep.ref, LRPlacement):
                            gr_with_lr_deps.append((pi, slot.subIterK, gr, dep))

        if len(gr_with_lr_deps) <= 1:
            self._completed.add('remove_lr_deps')
            return

        mfma_by_pos = {}
        for pi, slots in enumerate(self._partitions):
            for slot in slots:
                if slot.mfma:
                    mfma_by_pos[(pi, slot.subIterK)] = slot.mfma

        gr_with_lr_deps.sort(key=lambda x: (x[0], x[1]))

        prev_gr_pos = (gr_with_lr_deps[-1][0], gr_with_lr_deps[-1][1])
        # Track the max LR exec order synced per tensor by previous GRs
        prev_gr_lr_eo = {}

        for pi, subIterK, gr, dep in gr_with_lr_deps:
            eo = _dep_exec_order(dep)
            tensor = dep.ref.tensor
            removed = False
            # Check 1: MFMA at previous GR position syncs on a later LR
            mfma = mfma_by_pos.get(prev_gr_pos)
            if mfma and mfma.deps:
                same_tensor_deps = [d for d in mfma.deps
                                    if isinstance(d.ref, LRPlacement)
                                    and d.ref.tensor == tensor]
                if same_tensor_deps:
                    mfma_max_eo = max(_dep_exec_order(d)
                                      for d in same_tensor_deps)
                    if mfma_max_eo > eo:
                        gr.deps.clear()
                        removed = True
            # Check 2: a previous GR already synced on same or later LR
            if not removed and tensor in prev_gr_lr_eo:
                if prev_gr_lr_eo[tensor] >= eo:
                    gr.deps.clear()
                    removed = True

            if not removed:
                prev_gr_lr_eo[tensor] = eo
            prev_gr_pos = (pi, subIterK)

        self._completed.add('remove_lr_deps')

    # ── Remove cross-subIterK deps ─────────────────────────

    def _gr_granularity(self, tensor: str) -> ReadGranularity:
        """Return GR granularity for a tensor."""
        return {'A': self.config.grA, 'B': self.config.grB,
                'SA': self.config.grSA, 'SB': self.config.grSB}[tensor]

    def _compute_inflight_loads(self, consumer_pi: int, consumer_slot: int,
                                tensor: str, dep_ref: DepRef) -> WaitGRCounts:
        """Count inflight GR atomic loads between a dep GR and the consumer.

        Walks backward through the flattened schedule (all partitions x subIterK)
        from the consumer position, counting atomic GR loads for all tensors.
        Stops when reaching the dependency GR (dep_ref.ref) after accounting
        for mt_offset wraps.

        Returns per-tensor inflight load counts.
        """
        numP = len(self._partitions)
        numK = len(self._partitions[0])
        flat_len = numP * numK

        consumer_flat = consumer_pi * numK + consumer_slot

        wraps_needed = abs(dep_ref.mt_offset)

        counts = WaitGRCounts()
        wraps_completed = 0
        pos = consumer_flat

        max_steps = (wraps_needed + 1) * flat_len
        for _ in range(max_steps):
            pos = (pos - 1) % flat_len
            if pos == flat_len - 1 and _ > 0:
                wraps_completed += 1

            pi = pos // numK
            slot_k = pos % numK
            slot = self._partitions[pi][slot_k]

            for gr in slot.grs:
                if gr.tensor == tensor and gr is dep_ref.ref and wraps_completed >= wraps_needed:
                    return counts
                gr_gran = self._gr_granularity(gr.tensor)
                tiles = gr.tiles
                n_tile = (tiles.tileId_end - tiles.tileId_start) // gr_gran.size.mn
                n_k = (tiles.subIterK_end - tiles.subIterK_start) // gr_gran.size.k
                cur = getattr(counts, gr.tensor)
                setattr(counts, gr.tensor, cur + n_tile * n_k)

        return counts

    def remove_cross_deps(self):
        """Replace cross-subIterK deps with wait preOps.

        For each placement, separates deps into same-subIterK (kept) and
        cross-subIterK (converted to preOps):
          - MFMA depending on LRs → single wait_lr
          - GR depending on LRs   → single wait_lr_sync
          - LR depending on GRs   → single wait_gr_sync with per-tensor inflight counts
        """
        if 'remove_lr_deps' not in self._completed:
            self.remove_unnecessary_lr_deps()

        for pi, slots in enumerate(self._partitions):
            for slot in slots:
                # ── MFMA ──
                if slot.mfma:
                    same, cross = self._split_deps(slot.mfma.deps, pi, slot.subIterK)
                    slot.mfma.deps = same
                    slot.mfma.preOps = []
                    if cross:
                        slot.mfma.preOps.append(DepOp(kind='wait_lr'))

                # ── LRs ──
                for lr in slot.lrs:
                    same, cross = self._split_deps(lr.deps, pi, lr.subIterK_slot)
                    lr.deps = same
                    lr.preOps = []
                    if cross:
                        dep = cross[0]
                        counts = self._compute_inflight_loads(
                            pi, lr.subIterK_slot, dep.ref.tensor, dep)
                        lr.preOps.append(DepOp(kind='wait_gr_sync',
                                               wait_gr_counts=counts))

                # ── GRs ──
                for gr in slot.grs:
                    same, cross = self._split_deps(gr.deps, pi, gr.subIterK_slot)
                    gr.deps = same
                    gr.preOps = [DepOp(kind='wait_lr_sync')]

        self._completed.add('remove_deps')

    def insert_gr_lr_inc(self):
        """Insert gr_inc/lr_inc preOps at MacroTile iteration transitions.

        Walks all LR and GR placements in global execution order
        (partition 0 slots → partition 1 slots → ..., within each slot: LR then GR).
        Tracks per-tensor the last-seen mtIteration. When a tensor's mtIteration
        changes, inserts a DepOp into that placement's preOps:
          - lr_inc for LR placements
          - gr_inc for GR placements
        """
        if 'remove_deps' not in self._completed:
            self.remove_cross_deps()

        last_lr_mt = {}  # tensor -> mtIteration for LR only
        last_gr_mt = {}  # tensor -> mtIteration for GR only
        first_lr = {}  # tensor -> first LR placement seen
        lr_inc_tensors = set()  # tensors that already received lr_inc

        for pi, slots in enumerate(self._partitions):
            for slot in slots:
                for lr in slot.lrs:
                    tensor = lr.tensor
                    mt = lr.mtIteration
                    if tensor not in first_lr:
                        first_lr[tensor] = lr
                    if tensor in last_lr_mt and last_lr_mt[tensor] != mt:
                        lr.preOps.append(DepOp(kind='lr_inc', tensor=tensor))
                        lr_inc_tensors.add(tensor)
                    last_lr_mt[tensor] = mt
                for gr in slot.grs:
                    tensor = gr.tensor
                    mt = gr.mtIteration
                    prev_mt = last_gr_mt.get(tensor, last_lr_mt.get(tensor))
                    if prev_mt is not None and prev_mt != mt:
                        if gr.tiles.tileId_start == 0:
                            gr.preOps.append(DepOp(kind='gr_inc', tensor=tensor))
                    last_gr_mt[tensor] = mt

        # Handle wrap-around: tensors with a single LR per iteration (e.g. SA, SB)
        # still need lr_inc because the GR at end-of-iteration writes to the other
        # LDS buffer, and the next iteration's LR must swap to read from it.
        for tensor, lr in first_lr.items():
            if tensor not in lr_inc_tensors:
                last = last_gr_mt.get(tensor, last_lr_mt.get(tensor))
                if last is not None and last != lr.mtIteration:
                    lr.preOps.append(DepOp(kind='lr_inc', tensor=tensor))

        self._completed.add('gr_inc')

    # ── Group LR/GR chains ─────────────────────────────────────

    _LR_GR_ORDER = ['A', 'B', 'SA', 'SB']

    @staticmethod
    def _merge_preops(all_preops: List[List['DepOp']]) -> List['DepOp']:
        """Merge preOps from multiple placements.

        Combines wait_gr/wait_gr_sync counts into a single DepOp, deduplicates barrier ops
        (wait_lr_sync, wait_lr), and collects the rest.
        """
        wait_gr_ops = []
        has_wait_gr_sync = False
        seen_kinds = set()
        others = []
        for preops in all_preops:
            for op in preops:
                if op.kind in ('wait_gr', 'wait_gr_sync') and op.wait_gr_counts:
                    if op.kind == 'wait_gr_sync':
                        has_wait_gr_sync = True
                    wait_gr_ops.append(op.wait_gr_counts)
                elif op.kind in ('wait_lr_sync', 'wait_lr'):
                    if op.kind not in seen_kinds:
                        seen_kinds.add(op.kind)
                        others.append(op)
                else:
                    others.append(op)
        result = []
        if wait_gr_ops:
            merged_counts = WaitGRCounts()
            for t in ('A', 'B', 'SA', 'SB'):
                setattr(merged_counts, t, min(getattr(c, t) for c in wait_gr_ops))
            merged_kind = 'wait_gr_sync' if has_wait_gr_sync else 'wait_gr'
            result.append(DepOp(kind=merged_kind, wait_gr_counts=merged_counts))
        result.extend(others)
        return result

    def group_lr_gr(self):
        """Group LR and GR placements into chains within each subIterK.

        Phase 1 — LR chain:
          Sort LRs by tensor order (A, B, SA, SB).  Build a dep chain so each
          LR depends on the previous one.  Merge all preOps onto the first LR
          (wait_gr counts are combined, other preOps are collected).

        Phase 2 — GR chain:
          Sort GRs by tensor order (A, B, SA, SB).  Build a dep chain.  If any
          GR originally had same-subIterK deps, replace the first GR's deps with
          a single dep on the last LR of the phase-1 chain.  Merge all preOps
          onto the first GR.
        """
        if 'gr_inc' not in self._completed:
            self.insert_gr_lr_inc()

        order = self._LR_GR_ORDER

        for pi, slots in enumerate(self._partitions):
            for slot in slots:
                # ── Phase 1: LR chain ──
                ordered_lrs = sorted(
                    slot.lrs,
                    key=lambda lr: order.index(lr.tensor))

                if len(ordered_lrs) > 1:
                    # Merge preOps onto first LR
                    merged = self._merge_preops(
                        [lr.preOps for lr in ordered_lrs])
                    ordered_lrs[0].preOps = merged
                    for lr in ordered_lrs[1:]:
                        lr.preOps = []

                    # Build chain: each LR depends on the previous
                    for i in range(1, len(ordered_lrs)):
                        ordered_lrs[i].deps = [
                            DepRef(ref=ordered_lrs[i - 1], mt_offset=0)]

                last_lr = ordered_lrs[-1] if ordered_lrs else None

                # ── Phase 2: GR chain ──
                ordered_grs = sorted(
                    slot.grs,
                    key=lambda gr: order.index(gr.tensor))

                if len(ordered_grs) > 1:
                    # Check if any GR has same-subIterK deps
                    any_deps = any(gr.deps for gr in ordered_grs)

                    # Merge preOps onto first GR
                    merged = self._merge_preops(
                        [gr.preOps for gr in ordered_grs])
                    ordered_grs[0].preOps = merged
                    for gr in ordered_grs[1:]:
                        gr.preOps = []

                    # First GR: if any GR had deps, point to last LR
                    if any_deps and last_lr is not None:
                        ordered_grs[0].deps = [
                            DepRef(ref=last_lr, mt_offset=0)]
                    else:
                        ordered_grs[0].deps = []

                    # Build chain: each GR depends on the previous
                    for i in range(1, len(ordered_grs)):
                        ordered_grs[i].deps = [
                            DepRef(ref=ordered_grs[i - 1], mt_offset=0)]
                elif len(ordered_grs) == 1:
                    # Single GR: still consolidate dep to last LR if it had deps
                    if ordered_grs[0].deps and last_lr is not None:
                        ordered_grs[0].deps = [
                            DepRef(ref=last_lr, mt_offset=0)]

        self._completed.add('group_lr_gr')

    def remove_unnecessary_wait_lr_sync(self):
        """Remove redundant wait_lr_sync from GRs after grouping.
        Given that we always use wait_lr cnt=0, grouping can guarantee future wait_lr_sync.

        A GR's wait_lr_sync is unnecessary when:
          1. The GR has no same-subIterK deps (deps is empty after grouping)
          2. The previous subIterK's GRs already have a wait_lr_sync
          3. That previous wait_lr_sync is ordered after all LRs in the
             previous subIterK (the GR has deps on the LR chain)

        In that case, all prior LR reads were already synced by the previous
        subIterK's barrier, and the current GR doesn't conflict with any LRs
        in its own subIterK, so the second wait_lr_sync is redundant.
        """
        if 'group_lr_gr' not in self._completed:
            self.group_lr_gr()

        for pi, slots in enumerate(self._partitions):
            for si, slot in enumerate(slots):
                if not slot.grs:
                    continue
                first_gr = slot.grs[0]
                has_wait_lr_sync = any(
                    op.kind == 'wait_lr_sync' for op in first_gr.preOps)
                if not has_wait_lr_sync:
                    continue
                has_deps = bool(first_gr.deps)
                if has_deps:
                    continue
                # Check previous subIterK in the same partition
                if si == 0:
                    continue
                prev_slot = slots[si - 1]
                if not prev_slot.grs:
                    continue
                prev_first_gr = prev_slot.grs[0]
                prev_has_wait_lr_sync = any(
                    op.kind == 'wait_lr_sync' for op in prev_first_gr.preOps)
                prev_deps_on_lrs = bool(prev_first_gr.deps)
                if prev_has_wait_lr_sync and prev_deps_on_lrs:
                    first_gr.preOps = [
                        op for op in first_gr.preOps
                        if op.kind != 'wait_lr_sync']

        self._completed.add('remove_wait_lr_sync')

    def _split_deps(self, deps: List[DepRef], consumer_pi: int,
                    consumer_slot: int) -> Tuple[List[DepRef], List[DepRef]]:
        """Split deps into same-subIterK and cross-subIterK lists.

        A dep is "same subIterK" if mt_offset == 0 AND the producer is in the
        same partition and same subIterK slot as the consumer.
        """
        same, cross = [], []
        for dep in deps:
            if (dep.mt_offset == 0 and
                    dep.ref.partition == consumer_pi and
                    dep.ref.subIterK_slot == consumer_slot):
                same.append(dep)
            else:
                cross.append(dep)
        return same, cross

    def emit(self) -> List[List[List[EmittedModule]]]:
        """Convert placements into EmittedModule chains per partition per subIterK.

        Returns [partition][subIterK][EmittedModule].

        Each subIterK list contains:
          - Primary modules (MFMA, LRs, GRs) with opType and label
          - Dependency modules (wait_gr, wait_lr, sync, lr_inc, gr_inc, gr_scale)
            emitted from preOps, chained via before-links

        The before-link topology matches SubtileBasedScheduler._buildEmittedModules:
          - wait_gr is standalone (no incoming before-link), but later deps chain from it
          - wait_gr_sync expands to two modules: wait_gr then sync
          - wait_lr_sync expands to two modules: wait_lr then sync
          - Same-subIterK DepRef deps become ordering constraints (no new module)
        """
        if 'remove_wait_lr_sync' not in self._completed:
            self.remove_unnecessary_wait_lr_sync()

        all_partitions = []
        for pi, slots in enumerate(self._partitions):
            partition_emitted = []
            for slot in slots:
                emitted: List[EmittedModule] = []
                placement_to_id = {}

                def add(opType: str, label: str, source: object = None) -> int:
                    mid = len(emitted)
                    emitted.append(EmittedModule(
                        moduleId=mid, opType=opType, label=label, source=source))
                    return mid

                def setBefore(moduleId: int, beforeId: int) -> None:
                    if beforeId is None or beforeId == moduleId:
                        return
                    cur = emitted[moduleId].before
                    if cur is None:
                        emitted[moduleId].before = beforeId
                        return
                    assert cur == beforeId, \
                        f"EmittedModule {moduleId} has multiple before deps: {cur} and {beforeId}"

                # Step 1: emit primary modules
                placements = []
                if slot.mfma:
                    placements.append(('mfma', slot.mfma))
                for lr in slot.lrs:
                    placements.append(('lr', lr))
                for gr in slot.grs:
                    placements.append(('gr', gr))

                for opType, placement in placements:
                    label = self._format_placement_label(placement, slot)
                    mid = add(opType, label, source=placement)
                    placement_to_id[id(placement)] = mid

                # Step 2: wire before-chains from preOps + deps
                for opType, placement in placements:
                    curId = placement_to_id[id(placement)]
                    prevId = None
                    lastDepId = None
                    firstPreOpId = None

                    # preOps
                    for preOp in placement.preOps:
                        if preOp.kind == 'wait_gr':
                            # Standalone: no incoming before-link, but later
                            # deps chain from it
                            depId = add('wait_gr', str(preOp), source=preOp)
                            prevId = depId
                            if firstPreOpId is None:
                                firstPreOpId = depId
                            continue
                        elif preOp.kind == 'wait_gr_sync':
                            # Expand to wait_gr + sync
                            depId = add('wait_gr', str(preOp), source=preOp)
                            prevId = depId
                            if firstPreOpId is None:
                                firstPreOpId = depId
                            depId = add('sync', 'sync',
                                        source=DepOp(kind='sync'))
                            setBefore(depId, prevId)
                            prevId = depId
                            lastDepId = depId
                            continue
                        elif preOp.kind == 'wait_lr_sync':
                            # Expand to wait_lr + sync
                            depId = add('wait_lr', 'wait_lr',
                                        source=DepOp(kind='wait_lr'))
                            setBefore(depId, prevId)
                            prevId = depId
                            lastDepId = depId
                            if firstPreOpId is None:
                                firstPreOpId = depId
                            depId = add('sync', 'sync',
                                        source=DepOp(kind='sync'))
                            setBefore(depId, prevId)
                            prevId = depId
                            lastDepId = depId
                            continue
                        else:
                            depId = add(preOp.kind, str(preOp), source=preOp)
                            setBefore(depId, prevId)
                            prevId = depId
                            lastDepId = depId
                            if firstPreOpId is None:
                                firstPreOpId = depId

                    # deps (same-subIterK DepRefs — ordering constraints)
                    # Wire dep refs as roots of the preOp chain so the
                    # dependency is not lost when preOps are present.
                    for dep in placement.deps:
                        ref_id = placement_to_id.get(id(dep.ref))
                        if ref_id is not None:
                            if firstPreOpId is not None:
                                setBefore(firstPreOpId, ref_id)
                            else:
                                prevId = ref_id

                    # Final link: primary module points to last dep
                    if lastDepId is not None:
                        setBefore(curId, lastDepId)
                    elif prevId is not None:
                        setBefore(curId, prevId)

                partition_emitted.append(emitted)
            all_partitions.append(partition_emitted)

        self._emitted = all_partitions
        self._completed.add('emit')
        return all_partitions

    def build(self):
        """Build mainloop """
        self.emit()
        self._completed.add('build')

    # ── Loop variant derivation ────────────────────────────

    @staticmethod
    def _rewire_before(emitted: List[EmittedModule],
                       removed_ids: set) -> List[EmittedModule]:
        """Rewire before-links that point to removed modules.

        If em.before points to a removed module, follow that module's own
        before link until we find a non-removed module (or None).
        """
        id_to_em = {em.moduleId: em for em in emitted}
        for em in emitted:
            if em.moduleId in removed_ids:
                continue
            b = em.before
            while b is not None and b in removed_ids:
                b = id_to_em[b].before
            em.before = b
        return [em for em in emitted if em.moduleId not in removed_ids]

    def build_ngll(self) -> List[List[List[EmittedModule]]]:
        """NGLL (No Global Load Loop): mainloop without GR(n+2), GR_INC, GRScale(n+2).

        WaitGR inflight counts are zeroed since no new GRs are in flight.
        """
        if 'emit' not in self._completed:
            self.emit()

        ngll = []
        for partition_emitted in self._emitted:
            part_ngll = []
            for emitted in partition_emitted:
                new_emitted = copy.deepcopy(emitted)
                removed = set()
                for em in new_emitted:
                    src = em.source
                    if em.opType == 'gr' and isinstance(src, GRPlacement) \
                            and src.mtIteration == 'n+2':
                        removed.add(em.moduleId)
                    elif em.opType == 'gr_inc':
                        removed.add(em.moduleId)
                    elif em.opType == 'gr_scale' and isinstance(src, DepOp) \
                            and hasattr(src, 'mtIteration') \
                            and getattr(src, 'mtIteration', None) == 'n+2':
                        removed.add(em.moduleId)
                    elif em.opType == 'wait_gr':
                        if isinstance(src, DepOp) and src.wait_gr_counts is not None:
                            src.wait_gr_counts = WaitGRCounts()
                part_ngll.append(self._rewire_before(new_emitted, removed))
            ngll.append(part_ngll)

        self._ngll_emitted = ngll
        return ngll

    def build_nll(self) -> List[List[List[EmittedModule]]]:
        """NLL (No Load Loop): mainloop without GR, LR(n+1), GR_INC, LR_INC,
        GRScale, WaitGR(n+1)+Sync. Keeps LR(n), MFMAs, WaitGR(n) with zeroed counts."""
        if 'emit' not in self._completed:
            self.emit()

        nll = []
        for partition_emitted in self._emitted:
            part_nll = []
            for emitted in partition_emitted:
                new_emitted = copy.deepcopy(emitted)
                removed = set()

                for em in new_emitted:
                    src = em.source
                    if em.opType == 'gr':
                        removed.add(em.moduleId)
                    elif em.opType == 'lr' and isinstance(src, LRPlacement) \
                            and src.mtIteration == 'n+1':
                        removed.add(em.moduleId)
                    elif em.opType in ('gr_inc', 'lr_inc', 'gr_scale'):
                        removed.add(em.moduleId)

                # Remove WaitGR(n+1) and its paired Sync.
                # Also zero inflight counts on remaining WaitGR(n).
                wait_gr_to_remove = set()
                for em in new_emitted:
                    if em.opType == 'wait_gr' and isinstance(em.source, DepOp):
                        cnts = em.source.wait_gr_counts
                        if cnts is not None and cnts.A == 0 and cnts.B == 0 \
                                and cnts.SA == 0 and cnts.SB == 0:
                            pass
                        if em.moduleId not in removed:
                            em.source.wait_gr_counts = WaitGRCounts()

                # Find Sync modules paired with removed wait_gr
                for em in new_emitted:
                    if em.opType == 'sync' and em.before is not None \
                            and em.before in removed:
                        removed.add(em.moduleId)

                # Remove WaitLR if no LR remains in this subIterK
                has_lr = any(em.opType == 'lr' and em.moduleId not in removed
                             for em in new_emitted)
                if not has_lr:
                    for em in new_emitted:
                        if em.opType == 'wait_lr':
                            removed.add(em.moduleId)

                part_nll.append(self._rewire_before(new_emitted, removed))
            nll.append(part_nll)

        self._nll_emitted = nll
        return nll

    @staticmethod
    def _to_emitted(ops) -> List[EmittedModule]:
        """Wrap GRPlacement/LRPlacement/DepOp objects into EmittedModules."""
        result = []
        for mid, op in enumerate(ops):
            if isinstance(op, GRPlacement):
                t = op.tiles
                em = EmittedModule(moduleId=mid, opType='gr',
                                   label=f'GR {op.tensor} (MT {op.mtIteration}, subIterK {t.fmt_k()}) ids {t.fmt_tiles()}',
                                   source=op)
            elif isinstance(op, LRPlacement):
                t = op.tiles
                em = EmittedModule(moduleId=mid, opType='lr',
                                   label=f'LR {op.tensor} (MT {op.mtIteration}, subIterK {t.fmt_k()}) ids {t.fmt_tiles()}',
                                   source=op)
            elif isinstance(op, DepOp):
                if op.kind == 'skip':
                    parts = op.tensor.split(':')
                    label = f'skip {parts[0]} {parts[1]} → {parts[2]}'
                else:
                    label = f'{op.kind.upper()} {op.tensor}'.strip()
                em = EmittedModule(moduleId=mid, opType=op.kind,
                                   label=label, source=op)
            else:
                raise ValueError(f"Unknown op type: {type(op)}")
            result.append(em)
        return result

    def _preloop_make_gr(self, mt: str, tiles: dict) -> List[GRPlacement]:
        """Create GR placements for all tensors at the given MT iteration.

        tiles: {'A': MFMATileRange, 'B': MFMATileRange}
        """
        return [GRPlacement(tensor=tensor, mtIteration=mt,
                            tiles=tiles['A' if tensor in ('A', 'SA') else 'B'],
                            subIterK_slot=0)
                for tensor in self.tensors]

    def _preloop_make_lr(self, tiles: dict) -> List[LRPlacement]:
        """Create LR placements for first partition.

        tiles: per-tensor MFMATileRange, e.g. {'A': MFMATileRange(0, k, mn0, mn1), ...}

        Uses the first MFMA's vgpr tile maps (the preloop loads data consumed
        by the first MFMA, not the next subIterK like mainloop LRs).
        """
        first_mfma = self._partitions[0][0].mfma
        mfma_maps = {'A': first_mfma.vgpr_tile_map_A,
                     'B': first_mfma.vgpr_tile_map_B}
        if self.config.hasScale:
            mfma_maps['SA'] = first_mfma.vgpr_tile_map_SA
            mfma_maps['SB'] = first_mfma.vgpr_tile_map_SB

        placements = []
        for tensor in self.tensors:
            lr = LRPlacement(
                tensor=tensor, mtIteration='0',
                tiles=tiles[tensor],
                subIterK_slot=0, partition=0)
            if tensor in mfma_maps:
                lr.vgpr_tile_map = copy.deepcopy(mfma_maps[tensor])
            placements.append(lr)
        return placements

    def _make_tensor_depops(self, kind: str) -> List[DepOp]:
        """Create a DepOp of the given kind for each tensor."""
        return [DepOp(kind=kind, tensor=tensor)
                for tensor in self.tensors]

    def build_preloop(self) -> List[List[List[EmittedModule]]]:
        """Build preloop: pipeline initialization sequence before mainloop.

        High-level sequence (waits/syncs auto-inserted by _insert_preloop_waits):
          GR(MT 0)  — all tensors, all tiles
          GR_INC
          LR        — first partition, subIterK=0
          LR_INC
          skip(LE 1, NLLEarly/NLL)
          GR(MT 1)  — first partition tiles
          GR_INC
          skip(LE 2, NGLL)

        Returns [1 partition][1 subIterK][EmittedModules] to match emit() shape.
        """
        cfg = self.config
        numK = cfg.numSubIterK
        part0 = self._partition_tile_range(0)
        all_tiles = {
            'A': MFMATileRange(0, numK, 0, cfg.numMFMATilesM),
            'B': MFMATileRange(0, numK, 0, cfg.numMFMATilesN),
        }
        part0_tiles = {
            'A': MFMATileRange(0, numK, *part0['A']),
            'B': MFMATileRange(0, numK, *part0['B']),
        }
        lr_tiles = {
            'A':  MFMATileRange(0, cfg.lrA.size.k, *part0['A']),
            'B':  MFMATileRange(0, cfg.lrB.size.k, *part0['B']),
        }
        if cfg.hasScale:
            lr_tiles['SA'] = MFMATileRange(0, cfg.lrSA.size.k, *part0['A'])
            lr_tiles['SB'] = MFMATileRange(0, cfg.lrSB.size.k, *part0['B'])

        emitted = self._to_emitted([
            *self._preloop_make_gr('0', all_tiles),
            *self._make_tensor_depops('gr_inc'),
            DepOp(kind='wait_gr', wait_gr_counts=WaitGRCounts()),
            DepOp(kind='sync'),
            *self._preloop_make_lr(lr_tiles),
            DepOp(kind='wait_lr'),
            DepOp(kind='skip', tensor='LE:1:NLL'),
            *self._preloop_make_gr('1', part0_tiles),
            # *self._make_tensor_depops('gr_inc'),
            DepOp(kind='skip', tensor='LE:2:NGLL'),
        ])

        self._preloop_emitted = [[emitted]]
        return self._preloop_emitted

    def _emitLoop(self, writer, kernel, label, emitted_3d):
        """Emit a loop section from a 3D emitted structure.

        emitted_3d: [partition][subIterK][EmittedModule]

        For subIterKs with MFMAs: calls instructionSchedule for interleaving.
        For subIterKs without MFMAs (preloop): emits instructions sequentially.
        """
        from Tensile.Components.SubtileBasedScheduler import SubtileBasedScheduler
        from rocisa.code import Module

        module = Module(label)
        module.addComment0(f"{label} start")
        for pi, partition_emitted in enumerate(emitted_3d):
            for k, em_list in enumerate(partition_emitted):
                module.addComment0(f"partition={pi} subIterK={k}")
                hasMFMA = any(em.opType == 'mfma' for em in em_list)
                if hasMFMA:
                    scheduled = SubtileBasedScheduler.instructionSchedule(em_list)
                    module.add(scheduled)
                else:
                    for em in em_list:
                        for inst in em.instructions:
                            module.add(inst)
        module.addComment0(f"{label} end")
        return module

    def emitAllLoops(self, writer, kernel):
        """Emit complete loop structure: preloop + mainloop + NGLL + NLL.

        Owns all control flow (labels, branches, counter management).
        For unroll_factor > 1, emits per-unroll copies with correct vgpr tiles.
        Each mainloop exit jumps to its corresponding NGLL→NLL pair.
        """
        from rocisa.code import Module, Label
        from rocisa.instruction import (SSubU32, SCmpEQU32, SCBranchSCC0,
                                        SCBranchSCC1, SBranch)
        from rocisa.container import sgpr

        assert 'populate' in self._completed, \
            "populate_instructions() must be called before emitAllLoops()"

        module = Module("AllLoops")
        uf = self.unroll_factor

        # ── Preloop ──
        module.add(self._emitLoop(writer, kernel, "PRELOOP",
                                  self._preloop_emitted))

        # ── Mainloop ──
        module.addComment0("MAINLOOP")
        loopBegin = Label("LoopBeginL", "")

        if uf == 1:
            module.add(loopBegin)
            module.add(self._emitLoop(writer, kernel, "MAINLOOP",
                                      self._emitted_per_unroll[0]))
            module.add(SSubU32(dst=sgpr("LoopCounterL"),
                               src0=sgpr("LoopCounterL"), src1=1,
                               comment="dec counterL"))
            module.add(SCmpEQU32(src0=sgpr("LoopCounterL"), src1=2,
                                 comment="counterL == 2?"))
            module.add(SCBranchSCC0(labelName=loopBegin.getLabelName(),
                                    comment="restart mainloop"))
        else:
            exitLabels = [Label(f"ExitC{ui}", "") for ui in range(uf - 1)]
            module.add(loopBegin)
            for ui in range(uf):
                module.add(self._emitLoop(writer, kernel, f"MAINLOOP_C{ui}",
                                          self._emitted_per_unroll[ui]))
                module.add(SSubU32(dst=sgpr("LoopCounterL"),
                                   src0=sgpr("LoopCounterL"), src1=1,
                                   comment=f"dec counterL (copy {ui})"))
                module.add(SCmpEQU32(src0=sgpr("LoopCounterL"), src1=2,
                                     comment=f"counterL == 2? (copy {ui} exit)"))
                if ui < uf - 1:
                    module.add(SCBranchSCC1(
                        labelName=exitLabels[ui].getLabelName(),
                        comment=f"copy {ui} exit → NGLL_C{ui}"))
                else:
                    module.add(SCBranchSCC0(
                        labelName=loopBegin.getLabelName(),
                        comment="restart mainloop"))

        # ── NGLL + NLL exit paths ──
        endLabel = Label("SkipToEnd", "")
        module.add(Label("SkipMainloop", ""))
        module.add(Label("SkipToNGLL", ""))

        if uf == 1:
            module.addComment0("NGLL")
            module.add(self._emitLoop(writer, kernel, "NGLL",
                                      self._ngll_per_unroll[0]))
            module.addComment0("NLL")
            module.add(Label("SkipToNLL", ""))
            module.add(self._emitLoop(writer, kernel, "NLL",
                                      self._nll_per_unroll[0]))
        else:
            # Fall-through from last mainloop copy
            last = uf - 1
            module.addComment0(f"NGLL_C{last}")
            module.add(self._emitLoop(writer, kernel, f"NGLL_C{last}",
                                      self._ngll_per_unroll[last]))
            module.addComment0(f"NLL_C{last}")
            module.add(self._emitLoop(writer, kernel, f"NLL_C{last}",
                                      self._nll_per_unroll[last]))
            module.add(SBranch(labelName=endLabel.getLabelName(),
                               comment="skip other exit paths"))

            for ui in range(uf - 1):
                module.add(exitLabels[ui])
                module.addComment0(f"NGLL_C{ui}")
                module.add(self._emitLoop(writer, kernel, f"NGLL_C{ui}",
                                          self._ngll_per_unroll[ui]))
                module.addComment0(f"NLL_C{ui}")
                module.add(self._emitLoop(writer, kernel, f"NLL_C{ui}",
                                          self._nll_per_unroll[ui]))
                if ui < uf - 2:
                    module.add(SBranch(labelName=endLabel.getLabelName(),
                                       comment="skip other exit paths"))

            # NLLEarly: reached from preloop when LoopCounterL <= 1
            module.add(SBranch(labelName=endLabel.getLabelName(),
                               comment="skip NLLEarly"))
            module.addComment0("NLLEarly")
            module.add(Label("SkipToNLL", ""))
            module.add(self._emitLoop(writer, kernel, "NLLEarly",
                                      self._nll_per_unroll[0]))
            module.add(endLabel)

        return module

    # ── VGPR tile allocation ──────────────────────────────

    def getNumVgpr(self, tileInfoA, tileInfoB,
                        scaleTileInfoA=None, scaleTileInfoB=None) -> int:
        """Return the total number of VGPRs needed across all tensors (A, B, SA, SB)
        without performing any allocation.

        Must be called after scheduling is complete.
        """
        if 'vgpr_tiles' not in self._completed:
            self.assign_vgpr_tiles()

        cfg = self.config

        def _tile_vgpr_count(tileInfo, lrGran):
            return int(math.ceil(tileInfo.mmaTileRegCount * lrGran.size.k * lrGran.size.mn))

        total = self.tile_peaks.get('A', 0) * _tile_vgpr_count(tileInfoA, cfg.lrA) \
              + self.tile_peaks.get('B', 0) * _tile_vgpr_count(tileInfoB, cfg.lrB)

        if cfg.hasScale and scaleTileInfoA and scaleTileInfoB:
            total += self.tile_peaks.get('SA', 0) * _tile_vgpr_count(scaleTileInfoA, cfg.lrSA) \
                   + self.tile_peaks.get('SB', 0) * _tile_vgpr_count(scaleTileInfoB, cfg.lrSB)

        return total

    def allocVgprTiles(self, writer, tileInfoA, tileInfoB,
                       scaleTileInfoA=None, scaleTileInfoB=None):
        """Allocate physical VGPR tiles based on assign_vgpr_tiles() peaks.

        Each vgprTile holds one LR granularity worth of data:
          size = ceil(mmaTileRegCount * lrGranularity.k * lrGranularity.mn)

        Ex: 4 VGPRs for A/B for 1 MFMATile, and 1 VGPR for a 2x2 MFMA tile for SA/SB if hasScale.

        Produces per-tensor lists indexed by vgprTileId:
          vgprTilesA/B:   List[RegisterTileInfo]
          vgprTilesSA/SB: List[RegisterTileInfo]
        """
        if 'vgpr_tiles' not in self._completed:
            self.assign_vgpr_tiles()

        from Tensile.Components.SubtileBasedKernel import TileInfo

        cfg = self.config

        def _tile_vgpr_count(tileInfo, lrGran):
            return int(math.ceil(tileInfo.mmaTileRegCount * lrGran.size.k * lrGran.size.mn))

        def _alloc_tiles(count, numRegs):
            tiles = []
            for _ in range(count):
                tile = TileInfo.RegisterTileInfo(writer.vgprPool)
                for j in range(0, numRegs, 4):
                    blockSize = min(4, numRegs - j)
                    vstart = writer.vgprPool.checkOutAligned(blockSize, blockSize)
                    for k in range(blockSize):
                        tile.append(vstart + k)
                tiles.append(tile)
            return tiles

        self.vgprTilesA = _alloc_tiles(self.tile_peaks.get('A', 0),
                                       _tile_vgpr_count(tileInfoA, cfg.lrA))
        self.vgprTilesB = _alloc_tiles(self.tile_peaks.get('B', 0),
                                       _tile_vgpr_count(tileInfoB, cfg.lrB))

        if cfg.hasScale and scaleTileInfoA and scaleTileInfoB:
            self.vgprTilesSA = _alloc_tiles(self.tile_peaks.get('SA', 0),
                                            _tile_vgpr_count(scaleTileInfoA, cfg.lrSA))
            self.vgprTilesSB = _alloc_tiles(self.tile_peaks.get('SB', 0),
                                            _tile_vgpr_count(scaleTileInfoB, cfg.lrSB))
        else:
            self.vgprTilesSA = []
            self.vgprTilesSB = []

    def deallocVgprTiles(self, writer):
        """Deallocate VGPR tiles allocated by allocVgprTiles."""
        def _dealloc_tiles(tiles):
            for tile in tiles:
                pool = tile.regList.regPool
                for val in tile:
                    if tile.index(val) % 4 == 0:
                        pool.checkIn(val)

        _dealloc_tiles(self.vgprTilesA)
        _dealloc_tiles(self.vgprTilesB)
        _dealloc_tiles(self.vgprTilesSA)
        _dealloc_tiles(self.vgprTilesSB)
        self.vgprTilesA = []
        self.vgprTilesB = []
        self.vgprTilesSA = []
        self.vgprTilesSB = []

    # ── Populate instructions ──────────────────────────────

    def populate_instructions(self, writer, kernel,
                              tileInfoA, tileInfoB, dtileInfo,
                              scaleTileInfoA=None, scaleTileInfoB=None) -> None:
        """Populate EmittedModule.instructions from placements and preOps.

        Uses per-tensor VGPR tile lists (vgprTilesA/B/SA/SB) indexed by
        vgprTileId from placement tile maps.
        """
        if self._preloop_emitted is None or self._ngll_emitted is None \
                or self._nll_emitted is None:
            self.build()

        from Tensile.Components.InstructionEmitter import InstructionEmitter

        emitter = InstructionEmitter(
            writer, kernel, self.config,
            tileInfoA, tileInfoB, dtileInfo,
            self.vgprTilesA, self.vgprTilesB,
            scaleTileInfoA, scaleTileInfoB,
            self.vgprTilesSA, self.vgprTilesSB,
        )

        # Rebuild all loop variants from current _emitted (which now has
        # vgpr_tile_maps populated by assign_vgpr_tiles, unlike the stale
        # copies from build()).
        self.build_preloop()
        self.build_ngll()
        self.build_nll()

        emitter.populate(self._preloop_emitted, unroll_iter=0)

        self._emitted_per_unroll = []
        self._ngll_per_unroll = []
        self._nll_per_unroll = []
        for ui in range(self.unroll_factor):
            em_copy = copy.deepcopy(self._emitted)
            emitter.populate(em_copy, unroll_iter=ui)
            self._emitted_per_unroll.append(em_copy)

            ngll_copy = copy.deepcopy(self._ngll_emitted)
            ngll_ui = (ui + 1) % self.unroll_factor
            emitter.populate(ngll_copy, unroll_iter=ngll_ui)
            self._ngll_per_unroll.append(ngll_copy)

            nll_copy = copy.deepcopy(self._nll_emitted)
            nll_ui = (ui + 2) % self.unroll_factor
            emitter.populate(nll_copy, unroll_iter=nll_ui)
            self._nll_per_unroll.append(nll_copy)

        self._completed.add('populate')

    # ── Print helpers ───────────────────────────────────────

    @staticmethod
    def _fmt_tensor(tensor: str) -> str:
        """Pad tensor name to 2 chars for alignment: 'A' -> 'A ', 'SA' -> 'SA'."""
        return tensor.ljust(2)


    def print_lr(self, partitions: List[List[SubIterKSlot]] = None) -> str:
        """Print place_LRs output in design doc format."""
        if partitions is None:
            partitions = self._partitions
        buf = io.StringIO()
        buf.write("MAINLOOP:\n")
        for pi, slots in enumerate(partitions):
            buf.write(f"  Partition {pi}:\n")
            self._print_lr_partition(buf, slots)
        return buf.getvalue()

    def _print_lr_partition(self, buf, slots):
        for slot in slots:
            buf.write(f"    subIterK={slot.subIterK}:\n")
            if slot.mfma:
                m = slot.mfma
                buf.write(f"      MFMAs (MT n, subIterK {m.subIterK}  ) "
                          f"A : {m.tileA.fmt_tiles()} , B : {m.tileB.fmt_tiles()}\n")
            for lr in slot.lrs:
                t = self._fmt_tensor(lr.tensor)
                buf.write(f"      LR {t} (MT {lr.mtIteration}, "
                          f"subIterK {lr.tiles.fmt_k()}) "
                          f"{lr.tiles.fmt_tiles()}\n")
        return buf.getvalue()

    def print_vgpr(self) -> str:
        """Print assign_vgpr_tiles output: LRs + MFMAs with vgprTileId annotations."""
        partitions = self._partitions
        buf = io.StringIO()
        buf.write(f"needsUnrolling: {self.needs_unrolling}, "
                  f"unrollFactor: {self.unroll_factor}\n")
        peaks_str = ", ".join(f"{t}: {cnt}" for t, cnt in sorted(self.tile_peaks.items()))
        buf.write(f"vgprTiles: {peaks_str}\n")
        for ui in range(self.unroll_factor):
            if self.unroll_factor > 1:
                buf.write(f"MAINLOOP (unroll {ui}):\n")
            else:
                buf.write("MAINLOOP:\n")
            for pi, slots in enumerate(partitions):
                buf.write(f"  Partition {pi}:\n")
                for slot in slots:
                    buf.write(f"    subIterK={slot.subIterK}:\n")
                    if slot.mfma:
                        m = slot.mfma
                        tiles_str = ""
                        parts = []
                        if m.vgpr_tile_map_A:
                            parts.append("A:" + str(m.vgpr_tile_map_A[ui]))
                        if m.vgpr_tile_map_B:
                            parts.append("B:" + str(m.vgpr_tile_map_B[ui]))
                        if m.vgpr_tile_map_SA:
                            parts.append("SA:" + str(m.vgpr_tile_map_SA[ui]))
                        if m.vgpr_tile_map_SB:
                            parts.append("SB:" + str(m.vgpr_tile_map_SB[ui]))
                        if parts:
                            tiles_str = " " + ", ".join(parts)
                        buf.write(f"      MFMAs (MT n, subIterK {m.subIterK}  ) "
                                  f"A : {m.tileA.fmt_tiles()} , "
                                  f"B : {m.tileB.fmt_tiles()}{tiles_str}\n")
                    for lr in slot.lrs:
                        tile_str = ""
                        if lr.vgpr_tile_map:
                            tile_str = f" tiles:{lr.vgpr_tile_map[ui]}"
                        t = self._fmt_tensor(lr.tensor)
                        buf.write(f"      LR {t} (MT {lr.mtIteration}, "
                                  f"subIterK {lr.tiles.fmt_k()}) "
                                  f"{lr.tiles.fmt_tiles()}{tile_str}\n")
        return buf.getvalue()

    def print_gr(self) -> str:
        """Print place_GRs output: LRs + MFMAs + GR placements, all partitions."""
        partitions = self._partitions
        buf = io.StringIO()
        buf.write("MAINLOOP:\n")
        for pi, slots in enumerate(partitions):
            buf.write(f"  Partition {pi}:\n")
            for slot in slots:
                buf.write(f"    subIterK={slot.subIterK}:\n")
                if slot.mfma:
                    m = slot.mfma
                    buf.write(f"      MFMAs (MT n, subIterK {m.subIterK}  ) "
                              f"A : {m.tileA.fmt_tiles()} , "
                              f"B : {m.tileB.fmt_tiles()}\n")
                for lr in slot.lrs:
                    t = self._fmt_tensor(lr.tensor)
                    buf.write(f"      LR {t} (MT {lr.mtIteration}, "
                              f"subIterK {lr.tiles.fmt_k()}) "
                              f"{lr.tiles.fmt_tiles()}\n")
                for gr in slot.grs:
                    buf.write(f"      GR {gr.tensor} (MT {gr.mtIteration}, "
                              f"subIterK {gr.tiles.fmt_k()}) "
                              f"ids {gr.tiles.fmt_tiles()}\n")
        return buf.getvalue()

    def print_deps(self) -> str:
        """Print annotate_deps output: placements with their before-dependencies."""
        buf = io.StringIO()
        buf.write("MAINLOOP:\n")
        for pi, slots in enumerate(self._partitions):
            buf.write(f"  Partition {pi}:\n")
            for slot in slots:
                buf.write(f"    subIterK={slot.subIterK}:\n")
                if slot.mfma:
                    self._print_placement_with_deps(buf, slot.mfma, slot)
                for lr in slot.lrs:
                    self._print_placement_with_deps(buf, lr, slot)
                for gr in slot.grs:
                    self._print_placement_with_deps(buf, gr, slot)
        return buf.getvalue()

    def _print_placement_with_deps(self, buf, placement, slot: SubIterKSlot):
        """Print a placement label followed by its deps."""
        label = self._format_placement_label(placement, slot)
        buf.write(f"      {label}\n")
        if placement.deps:
            buf.write("        deps:\n")
            for dep in placement.deps:
                dep_str = self._format_dep_ref(dep)
                buf.write(f"            - {dep_str}\n")

    def _format_placement_label(self, placement, slot: SubIterKSlot) -> str:
        """Format a placement (MFMA/LR/GR) into a human-readable label."""
        if isinstance(placement, MFMAPlacement):
            m = placement
            label = (f"MFMAs (MT n, subIterK {m.subIterK}  ) "
                     f"A : {m.tileA.fmt_tiles()} , B : {m.tileB.fmt_tiles()}")
            return label
        elif isinstance(placement, LRPlacement):
            t = self._fmt_tensor(placement.tensor)
            label = (f"LR {t} (MT {placement.mtIteration}, "
                     f"subIterK {placement.tiles.fmt_k()}) "
                     f"{placement.tiles.fmt_tiles()}")
            return label
        elif isinstance(placement, GRPlacement):
            return (f"GR {placement.tensor} (MT {placement.mtIteration}, "
                    f"subIterK {placement.tiles.fmt_k()}) "
                    f"ids {placement.tiles.fmt_tiles()}")
        return str(placement)

    def print_remove_deps(self) -> str:
        """Print remove_cross_deps output: placements with preOps and remaining deps."""
        buf = io.StringIO()
        buf.write("MAINLOOP:\n")
        for pi, slots in enumerate(self._partitions):
            buf.write(f"  Partition {pi}:\n")
            for slot in slots:
                buf.write(f"    subIterK={slot.subIterK}:\n")
                if slot.mfma:
                    self._print_placement_with_preops(buf, slot.mfma, slot)
                for lr in slot.lrs:
                    self._print_placement_with_preops(buf, lr, slot)
                for gr in slot.grs:
                    self._print_placement_with_preops(buf, gr, slot)
        return buf.getvalue()

    def print_group_lr_gr(self) -> str:
        """Print group_lr_gr output: placements with chained deps and merged preOps."""
        buf = io.StringIO()
        buf.write("MAINLOOP:\n")
        for pi, slots in enumerate(self._partitions):
            buf.write(f"  Partition {pi}:\n")
            for slot in slots:
                buf.write(f"    subIterK={slot.subIterK}:\n")
                if slot.mfma:
                    self._print_placement_with_preops(buf, slot.mfma, slot)
                for lr in slot.lrs:
                    self._print_placement_with_preops(buf, lr, slot)
                for gr in slot.grs:
                    self._print_placement_with_preops(buf, gr, slot)
        return buf.getvalue()

    def _print_placement_with_preops(self, buf, placement, slot: SubIterKSlot):
        """Print a placement label followed by its preOps and remaining deps."""
        label = self._format_placement_label(placement, slot)
        buf.write(f"      {label}\n")
        if placement.preOps:
            buf.write("        preOps:\n")
            for op in placement.preOps:
                buf.write(f"            - {op}\n")
        if placement.deps:
            buf.write("        deps:\n")
            for dep in placement.deps:
                dep_str = self._format_dep_ref(dep)
                buf.write(f"            - {dep_str}\n")
    

    def _format_dep_ref(self, dep: DepRef) -> str:
        """Format a DepRef for display."""
        p = dep.ref
        slot = p.subIterK_slot if hasattr(p, 'subIterK_slot') else '?'
        part = p.partition if hasattr(p, 'partition') else 0
        kind = 'LR' if isinstance(p, LRPlacement) else 'GR'
        mt = f" (MT{dep.mt_offset})" if dep.mt_offset != 0 else ""
        return f"{kind} {p.tensor} @P{part}:subIterK={slot}{mt}"


    def print_emit(self, all_partitions: List[List[List[EmittedModule]]] = None) -> str:
        """Print emit output: EmittedModule list with before-links."""
        if all_partitions is None:
            all_partitions = self._emitted
        buf = io.StringIO()
        buf.write("MAINLOOP:\n")
        for pi, partition_emitted in enumerate(all_partitions):
            buf.write(f"  Partition {pi}:\n")
            for k, emitted in enumerate(partition_emitted):
                buf.write(f"    subIterK={k}:\n")
                for em in emitted:
                    before_str = f" <- [{em.before}]" if em.before is not None else ""
                    buf.write(f"      [{em.moduleId:2d}] {em.opType:10s} {em.label}{before_str}\n")
        return buf.getvalue()

    def print_emit_dep_order(self, all_partitions: List[List[List[EmittedModule]]] = None) -> str:
        """Print emit output as dependency paths (same decomposition as _extractPathsFromBeforeDeps)."""
        from Tensile.Components.SubtileBasedScheduler import SubtileBasedScheduler
        if all_partitions is None:
            all_partitions = self._emitted
        buf = io.StringIO()
        buf.write("MAINLOOP (dependency paths):\n")
        for pi, partition_emitted in enumerate(all_partitions):
            buf.write(f"  Partition {pi}:\n")
            for k, emitted in enumerate(partition_emitted):
                buf.write(f"    subIterK={k}:\n")
                mfmaIdx, paths, preMfmaPaths = SubtileBasedScheduler._extractPathsFromBeforeDeps(emitted)
                em = emitted[mfmaIdx]
                buf.write(f"      MFMA: [{em.moduleId:2d}] {em.label}")
                if em.before is not None:
                    buf.write(f" <- [{em.before}]")
                buf.write("\n")
                for i, path in enumerate(preMfmaPaths):
                    buf.write(f"      preMFMA path {i}:\n")
                    for idx in path:
                        buf.write(f"        [{emitted[idx].moduleId:2d}] {emitted[idx].opType:10s} {emitted[idx].label}\n")
                for i, path in enumerate(paths):
                    buf.write(f"      path {i}:\n")
                    for idx in path:
                        buf.write(f"        [{emitted[idx].moduleId:2d}] {emitted[idx].opType:10s} {emitted[idx].label}\n")
        return buf.getvalue()
