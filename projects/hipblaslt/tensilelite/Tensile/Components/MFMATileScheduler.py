"""MFMATile-based logical scheduler.

Builds a logical schedule using MFMA tile indices as the core primitive,
with explicit per-operation load granularity for GR/LR on A, B, SA, SB.

The schedule is built in 7 passes:
  place_LRs        — place LRs based on their granularities
  assign_vgpr_sets — assign VGPR tile sets (ping-pong) based on subIterK dependencies
  place_GRs        — place GRs
  annotate_deps    — annotate raw per-op dependencies
  remove_cross_deps— replace cross-subIterK deps with wait preOps
  insert_gr_lr_inc    — insert lr_inc/gr_inc preOps at MT transitions
  group            — serialize and group (produce paths for instructionSchedule)
  emit             — produce List[EmittedModule] with before-link chains
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Tuple
import io


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
    # VGPR set annotations. TODO. Check if we keep this info like this. TBD when implementing unrolling.
    mfma_sets: Optional[dict] = None   # {'A': int, 'B': int, 'SA': int, 'SB': int}
    lr_sets: Optional[dict] = None     # tensor -> set id per LR


# ── VGPR set assignment output ───────────────────────────────

@dataclass
class VGPRSetAssignment:
    """VGPR set (0 or 1) for each tensor at each subIterK."""
    mfma_sets: dict   # {'A': int, 'B': int, 'SA': int, 'SB': int}
    lr_sets: dict     # tensor -> set id


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
      'wait_gr'      — wait for global reads (includes implicit sync)
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
        if self.kind == 'wait_gr' and self.wait_gr_counts:
            return f"wait_gr({self.wait_gr_counts})"
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


# ── Main scheduler class ───────────────────────────────────

class MFMATileScheduler:
    """MFMATile-based logical scheduler.

    Builds the schedule in 6 passes, each producing testable intermediate output.
    Each pass auto-runs its prerequisites if needed (tracked via self._completed).
    """

    def __init__(self, config: SchedulerConfig):
        self.config = config
        self._completed: set = set()   # tracks which passes have run: {'lr', 'vgpr', 'gr', 'deps', 'group', 'emit'}
        self._partitions: Optional[List[List[SubIterKSlot]]] = None  # shared mutable state across passes
        self._grouped: Optional[List[GroupedSubIterK]] = None
        self._emitted: Optional[List[List[EmittedModule]]] = None

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
                lr_mt = ("n+1" if is_last else "n") if multi_part else \
                         "n+1" if is_wrap else "n"
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

        return slots

    # ── Assign VGPR sets ──────────────────────────────────

    def assign_vgpr_sets(self) -> List[SubIterKSlot]:
        """Assign VGPR set IDs (0 or 1) to MFMAs and LRs.

        Rule: MFMA and LR in the same subIterK never share a set.
        The next MFMA reads from whichever set the previous LR wrote to.

        Single sequential scan across all partitions and subIterKs.
        A map tracks (tensor, tile_range, k_chunk_start) → set_id.
        Wrapping LRs (that reload a chunk already consumed) are deferred
        until the end of the partition so they don't affect remaining
        MFMAs within the same partition.
        """
        if 'lr' not in self._completed:
            self.place_LRs()

        cfg = self.config
        numK = cfg.numSubIterK

        tensor_names = ['A', 'B']
        if cfg.hasScale:
            tensor_names += ['SA', 'SB']

        tensor_k_gran = {}
        for t, gran in [('A', cfg.lrA), ('B', cfg.lrB)]:
            tensor_k_gran[t] = gran.size.k
        if cfg.hasScale:
            tensor_k_gran['SA'] = cfg.lrSA.size.k
            tensor_k_gran['SB'] = cfg.lrSB.size.k

        # Initialize all slots across all partitions
        for partition_slots in self._partitions:
            for slot in partition_slots:
                slot.mfma_sets = {}
                slot.lr_sets = {}

        part_ranges = [self._partition_tile_range(pi)
                       for pi in range(cfg.numPartitions)]

        # Seed: preloop loads partition 0's tiles, chunk 0, into set 0.
        # Key: (tensor, tile_range, k_chunk_start) → set_id
        set_map = {}
        for t in tensor_names:
            side_key = 'A' if t in ('A', 'SA') else 'B'
            set_map[(t, part_ranges[0][side_key], 0)] = 0

        # Sequential scan across partitions and subIterKs
        for pi, slots in enumerate(self._partitions):
            deferred = []  # wrapping LR writes, applied after partition
            for k in range(numK):
                slot = slots[k]
                # Assign MFMA sets
                for t in tensor_names:
                    k_gran = tensor_k_gran[t]
                    side_key = 'A' if t in ('A', 'SA') else 'B'
                    tile_range = part_ranges[pi][side_key]
                    k_chunk_start = (k // k_gran) * k_gran
                    mfma_set = set_map.get((t, tile_range, k_chunk_start), 0)
                    slot.mfma_sets[t] = mfma_set

                # Assign LR sets
                for lr in slot.lrs:
                    t = lr.tensor
                    k_gran = tensor_k_gran[t]
                    mfma_set = slot.mfma_sets[t]
                    lr_set = 1 - mfma_set
                    slot.lr_sets[t] = lr_set

                    # Record what this LR loaded
                    lr_tiles = (lr.tiles.tileId_start, lr.tiles.tileId_end)
                    lr_k_start = lr.tiles.subIterK_start
                    key = (t, lr_tiles, lr_k_start)

                    # Wrapping LR: reloads a chunk at or before current chunk.
                    # Defer so it doesn't affect this partition's remaining MFMAs.
                    current_chunk = k // k_gran
                    lr_chunk = lr_k_start // k_gran
                    if lr_chunk <= current_chunk:
                        deferred.append((key, lr_set))
                    else:
                        set_map[key] = lr_set

            # Apply deferred wrapping writes for next partition / iteration
            for key, val in deferred:
                set_map[key] = val

        self._completed.add('vgpr')
        return self._partitions[0]

    # ── Place GRs ─────────────────────────────────────────

    def _build_gr_list(self, part_ranges, offsetMT, offsetPartition,
                             debug=False):
        """Phase 1: Build ordered GR list from assign_vgpr_sets MFMAs.

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
        if 'vgpr' not in self._completed:
            self.assign_vgpr_sets()


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

        def _slot_offset(consumer_slot, consumer_type, producer):
            """Offset from slot ordering alone: 0 if producer ran first, -1 otherwise."""
            prod_slot = producer.subIterK_slot
            if prod_slot < consumer_slot:
                return 0
            if prod_slot > consumer_slot:
                return -1
            prod_type = 'LR' if isinstance(producer, LRPlacement) else 'GR'
            return -1 if _order[prod_type] >= _order[consumer_type] else 0

        def _mt_offset(consumer_slot, consumer_type, producer, consumer=None):
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
            # Same effective mt: slot ordering decides.
            return _slot_offset(consumer_slot, consumer_type, producer)

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

        for k, slot in enumerate(slots):
            # MFMA: depends on LRs that loaded subIterK=k data with matching tiles
            if slot.mfma:
                tensor_names = ['A', 'B']
                if cfg.hasScale:
                    tensor_names += ['SA', 'SB']
                for t in tensor_names:
                    for lr in lr_by_data[slot.mfma.subIterK].get(t, []):
                        if _tiles_overlap(slot.mfma, t, lr.tiles):
                            slot.mfma.deps.append(DepRef(
                                ref=lr, mt_offset=_mt_offset(k, 'MFMA', lr)))

            # LR: depends on GR (data must be in LDS before reading)
            # Cross-partition: the GR that loaded the matching tiles may be
            # in a different partition. Filter by tile overlap.
            for lr in slot.lrs:
                for gr in gr_by_tensor.get(lr.tensor, []):
                    if _range_overlaps(lr.tiles, gr.tiles):
                        lr.deps.append(DepRef(
                            ref=gr, mt_offset=_mt_offset(k, 'LR', gr, consumer=lr)))

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

        # ── Dedup: keep only the single last dep ──
        # Execution order is (MT offset, partition, subIterK). Waiting for the
        # last one guarantees all earlier ones have completed.
        def _dedup_deps(deps):
            if len(deps) <= 1:
                return deps
            def _exec_order(dep):
                return (dep.mt_offset, dep.ref.partition, dep.ref.subIterK_slot)
            return [max(deps, key=_exec_order)]

        for slot in slots:
            for lr in slot.lrs:
                lr.deps = _dedup_deps(lr.deps)
            for gr in slot.grs:
                gr.deps = _dedup_deps(gr.deps)

    # ── Remove cross-subIterK deps ─────────────────────────

    def _gr_granularity(self, tensor: str) -> ReadGranularity:
        """Return GR granularity for a tensor."""
        return {'A': self.config.grA, 'B': self.config.grB,
                'SA': self.config.grSA, 'SB': self.config.grSB}[tensor]

    def _compute_inflight_loads(self, consumer_pi: int, consumer_slot: int,
                                tensor: str, dep_ref: DepRef) -> int:
        """Count inflight GR atomic loads between a dep GR and the consumer.

        Walks backward through the flattened schedule (all partitions x subIterK)
        from the consumer position, counting atomic GR loads for `tensor`.
        Stops when reaching the dependency GR (dep_ref.ref) after accounting
        for mt_offset wraps.

        Returns the number of inflight atomic loads.
        """
        gr_gran = self._gr_granularity(tensor)
        numP = len(self._partitions)
        numK = len(self._partitions[0])
        flat_len = numP * numK

        # Flatten: flat_idx = pi * numK + slot_k
        consumer_flat = consumer_pi * numK + consumer_slot

        # How many full wraps we need before stopping at the dep.
        # mt_offset is negative (e.g., -1 = previous MT, -2 = two MTs back).
        wraps_needed = abs(dep_ref.mt_offset)
        # If the dep is in the same MT (mt_offset == 0) we still walk backward
        # up to the dep within the current "unwrapped" iteration.
        # wraps_completed tracks how many full-loop wraps we've done.

        count = 0
        wraps_completed = 0
        pos = consumer_flat

        # Walk backward; maximum steps = wraps_needed * flat_len + flat_len
        # (at most wraps_needed full loops + the partial first loop).
        max_steps = (wraps_needed + 1) * flat_len
        for _ in range(max_steps):
            pos = (pos - 1) % flat_len
            if pos == flat_len - 1 and _ > 0:
                # We just wrapped around the loop boundary
                wraps_completed += 1

            pi = pos // numK
            slot_k = pos % numK
            slot = self._partitions[pi][slot_k]

            for gr in slot.grs:
                if gr.tensor != tensor:
                    continue
                # Check if this is the dependency GR
                if gr is dep_ref.ref and wraps_completed >= wraps_needed:
                    return count
                # Count atomic loads for this GR
                tiles = gr.tiles
                n_tile = (tiles.tileId_end - tiles.tileId_start) // gr_gran.size.mn
                n_k = (tiles.subIterK_end - tiles.subIterK_start) // gr_gran.size.k
                count += n_tile * n_k

        return count

    def remove_cross_deps(self):
        """Replace cross-subIterK deps with wait preOps.

        For each placement, separates deps into same-subIterK (kept) and
        cross-subIterK (converted to preOps):
          - MFMA depending on LRs → single wait_lr
          - GR depending on LRs   → single wait_lr_sync
          - LR depending on GRs   → single wait_gr with per-tensor inflight counts
        """
        if 'deps' not in self._completed:
            self.annotate_deps()

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
                        counts = WaitGRCounts()
                        for dep in cross:
                            t = dep.ref.tensor
                            inflight = self._compute_inflight_loads(
                                pi, lr.subIterK_slot, t, dep)
                            setattr(counts, t, inflight)
                        lr.preOps.append(DepOp(kind='wait_gr',
                                               wait_gr_counts=counts))

                # ── GRs ──
                for gr in slot.grs:
                    same, cross = self._split_deps(gr.deps, pi, gr.subIterK_slot)
                    gr.deps = same
                    gr.preOps = []
                    if cross:
                        gr.preOps.append(DepOp(kind='wait_lr_sync'))

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

        last_mt = {}  # tensor -> mtIteration string

        for pi, slots in enumerate(self._partitions):
            for slot in slots:
                for lr in slot.lrs:
                    tensor = lr.tensor
                    mt = lr.mtIteration
                    if tensor in last_mt and last_mt[tensor] != mt:
                        lr.preOps.append(DepOp(kind='lr_inc', tensor=tensor))
                    last_mt[tensor] = mt
                for gr in slot.grs:
                    tensor = gr.tensor
                    mt = gr.mtIteration
                    if tensor in last_mt and last_mt[tensor] != mt:
                        gr.preOps.append(DepOp(kind='gr_inc', tensor=tensor))
                    last_mt[tensor] = mt

        self._completed.add('gr_inc')

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

    # ── Group and serialize (commented out — will be reworked) ──

    # def group(self) -> List[GroupedSubIterK]:
    #     """Serialize operations within each subIterK.
    #
    #     Takes annotate_deps' raw deps and transforms them:
    #     - Cross-subIterK LR deps → WaitLROp barrier
    #     - Cross-subIterK GR deps → WaitGROp barrier
    #     - Same-subIterK deps → node refs for serialization
    #     - LRs serialized: A → B → SA → SB with WaitGROp before first
    #     - GRs serialized: SA → A (or SB → B) with merged deps
    #     """
    #     if 'gr' not in self._completed:
    #         self.place_GRs()
    #     slots = self._partitions[0]
    #     if 'deps' not in self._completed:
    #         self.annotate_deps()
    #     cfg = self.config
    #     numK = cfg.numSubIterK
    #
    #     grouped = []
    #     for k, slot in enumerate(slots):
    #         gslot = GroupedSubIterK(subIterK=k)
    #
    #         # MFMA with WaitLROp before
    #         if slot.mfma:
    #             gslot.ops.append(AnnotatedOp(
    #                 kind='MFMA',
    #                 before=[DepOp(kind='wait_lr')],
    #                 placement=slot.mfma,
    #             ))
    #
    #         # LRs serialized: A → B → SA → SB
    #         # First LR gets WaitGROp + any LR_INCOps
    #         lr_order = ['A', 'B', 'SA', 'SB']
    #         ordered_lrs = sorted(slot.lrs, key=lambda lr: lr_order.index(lr.tensor))
    #
    #         prev_lr_op = None
    #         for i, lr in enumerate(ordered_lrs):
    #             before = []
    #             if i == 0:
    #                 before.append(DepOp(kind='wait_gr'))
    #                 for lr2 in ordered_lrs:
    #                     if lr2.mtIteration != "n":
    #                         before.append(DepOp(kind='lr_inc', tensor=lr2.tensor))
    #             else:
    #                 before.append(DepOp(kind='ref', ref=prev_lr_op))
    #
    #             lr_op = AnnotatedOp(
    #                 kind='LR', before=before, placement=lr,
    #             )
    #             gslot.ops.append(lr_op)
    #             prev_lr_op = lr_op
    #
    #         # GRs serialized: SA → A or SB → B
    #         gr_order = ['SA', 'A', 'SB', 'B']
    #         ordered_grs = sorted(slot.grs, key=lambda gr: gr_order.index(gr.tensor))
    #
    #         # Only subIterK=0 needs collision wait (same-subIterK LR still async).
    #         # Later subIterKs: collision LR covered by MFMAs' WaitLROp.
    #         needs_collision_wait = self._needs_collision_wait(k, ordered_lrs)
    #         collision_lr_op = None
    #         if needs_collision_wait:
    #             collision_lr_op = self._find_collision_LR_op(gslot, ordered_lrs)
    #
    #         prev_gr_op = None
    #         for i, gr in enumerate(ordered_grs):
    #             before = []
    #             if i == 0:
    #                 for gr2 in ordered_grs:
    #                     before.append(DepOp(kind='gr_inc', tensor=gr2.tensor))
    #                 if collision_lr_op:
    #                     before.append(DepOp(kind='ref', ref=collision_lr_op))
    #                     before.append(DepOp(kind='wait_lr_sync'))
    #             else:
    #                 before.append(DepOp(kind='ref', ref=prev_gr_op))
    #
    #             gr_op = AnnotatedOp(
    #                 kind='GR', before=before, placement=gr,
    #             )
    #             gslot.ops.append(gr_op)
    #             prev_gr_op = gr_op
    #
    #         grouped.append(gslot)
    #
    #     self._grouped = grouped
    #     self._completed.add('group')
    #     return grouped
    #
    # def _needs_collision_wait(self, subIterK: int, ordered_lrs: list) -> bool:
    #     """Check if GRs at this subIterK need explicit wait_lr_sync."""
    #     return subIterK == 0
    #
    # def _find_collision_LR_op(self, gslot: GroupedSubIterK, ordered_lrs: list) -> Optional[AnnotatedOp]:
    #     """Find the last LR AnnotatedOp in this subIterK that GR must wait for."""
    #     last_lr = None
    #     for op in gslot.ops:
    #         if op.kind == 'LR':
    #             last_lr = op
    #     return last_lr
    #
    # def emit(self) -> List[List[EmittedModule]]:
    #     """Convert grouped ops into a flat List[EmittedModule] per subIterK."""
    #     if 'group' not in self._completed:
    #         self.group()
    #     grouped = self._grouped
    #
    #     all_emitted = []
    #     for gslot in grouped:
    #         emitted: List[EmittedModule] = []
    #         op_to_id = {}
    #
    #         def add(opType, label, before=None):
    #             mid = len(emitted)
    #             emitted.append(EmittedModule(
    #                 moduleId=mid, opType=opType,
    #                 label=label, before=before))
    #             return mid
    #
    #         for op in gslot.ops:
    #             opType = op.kind.lower()
    #
    #             prev_id = None
    #             for dep in op.before:
    #                 if dep.kind == 'ref':
    #                     ref_id = op_to_id.get(id(dep.ref))
    #                     if ref_id is not None:
    #                         prev_id = ref_id
    #                 elif dep.kind == 'wait_lr_sync':
    #                     prev_id = add('wait_lr', 'wait_lr', before=prev_id)
    #                     prev_id = add('sync', 'sync', before=prev_id)
    #                 else:
    #                     prev_id = add(dep.kind, str(dep), before=prev_id)
    #
    #             label = self._format_op_label(op, gslot)
    #             mid = add(opType, label, before=prev_id)
    #             op_to_id[id(op)] = mid
    #
    #         all_emitted.append(emitted)
    #
    #     self._emitted = all_emitted
    #     self._completed.add('emit')
    #     return all_emitted

    # ── Print helpers ───────────────────────────────────────

    @staticmethod
    def _fmt_tensor(tensor: str) -> str:
        """Pad tensor name to 2 chars for alignment: 'A' -> 'A ', 'SA' -> 'SA'."""
        return tensor.ljust(2)

    def _get_slot_data(self, subIterK: int) -> Optional[SubIterKSlot]:
        """Get slot for VGPR set annotations (mfma_sets, lr_sets)."""
        if self._partitions and subIterK < len(self._partitions[0]):
            return self._partitions[0][subIterK]
        return None

    # def _format_op_label(self, op: AnnotatedOp, gslot) -> str:
    #     """Compute a human-readable label from an AnnotatedOp's placement."""
    #     p = op.placement
    #     k = gslot.subIterK if hasattr(gslot, 'subIterK') else 0
    #     s3 = self._get_slot_data(k)
    #
    #     if op.kind == 'MFMA':
    #         label = (f"MFMAs (MT n, subIterK {p.subIterK}  ) "
    #                  f"A : {p.tileA.fmt_tiles()} , B : {p.tileB.fmt_tiles()}")
    #         if s3 and s3.mfma_sets:
    #             label += " " + ", ".join(
    #                 f"set{t}:{s3.mfma_sets[t]}"
    #                 for t in sorted(s3.mfma_sets.keys()))
    #         return label
    #     elif op.kind == 'LR':
    #         t = self._fmt_tensor(p.tensor)
    #         label = (f"LR {t} (MT {p.mtIteration}, "
    #                  f"subIterK {p.tiles.fmt_k()}) "
    #                  f"{p.tiles.fmt_tiles()}")
    #         if s3 and s3.lr_sets and p.tensor in s3.lr_sets:
    #             label += f" set{p.tensor}:{s3.lr_sets[p.tensor]}"
    #         return label
    #     elif op.kind == 'GR':
    #         return (f"GR {p.tensor} (MT {p.mtIteration}, "
    #                 f"subIterK {p.tiles.fmt_k()}) "
    #                 f"ids {p.tiles.fmt_tiles()}")
    #     return op.kind

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
        """Print assign_vgpr_sets output: LRs + MFMAs with set annotations."""
        partitions = self._partitions
        buf = io.StringIO()
        buf.write("MAINLOOP:\n")
        for pi, slots in enumerate(partitions):
            buf.write(f"  Partition {pi}:\n")
            for slot in slots:
                buf.write(f"    subIterK={slot.subIterK}:\n")
                if slot.mfma:
                    m = slot.mfma
                    sets_str = ""
                    if slot.mfma_sets:
                        sets_str = " " + ", ".join(
                            f"set{t}:{slot.mfma_sets[t]}"
                            for t in sorted(slot.mfma_sets.keys())
                        )
                    buf.write(f"      MFMAs (MT n, subIterK {m.subIterK}  ) "
                              f"A : {m.tileA.fmt_tiles()} , "
                              f"B : {m.tileB.fmt_tiles()}{sets_str}\n")
                for lr in slot.lrs:
                    set_str = ""
                    if slot.lr_sets and lr.tensor in slot.lr_sets:
                        set_str = f" set{lr.tensor}:{slot.lr_sets[lr.tensor]}"
                    t = self._fmt_tensor(lr.tensor)
                    buf.write(f"      LR {t} (MT {lr.mtIteration}, "
                              f"subIterK {lr.tiles.fmt_k()}) "
                              f"{lr.tiles.fmt_tiles()}{set_str}\n")
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
                    sets_str = ""
                    if slot.mfma_sets:
                        sets_str = " " + ", ".join(
                            f"set{t}:{slot.mfma_sets[t]}"
                            for t in sorted(slot.mfma_sets.keys())
                        )
                    buf.write(f"      MFMAs (MT n, subIterK {m.subIterK}  ) "
                              f"A : {m.tileA.fmt_tiles()} , "
                              f"B : {m.tileB.fmt_tiles()}{sets_str}\n")
                for lr in slot.lrs:
                    set_str = ""
                    if slot.lr_sets and lr.tensor in slot.lr_sets:
                        set_str = f" set{lr.tensor}:{slot.lr_sets[lr.tensor]}"
                    t = self._fmt_tensor(lr.tensor)
                    buf.write(f"      LR {t} (MT {lr.mtIteration}, "
                              f"subIterK {lr.tiles.fmt_k()}) "
                              f"{lr.tiles.fmt_tiles()}{set_str}\n")
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
            if slot.mfma_sets:
                label += " " + ", ".join(
                    f"set{t}:{slot.mfma_sets[t]}"
                    for t in sorted(slot.mfma_sets.keys()))
            return label
        elif isinstance(placement, LRPlacement):
            t = self._fmt_tensor(placement.tensor)
            label = (f"LR {t} (MT {placement.mtIteration}, "
                     f"subIterK {placement.tiles.fmt_k()}) "
                     f"{placement.tiles.fmt_tiles()}")
            if slot.lr_sets and placement.tensor in slot.lr_sets:
                label += f" set{placement.tensor}:{slot.lr_sets[placement.tensor]}"
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

    # def print_group(self, grouped: List[GroupedSubIterK] = None) -> str:
    #     """Print group output: serialized ops with dependencies."""
    #     if grouped is None:
    #         grouped = self._grouped
    #     buf = io.StringIO()
    #     buf.write("MAINLOOP:\n")
    #     buf.write("  Partition 0:\n")
    #     self._print_grouped_slots(buf, grouped)
    #     return buf.getvalue()
    #
    # def _print_grouped_slots(self, buf, grouped: List[GroupedSubIterK]):
    #     """Print a list of GroupedSubIterK slots with their deps."""
    #     for gslot in grouped:
    #         buf.write(f"    subIterK={gslot.subIterK}:\n")
    #         for op in gslot.ops:
    #             label = self._format_op_label(op, gslot)
    #             buf.write(f"      {label}\n")
    #             if op.before:
    #                 buf.write("        before:\n")
    #                 for dep in op.before:
    #                     dep_str = self._format_dep(dep)
    #                     buf.write(f"            - {dep_str}\n")

    def _format_dep_ref(self, dep: DepRef) -> str:
        """Format a DepRef for display."""
        p = dep.ref
        slot = p.subIterK_slot if hasattr(p, 'subIterK_slot') else '?'
        part = p.partition if hasattr(p, 'partition') else 0
        kind = 'LR' if isinstance(p, LRPlacement) else 'GR'
        mt = f" (MT{dep.mt_offset})" if dep.mt_offset != 0 else ""
        return f"{kind} {p.tensor} @P{part}:subIterK={slot}{mt}"

    # def _format_dep(self, dep: DepOp) -> str:
    #     """Format a DepOp for display (used by group/emit)."""
    #     if dep.kind == 'ref' and dep.ref and isinstance(dep.ref, AnnotatedOp):
    #         p = dep.ref.placement
    #         slot = self._get_slot_data(p.subIterK_slot) if hasattr(p, 'subIterK_slot') else None
    #         if slot:
    #             return self._format_placement_label(p, slot)
    #         return self._format_placement_label(p, SubIterKSlot(subIterK=0))
    #     return str(dep)
    #
    # def print_emit(self, all_emitted: List[List[EmittedModule]] = None) -> str:
    #     """Print emit output: EmittedModule list with before-links."""
    #     if all_emitted is None:
    #         all_emitted = self._emitted
    #     buf = io.StringIO()
    #     for k, emitted in enumerate(all_emitted):
    #         buf.write(f"subIterK={k}:\n")
    #         for em in emitted:
    #             before_str = f" ← [{em.before}]" if em.before is not None else ""
    #             buf.write(f"  [{em.moduleId:2d}] {em.opType:8s} {em.label}{before_str}\n")
    #         buf.write("\n")
    #     return buf.getvalue()
