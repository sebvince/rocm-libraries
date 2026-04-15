"""MFMATile-based logical scheduler.

Builds a logical schedule using MFMA tile indices as the core primitive,
with explicit per-operation load granularity for GR/LR on A, B, SA, SB.

The schedule is built in 6 passes:
  place_LRs        — place LRs based on their granularities
  assign_vgpr_sets — assign VGPR tile sets (ping-pong) based on subIterK dependencies
  place_GRs        — place GRs
  annotate_deps    — annotate raw per-op dependencies
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


@dataclass
class LRPlacement:
    """Local Read placement for one tensor in one subIterK slot."""
    tensor: str                # 'A', 'B', 'SA', 'SB'
    mtIteration: str           # 'n', 'n+1'
    tiles: MFMATileRange
    subIterK_slot: int         # which subIterK this LR is placed in


@dataclass
class GRPlacement:
    """Global Read placement for one tensor in one subIterK slot."""
    tensor: str                # 'A', 'B', 'SA', 'SB'
    mtIteration: str           # 'n+2'
    tiles: MFMATileRange
    subIterK_slot: int         # which subIterK this GR is placed in


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
class DepOp:
    """A typed dependency in a before-chain.

    Kinds:
      'wait_gr'      — wait for global reads (includes implicit sync)
      'wait_lr'      — wait for local reads (no sync, used before MFMAs)
      'wait_lr_sync' — wait for local reads + sync barrier (used before GRs,
                        to ensure LR finished reading LDS before GR overwrites it)
      'lr_inc'       — LDS buffer swap for LR (needs tensor)
      'gr_inc'       — pointer update + LDS swap for GR (needs tensor)
      'ref'          — reference to another AnnotatedOp in same subIterK
      'lr_ref'       — dependency on LR for a tensor (annotate_deps, cross-subIterK)
      'gr_ref'       — dependency on GR for a tensor (annotate_deps, cross-subIterK)
    """
    kind: str
    tensor: str = ""
    ref: Optional['AnnotatedOp'] = None

    def __str__(self):
        if self.tensor:
            return f"{self.kind}({self.tensor})"
        return self.kind


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
        self._deps: Optional[List[GroupedSubIterK]] = None
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
                    subIterK_slot=si))

    def place_GRs(self) -> List[SubIterKSlot]:
        """Place Global Reads by iterating MFMAs across partitions.

        Phase 1: Build ordered GR list from partition traversal respecting gr granularities.
        Phase 2: Distribute evenly GR atoms across all (partition, subIterK) slots. GR atoms being the smallest load granularity for a specific tensor.

        This should give a sheduling respecting the following rules:
         - GR are in the order we expect them from the LR pov
         - we respect the GR granularities (can change the above rule a bit)
         - Overall loads are spread accross all subIterKs of all partitions.
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

    def annotate_deps(self) -> List[GroupedSubIterK]:
        """Annotate each operation with its raw before-dependencies.

        These are the per-op dependencies before group() serialization. They may
        reference operations in other subIterKs or iterations (descriptive).

        Rules:
        - MFMA(subIterK=k) depends on all LRs that loaded subIterK=k data
        - LR depends on GR for same tensor (data must be in LDS)
        - LR cross-MT depends on LR_INCOp (buffer swap)
        - GR depends on GRInc for same tensor (pointer update)
        - GR depends on collision LR for same tensor (LDS double-buffer)
        """
        if 'gr' not in self._completed:
            self.place_GRs()
        slots = self._partitions[0]
        cfg = self.config
        numK = cfg.numSubIterK

        grouped = []
        for k, slot in enumerate(slots):
            gslot = GroupedSubIterK(subIterK=k)

            # ── MFMA deps ──
            # MFMA(subIterK=k) depends on all LRs that loaded subIterK=k data.
            # These are from the previous iteration (MT n).
            if slot.mfma:
                before = []
                tensor_names = ['A', 'B']
                if cfg.hasScale:
                    tensor_names += ['SA', 'SB']
                for t in tensor_names:
                    before.append(DepOp(kind='lr_ref', tensor=t))

                gslot.ops.append(AnnotatedOp(
                    kind='MFMA', before=before, placement=slot.mfma))

            # ── LR deps ──
            for lr in slot.lrs:
                before = []
                if lr.mtIteration != "n":
                    before.append(DepOp(kind='lr_inc', tensor=lr.tensor))
                before.append(DepOp(kind='gr_ref', tensor=lr.tensor))

                gslot.ops.append(AnnotatedOp(
                    kind='LR', before=before, placement=lr))

            # ── GR deps ──
            for gr in slot.grs:
                before = [
                    DepOp(kind='gr_inc', tensor=gr.tensor),
                    DepOp(kind='lr_ref', tensor=gr.tensor),
                ]
                gslot.ops.append(AnnotatedOp(
                    kind='GR', before=before, placement=gr))

            grouped.append(gslot)

        self._deps = grouped
        self._completed.add('deps')
        return grouped

    # ── Group and serialize ───────────────────────────────

    def group(self) -> List[GroupedSubIterK]:
        """Serialize operations within each subIterK.

        Takes annotate_deps' raw deps and transforms them:
        - Cross-subIterK LR deps → WaitLROp barrier
        - Cross-subIterK GR deps → WaitGROp barrier
        - Same-subIterK deps → node refs for serialization
        - LRs serialized: A → B → SA → SB with WaitGROp before first
        - GRs serialized: SA → A (or SB → B) with merged deps
        """
        if 'gr' not in self._completed:
            self.place_GRs()
        slots = self._partitions[0]
        if 'deps' not in self._completed:
            self.annotate_deps()
        cfg = self.config
        numK = cfg.numSubIterK

        grouped = []
        for k, slot in enumerate(slots):
            gslot = GroupedSubIterK(subIterK=k)

            # MFMA with WaitLROp before
            if slot.mfma:
                gslot.ops.append(AnnotatedOp(
                    kind='MFMA',
                    before=[DepOp(kind='wait_lr')],
                    placement=slot.mfma,
                ))

            # LRs serialized: A → B → SA → SB
            # First LR gets WaitGROp + any LR_INCOps
            lr_order = ['A', 'B', 'SA', 'SB']
            ordered_lrs = sorted(slot.lrs, key=lambda lr: lr_order.index(lr.tensor))

            prev_lr_op = None
            for i, lr in enumerate(ordered_lrs):
                before = []
                if i == 0:
                    before.append(DepOp(kind='wait_gr'))
                    for lr2 in ordered_lrs:
                        if lr2.mtIteration != "n":
                            before.append(DepOp(kind='lr_inc', tensor=lr2.tensor))
                else:
                    before.append(DepOp(kind='ref', ref=prev_lr_op))

                lr_op = AnnotatedOp(
                    kind='LR', before=before, placement=lr,
                )
                gslot.ops.append(lr_op)
                prev_lr_op = lr_op

            # GRs serialized: SA → A or SB → B
            gr_order = ['SA', 'A', 'SB', 'B']
            ordered_grs = sorted(slot.grs, key=lambda gr: gr_order.index(gr.tensor))

            # Only subIterK=0 needs collision wait (same-subIterK LR still async).
            # Later subIterKs: collision LR covered by MFMAs' WaitLROp.
            needs_collision_wait = self._needs_collision_wait(k, ordered_lrs)
            collision_lr_op = None
            if needs_collision_wait:
                collision_lr_op = self._find_collision_LR_op(gslot, ordered_lrs)

            prev_gr_op = None
            for i, gr in enumerate(ordered_grs):
                before = []
                if i == 0:
                    for gr2 in ordered_grs:
                        before.append(DepOp(kind='gr_inc', tensor=gr2.tensor))
                    if collision_lr_op:
                        before.append(DepOp(kind='ref', ref=collision_lr_op))
                        before.append(DepOp(kind='wait_lr_sync'))
                else:
                    before.append(DepOp(kind='ref', ref=prev_gr_op))

                gr_op = AnnotatedOp(
                    kind='GR', before=before, placement=gr,
                )
                gslot.ops.append(gr_op)
                prev_gr_op = gr_op

            grouped.append(gslot)

        self._grouped = grouped
        self._completed.add('group')
        return grouped

    def _needs_collision_wait(self, subIterK: int, ordered_lrs: list) -> bool:
        """Check if GRs at this subIterK need explicit wait_lr_sync.

        In group(), only same-subIterK node refs survive; cross-subIterK deps
        are absorbed by WaitLROp/WaitGROp barriers.

        At subIterK=0: the LRs in this slot just ran (async ds_reads). GR is
        about to write to the same LDS buffer. Need ref(LR) + wait_lr_sync.

        At later subIterKs: the collision LR is from a previous iteration,
        already covered by MFMAs' WaitLROp. No explicit wait needed.
        """
        return subIterK == 0

    def _find_collision_LR_op(self, gslot: GroupedSubIterK, ordered_lrs: list) -> Optional[AnnotatedOp]:
        """Find the last LR AnnotatedOp in this subIterK that GR must wait for.

        Returns the last LR node — since LRs are serialized (A → B → SA),
        referencing the last one ensures GR is sequenced after all LRs
        and the wait_lr_sync covers all pending ds_reads.
        """
        last_lr = None
        for op in gslot.ops:
            if op.kind == 'LR':
                last_lr = op
        return last_lr

    # ── Produce EmittedModules ────────────────────────────

    def emit(self) -> List[List[EmittedModule]]:
        """Convert grouped ops into a flat List[EmittedModule] per subIterK.

        Each AnnotatedOp becomes one EmittedModule. Its DepOp before-deps are
        flattened into chained EmittedModules, with the primary op's .before
        pointing to the last dep in the chain.

        'ref' deps resolve to the referenced AnnotatedOp's moduleId.
        'wait_lr_sync' expands to two modules (wait_lr → sync).
        All other dep kinds become one EmittedModule each.
        """
        if 'group' not in self._completed:
            self.group()
        grouped = self._grouped

        all_emitted = []
        for gslot in grouped:
            emitted: List[EmittedModule] = []
            op_to_id = {}  # AnnotatedOp id() → EmittedModule moduleId

            def add(opType, label, before=None):
                mid = len(emitted)
                emitted.append(EmittedModule(
                    moduleId=mid, opType=opType,
                    label=label, before=before))
                return mid

            for op in gslot.ops:
                opType = op.kind.lower()

                prev_id = None
                for dep in op.before:
                    if dep.kind == 'ref':
                        ref_id = op_to_id.get(id(dep.ref))
                        if ref_id is not None:
                            prev_id = ref_id
                    elif dep.kind == 'wait_lr_sync':
                        # Expand to two modules: wait_lr → sync
                        prev_id = add('wait_lr', 'wait_lr', before=prev_id)
                        prev_id = add('sync', 'sync', before=prev_id)
                    else:
                        prev_id = add(dep.kind, str(dep), before=prev_id)

                label = self._format_op_label(op, gslot)
                mid = add(opType, label, before=prev_id)
                op_to_id[id(op)] = mid

            all_emitted.append(emitted)

        self._emitted = all_emitted
        self._completed.add('emit')
        return all_emitted

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

    def _format_op_label(self, op: AnnotatedOp, gslot) -> str:
        """Compute a human-readable label from an AnnotatedOp's placement.

        Labels are only for display — never stored in the data structure.
        Set annotations come from the assign_vgpr_sets slot data.
        """
        p = op.placement
        k = gslot.subIterK if hasattr(gslot, 'subIterK') else 0
        s3 = self._get_slot_data(k)

        if op.kind == 'MFMA':
            label = (f"MFMAs (MT n, subIterK {p.subIterK}  ) "
                     f"A : {p.tileA.fmt_tiles()} , B : {p.tileB.fmt_tiles()}")
            if s3 and s3.mfma_sets:
                label += " " + ", ".join(
                    f"set{t}:{s3.mfma_sets[t]}"
                    for t in sorted(s3.mfma_sets.keys()))
            return label
        elif op.kind == 'LR':
            t = self._fmt_tensor(p.tensor)
            label = (f"LR {t} (MT {p.mtIteration}, "
                     f"subIterK {p.tiles.fmt_k()}) "
                     f"{p.tiles.fmt_tiles()}")
            if s3 and s3.lr_sets and p.tensor in s3.lr_sets:
                label += f" set{p.tensor}:{s3.lr_sets[p.tensor]}"
            return label
        elif op.kind == 'GR':
            return (f"GR {p.tensor} (MT {p.mtIteration}, "
                    f"subIterK {p.tiles.fmt_k()}) "
                    f"ids {p.tiles.fmt_tiles()}")
        return op.kind

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

    def print_deps(self, grouped: List[GroupedSubIterK] = None) -> str:
        """Print annotate_deps output: ops with raw per-op dependencies."""
        return self._print_grouped(grouped or self._deps)

    def print_group(self, grouped: List[GroupedSubIterK] = None) -> str:
        """Print group output: serialized ops with dependencies."""
        return self._print_grouped(grouped or self._grouped)

    def _print_grouped(self, grouped: List[GroupedSubIterK]) -> str:
        """Shared format for annotate_deps and group output."""
        buf = io.StringIO()
        buf.write("MAINLOOP:\n")
        buf.write("  Partition 0:\n")
        for gslot in grouped:
            buf.write(f"    subIterK={gslot.subIterK}:\n")
            for op in gslot.ops:
                label = self._format_op_label(op, gslot)
                buf.write(f"      {label}\n")
                if op.before:
                    buf.write("        before:\n")
                    for dep in op.before:
                        dep_str = self._format_dep(dep, gslot)
                        buf.write(f"            - {dep_str}\n")
        return buf.getvalue()

    def _format_dep(self, dep: DepOp, gslot) -> str:
        """Format a DepOp for display. Refs are resolved to labels."""
        if dep.kind == 'ref' and dep.ref:
            return self._format_op_label(dep.ref, gslot)
        return str(dep)

    def print_emit(self, all_emitted: List[List[EmittedModule]] = None) -> str:
        """Print emit output: EmittedModule list with before-links."""
        if all_emitted is None:
            all_emitted = self._emitted
        buf = io.StringIO()
        for k, emitted in enumerate(all_emitted):
            buf.write(f"subIterK={k}:\n")
            for em in emitted:
                before_str = f" ← [{em.before}]" if em.before is not None else ""
                buf.write(f"  [{em.moduleId:2d}] {em.opType:8s} {em.label}{before_str}\n")
            buf.write("\n")
        return buf.getvalue()
