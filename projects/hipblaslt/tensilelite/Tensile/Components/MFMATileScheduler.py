"""MFMATile-based logical scheduler.

Builds a logical schedule using MFMA tile indices as the core primitive,
with explicit per-operation load granularity for GR/LR on A, B, SA, SB.

The schedule is built in 6 steps:
  1. Place LRs based on their granularities
  2. Assign VGPR tile sets (ping-pong) based on subIterK dependencies
  3. Place GRs
  4. Annotate dependencies
  5. Group and serialize (produce paths for instructionSchedule)
  6. Produce List[EmittedModule] with before-link chains
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
    # Step 2 annotations
    mfma_sets: Optional[dict] = None   # {'A': int, 'B': int, 'SA': int, 'SB': int}
    lr_sets: Optional[dict] = None     # tensor -> set id per LR


# ── Step 2 output ───────────────────────────────────────────

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
      'lr_ref'       — dependency on LR for a tensor (Step 4, cross-subIterK)
      'gr_ref'       — dependency on GR for a tensor (Step 4, cross-subIterK)
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


# ── Step 5 grouped output ──────────────────────────────────

@dataclass
class GroupedSubIterK:
    """Step 5 output: serialized ops within one subIterK."""
    subIterK: int
    ops: List[AnnotatedOp] = field(default_factory=list)


# ── Step 6 output ──────────────────────────────────────────

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

    Builds the schedule in 5 steps, each producing testable intermediate output.
    """

    def __init__(self, config: SchedulerConfig):
        self.config = config
        self._step1_result: Optional[List[List[SubIterKSlot]]] = None
        self._step2_result: Optional[List[SubIterKSlot]] = None
        self._step3_result: Optional[List[SubIterKSlot]] = None
        self._step4_result: Optional[List[SubIterKSlot]] = None
        self._step5_result: Optional[List[GroupedSubIterK]] = None
        self._step6_result: Optional[List[EmittedModule]] = None

    # ── Step 1: Place LRs ─────────────────────────────────

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

    def step1_place_LRs(self) -> List[List[SubIterKSlot]]:
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

        self._step1_result = partitions
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

        # Always include A and B. Scales only when their side needs loading.
        tensors = [('A', cfg.lrA), ('B', cfg.lrB)]
        if cfg.hasScale:
            if load['A']:
                tensors.append(('SA', cfg.lrSA))
            if load['B']:
                tensors.append(('SB', cfg.lrSB))

        # Place LRs grouped by k_gran.
        # - Non-wrapping (K-prefetch): check placed set, skip if already loaded.
        # - Wrapping (cross-partition): check load[side], skip if tiles unchanged.
        for k_gran in sorted(set(g.size.k for _, g in tensors)):
            group = [(t, g) for t, g in tensors if g.size.k == k_gran]
            num_chunks = numK // k_gran
            for chunk_idx in range(num_chunks):
                next_chunk = (chunk_idx + 1) % num_chunks
                is_wrap = (next_chunk == 0)
                lr_mt = ("n+1" if is_last else "n") if multi_part else \
                         "n+1" if is_wrap else "n"
                lr_k_start = next_chunk * k_gran
                lr_k_end = lr_k_start + k_gran
                base_slot = chunk_idx * k_gran

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

    # ── Step 2: Assign VGPR sets ──────────────────────────

    def step2_assign_vgpr_sets(self) -> List[SubIterKSlot]:
        """Assign VGPR set IDs (0 or 1) to MFMAs and LRs.

        Rule: MFMA at subIterK=k reads from the set that was written by the
        LR that loaded that subIterK's data. The LR writes to the *other* set
        so that MFMA and LR never collide.
        """
        if self._step1_result is None:
            self.step1_place_LRs()
        slots = self._step1_result[0]  # operate on partition 0
        cfg = self.config
        numK = cfg.numSubIterK

        # For each tensor, track which set the MFMA reads at each subIterK.
        # subIterK=0 MFMA always reads set 0.
        # If LR at subIterK=k loads data for subIterK=k+1,
        # then LR writes to set != MFMA[k].set (the other set),
        # and MFMA[k+1] reads from that written set.

        tensor_names = ['A', 'B']
        if cfg.hasScale:
            tensor_names += ['SA', 'SB']

        for slot in slots:
            slot.mfma_sets = {}
            slot.lr_sets = {}

        # Determine k granularity per tensor for set-advance logic
        tensor_k_gran = {}
        for t, gran, _ in [('A', cfg.lrA, 0), ('B', cfg.lrB, 0)]:
            tensor_k_gran[t] = gran.size.k
        if cfg.hasScale:
            tensor_k_gran['SA'] = cfg.lrSA.size.k
            tensor_k_gran['SB'] = cfg.lrSB.size.k

        # Assign sets per tensor across subIterKs
        for t in tensor_names:
            current_set = 0
            for k in range(numK):
                slots[k].mfma_sets[t] = current_set
                # Find LR in this slot for this tensor
                lr_for_t = [lr for lr in slots[k].lrs if lr.tensor == t]
                if lr_for_t:
                    # LR writes to the *other* set
                    lr_set = 1 - current_set
                    slots[k].lr_sets[t] = lr_set
                    # Only advance MFMA set if k_gran == 1 (per-subIterK loads).
                    # For k_gran == numSubIterK, the LR loads for next MT,
                    # so current MT's MFMAs keep reading the same set.
                    if tensor_k_gran[t] == 1:
                        current_set = lr_set

        self._step2_result = slots
        return slots

    # ── Step 3: Place GRs ─────────────────────────────────

    def step3_place_GRs(self) -> List[SubIterKSlot]:
        """Place Global Reads for MT n+2.

        GRs are split across subIterKs:
        - subIterK=0: GR SA + GR A
        - subIterK=1: GR SB + GR B
        (or further split if more subIterKs available)
        """
        if self._step2_result is None:
            self.step2_assign_vgpr_sets()
        slots = self._step2_result
        cfg = self.config
        numK = cfg.numSubIterK

        # GR for A and SA go in subIterK=0, B and SB go in subIterK=1
        # Each tensor gets one GR per subIterK covering all its M/N tiles.
        # The mn granularity is metadata for emission (how many buffer_loads),
        # not for splitting at the logical level.

        gr_a_slot = 0
        gr_b_slot = min(1, numK - 1)

        # Place GR A (all M tiles)
        slots[gr_a_slot].grs.append(GRPlacement(
            tensor='A', mtIteration='n+2',
            tiles=MFMATileRange(0, cfg.grA.size.k, 0, cfg.numMFMATilesM),
            subIterK_slot=gr_a_slot))

        # Place GR B (all N tiles)
        slots[gr_b_slot].grs.append(GRPlacement(
            tensor='B', mtIteration='n+2',
            tiles=MFMATileRange(0, cfg.grB.size.k, 0, cfg.numMFMATilesN),
            subIterK_slot=gr_b_slot))

        if cfg.hasScale:
            # GR SA in same slot as GR A
            slots[gr_a_slot].grs.append(GRPlacement(
                tensor='SA', mtIteration='n+2',
                tiles=MFMATileRange(0, cfg.grSA.size.k, 0, cfg.numMFMATilesM),
                subIterK_slot=gr_a_slot))

            # GR SB in same slot as GR B
            slots[gr_b_slot].grs.append(GRPlacement(
                tensor='SB', mtIteration='n+2',
                tiles=MFMATileRange(0, cfg.grSB.size.k, 0, cfg.numMFMATilesN),
                subIterK_slot=gr_b_slot))

        self._step3_result = slots
        return slots

    # ── Step 4: Annotate dependencies ─────────────────────

    def step4_annotate_deps(self) -> List[GroupedSubIterK]:
        """Annotate each operation with its raw before-dependencies.

        These are the per-op dependencies before Step 5 grouping. They may
        reference operations in other subIterKs or iterations (descriptive).

        Rules:
        - MFMA(subIterK=k) depends on all LRs that loaded subIterK=k data
        - LR depends on GR for same tensor (data must be in LDS)
        - LR cross-MT depends on LR_INCOp (buffer swap)
        - GR depends on GRInc for same tensor (pointer update)
        - GR depends on collision LR for same tensor (LDS double-buffer)
        """
        if self._step3_result is None:
            self.step3_place_GRs()
        slots = self._step3_result
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

        self._step4_result = grouped
        return grouped

    # ── Step 5: Group and serialize ───────────────────────

    def step5_group(self) -> List[GroupedSubIterK]:
        """Serialize operations within each subIterK.

        Takes Step 4's raw deps and transforms them:
        - Cross-subIterK LR deps → WaitLROp barrier
        - Cross-subIterK GR deps → WaitGROp barrier
        - Same-subIterK deps → node refs for serialization
        - LRs serialized: A → B → SA → SB with WaitGROp before first
        - GRs serialized: SA → A (or SB → B) with merged deps
        """
        if self._step3_result is None:
            self.step3_place_GRs()
        # Step 5 operates on the raw slot placements from Step 3
        slots = self._step3_result
        # Ensure Step 4 has been run (for its own output/display)
        if self._step4_result is None:
            self.step4_annotate_deps()
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

        self._step5_result = grouped
        return grouped

    def _needs_collision_wait(self, subIterK: int, ordered_lrs: list) -> bool:
        """Check if GRs at this subIterK need explicit wait_lr_sync.

        At Step 5, only same-subIterK node refs survive; cross-subIterK deps
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

    # ── Step 6: Produce EmittedModules ────────────────────

    def step6_emit(self) -> List[List[EmittedModule]]:
        """Convert Step 5 grouped ops into a flat List[EmittedModule] per subIterK.

        Each AnnotatedOp becomes one EmittedModule. Its DepOp before-deps are
        flattened into chained EmittedModules, with the primary op's .before
        pointing to the last dep in the chain.

        'ref' deps resolve to the referenced AnnotatedOp's moduleId.
        'wait_lr_sync' expands to two modules (wait_lr → sync).
        All other dep kinds become one EmittedModule each.
        """
        if self._step5_result is None:
            self.step5_group()
        grouped = self._step5_result

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

        self._step6_result = all_emitted
        return all_emitted

    # ── Print helpers ───────────────────────────────────────

    @staticmethod
    def _fmt_tensor(tensor: str) -> str:
        """Pad tensor name to 2 chars for alignment: 'A' -> 'A ', 'SA' -> 'SA'."""
        return tensor.ljust(2)

    def _get_slot_data(self, subIterK: int) -> Optional[SubIterKSlot]:
        """Get Step 3 slot for set annotations (mfma_sets, lr_sets)."""
        if self._step3_result and subIterK < len(self._step3_result):
            return self._step3_result[subIterK]
        return None

    def _format_op_label(self, op: AnnotatedOp, gslot) -> str:
        """Compute a human-readable label from an AnnotatedOp's placement.

        Labels are only for display — never stored in the data structure.
        Set annotations come from the Step 3 slot data.
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

    def print_step1(self, partitions: List[List[SubIterKSlot]] = None) -> str:
        """Print Step 1 output in design doc format."""
        if partitions is None:
            partitions = self._step1_result
        buf = io.StringIO()
        buf.write("MAINLOOP:\n")
        for pi, slots in enumerate(partitions):
            buf.write(f"  Partition {pi}:\n")
            self._print_step1_partition(buf, slots)
        return buf.getvalue()

    def _print_step1_partition(self, buf, slots):
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

    def print_step2(self, slots: List[SubIterKSlot] = None) -> str:
        """Print Step 2 output: same as Step 1 but with set annotations."""
        if slots is None:
            slots = self._step2_result
        buf = io.StringIO()
        buf.write("MAINLOOP:\n")
        buf.write("  Partition 0:\n")
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

    def print_step3(self, slots: List[SubIterKSlot] = None) -> str:
        """Print Step 3 output: Step 2 + GR placements."""
        if slots is None:
            slots = self._step3_result
        buf = io.StringIO()
        buf.write("MAINLOOP:\n")
        buf.write("  Partition 0:\n")
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

    def print_step4(self, grouped: List[GroupedSubIterK] = None) -> str:
        """Print Step 4 output: ops with raw per-op dependencies."""
        if grouped is None:
            grouped = self._step4_result
        return self.print_step5(grouped)

    def print_step5(self, grouped: List[GroupedSubIterK] = None) -> str:
        """Print Step 5 output: grouped and serialized with dependencies."""
        if grouped is None:
            grouped = self._step5_result
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

    def print_step6(self, all_emitted: List[List[EmittedModule]] = None) -> str:
        """Print Step 6 output: EmittedModule list with before-links."""
        if all_emitted is None:
            all_emitted = self._step6_result
        buf = io.StringIO()
        for k, emitted in enumerate(all_emitted):
            buf.write(f"subIterK={k}:\n")
            for em in emitted:
                before_str = f" ← [{em.before}]" if em.before is not None else ""
                buf.write(f"  [{em.moduleId:2d}] {em.opType:8s} {em.label}{before_str}\n")
            buf.write("\n")
        return buf.getvalue()
