"""MFMATile-based logical scheduler.

Builds a logical schedule using MFMA tile indices as the core primitive,
with explicit per-operation load granularity for GR/LR on A, B, SA, SB.

The schedule is built in 5 steps:
  1. Place LRs based on their granularities
  2. Assign VGPR tile sets (ping-pong) based on subIterK dependencies
  3. Place GRs
  4. Annotate dependencies
  5. Group and serialize (produce paths for instructionSchedule)
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

    @property
    def hasScale(self) -> bool:
        return self.lrSA is not None and self.lrSB is not None

    @classmethod
    def from_tile_info(cls, tileInfoA, tileInfoB,
                       lrA: ReadGranularity, lrB: ReadGranularity,
                       grA: ReadGranularity, grB: ReadGranularity,
                       scaleTileInfoA=None, scaleTileInfoB=None,
                       lrSA: Optional[ReadGranularity] = None,
                       lrSB: Optional[ReadGranularity] = None,
                       grSA: Optional[ReadGranularity] = None,
                       grSB: Optional[ReadGranularity] = None):
        """Build config from TileInfo objects.

        Derives numMFMATilesM/N/K from the tile info:
        - numMFMATilesM = tileInfoA.localMMATileGrid[0]
        - numMFMATilesN = tileInfoB.localMMATileGrid[0]
        - numSubIterK   = tileInfoA.subtileShape[1]
        """
        numMFMATilesM = tileInfoA.localMMATileGrid[0]
        numMFMATilesN = tileInfoB.localMMATileGrid[0]
        numSubIterK = tileInfoA.subtileShape[1]

        assert tileInfoA.subtileShape[1] == tileInfoB.subtileShape[1], \
            "A and B must have same subtileShape[1]"

        return cls(
            numMFMATilesM=numMFMATilesM,
            numMFMATilesN=numMFMATilesN,
            numSubIterK=numSubIterK,
            lrA=lrA, lrB=lrB,
            grA=grA, grB=grB,
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


# ── Step 4 dependency types ─────────────────────────────────

@dataclass
class DepRef:
    """A dependency reference: either an op string or an LR/GR reference."""
    label: str   # human-readable label e.g. "LR A (MT n, subIterK [0])"


@dataclass
class AnnotatedOp:
    """An operation with its before-dependencies."""
    kind: str        # 'MFMA', 'LR', 'GR', etc.
    label: str       # human-readable
    before: List[str] = field(default_factory=list)
    # Original placement reference
    placement: object = None


# ── Step 5 grouped output ──────────────────────────────────

@dataclass
class GroupedSubIterK:
    """Step 5 output: serialized ops within one subIterK."""
    subIterK: int
    ops: List[AnnotatedOp] = field(default_factory=list)


# ── Main scheduler class ───────────────────────────────────

class MFMATileScheduler:
    """MFMATile-based logical scheduler.

    Builds the schedule in 5 steps, each producing testable intermediate output.
    """

    def __init__(self, config: SchedulerConfig):
        self.config = config
        self._step1_result: Optional[List[SubIterKSlot]] = None
        self._step2_result: Optional[List[SubIterKSlot]] = None
        self._step3_result: Optional[List[SubIterKSlot]] = None
        self._step4_result: Optional[List[SubIterKSlot]] = None
        self._step5_result: Optional[List[GroupedSubIterK]] = None

    # ── Step 1: Place LRs ─────────────────────────────────

    def step1_place_LRs(self) -> List[SubIterKSlot]:
        """Place MFMAs and LRs based on read granularities.

        For each subIterK:
        - Place MFMA that consumes all M/N tiles at this subIterK
        - Place LR for each tensor based on its granularity:
          - k=1: one LR per subIterK, loading next subIterK
          - k=numSubIterK: one LR per MT switch, split across subIterKs
        """
        cfg = self.config
        numK = cfg.numSubIterK
        slots = [SubIterKSlot(subIterK=k) for k in range(numK)]

        # Place MFMAs: each subIterK gets one MFMA consuming all M/N tiles
        for k in range(numK):
            slots[k].mfma = MFMAPlacement(
                subIterK=k,
                tileA=MFMATileRange(k, k + 1, 0, cfg.numMFMATilesM),
                tileB=MFMATileRange(k, k + 1, 0, cfg.numMFMATilesN),
            )

        # Place LRs for each tensor
        tensors_and_grans = [
            ('A',  cfg.lrA,  cfg.numMFMATilesM),
            ('B',  cfg.lrB,  cfg.numMFMATilesN),
        ]
        if cfg.hasScale:
            tensors_and_grans.append(('SA', cfg.lrSA, cfg.numMFMATilesM))
            tensors_and_grans.append(('SB', cfg.lrSB, cfg.numMFMATilesN))

        for tensor, gran, numTiles in tensors_and_grans:
            self._place_LRs_for_tensor(slots, tensor, gran, numTiles, numK)

        self._step1_result = slots
        return slots

    def _place_LRs_for_tensor(self, slots, tensor, gran, numTiles, numK):
        """Place LR operations for one tensor across subIterK slots.

        The LR always loads data for the *next* consumption point:
        - k granularity = 1: LR at subIterK=i loads subIterK=i+1 data
          - At last subIterK, loads subIterK=0 of next MT (n+1)
        - k granularity = numSubIterK: LR loads all subIterKs at once
          - Only placed once per MT boundary, split across subIterKs by tensor
        """
        k_gran = gran.size.k
        mn_gran = gran.size.mn

        if k_gran == 1:
            # One LR per subIterK, loading the next subIterK's data
            # Each LR covers ALL M/N tiles for that tensor
            for k in range(numK):
                next_k = (k + 1) % numK
                is_mt_switch = (next_k == 0)
                mt_iter = "n+1" if is_mt_switch else "n"

                if is_mt_switch:
                    lr_k_start = 0
                    lr_k_end = k_gran
                else:
                    lr_k_start = next_k
                    lr_k_end = next_k + k_gran

                lr = LRPlacement(
                    tensor=tensor,
                    mtIteration=mt_iter,
                    tiles=MFMATileRange(lr_k_start, lr_k_end, 0, numTiles),
                    subIterK_slot=k,
                )
                slots[k].lrs.append(lr)

        elif k_gran == numK:
            # One LR covers all subIterKs — placed once, at MT switch
            # Split across subIterKs by tensor type:
            # SA goes in the first subIterK that has room, SB in the next, etc.
            # For the design doc example: SA at subIterK=0, SB at subIterK=1
            if tensor == 'SA':
                slot_k = 0
            elif tensor == 'SB':
                slot_k = numK - 1
            else:
                slot_k = 0

            lr = LRPlacement(
                tensor=tensor,
                mtIteration="n+1",
                tiles=MFMATileRange(0, numK, 0, numTiles),
                subIterK_slot=slot_k,
            )
            slots[slot_k].lrs.append(lr)

        else:
            raise NotImplementedError(
                f"LR k granularity {k_gran} not yet supported (must be 1 or numSubIterK={numK})")

    # ── Step 2: Assign VGPR sets ──────────────────────────

    def step2_assign_vgpr_sets(self) -> List[SubIterKSlot]:
        """Assign VGPR set IDs (0 or 1) to MFMAs and LRs.

        Rule: MFMA at subIterK=k reads from the set that was written by the
        LR that loaded that subIterK's data. The LR writes to the *other* set
        so that MFMA and LR never collide.
        """
        if self._step1_result is None:
            self.step1_place_LRs()
        slots = self._step1_result
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

    def step4_annotate_deps(self) -> List[SubIterKSlot]:
        """Annotate each operation with its before-dependencies.

        Rules:
        - MFMA(subIterK=k) depends on all LRs that loaded subIterK=k data
        - LR(n) depends on GR(n) (data must be in LDS before local read)
        - GR(n+2) depends on LR(n) (LDS double-buffer collision avoidance)
        - LR cross-MT depends on LR_INCOp (buffer swap)
        - GR last-for-MT depends on GR_INCOp (pointer update)
        """
        if self._step3_result is None:
            self.step3_place_GRs()
        slots = self._step3_result

        result = []
        for k, slot in enumerate(slots):
            annotated_slot = SubIterKSlot(
                subIterK=k,
                mfma=slot.mfma,
                lrs=list(slot.lrs),
                grs=list(slot.grs),
                mfma_sets=slot.mfma_sets,
                lr_sets=slot.lr_sets,
            )
            result.append(annotated_slot)

        self._step4_result = result
        return result

    # ── Step 5: Group and serialize ───────────────────────

    def step5_group(self) -> List[GroupedSubIterK]:
        """Serialize operations within each subIterK.

        - MFMAs: single WaitLROp before
        - LRs: serialized A → B → SA → SB with WaitGROp+Sync before first
        - GRs: serialized SA → A (or SB → B) with merged deps
        """
        if self._step4_result is None:
            self.step4_annotate_deps()
        slots = self._step4_result
        cfg = self.config
        numK = cfg.numSubIterK

        grouped = []
        for k, slot in enumerate(slots):
            gslot = GroupedSubIterK(subIterK=k)

            # MFMA with WaitLROp before
            if slot.mfma:
                mfma_label = (f"MFMAs (MT n, subIterK {k}  ) "
                              f"A : {slot.mfma.tileA.fmt_tiles()} , "
                              f"B : {slot.mfma.tileB.fmt_tiles()}")
                sets_str = ", ".join(
                    f"set{t}:{slot.mfma_sets[t]}"
                    for t in sorted(slot.mfma_sets.keys())
                ) if slot.mfma_sets else ""
                mfma_label += f" {sets_str}"

                mfma_op = AnnotatedOp(
                    kind='MFMA', label=mfma_label,
                    before=["WaitLROp"],
                    placement=slot.mfma,
                )
                gslot.ops.append(mfma_op)

            # LRs serialized: A → B → SA → SB
            # First LR gets WaitGROp + Sync + any LR_INCOps
            lr_order = ['A', 'B', 'SA', 'SB']
            ordered_lrs = sorted(slot.lrs, key=lambda lr: lr_order.index(lr.tensor))

            prev_lr_label = None
            for i, lr in enumerate(ordered_lrs):
                lr_label = (f"LR {lr.tensor}  (MT {lr.mtIteration}, "
                            f"subIterK {lr.tiles.fmt_k()}) "
                            f"{lr.tiles.fmt_tiles()}")
                if lr.tensor in (slot.lr_sets or {}):
                    lr_label += f" set{lr.tensor}:{slot.lr_sets[lr.tensor]}"

                before = []
                if i == 0:
                    # First LR: WaitGROp + Sync
                    before.append("WaitGROp(A=X,B=Y,SA=W SB=Z), SyncOp")
                    # Add LR_INCOps for cross-MT reads
                    for lr2 in ordered_lrs:
                        if lr2.mtIteration != "n":
                            before.append(f"LR_INCOp {lr2.tensor}")
                else:
                    before.append(prev_lr_label)

                lr_op = AnnotatedOp(
                    kind='LR', label=lr_label,
                    before=before,
                    placement=lr,
                )
                gslot.ops.append(lr_op)
                prev_lr_label = lr_label

            # GRs serialized: SA → A or SB → B
            gr_order = ['SA', 'A', 'SB', 'B']
            ordered_grs = sorted(slot.grs, key=lambda gr: gr_order.index(gr.tensor))

            prev_gr_label = None
            for i, gr in enumerate(ordered_grs):
                gr_label = (f"GR {gr.tensor} (MT {gr.mtIteration}, "
                            f"subIterK {gr.tiles.fmt_k()}) "
                            f"ids {gr.tiles.fmt_tiles()}")

                before = []
                if i == 0:
                    # First GR: depends on GRInc + LR collision avoidance
                    for gr2 in ordered_grs:
                        before.append(f"GRInc {gr2.tensor}")
                    # LDS collision: GR(n+2) depends on LR(n)
                    collision_lrs = self._find_collision_LRs(slot, slots)
                    for clr in collision_lrs:
                        before.append(
                            f"LR {clr.tensor} (MT n, subIterK {clr.tiles.fmt_k()}) "
                            f"{clr.tiles.fmt_tiles()}, WaitLROp, Sync")
                else:
                    before.append(prev_gr_label)

                gr_op = AnnotatedOp(
                    kind='GR', label=gr_label,
                    before=before,
                    placement=gr,
                )
                gslot.ops.append(gr_op)
                prev_gr_label = gr_label

            grouped.append(gslot)

        self._step5_result = grouped
        return grouped

    def _find_collision_LRs(self, current_slot, all_slots):
        """Find LRs from current subIterK that GR(n+2) must wait for (LDS collision)."""
        # GR(n+2) writes to LDS buffer that LR(n) reads from.
        # So GR depends on the LR in the same slot that reads from MT n.
        result = []
        for lr in current_slot.lrs:
            result.append(lr)
        return result

    # ── Print helpers ───────────────────────────────────────

    @staticmethod
    def _fmt_tensor(tensor: str) -> str:
        """Pad tensor name to 2 chars for alignment: 'A' -> 'A ', 'SA' -> 'SA'."""
        return tensor.ljust(2)

    def print_step1(self, slots: List[SubIterKSlot] = None) -> str:
        """Print Step 1 output in design doc format."""
        if slots is None:
            slots = self._step1_result
        buf = io.StringIO()
        buf.write("MAINLOOP:\n")
        buf.write("  Partition 0:\n")
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

    def print_step4(self, slots: List[SubIterKSlot] = None) -> str:
        """Print Step 4 output: Step 3 + dependency annotations."""
        # For now, Step 4 produces the same structure; deps are shown in Step 5
        return self.print_step3(slots)

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
                buf.write(f"      {op.label}\n")
                if op.before:
                    buf.write("        before:\n")
                    for dep in op.before:
                        buf.write(f"            - {dep}\n")
        return buf.getvalue()
