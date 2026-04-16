"""Instruction emitter for MFMATileScheduler.

Converts the logical schedule (EmittedModule chains) into concrete GPU
instructions by dispatching each opType to its emit method.
"""

from __future__ import annotations

from Tensile.Components.SubtileBasedKernel import (
    emitMfmaInstruction, emitSingleDsRead, emitSingleBufferLoad,
    globalReadPtrUpdates, globalReadLDSBufferSwap,
    localReadLDSBufferSwap,
    globalReadDoScaleSubtile, globalReadScalePtrUpdates,
)
from rocisa.code import Module
from rocisa.instruction import SWaitCnt, SBarrier, DSLoadB32
from rocisa.container import vgpr, DSModifiers


class InstructionEmitter:
    """Emits GPU instructions for each opType in the MFMATileScheduler output.

    VGPR tile indexing uses flat 2x arrays with per-placement set selection:
      vgprTilesA[tileId + set * numTilesA]
      vgprTilesB[tileId + set * numTilesB]
    where set is read from placement.vgpr_sets (MFMA) or placement.vgpr_set (LR).
    """

    def __init__(self, writer, kernel, config,
                 tileInfoA, tileInfoB, dtileInfo,
                 vgprTilesA, vgprTilesB,
                 numTilesA, numTilesB,
                 scaleTileInfoA=None, scaleTileInfoB=None,
                 scaleVgprTiles=None, scaleVgprTilesAlt=None,
                 numScaleGroupsA=0):
        self.writer = writer
        self.kernel = kernel
        self.config = config
        self.tileInfoA = tileInfoA
        self.tileInfoB = tileInfoB
        self.dtileInfo = dtileInfo
        self.vgprTilesA = vgprTilesA
        self.vgprTilesB = vgprTilesB
        self.numTilesA = numTilesA
        self.numTilesB = numTilesB
        self.scaleVgprTiles = scaleVgprTiles
        self.scaleVgprTilesAlt = scaleVgprTilesAlt
        self.numScaleGroupsA = numScaleGroupsA

        # Derived state
        self.hasScale = scaleTileInfoA is not None and scaleTileInfoB is not None
        self.subtileShapeK = tileInfoA.subtileShape[1]
        self.tileInfoMap = {'A': tileInfoA, 'B': tileInfoB}
        if self.hasScale:
            self.tileInfoMap['SA'] = scaleTileInfoA
            self.tileInfoMap['SB'] = scaleTileInfoB

        # Dispatch table
        self._dispatch = {
            'mfma':     lambda em: self.emit_mfma(em.source),
            'lr':       lambda em: self.emit_lr(em.source),
            'gr':       lambda em: self.emit_gr(em.source),
            'wait_gr':  lambda em: self.emit_wait_gr(em.source),
            'wait_lr':  lambda em: self.emit_wait_lr(),
            'sync':     lambda em: self.emit_sync(),
            'lr_inc':   lambda em: self.emit_lr_inc(em.source),
            'gr_inc':   lambda em: self.emit_gr_inc(em.source),
            'gr_scale': lambda em: self.emit_gr_scale(em.source),
        }

    def _get_vgpr_tile(self, tensor, tileId, vgpr_set):
        """Look up a data VGPR tile by tensor, tileId, and set."""
        if tensor in ('A', 'SA'):
            return self.vgprTilesA[tileId + vgpr_set * self.numTilesA]
        else:
            return self.vgprTilesB[tileId + vgpr_set * self.numTilesB]

    def _get_scale_tiles(self, vgpr_set):
        """Return the scale VGPR list for the given set."""
        return self.scaleVgprTiles if vgpr_set == 0 else self.scaleVgprTilesAlt

    def emit_mfma(self, placement):
        """Emit MFMA instructions from MFMAPlacement."""
        module = Module()
        vgpr_sets = placement.vgpr_sets or {}
        setA = vgpr_sets.get('A', 0)
        setB = vgpr_sets.get('B', 0)
        setSA = vgpr_sets.get('SA', 0)
        subIterK = placement.subIterK

        scaleTiles = self._get_scale_tiles(setSA) if self.hasScale else None

        for a in placement.tileA.tileId_list:
            for b in placement.tileB.tileId_list:
                aTile = self.vgprTilesA[a + setA * self.numTilesA]
                bTile = self.vgprTilesB[b + setB * self.numTilesB]
                dTile = self.dtileInfo.vgprTiles[a + b * self.dtileInfo.localMMATileGrid[0]]

                if self.hasScale:
                    scaleGroupA = a // 2
                    scaleGroupB = b // 2
                    scaleAVgpr = scaleTiles[scaleGroupA]
                    scaleBVgpr = scaleTiles[self.numScaleGroupsA + scaleGroupB]
                    sAsel = (a % 2) + 2 * subIterK
                    sBsel = (b % 2) + 2 * subIterK
                else:
                    scaleAVgpr = scaleBVgpr = -1
                    sAsel = sBsel = 0

                module.add(emitMfmaInstruction(
                    self.writer, self.kernel, aTile, bTile, dTile, dTile,
                    scaleAVgpr=scaleAVgpr, scaleBVgpr=scaleBVgpr,
                    scaleAsel=sAsel, scaleBsel=sBsel,
                    comment=f"MFMA C[{a},{b}] += A[{a},K={subIterK}] * B[{b},K={subIterK}]"))
        return list(module.flatitems())

    def emit_lr(self, placement):
        """Emit LR (ds_read) instructions from LRPlacement."""
        module = Module()
        tensor = placement.tensor
        lr_set = placement.vgpr_set if placement.vgpr_set is not None else 0

        if tensor in ('A', 'B'):
            ti = self.tileInfoMap[tensor]
            for tileId in placement.tiles.tileId_list:
                for k in placement.tiles.subIterK_list:
                    subtileK = k // self.subtileShapeK
                    subIterK_within = k % self.subtileShapeK
                    dstTile = self._get_vgpr_tile(tensor, tileId, lr_set)
                    module.add(emitSingleDsRead(
                        ti, tileId, subtileK, subIterK_within, dstTile))
        elif tensor in ('SA', 'SB'):
            # Scale LR: DSLoadB32
            tc = 'MXSA' if tensor == 'SA' else 'MXSB'
            ti = self.tileInfoMap[tensor]
            scaleTiles = self._get_scale_tiles(lr_set)
            groupStride = 2 * ti.subtileSize
            for tileId in placement.tiles.tileId_list:
                scaleGroupIdx = tileId // 2
                vid = scaleGroupIdx if tensor == 'SA' else self.numScaleGroupsA + scaleGroupIdx
                for k in placement.tiles.subIterK_list:
                    subtileK = k // self.subtileShapeK
                    dsOffset = groupStride * (scaleGroupIdx * (self.config.numSubIterK // self.subtileShapeK) + subtileK)
                    vdst = scaleTiles[vid]
                    module.add(DSLoadB32(
                        dst=vgpr(vdst),
                        src=vgpr(ti.sharedVgprLROffset[0]),
                        ds=DSModifiers(offset=dsOffset),
                        comment=f"scale{tc}[group{scaleGroupIdx},K={k}]: load 4B from LDS"))
        return list(module.flatitems())

    def emit_gr(self, placement):
        """Emit GR (buffer_load) instructions from GRPlacement."""
        module = Module()
        tensor = placement.tensor
        if tensor in ('A', 'B'):
            ti = self.tileInfoMap[tensor]
            for tileId in placement.tiles.tileId_list:
                for k in placement.tiles.subIterK_list:
                    subtileK = k // self.subtileShapeK
                    module.add(emitSingleBufferLoad(ti, self.kernel, tileId, subtileK))
        elif tensor in ('SA', 'SB'):
            tc = 'MXSA' if tensor == 'SA' else 'MXSB'
            module.add(globalReadDoScaleSubtile(tc, self.writer, self.kernel))
        return list(module.flatitems())

    def emit_wait_gr(self, source):
        """Emit SWaitCnt for wait_gr from DepOp with wait_gr_counts."""
        counts = source.wait_gr_counts
        if counts is None:
            return []
        grCnt = (int(counts.A / self.tileInfoA.loadRatioGR) +
                 int(counts.B / self.tileInfoB.loadRatioGR) +
                 counts.SA + counts.SB)
        return [SWaitCnt(vlcnt=grCnt, vscnt=-1,
                         comment=f"Wait GR: A={counts.A} B={counts.B} SA={counts.SA} SB={counts.SB} => vlcnt={grCnt}")]

    def emit_wait_lr(self):
        return [SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1,
                         comment="Wait for LR to complete")]

    def emit_sync(self):
        return [SBarrier(comment="Barrier")]

    def emit_lr_inc(self, source):
        """Emit localReadLDSBufferSwap for a single tensor."""
        tensor = source.tensor
        tc = {'A': 'A', 'B': 'B', 'SA': 'MXSA', 'SB': 'MXSB'}.get(tensor, tensor)
        module = Module()
        module.add(localReadLDSBufferSwap(tc, self.writer, self.kernel))
        return list(module.flatitems())

    def emit_gr_inc(self, source):
        """Emit globalReadPtrUpdates + globalReadLDSBufferSwap for a single tensor."""
        tensor = source.tensor
        tc = {'A': 'A', 'B': 'B', 'SA': 'MXSA', 'SB': 'MXSB'}.get(tensor, tensor)
        module = Module()
        module.add(globalReadPtrUpdates(tc, self.writer, self.kernel))
        module.add(globalReadLDSBufferSwap(tc, self.writer, self.kernel))
        if tensor in ('SA', 'SB'):
            module.add(globalReadScalePtrUpdates(tc, self.writer, self.kernel))
        return list(module.flatitems())

    def emit_gr_scale(self, source):
        """Emit scale global reads."""
        module = Module()
        module.add(globalReadDoScaleSubtile('MXSA', self.writer, self.kernel))
        module.add(globalReadDoScaleSubtile('MXSB', self.writer, self.kernel))
        return list(module.flatitems())

    def populate(self, emitted):
        """Walk emitted partitions and fill em.instructions."""
        for partition_emitted in emitted:
            for emitted_group in partition_emitted:
                for em in emitted_group:
                    handler = self._dispatch.get(em.opType)
                    if handler:
                        em.instructions = handler(em)
