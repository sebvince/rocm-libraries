# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Cluster-scope barrier handshake for the subtile mainloop.

The handshake is split into a *signal* half and a *wait* half so the wait can be
moved away from the signal, hiding the cluster barrier's cross-CU latency behind
the WMMAs that issue in between instead of exposing it as a stall.
"""

from __future__ import annotations

from rocisa.code import Label, Module
from rocisa.container import sgpr
from rocisa.instruction import (SBarrier, SCBranchSCC0, SCmpEQU32,
                                MFMAInstruction, MXMFMAInstruction)

_isWgBarrier = lambda x: isinstance(x, SBarrier) and "s_barrier_wait -1" in str(x)

# Number of WMMAs to let issue after the signal before closing the handshake
# with the wait. The wait is placed after this many WMMAs so the cluster
# barrier's cross-CU latency hides behind them instead of exposing a stall.
_WMMA_SIGNAL_TO_WAIT = 8


def _findNextMFMA(items, start):
    """Index of the first MFMA at/after ``start``, or ``None`` if none follows."""
    for j in range(start, len(items)):
        if isinstance(items[j], (MFMAInstruction, MXMFMAInstruction)):
            return j
    return None


def _findNthMFMAEnd(items, start, n):
    """Index just after the ``n``-th MFMA at/after ``start``.

    Returns ``None`` if fewer than ``n`` MFMAs follow ``start``.
    """
    seen = 0
    for j in range(start, len(items)):
        if isinstance(items[j], (MFMAInstruction, MXMFMAInstruction)):
            seen += 1
            if seen == n:
                return j + 1
    return None


def subtileClusterBarrierSignal(writer, kernel) -> Module:
    """Wave-0-only cluster_barrier signal.

    Wave 0 alone issues the cluster_barrier signal; all other waves branch over
    it. Ends at the ``skipPreSignal`` label so all waves fall through to whatever
    work follows; the matching wait is emitted later by ``subtileClusterBarrierWait``.
    """
    mod = Module("subtile_cluster_barrier_signal")
    skipPreSignal = Label(writer.labels.getUniqueNamePrefix("skipCBPreSignal"), "", 16)
    # Elect wave 0 to issue the single cluster_barrier signal.
    mod.add(SCmpEQU32(sgpr("WaveIdx"), 0, "wave 0?"))
    mod.add(SCBranchSCC0(skipPreSignal.getLabelName(), "only wave 0 signals the cluster"))
    mod.add(SBarrier(True, False, True, "cluster_barrier signal"))
    mod.add(skipPreSignal)
    return mod


def subtileClusterBarrierWait(writer, kernel) -> Module:
    """The all-waves cluster_barrier wait that closes the handshake."""
    mod = Module("subtile_cluster_barrier_wait")
    mod.add(SBarrier(True, True, True, "cluster_barrier wait"))
    return mod


def insertClusterBarrier(module, writer, kernel):
    """Splice the cluster-scope barrier handshake into the post-schedule order.

    No-op unless ``ClusterBarrier`` is enabled. The signal is spliced in right
    after the mainloop's existing workgroup barrier (reusing that sync instead of
    emitting a second one); the wait is placed ``_WMMA_SIGNAL_TO_WAIT`` WMMAs
    after the signal, so the barrier's cross-CU latency overlaps just those WMMAs
    before the handshake is closed. If fewer than that many WMMAs follow the
    signal, the wait falls back to the end of the section.

    If no workgroup barrier is found in this section, the signal is prepended at
    the start so the handshake is still opened (correctness over reuse).

    Returns a rebuilt Module; the input is left untouched.
    """
    if not kernel.get("ClusterBarrier"):
        return module

    signalItems = subtileClusterBarrierSignal(writer, kernel).flatitems()
    waitItems = subtileClusterBarrierWait(writer, kernel).flatitems()

    # ClusterBarrier is only supported on gfx1250.
    assert writer.states.asmCaps.get("HasClusterBarrier", False), \
        "ClusterBarrier requires the HasClusterBarrier asm capability"

    # Place the wave-0-election branch right after a WMMA to hide branching
    # latency: keep s_cmp before the next scheduled MFMA and emit the branch
    # after it.

    items = module.flatitems()
    result = Module(module.name)
    done = False
    skip = set()
    for i, inst in enumerate(items):
        if i in skip:
            continue
        result.add(inst)
        if not done and _isWgBarrier(inst):
            done = True
            mfmaIdx = _findNextMFMA(items, i + 1)
            if mfmaIdx is None:
                # No following MFMA to pin the branch to: emit the block intact
                # (best-effort).
                for s in signalItems:
                    result.add(s)
            else:
                # Split the signal block at the wave-0 election branch. The
                # block is authored with exactly one conditional branch; assert
                # it so a future change that adds another fails loudly here.
                brIdxs = [k for k, s in enumerate(signalItems)
                          if isinstance(s, SCBranchSCC0)]
                assert len(brIdxs) == 1, \
                    "signal block must contain exactly one wave-0 election branch"
                brIdx = brIdxs[0]
                pre, post = signalItems[:brIdx], signalItems[brIdx:]
                # Everything up to the MFMA (incl. its s_set_vgpr_msb primer)
                # keeps its order, then s_cmp, the MFMA, and the branch. SCC
                # survives the MFMA and vgpr-msb is a persistent mode, so the
                # intervening compare disturbs neither.
                for k in range(i + 1, mfmaIdx):
                    result.add(items[k])
                    skip.add(k)
                for s in pre:
                    result.add(s)
                result.add(items[mfmaIdx])
                skip.add(mfmaIdx)
                for s in post:
                    result.add(s)
    if not done:  # no workgroup barrier: open the handshake at the start
        head = Module(module.name)
        for s in signalItems:
            head.add(s)
        for inst in result.flatitems():
            head.add(inst)
        result = head

    # Wait: close the handshake _WMMA_SIGNAL_TO_WAIT WMMAs after the signal so
    # the cluster barrier's cross-CU latency hides behind exactly those WMMAs.
    # The signal block ends at its last item (the skipPreSignal label); find it
    # by identity and count WMMAs from just after it.
    finalItems = result.flatitems()
    signalMarker = signalItems[-1]
    signalEnd = next((i for i, inst in enumerate(finalItems)
                      if inst is signalMarker), None)
    insertAt = (_findNthMFMAEnd(finalItems, signalEnd + 1, _WMMA_SIGNAL_TO_WAIT)
                if signalEnd is not None else None)

    if insertAt is None:
        # Fewer than _WMMA_SIGNAL_TO_WAIT WMMAs after the signal (or no marker):
        # fall back to closing the handshake at the end of the section.
        for w in waitItems:
            result.add(w)
        return result

    rebuilt = Module(module.name)
    for i, inst in enumerate(finalItems):
        rebuilt.add(inst)
        if i + 1 == insertAt:
            for w in waitItems:
                rebuilt.add(w)
    return rebuilt
