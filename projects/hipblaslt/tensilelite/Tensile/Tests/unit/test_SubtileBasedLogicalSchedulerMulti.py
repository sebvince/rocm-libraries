"""Multi-codepath (multi-wave) tests for LogicalScheduler.

Mirrors test_SubtileBasedLogicalScheduler.py's standalone driver but for the
new multi-wave path gated for gfx1250:
  Pass 0 — Place MFMAs (per-wave, global tile ids; ok for a wave to have 0)
  Pass 1 — Place LR    (per-wave, MFMA+LR using global ids; only for waves
                        that have MFMAs)

Example usage:
  PYTHONPATH=. python Tensile/Tests/unit/test_SubtileBasedLogicalSchedulerMulti.py
  PYTHONPATH=. python Tensile/Tests/unit/test_SubtileBasedLogicalSchedulerMulti.py --waves 4 --wg 2x2
"""
from Tensile.Components.Subtile.LogicalScheduler import (
    LogicalScheduler,
    ReadGranularity,
    SchedulerConfig,
)


def make_cfg_bf16_256x256_multi(numWaves=4, waveGroup=(2, 2)):
    """BF16 256x256x64 multi-wave config.

    With miWaveGroup=(2,2) on a 256x256 macrotile, each of the 4 waves owns
    one 128x128 quadrant of C → 4x4 MFMA tiles per wave in M and N.
    The global MFMA tile grid is 8x8, with subIterK=2.
    """
    return SchedulerConfig(
        # Per-wave tile counts (4x4 MFMA tiles per wave).
        numMFMATilesM=4,
        numMFMATilesN=4,
        numSubIterK=2,
        lrA=ReadGranularity(mn=1, k=1),
        lrB=ReadGranularity(mn=1, k=1),
        # Split-by-wave GR (gfx1250): mn=4 → one atom per A-loader wave / B-loader wave.
        grA=ReadGranularity(mn=4, k=2),
        grB=ReadGranularity(mn=4, k=2),
        splitGRByWave=True,
        numWaves=numWaves,
        waveGroup=waveGroup,
    )


# Tool to visualize the multi-codepath scheduling steps on a real kernel configuration.
# Runs only the first two passes (Place MFMAs + Place LR) for the gfx1250
# multi-wave path. Example:
#   PYTHONPATH=. python Tensile/Tests/unit/test_SubtileBasedLogicalSchedulerMulti.py --waves 4 --wg 2x2
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Visualize multi-wave SubtileBased LogicalScheduler "
                    "(passes 0 + 1 only).",
    )
    parser.add_argument("--mt0", type=int, default=256, help="MacroTile0 (default: 256)")
    parser.add_argument("--mt1", type=int, default=256, help="MacroTile1 (default: 256)")
    parser.add_argument("--du", type=int, default=64, help="DepthU (default: 64)")
    parser.add_argument("--waves", type=int, default=4, help="Number of waves (default: 4)")
    parser.add_argument("--wg", type=str, default="2x2",
                        help="waveGroup as MxN (default: 2x2)")
    parser.add_argument("--interactive", "-i", action="store_true",
                        help="Step through each phase interactively")
    args = parser.parse_args()

    wg_parts = args.wg.lower().split("x")
    if len(wg_parts) != 2:
        parser.error(f"--wg must be MxN (e.g. 2x2), got: {args.wg}")
    waveGroup = (int(wg_parts[0]), int(wg_parts[1]))

    if args.mt0 != 256 or args.mt1 != 256 or args.du != 64:
        parser.error("standalone driver currently fixed to BF16 256x256x64")

    cfg = make_cfg_bf16_256x256_multi(numWaves=args.waves, waveGroup=waveGroup)

    print(f"Config: MT={args.mt0}x{args.mt1}, DU={args.du}, dtype=bf16, "
          f"numWaves={args.waves}, waveGroup={waveGroup[0]}x{waveGroup[1]}")
    print(f"        per-wave numMFMATilesM={cfg.numMFMATilesM}, "
          f"numMFMATilesN={cfg.numMFMATilesN}, "
          f"numSubIterK={cfg.numSubIterK}")
    print()

    sched = LogicalScheduler(cfg)

    COL_W = 40  # width of each wave column body

    def _wave_header(w):
        wave_slots = sched._wave_partitions[w]
        a_ranges, b_ranges = [], []
        for slots in wave_slots:
            for slot in slots:
                if slot.mfma:
                    a_ranges.append(slot.mfma.tileA.fmt_tiles())
                    b_ranges.append(slot.mfma.tileB.fmt_tiles())
        a = a_ranges[0] if a_ranges else "-"
        b = b_ranges[0] if b_ranges else "-"
        return f"Wave {w} : A{a}  B{b}"

    def _fmt_dep(dep):
        p = dep.ref
        kind = type(p).__name__[:-len("Placement")]  # MFMA / LR / GR
        mt = f" MT{dep.mt_offset:+d}" if dep.mt_offset != 0 else ""
        wave = getattr(p, 'wave', 0)
        return f"{kind} {p.tensor} @W{wave}P{p.partition}K{p.subIterK_slot}{mt}"

    def _fmt_preops(placement):
        return [f"   ↳ pre: {op}" for op in placement.preOps]

    def _fmt_postops(placement):
        return [f"   ↱ post: {op}" for op in placement.postOps]

    def _cell_lines_for_slot(wave_slots, pi, slot_idx, show_vgpr=False,
                              show_deps=False, show_preops=False):
        """Return list of strings (one per visual line) for one wave's slot."""
        slot = wave_slots[pi][slot_idx]
        lines = []
        if slot.mfma:
            m = slot.mfma
            extra = ""
            if show_vgpr and m.vgpr_tile_maps:
                parts = []
                for tensor in ('A', 'B', 'SA', 'SB'):
                    maps = m.vgpr_tile_maps.get(tensor)
                    if maps:
                        parts.append(f"{tensor}:" + str(maps[0]))
                if parts:
                    extra = " " + ", ".join(parts)
            lines.append(f"MFMA (MT n, subK[{m.subIterK}]) "
                         f"A{m.tileA.fmt_tiles()} B{m.tileB.fmt_tiles()}{extra}")
            if show_preops:
                lines.extend(_fmt_preops(m))
            if show_preops:
                lines.extend(_fmt_postops(m))
            if show_deps:
                for dep in m.deps:
                    lines.append(f"   ← {_fmt_dep(dep)}")
        for lr in slot.lrs:
            mt = f"n+{lr.mtIteration}" if lr.mtIteration else "n"
            extra = ""
            if show_vgpr and lr.vgpr_tile_map:
                extra = f" tiles:{lr.vgpr_tile_map[0]}"
            lines.append(f"LR {lr.tensor:<2} (MT {mt}, subK{lr.tiles.fmt_k()}) "
                         f"{lr.tiles.fmt_tiles()}{extra}")
            if show_preops:
                lines.extend(_fmt_preops(lr))
                lines.extend(_fmt_postops(lr))
            if show_deps:
                for dep in lr.deps:
                    lines.append(f"   ← {_fmt_dep(dep)}")
        for gr in slot.grs:
            mt = f"n+{gr.mtIteration}" if gr.mtIteration else "n"
            lines.append(f"GR {gr.tensor:<2} (MT {mt}, subK{gr.tiles.fmt_k()}) "
                         f"{gr.tiles.fmt_tiles()}")
            if show_preops:
                lines.extend(_fmt_preops(gr))
                lines.extend(_fmt_postops(gr))
            if show_deps:
                for dep in gr.deps:
                    lines.append(f"   ← {_fmt_dep(dep)}")
        if not lines:
            lines.append("·")
        return lines

    LABEL_W = 11  # width of left label column ("subIterK=N")

    def _print_waves(title, show_vgpr=False, show_deps=False, show_preops=False):
        sep_total = 2 + LABEL_W + 2 + (COL_W + 3) * cfg.numWaves
        print("=" * sep_total)
        print(f"  {title}")
        print("=" * sep_total)
        print()

        # Header row
        header_cells = [_wave_header(w).ljust(COL_W) for w in range(cfg.numWaves)]
        print("  " + " " * LABEL_W + "│ " + " │ ".join(header_cells))
        sep = "─" * COL_W
        print("  " + "─" * LABEL_W + "┼─" + "─┼─".join(sep for _ in range(cfg.numWaves)))

        numP = cfg.numPartitions
        numK = cfg.numSubIterK
        for pi in range(numP):
            banner = f" Partition {pi} "
            pad = sep_total - 4 - len(banner)
            print(f"  ──{banner}" + "─" * max(0, pad))
            for k in range(numK):
                per_wave = [_cell_lines_for_slot(sched._wave_partitions[w], pi, k,
                                                 show_vgpr=show_vgpr,
                                                 show_deps=show_deps,
                                                 show_preops=show_preops)
                            for w in range(cfg.numWaves)]
                max_lines = max(len(c) for c in per_wave)
                for line_idx in range(max_lines):
                    cells = []
                    for w in range(cfg.numWaves):
                        cell = per_wave[w][line_idx] if line_idx < len(per_wave[w]) else ""
                        cells.append(cell.ljust(COL_W))
                    label = f"subIterK={k}".ljust(LABEL_W) if line_idx == 0 else " " * LABEL_W
                    print(f"  {label}│ " + " │ ".join(cells))
        print()

    def _print_waves_stacked(title):
        """One section per wave (vertical layout); fits wide vgpr tile maps."""
        sep = "=" * 100
        print(sep)
        print(f"  {title}")
        print(sep)
        peaks = getattr(sched, 'tile_peaks', {})
        uf = getattr(sched, 'unroll_factor', 1)
        print(f"  unrollFactor={uf}, tilePeaks="
              + ", ".join(f"{t}:{n}" for t, n in sorted(peaks.items())))
        for w in range(cfg.numWaves):
            wave_slots = sched._wave_partitions[w]
            print()
            print(f"  ── Wave {w} {_wave_header(w)[7:]} " + "─" * 40)
            for pi, slots in enumerate(wave_slots):
                if cfg.numPartitions > 1:
                    print(f"    Partition {pi}:")
                for slot in slots:
                    print(f"    subIterK={slot.subIterK}:")
                    for line in _cell_lines_for_slot(wave_slots, pi, slot.subIterK,
                                                     show_vgpr=True):
                        print(f"      {line}")
        print()

    steps = [
        ("Place MFMAs",        lambda: (sched.place_MFMAs(),
                                        _print_waves("Place MFMAs (global tile ids)"))),
        ("Place LRs",          lambda: (sched.place_LRs(),
                                        _print_waves("Place LRs (per-wave)"))),
        ("Assign VGPR tiles",  lambda: (sched.assign_vgpr_tiles(),
                                        _print_waves_stacked("Assign VGPR tiles (per-wave)"))),
        ("Place GRs",          lambda: (sched.place_GRs(),
                                        _print_waves("Place GRs (per-wave)"))),
        ("Annotate deps",      lambda: (sched.annotate_deps(),
                                        _print_waves("Annotate deps (per-wave)",
                                                     show_deps=True))),
        ("Remove unnecessary GR deps",
                               lambda: (sched.remove_unnecessary_gr_deps(),
                                        _print_waves("Remove unnecessary GR deps (per-wave)",
                                                     show_deps=True))),
        ("Remove unnecessary LR deps",
                               lambda: (sched.remove_unnecessary_lr_deps(),
                                        _print_waves("Remove unnecessary LR deps (per-wave)",
                                                     show_deps=True))),
        ("Remove cross deps",  lambda: (sched.remove_cross_deps(),
                                        _print_waves("Remove cross deps (per-wave)",
                                                     show_deps=True,
                                                     show_preops=True))),
        ("Insert gr/lr inc",   lambda: (sched.insert_gr_lr_inc(),
                                        _print_waves("Insert gr/lr inc (per-wave)",
                                                     show_preops=True))),
        ("Group LR/GR",        lambda: (sched.group_lr_gr(),
                                        _print_waves("Group LR/GR (per-wave)",
                                                     show_deps=True,
                                                     show_preops=True))),
    ]

    for i, (title, run) in enumerate(steps):
        run()
        if args.interactive and i < len(steps) - 1:
            input("Press Enter for next step...")
