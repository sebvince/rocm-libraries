################################################################################
#
# Copyright (C) 2024-2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
################################################################################

import pytest
from Tensile.Components.InterleavingScheduler import (
    schedule_interleaving,
    SlotAssignment,
    InterleavingResult,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def cost_len(instructions):
    """Cost = number of instructions in the slot."""
    return len(instructions)


def cost_quadratic(instructions):
    """Cost = len^2. Encourages even distribution across slots."""
    return len(instructions) ** 2


def cost_zero(instructions):
    """Zero cost — only gap matters."""
    return 0


def is_upper(s):
    """Target predicate: string starts with uppercase."""
    return s[0].isupper() if s else False


def _extract_order(result, instruction_lists):
    """Verify that instructions from each list appear in their original order."""
    flat = result.all_instructions()
    for lst in instruction_lists:
        indices = []
        for instr in lst:
            idx = None
            start = indices[-1] + 1 if indices else 0
            for j in range(start, len(flat)):
                if flat[j] is instr:
                    idx = j
                    break
            assert idx is not None, f"Instruction {instr!r} not found in output"
            indices.append(idx)
        # Indices must be strictly increasing.
        for a, b in zip(indices, indices[1:]):
            assert a < b, f"Order violated for list {lst}"


def _compute_actual_min_gap(result, is_target):
    """Compute the actual minimum gap between target instructions in the result."""
    last_target_slot = None
    min_gap = None
    for slot in result.slots:
        has_target = any(is_target(i) for i in slot.instructions)
        if has_target:
            if last_target_slot is not None:
                gap = slot.slot_index - last_target_slot
                if min_gap is None or gap < min_gap:
                    min_gap = gap
            last_target_slot = slot.slot_index
    return min_gap


# ---------------------------------------------------------------------------
# A: Basic functionality
# ---------------------------------------------------------------------------

class TestBasic:
    def test_single_list_no_targets(self):
        # 4 MFMAs -> 3 slots, 3 lowercase instructions, no targets.
        lists = [["a", "b", "c"]]
        result = schedule_interleaving(4, lists, cost_len, is_upper)
        assert result.total_cost == 3  # 3 instructions total, cost = len
        assert len(result.slots) == 3
        assert result.all_instructions() == ["a", "b", "c"]
        # No targets -> min_gap is vacuously maximal.
        assert result.min_gap == 3

    def test_single_list_all_in_one_slot(self):
        # 2 MFMAs -> 1 slot, everything goes there.
        lists = [["a", "b", "c"]]
        result = schedule_interleaving(2, lists, cost_len, is_upper)
        assert len(result.slots) == 1
        assert result.slots[0].instructions == ["a", "b", "c"]
        assert result.total_cost == 3

    def test_two_lists_no_targets(self):
        lists = [["a", "b"], ["c", "d"]]
        result = schedule_interleaving(4, lists, cost_len, is_upper)
        assert len(result.all_instructions()) == 4
        _extract_order(result, lists)

    def test_empty_lists(self):
        result = schedule_interleaving(4, [], cost_len, is_upper)
        assert result.total_cost == 0
        assert result.min_gap == 3
        assert len(result.slots) == 3

    def test_empty_lists_filtered(self):
        # Empty lists should be filtered out.
        result = schedule_interleaving(4, [[], ["a"]], cost_len, is_upper)
        assert "a" in result.all_instructions()

    def test_single_instruction_single_slot(self):
        lists = [["x"]]
        result = schedule_interleaving(2, lists, cost_len, is_upper)
        assert result.slots[0].instructions == ["x"]
        assert result.total_cost == 1


# ---------------------------------------------------------------------------
# B: Gap optimization
# ---------------------------------------------------------------------------

class TestGap:
    def test_two_targets_maximize_gap(self):
        # 6 MFMAs -> 5 slots. Two target instructions among fillers.
        # Targets should be spread as far apart as possible.
        lists = [["A", "a", "b", "c", "B"]]
        result = schedule_interleaving(6, lists, cost_zero, is_upper)
        assert result.min_gap >= 1
        actual = _compute_actual_min_gap(result, is_upper)
        assert actual == result.min_gap
        # With 5 slots and 2 targets, max possible gap is 4 (slot 0 and slot 4).
        assert result.min_gap == 4

    def test_three_targets_even_spread(self):
        # 7 MFMAs -> 6 slots. Three targets A, B, C among fillers.
        # Best spread: slots 0, 3, 6 would be gap=3, but slot 6 doesn't exist
        # (only 0-5). So slots 0, 2, 5 -> gap=2,3 -> min=2.
        # Or slots 0, 3, 5 -> gap 3,2 -> min=2. Max achievable min_gap = 2.
        lists = [["A", "a", "B", "b", "C"]]
        result = schedule_interleaving(7, lists, cost_zero, is_upper)
        assert result.min_gap >= 2
        actual = _compute_actual_min_gap(result, is_upper)
        assert actual == result.min_gap

    def test_single_target_vacuous_gap(self):
        # Only one target -> no pair -> gap is vacuously maximal.
        lists = [["A", "a", "b"]]
        result = schedule_interleaving(4, lists, cost_zero, is_upper)
        # With a single target, there are no consecutive pairs, so the reported
        # min_gap should be the max feasible G from binary search.
        # The DP with any G is feasible as long as there's only one target.
        assert result.min_gap >= 3

    def test_gap_constrained_by_order(self):
        # Two targets adjacent in the list — order forces them close.
        # 4 MFMAs -> 3 slots. List: [A, B, a].
        # A and B must appear in order. Best: A in slot 0, B in slot 1 -> gap=1.
        lists = [["A", "B", "a"]]
        result = schedule_interleaving(4, lists, cost_zero, is_upper)
        assert result.min_gap >= 1
        actual = _compute_actual_min_gap(result, is_upper)
        assert actual == result.min_gap

    def test_targets_in_different_lists(self):
        # Targets in separate lists can be placed independently.
        # 6 MFMAs -> 5 slots. List1: [A, a], List2: [B, b].
        # A and B are independent — can spread across slots.
        lists = [["A", "a"], ["B", "b"]]
        result = schedule_interleaving(6, lists, cost_zero, is_upper)
        assert result.min_gap >= 1
        actual = _compute_actual_min_gap(result, is_upper)
        assert actual == result.min_gap


# ---------------------------------------------------------------------------
# C: Cost minimization
# ---------------------------------------------------------------------------

class TestCost:
    def test_quadratic_cost_encourages_spread(self):
        # With quadratic cost, instructions should be spread evenly.
        # 4 MFMAs -> 3 slots, 3 instructions.
        lists = [["a", "b", "c"]]
        result = schedule_interleaving(4, lists, cost_quadratic, is_upper)
        # Optimal: 1 per slot -> cost = 1+1+1 = 3.
        assert result.total_cost == 3
        for slot in result.slots:
            assert len(slot.instructions) == 1

    def test_cost_minimized_at_fixed_gap(self):
        # Two placements with same gap but different costs.
        # 4 MFMAs -> 3 slots. List: [A, a, a, B].
        # Gap constraint forces A and B apart. With quadratic cost,
        # the filler 'a' instructions should be spread.
        lists = [["A", "a", "a", "B"]]
        result = schedule_interleaving(4, lists, cost_quadratic, is_upper)
        _extract_order(result, lists)
        # The gap should be maximized (A in slot 0, B in slot 2 -> gap=2).
        assert result.min_gap == 2

    def test_cost_fn_receives_correct_contents(self):
        # Verify cost_fn is called with correct instruction lists.
        calls = []
        def tracking_cost(instructions):
            calls.append(list(instructions))
            return len(instructions)
        lists = [["a", "b"]]
        result = schedule_interleaving(3, lists, tracking_cost, is_upper)
        # All instructions should appear in exactly one call.
        all_seen = []
        for call in calls:
            all_seen.extend(call)
        # The final solution's slots should match some subset of calls.
        assert "a" in all_seen and "b" in all_seen


# ---------------------------------------------------------------------------
# D: Order preservation
# ---------------------------------------------------------------------------

class TestOrder:
    def test_single_list_order(self):
        lists = [["a", "b", "c", "d", "e"]]
        result = schedule_interleaving(4, lists, cost_len, is_upper)
        _extract_order(result, lists)

    def test_two_lists_order(self):
        lists = [["a", "b", "c"], ["d", "e", "f"]]
        result = schedule_interleaving(5, lists, cost_len, is_upper)
        _extract_order(result, lists)

    def test_within_slot_list_order(self):
        # Within a slot, list 0 instructions come before list 1.
        # Force everything into one slot.
        lists = [["a"], ["b"]]
        result = schedule_interleaving(2, lists, cost_len, is_upper)
        assert result.slots[0].instructions == ["a", "b"]


# ---------------------------------------------------------------------------
# E: Edge cases
# ---------------------------------------------------------------------------

class TestEdge:
    def test_many_slots_few_instructions(self):
        # 10 MFMAs -> 9 slots, but only 2 instructions.
        lists = [["a", "b"]]
        result = schedule_interleaving(10, lists, cost_len, is_upper)
        total_instr = sum(len(s.instructions) for s in result.slots)
        assert total_instr == 2
        assert len(result.slots) == 9

    def test_four_lists(self):
        lists = [["a"], ["b"], ["c"], ["d"]]
        result = schedule_interleaving(3, lists, cost_len, is_upper)
        assert sorted(result.all_instructions()) == ["a", "b", "c", "d"]
        _extract_order(result, lists)

    def test_all_targets(self):
        # Every instruction is a target.
        # 6 MFMAs -> 5 slots. 3 targets -> max gap = 2.
        lists = [["A", "B", "C"]]
        result = schedule_interleaving(6, lists, cost_zero, is_upper)
        assert result.min_gap >= 2
        actual = _compute_actual_min_gap(result, is_upper)
        assert actual == result.min_gap

    def test_deterministic(self):
        lists = [["A", "a", "B", "b"]]
        r1 = schedule_interleaving(5, lists, cost_len, is_upper)
        r2 = schedule_interleaving(5, lists, cost_len, is_upper)
        assert r1.total_cost == r2.total_cost
        assert r1.min_gap == r2.min_gap
        assert r1.all_instructions() == r2.all_instructions()


# ---------------------------------------------------------------------------
# F: Validation
# ---------------------------------------------------------------------------

class TestValidation:
    def test_num_mfma_too_small(self):
        with pytest.raises(ValueError, match="num_mfma must be >= 2"):
            schedule_interleaving(1, [["a"]], cost_len, is_upper)

    def test_too_many_lists(self):
        lists = [["a"], ["b"], ["c"], ["d"], ["e"]]
        with pytest.raises(ValueError, match="At most 4"):
            schedule_interleaving(3, lists, cost_len, is_upper)
