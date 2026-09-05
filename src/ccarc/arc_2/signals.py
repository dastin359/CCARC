"""Mechanical generalization signals — a partial stand-in for the reviewer.

There is no independent reviewer agent here — nothing that can reject a
solution passing training 100% on generalization grounds alone. That leaves a
hole exactly where the hard tasks live: a rule that reproduces every training
pair and is still wrong.

A model-based reviewer is out of scope here. What is in scope is the cheap half
of what a reviewer does — noticing that a test prediction breaks a regularity
every training output obeys. That needs no model, no second context, and no
tokens beyond the finding itself.

Design rule: **only report a signal when every training output agrees and the
prediction disagrees.** A false positive here is expensive — it spends the
solver's attention and teaches it to discount the channel — so each check
abstains unless the training set is unanimous.
"""

from __future__ import annotations

from typing import Any

Grid = list[list[int]]


def _shape(grid: Grid) -> tuple[int, int]:
    return (len(grid), len(grid[0]) if grid else 0)


def _colours(grid: Grid) -> set[int]:
    return {int(cell) for row in grid for cell in row}


def _unanimous(values: list[Any]) -> Any | None:
    """The single shared value, or None when the training set does not agree."""
    if not values:
        return None
    first = values[0]
    return first if all(value == first for value in values[1:]) else None


def output_space_signals(
    train_pairs: list[dict[str, Any]],
    test_inputs: list[Grid],
    test_candidates: list[list[Grid]],
) -> list[str]:
    """Regularities every training output obeys that a test prediction breaks.

    ``test_candidates[i]`` is the list of candidate grids for ``test_inputs[i]``.
    Returns human-readable findings; an empty list means nothing stood out.
    """
    outputs = [pair["output"] for pair in train_pairs if pair.get("output")]
    inputs = [pair["input"] for pair in train_pairs if pair.get("input")]
    if not outputs or len(outputs) != len(inputs):
        return []

    findings: list[str] = []

    fixed_shape = _unanimous([_shape(o) for o in outputs])
    shape_equals_input = all(_shape(o) == _shape(i) for o, i in zip(outputs, inputs))
    all_square = all(_shape(o)[0] == _shape(o)[1] for o in outputs)
    # Only a palette that is *identical* across every training output says
    # anything about the test output. The union across outputs does not: plenty
    # of rules recolour from the input, and a test input legitimately carries
    # colours no training pair contained.
    fixed_palette = _unanimous([frozenset(_colours(o)) for o in outputs])
    palette_subset_of_input = all(_colours(o) <= _colours(i) for o, i in zip(outputs, inputs))

    for index, candidates in enumerate(test_candidates):
        test_input = test_inputs[index] if index < len(test_inputs) else None
        for position, candidate in enumerate(candidates, start=1):
            if not candidate:
                continue
            label = f"test {index}" + (f" candidate {position}" if len(candidates) > 1 else "")
            shape = _shape(candidate)

            if fixed_shape is not None and shape != fixed_shape:
                findings.append(
                    f"{label} is {shape[0]}x{shape[1]}, but every training output is "
                    f"{fixed_shape[0]}x{fixed_shape[1]}"
                )
            elif shape_equals_input and test_input is not None and shape != _shape(test_input):
                findings.append(
                    f"{label} is {shape[0]}x{shape[1]} from a "
                    f"{_shape(test_input)[0]}x{_shape(test_input)[1]} input, but every training "
                    "output has its input's shape"
                )
            elif all_square and shape[0] != shape[1]:
                findings.append(
                    f"{label} is {shape[0]}x{shape[1]}, but every training output is square"
                )

            if fixed_palette is not None:
                novel = _colours(candidate) - set(fixed_palette)
                if novel:
                    findings.append(
                        f"{label} uses colour(s) {sorted(novel)}, but every training output uses "
                        f"exactly {sorted(fixed_palette)}"
                    )
            if palette_subset_of_input and test_input is not None:
                invented = _colours(candidate) - _colours(test_input)
                if invented:
                    findings.append(
                        f"{label} introduces colour(s) {sorted(invented)} not present in its own "
                        "input, but every training output draws only on its input's palette"
                    )

    # Identical candidates are a wasted second attempt: ARC-AGI-2 scores two
    # predictions per test example, and duplicates throw one away.
    for index, candidates in enumerate(test_candidates):
        if len(candidates) > 1 and all(c == candidates[0] for c in candidates[1:]):
            findings.append(
                f"test {index} emitted identical candidates — the second attempt is wasted"
            )

    return findings
