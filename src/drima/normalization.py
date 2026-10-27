"""Movement pattern normalization schemes for DRIMA.

Project Overview & Why Normalization Matters
--------------------------------------------
When two agents interact in DRIMA, their movement probabilities are adjusted independently at
each of the 9 direction fields (`E, NE, N, NW, W, SW, S, SE, X`). Because one or more fields
are increased (or decreased), the sum of probabilities in an agent's movement pattern after an
interaction will generally exceed (or fall below) `1.0`.

As explained in Section 3.2 (Figures 2, 4, and 5 of de Oliveira, 2010):
1. **Without normalization (`"none"`, Figure 4):** R-agents exposed to two D-agents moving in
   different directions rapidly accumulate probability `1.0` in *both* directions, violating the
   axioms of probability and failing to resolve toward a single direction.
2. **Standard proportional normalization (`"proportional"`, Figure 5):** Dividing each field by
   the sum flattens out directional gains. In the presence of two D-agents, R-agents stagnate
   around `0.5` probability in each D-agent's direction without ever committing to one.
3. **Iterative Equal-Displacement Normalization (`"equal_displacement"`, Figure 2 - DRIMA's core
   algorithm):**
   Instead of scaling proportionally, DRIMA displaces every *strictly positive* probability field
   (`P_d > 0`) by the *same equal amount*:
   - Compute `excess = sum(pattern) - 1.0`.
   - Count how many fields currently have `P_d > 0` (say, `k` fields).
   - Subtract `excess / k` from each field with `P_d > 0`.
   - If subtracting `excess / k` pushes any field below `0.0`, clamp that field to `0.0`.
   - Because clamping a negative value up to `0.0` re-introduces a small excess above `1.0`,
     **repeat the process iteratively** until `sum(pattern) == 1.0` and all `P_d >= 0.0`.

Common Pitfalls & What NOT To Do
--------------------------------
- **Do NOT subtract `excess / 9` from fields that are already `0.0`:** Section 3.2 explicitly
  states: *"The amount of probability to be subtracted from (or added to) every movement pattern
  field is given by the division of the excess above 1 (or the amount less than 1) by the number
  of fields with probability values larger than 0."* Including zero-valued fields in the divisor
  immediately turns them negative and wastes an iteration.
- **Do NOT run only a single pass of equal displacement:** Whenever a small positive probability
  (e.g., `0.01`) has a larger displacement (e.g., `0.03`) subtracted from it and is clamped to
  `0.0`, the remaining positive fields still sum to `1.02`. You *must* iterate until convergence.
- **Do NOT allow `X` (index 8) to become positive under Full (`"F"`) mobility:** Even when the
  pre-normalized sum is `< 1.0`, under Full mobility `X` is fixed at `0.0` and only positive
  fields (which are among indices `0..7`) are displaced.

Example Usage
-------------
```python
from src.drima.normalization import normalize_equal_displacement

# Suppose R01 (Full mobility) had 0.125 in 8 directions and gained +1/9 (~0.1111) in East (index 0)
raw_pattern = [0.125 + 1.0 / 9.0, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.0]
normalized = normalize_equal_displacement(raw_pattern, mobility="F")
assert abs(sum(normalized) - 1.0) < 1e-12
assert normalized[8] == 0.0
```
"""

from typing import List, Sequence


NORMALIZATION_TOLERANCE: float = 1e-12
MAX_NORMALIZATION_ITERATIONS: int = 100


def normalize_equal_displacement(
    pattern: Sequence[float],
    mobility: str = "F",
    tolerance: float = NORMALIZATION_TOLERANCE,
) -> List[float]:
    """Normalize a 9-element movement pattern using DRIMA's iterative equal-displacement scheme.

    Implements the algorithm in Section 3.2 of de Oliveira (2010):
    1. Enforce `pattern[8] = 0.0` if `mobility == 'F'`, and clamp any negative inputs to `0.0`.
    2. While `|sum(pattern) - 1.0| > tolerance`:
       - Compute `diff = sum(pattern) - 1.0`.
       - Find indices `i` where `pattern[i] > 0.0`.
       - Subtract `diff / len(positive_indices)` from each positive field.
       - Clamp any resulting negative values to `0.0`.

    Args:
        pattern: Sequence of 9 probability values `[P_E, ..., P_SE, X]`.
        mobility: `'F'` (Full mobility: index 8 locked to `0.0`) or `'P'` (Partial mobility).
        tolerance: Convergence threshold for `|sum(pattern) - 1.0|`.

    Returns:
        New 9-element `List[float]` with non-negative values summing to `1.0`.
    """
    if len(pattern) != 9:
        raise ValueError(f"Movement pattern must have 9 elements, got {len(pattern)}.")

    mobility_upper = mobility.upper()
    if mobility_upper not in ("F", "P"):
        raise ValueError(f"mobility must be 'F' or 'P', got '{mobility}'.")

    probs = [max(0.0, float(p)) for p in pattern]
    if mobility_upper == "F":
        probs[8] = 0.0

    # Edge case: if all probabilities collapsed to 0, reinitialize uniformly over allowed fields
    if sum(probs) <= tolerance:
        if mobility_upper == "F":
            return [0.125] * 8 + [0.0]
        return [1.0 / 9.0] * 9

    for _ in range(MAX_NORMALIZATION_ITERATIONS):
        total = sum(probs)
        diff = total - 1.0
        if abs(diff) <= tolerance:
            break

        positive_indices = [idx for idx, val in enumerate(probs) if val > 0.0]
        if not positive_indices:
            if mobility_upper == "F":
                return [0.125] * 8 + [0.0]
            return [1.0 / 9.0] * 9

        displacement = diff / len(positive_indices)
        clamped_any = False
        for idx in positive_indices:
            new_val = probs[idx] - displacement
            if new_val <= tolerance:
                probs[idx] = 0.0
                clamped_any = True
            else:
                probs[idx] = new_val

        if not clamped_any and abs(sum(probs) - 1.0) <= tolerance:
            break

    # Final micro-adjustment on the largest field to eliminate floating-point rounding drift
    final_total = sum(probs)
    if final_total > 0.0 and abs(final_total - 1.0) > 0.0:
        max_idx = max(range(9), key=lambda idx: probs[idx])
        probs[max_idx] = max(0.0, probs[max_idx] + (1.0 - final_total))

    return probs


def normalize_proportional(
    pattern: Sequence[float],
    mobility: str = "F",
) -> List[float]:
    """Normalize a 9-element movement pattern proportionally (Section 3.2, Figure 5).

    Divides each field by the total probability sum, preserving relative ratios.

    Args:
        pattern: Sequence of 9 probability values.
        mobility: `'F'` (Full) or `'P'` (Partial).

    Returns:
        New 9-element `List[float]` summing to `1.0`.
    """
    if len(pattern) != 9:
        raise ValueError(f"Movement pattern must have 9 elements, got {len(pattern)}.")

    mobility_upper = mobility.upper()
    if mobility_upper not in ("F", "P"):
        raise ValueError(f"mobility must be 'F' or 'P', got '{mobility}'.")

    probs = [max(0.0, float(p)) for p in pattern]
    if mobility_upper == "F":
        probs[8] = 0.0

    total = sum(probs)
    if total <= NORMALIZATION_TOLERANCE:
        if mobility_upper == "F":
            return [0.125] * 8 + [0.0]
        return [1.0 / 9.0] * 9

    return [p / total for p in probs]


def normalize_pattern(
    pattern: Sequence[float],
    mobility: str = "F",
    method: str = "equal_displacement",
) -> List[float]:
    """Normalize a movement pattern according to the selected `method`.

    Args:
        pattern: Sequence of 9 floats.
        mobility: `'F'` or `'P'`.
        method: One of `'equal_displacement'` (DRIMA default), `'proportional'`, or `'none'`.

    Returns:
        Normalized (or clamped, if `method == 'none'`) 9-element list of floats.
    """
    method_lower = method.lower()
    if method_lower == "equal_displacement":
        return normalize_equal_displacement(pattern, mobility=mobility)
    if method_lower == "proportional":
        return normalize_proportional(pattern, mobility=mobility)
    if method_lower == "none":
        probs = [max(0.0, min(1.0, float(p))) for p in pattern]
        if mobility.upper() == "F":
            probs[8] = 0.0
        return probs
    raise ValueError(
        f"Unknown normalization method '{method}'. Choose from "
        f"'equal_displacement', 'proportional', or 'none'."
    )
