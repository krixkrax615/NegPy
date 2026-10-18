from __future__ import annotations

from typing import List


#: How many frames on each side of the open one are decoded ahead, nearest first.
NEIGHBOR_PREFETCH_RADIUS = 3


def display_neighbor_indices(display_order: List[int], current_index: int, radius: int = 1) -> List[int]:
    """Actual indices of the visible assets within `radius` of the current one, nearest first, previous before next."""
    try:
        position = display_order.index(current_index)
    except ValueError:
        return []
    neighbors: List[int] = []
    for distance in range(1, max(1, radius) + 1):
        if position - distance >= 0:
            neighbors.append(display_order[position - distance])
        if position + distance < len(display_order):
            neighbors.append(display_order[position + distance])
    return neighbors


def neighbor_assets(files: List[dict], display_order: List[int], current_index: int, radius: int = 1) -> List[dict]:
    """The assets within `radius` of the open one in filmstrip order."""
    return [files[index] for index in display_neighbor_indices(display_order, current_index, radius)]
