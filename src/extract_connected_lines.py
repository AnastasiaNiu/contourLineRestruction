# -*- coding: utf-8 -*-
"""
Connectivity analysis on a merged skeleton.

Reads a ``merged_skeleton_data.json`` (output of
:mod:`merge_skeleton_and_connections`), groups pixels into connected
components via 8-connectivity, then traces each component into an ordered
polyline. Each output line additionally tags both endpoints with an
``is_boundary`` flag (true when the endpoint lies within ``margin`` pixels of
the image border); the SA step later uses those flags to decide which
endpoints to match.
"""

import json
import os
import cv2
import numpy as np
from collections import deque


def load_skeleton_data(json_path):
    """
    Load the merged skeleton JSON.

    Parameters
    ----------
    json_path : str
        Path to ``merged_skeleton_data.json``.

    Returns
    -------
    dict
        Parsed JSON payload.
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def find_connected_components(pixels, width, height):
    """
    Group ``pixels`` into 8-connected components.

    Parameters
    ----------
    pixels : iterable of tuple
        ``(x, y)`` integer pixel coordinates.
    width, height : int
        Image width and height (used only for the BFS bounds).

    Returns
    -------
    list of set
        One ``set`` per component, each containing ``(x, y)`` tuples.
    """
    pixel_set = set(pixels)
    visited = set()
    components = []

    directions = [(-1, -1), (-1, 0), (-1, 1),
                  (0, -1),          (0, 1),
                  (1, -1),  (1, 0),  (1, 1)]

    for p in pixel_set:
        if p in visited:
            continue

        component = set()
        queue = deque([p])

        while queue:
            x, y = queue.popleft()
            if (x, y) in visited or (x, y) not in pixel_set:
                continue

            visited.add((x, y))
            component.add((x, y))

            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < width and 0 <= ny < height:
                    if (nx, ny) in pixel_set and (nx, ny) not in visited:
                        queue.append((nx, ny))

        if component:
            components.append(component)

    return components


def get_endpoints(component):
    """
    Return every pixel in ``component`` with exactly one 8-connected neighbour.

    Parameters
    ----------
    component : set of tuple
        ``(x, y)`` pixels forming the component.

    Returns
    -------
    list of tuple
        ``(x, y)`` of every endpoint pixel.
    """
    endpoints = []
    for px, py in component:
        neighbors = 0
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                if (px + dx, py + dy) in component:
                    neighbors += 1
        if neighbors == 1:
            endpoints.append((px, py))
    return endpoints


def trace_line_order(start, component):
    """
    Trace ``component`` into an ordered polyline starting at ``start``.

    Algorithm
    ---------
    1. BFS from ``start`` to the farthest reachable pixel; the path is the
       "spine" of the component (both endpoints are true endpoints, i.e. have
       degree 1).
    2. For every spine pixel, recursively trace the unvisited neighbours and
       splice the resulting branch right after the spine node.

    The output therefore contains every pixel in the component, ordered so
    that intermediate branching points have their branches appended in the
    right place.

    Parameters
    ----------
    start : tuple
        ``(x, y)`` of the first endpoint of the component.
    component : set or iterable of tuple
        Pixels forming the component.

    Returns
    -------
    list of tuple
        The traced polyline as ``(x, y)`` coordinates.
    """
    from collections import deque

    if start not in component:
        return []

    comp_set = component if isinstance(component, set) else set(component)

    def neighbors(p):
        x, y = p
        result = []
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                np_ = (x + dx, y + dy)
                if np_ in comp_set:
                    result.append(np_)
        return result

    # BFS to the farthest pixel; rebuild the spine from parent pointers.
    parent = {start: None}
    depth = {start: 0}
    queue = deque([start])
    far = start
    while queue:
        cur = queue.popleft()
        for n in neighbors(cur):
            if n not in parent:
                parent[n] = cur
                depth[n] = depth[cur] + 1
                queue.append(n)
                if depth[n] > depth[far]:
                    far = n
    spine = []
    cur = far
    while cur is not None:
        spine.append(cur)
        cur = parent[cur]
    spine.reverse()  # From ``start`` to the far endpoint.

    visited_global = set(spine)
    result = list(spine)

    def trace_branch(node, head):
        """
        Trace from ``head`` (a neighbour of ``node`` that is not on the spine)
        and insert the resulting trail right after ``node`` in ``result``.
        """
        nonlocal result
        trail = [head]
        visited_global.add(head)
        cur = head
        prev = node
        while True:
            candidates = [x for x in neighbors(cur)
                          if x not in visited_global and x != prev]
            if not candidates:
                break
            # Prefer the neighbour with the lowest degree (continue along
            # the path, avoid thick branches).
            candidates.sort(key=lambda c: len(neighbors(c)))
            nxt = candidates[0]
            trail.append(nxt)
            visited_global.add(nxt)
            prev = cur
            cur = nxt
        idx = result.index(node)
        result = result[:idx + 1] + trail + result[idx + 1:]

    # For every spine pixel, walk its non-spine neighbours and trace branches.
    for node in spine:
        for n in neighbors(node):
            if n not in visited_global:
                trace_branch(node, n)

    return result


def main(data_dir=None,
         merged_filename='merged_skeleton_data.json',
         output_filename='connected_lines.json'):
    """
    Build the connected-line JSON for the merged skeleton.

    Parameters
    ----------
    data_dir : str or None, optional
        Directory holding ``merged_skeleton_data.json``. ``None`` falls back
        to the legacy ``data/test_2025`` directory.
    merged_filename : str, optional
        Filename of the input merged-skeleton JSON
        (default ``'merged_skeleton_data.json'``).
    output_filename : str, optional
        Filename of the output connected-lines JSON
        (default ``'connected_lines.json'``).
    """
    if data_dir is None:
        # Default to test_2025 (backwards compatibility).
        base = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'test_2025')
        data_dir = base
    json_path = os.path.join(data_dir, merged_filename)
    output_path = os.path.join(data_dir, output_filename)

    # Load data.
    data = load_skeleton_data(json_path)
    lines_data = data['lines']

    width = data['metadata']['image_size']['width']
    height = data['metadata']['image_size']['height']

    # Collect every pixel.
    all_pixels = []
    for key, line in lines_data.items():
        for coord in line['pixel_coords']:
            all_pixels.append((coord['x'], coord['y']))

    # Connected-components analysis.
    components = find_connected_components(all_pixels, width, height)

    # Build the result.
    result = {
        "metadata": {
            "image_size": {"width": width, "height": height},
            "total_lines": len(components)
        },
        "lines": {}
    }

    margin = 10  # Boundary threshold in pixels; endpoints closer than this
    # to any image border are flagged as boundary endpoints and excluded
    # from SA matching.

    for i, comp in enumerate(components):
        endpoints = get_endpoints(comp)

        # Pick the two endpoints to use as line endpoints.
        if len(endpoints) >= 2:
            first_ep, second_ep = endpoints[0], endpoints[1]
            line_pixels = trace_line_order(first_ep, comp)
            if len(line_pixels) < 2:
                line_pixels = trace_line_order(second_ep, comp)
        elif len(endpoints) == 1:
            # Closed line: the single endpoint is both ends.
            first_ep = endpoints[0]
            second_ep = endpoints[0]
            line_pixels = trace_line_order(first_ep, comp)
        else:
            # No endpoints (should not happen in practice).
            first_ep = list(comp)[0]
            second_ep = first_ep
            line_pixels = list(comp)

        # Check whether either endpoint lies on the image boundary.
        first_is_boundary = (first_ep[0] < margin or first_ep[0] >= width - margin or
                             first_ep[1] < margin or first_ep[1] >= height - margin)
        second_is_boundary = (second_ep[0] < margin or second_ep[0] >= width - margin or
                              second_ep[1] < margin or second_ep[1] >= height - margin)

        result["lines"][str(i + 1)] = {
            "pixel_count": len(comp),
            "first_point": {"x": first_ep[0], "y": first_ep[1]},
            "second_point": {"x": second_ep[0], "y": second_ep[1]},
            "first_is_boundary": first_is_boundary,
            "second_is_boundary": second_is_boundary,
            "pixel_coords": [{"x": p[0], "y": p[1]} for p in line_pixels]
        }

    # Save.
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"Saved to: {output_path}")
    print(f"Total lines: {len(components)}")
    print("\nLines with non-boundary endpoints:")
    for line_id, line_data in result["lines"].items():
        if not line_data["first_is_boundary"] or not line_data["second_is_boundary"]:
            first = line_data["first_point"]
            second = line_data["second_point"]
            print(f"  Line {line_id}: ({first['x']},{first['y']}) <-> ({second['x']},{second['y']})")


if __name__ == '__main__':
    import sys
    # CLI: python extract_connected_lines.py [data_dir] [merged_filename] [output_filename]
    args = sys.argv[1:]
    data_dir_arg = args[0] if len(args) > 0 else None
    merged_arg = args[1] if len(args) > 1 else 'merged_skeleton_data.json'
    output_arg = args[2] if len(args) > 2 else 'connected_lines.json'
    main(data_dir=data_dir_arg, merged_filename=merged_arg, output_filename=output_arg)