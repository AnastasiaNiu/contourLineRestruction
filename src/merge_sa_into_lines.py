# -*- coding: utf-8 -*-
"""
Merge ``connected_lines_v4.json`` line segments into longer lines by
applying the simulated-annealing (SA) connections.

Inputs
------
connected_lines_v4.json       28 lines after the v4 breakpoint-matching step
annealing_connections.json    SA output: 5 connections, each linking two
                              endpoints

Output
------
connected_lines_sa_merged.json  Stitched lines (28 - 5 * 2 + 5 = 23 lines)

Stitching rules
---------------
Every line's ``pixel_coords`` is stored in the order first_point -> second_point.
For an SA connection (L_A, typeA) <-> (L_B, typeB):

- L_A is "entered" from its non-``typeA`` side and walked toward the
  ``typeA`` side (which then leaves toward L_B).
- L_B is "entered" at its ``typeB`` side and walked toward the other end
  (which is the tail of the chain).

The ``direction`` field determines the orientation:

- ``'forward'``: pixels kept in the original (first -> second) order.
- ``'reverse'``: pixels reversed (second -> first).

For a chain head (no incoming SA):

- If the SA touches the 'first' end of the segment, ``direction='reverse'``
  (we start at the 'second' end).
- Otherwise (``typeA == 'second'``), ``direction='forward'``.

For a middle / tail node (has an incoming SA):

- Entering via 'first'  -> ``direction='forward'``.
- Entering via 'second' -> ``direction='reverse'``.

Multi-SA chains (A -> B -> C -> ...) are handled by walking the SA adjacency
list in order.
"""
import argparse
import json
import os
from collections import defaultdict


def pixel_dist(a, b):
    """
    Euclidean distance between two ``(x, y)`` points.

    Parameters
    ----------
    a, b : tuple
        ``(x, y)`` coordinates.

    Returns
    -------
    float
        ``sqrt((a.x - b.x)**2 + (a.y - b.y)**2)``.
    """
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def orient_curve(curve, target_start):
    """
    Orient ``curve`` so that its starting endpoint is close to ``target_start``.

    Parameters
    ----------
    curve : list of tuple
        ``(x, y)`` samples along the curve.
    target_start : tuple
        ``(x, y)`` of the desired starting endpoint.

    Returns
    -------
    list of tuple
        ``curve`` either in original or reversed order.
    """
    d0 = pixel_dist(curve[0], target_start)
    d1 = pixel_dist(curve[-1], target_start)
    if d1 < d0:
        return list(reversed(curve))
    return list(curve)


def merge_lines_by_sa(v4_json_path, sa_json_path, output_json_path):
    """
    Stitch the v4 line segments together using the SA connections.

    Parameters
    ----------
    v4_json_path : str
        Path to ``connected_lines_v4.json``.
    sa_json_path : str
        Path to ``annealing_connections.json``.
    output_json_path : str
        Destination path for the merged JSON.

    Returns
    -------
    dict
        The merged output payload (the same object that is serialised to
        ``output_json_path``).
    """
    with open(v4_json_path, 'r', encoding='utf-8') as f:
        v4 = json.load(f)
    with open(sa_json_path, 'r', encoding='utf-8') as f:
        sa = json.load(f)

    width = v4['metadata']['image_size']['width']
    height = v4['metadata']['image_size']['height']
    lines = v4['lines']

    # ── Union-Find ──
    parent = {k: k for k in lines.keys()}

    def find(x):
        """Find with path compression."""
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        """Union two roots."""
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    sa_conns = sa.get('connections', [])
    ep_info = {}
    for c in sa_conns:
        e1 = c['endpoint1']
        e2 = c['endpoint2']
        la = str(e1['line_id'])
        lb = str(e2['line_id'])
        if la not in lines or lb not in lines:
            continue
        union(la, lb)
        ep_info[(la, lb)] = {
            'typeA': e1['type'],
            'typeB': e2['type'],
            'curve': [(p['x'], p['y']) for p in c['pixel_coords']],
        }

    # ── Grouping by root ──
    groups = defaultdict(list)
    for k in lines.keys():
        groups[find(k)].append(k)

    # ── Build the SA adjacency list
    #     line_id -> [(next_line_id, this_ep_type, other_ep_type, curve)]
    adj = defaultdict(list)
    for (la, lb), info in ep_info.items():
        adj[la].append((lb, info['typeA'], info['typeB'], info['curve']))
        adj[lb].append((la, info['typeB'], info['typeA'], info['curve']))

    degree = defaultdict(int)
    for la, lb in ep_info.keys():
        degree[la] += 1
        degree[lb] += 1

    sa_used_eps = defaultdict(set)
    for (la, lb), info in ep_info.items():
        sa_used_eps[la].add(info['typeA'])
        sa_used_eps[lb].add(info['typeB'])

    result_lines = {}
    next_id = 1

    for root, members in groups.items():
        if len(members) == 1:
            k = members[0]
            line = lines[k]
            merged_pixels = [(p['x'], p['y']) for p in line['pixel_coords']]
            result_lines[str(next_id)] = {
                'source_line_ids': [k],
                'merged_via_sa': [],
                'pixel_count': len(merged_pixels),
                'first_point': line['first_point'],
                'second_point': line['second_point'],
                'first_is_boundary': line.get('first_is_boundary', False),
                'second_is_boundary': line.get('second_is_boundary', False),
                'pixel_coords': [{'x': p[0], 'y': p[1]} for p in merged_pixels],
            }
            next_id += 1
            continue

        # Multi-line group: find the chain head (degree == 1).
        endpoints = [m for m in members if degree[m] == 1]
        if not endpoints:
            endpoints = [members[0]]
        head = endpoints[0]

        # Walk the SA adjacency list to build the ordered chain
        # (line_id, direction, curve_to_next).
        chain = []
        visited = set()
        cur = head
        entered_at = None
        while True:
            visited.add(cur)
            next_neighbors = [n for n in adj[cur] if n[0] not in visited]
            if not next_neighbors:
                # Chain tail.
                if entered_at is None:
                    this_sa_ep = next(iter(sa_used_eps[cur]))
                    direction = 'forward' if this_sa_ep == 'second' else 'reverse'
                else:
                    direction = 'forward' if entered_at == 'first' else 'reverse'
                chain.append((cur, direction, None))
                break

            neighbor_id, this_ep_type, other_ep_type, curve = next_neighbors[0]
            if entered_at is None:
                # Chain head: start at the non-SA end.
                direction = 'forward' if this_ep_type == 'second' else 'reverse'
            else:
                # Middle: enter via entered_at and walk to the other end.
                direction = 'forward' if entered_at == 'first' else 'reverse'
            chain.append((cur, direction, curve))
            entered_at = other_ep_type
            cur = neighbor_id

        # ── Stitch pixels along the chain ──
        merged_pixels = []
        sa_used = []
        for i, (lid, direction, curve) in enumerate(chain):
            line = lines[lid]
            pix = [(p['x'], p['y']) for p in line['pixel_coords']]
            if direction == 'reverse':
                pix = list(reversed(pix))
            merged_pixels.extend(pix)

            if curve is not None and i < len(chain) - 1:
                next_lid, next_dir, _ = chain[i + 1]
                next_line = lines[next_lid]
                next_pix = [(p['x'], p['y']) for p in next_line['pixel_coords']]
                if next_dir == 'reverse':
                    next_pix = list(reversed(next_pix))
                last_pt = merged_pixels[-1]
                first_pt_next = next_pix[0]
                curve_oriented = orient_curve(curve, last_pt)
                d_start = pixel_dist(curve_oriented[0], last_pt)
                d_end = pixel_dist(curve_oriented[-1], first_pt_next)
                if d_start < 5 and d_end < 5:
                    merged_pixels.extend(curve_oriented[1:-1])
                else:
                    merged_pixels.extend(curve_oriented)
                sa_used.append(sa_conns[i]['id'] if i < len(sa_conns) else None)

        fp = {'x': merged_pixels[0][0], 'y': merged_pixels[0][1]}
        sp = {'x': merged_pixels[-1][0], 'y': merged_pixels[-1][1]}
        margin = 10
        fp_b = fp['x'] < margin or fp['x'] >= width - margin or fp['y'] < margin or fp['y'] >= height - margin
        sp_b = sp['x'] < margin or sp['x'] >= width - margin or sp['y'] < margin or sp['y'] >= height - margin

        result_lines[str(next_id)] = {
            'source_line_ids': members,
            'merged_via_sa': [x for x in sa_used if x is not None],
            'pixel_count': len(merged_pixels),
            'first_point': fp,
            'second_point': sp,
            'first_is_boundary': fp_b,
            'second_is_boundary': sp_b,
            'pixel_coords': [{'x': p[0], 'y': p[1]} for p in merged_pixels],
        }
        next_id += 1

    output = {
        'metadata': {
            'image_size': {'width': width, 'height': height},
            'total_lines': len(result_lines),
            'source': 'merged_from_v4+SA',
            'v4_lines': len(lines),
            'sa_connections': len(sa_conns),
        },
        'lines': result_lines,
    }

    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"[OK] {len(lines)} v4 lines + {len(sa_conns)} SA connections -> {len(result_lines)} merged lines")
    print(f"[OK] Saved: {output_json_path}")
    return output


def main():
    """CLI entry point."""
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', required=True)
    ap.add_argument('--v4', default='connected_lines_v4.json')
    ap.add_argument('--sa', default='annealing_connections.json')
    ap.add_argument('--out', default='connected_lines_sa_merged.json')
    args = ap.parse_args()

    merge_lines_by_sa(
        os.path.join(args.data_dir, args.v4),
        os.path.join(args.data_dir, args.sa),
        os.path.join(args.data_dir, args.out),
    )


if __name__ == '__main__':
    main()