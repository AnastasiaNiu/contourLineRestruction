"""
Contour-line breakpoint connection using Simulated Annealing.

The algorithm connects non-boundary endpoints of merged contour lines with
cubic Bezier curves. It builds on the same direction-coding rules used by
``breakpoint_v4_debug.py`` and optimizes the assignment with simulated
annealing.

Pipeline
--------
1. Load the connected-lines JSON and the merged-skeleton JSON.
2. Identify every non-boundary endpoint.
3. Build a candidate graph where each pair of endpoints within ``max_distance``
   is annotated with a direction-compatibility score (see
   :data:`DIRECTION_MATCH_RULES`).
4. Seed the initial assignment from "mutual best partner" and the remaining
   orphan endpoints.
5. Run simulated annealing (with single-endpoint swap and double-end
   disconnect/reconnect moves) and minimize a weighted cost function
   combining angle cost, intersection cost and distance cost.
6. Drop any connection that crosses another connection or an unrelated
   skeleton line.
7. Visualise the result and write JSON + PNG outputs.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import cv2
import random
import math
import os


def load_json(json_path):
    """Load JSON data from ``json_path``.

    Parameters
    ----------
    json_path : str
        Path to the JSON file.

    Returns
    -------
    dict
        Parsed JSON content.
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def find_non_boundary_endpoints(lines_data, width, height):
    """Collect every non-boundary endpoint (deduplicated).

    Parameters
    ----------
    lines_data : dict
        Mapping ``line_id -> line_record`` where ``line_record`` contains
        ``first_point``, ``second_point`` and ``first_is_boundary`` /
        ``second_is_boundary`` flags.
    width : int
        Image width (currently unused; reserved for future extension).
    height : int
        Image height (currently unused; reserved for future extension).

    Returns
    -------
    list of dict
        Each entry is ``{'x', 'y', 'line_id', 'endpoint_type'}`` and only
        includes endpoints whose ``is_boundary`` flag is ``False``. Endpoints
        appearing in multiple lines are deduplicated by ``(x, y)``.
    """
    endpoints = []
    seen = set()

    for line_id, line_info in lines_data.items():
        first = line_info['first_point']
        second = line_info['second_point']

        # First endpoint.
        first_key = (first['x'], first['y'])
        if not line_info.get('first_is_boundary', False) and first_key not in seen:
            endpoints.append({
                'x': first['x'],
                'y': first['y'],
                'line_id': line_id,
                'endpoint_type': 'first'
            })
            seen.add(first_key)

        # Second endpoint.
        second_key = (second['x'], second['y'])
        if not line_info.get('second_is_boundary', False) and second_key not in seen:
            endpoints.append({
                'x': second['x'],
                'y': second['y'],
                'line_id': line_id,
                'endpoint_type': 'second'
            })
            seen.add(second_key)

    return endpoints


def calculate_distance(p1, p2):
    """Compute the Euclidean distance between two endpoint dicts.

    Parameters
    ----------
    p1, p2 : dict
        Endpoint dicts with keys ``'x'`` and ``'y'``.

    Returns
    -------
    float
        Euclidean distance in pixels.
    """
    return math.sqrt((p1['x'] - p2['x']) ** 2 + (p1['y'] - p2['y']) ** 2)


def get_trace_direction(endpoint, lines_data):
    """Compute the unit tangent of the contour at ``endpoint``.

    Identical strategy to ``breakpoint_v3_debug.py``'s
    ``get_endpoint_direction_code``: walk up to ``num_steps=30`` skeleton
    pixels along the 8-neighbour direction list and return the unit vector
    defined by the mean displacement of those pixels relative to the start.

    Parameters
    ----------
    endpoint : dict
        Endpoint record with ``'x'``, ``'y'`` and ``'line_id'``.
    lines_data : dict
        Mapping ``line_id -> line_record`` containing ``pixel_coords``.

    Returns
    -------
    tuple[float, float]
        Unit tangent vector ``(dx, dy)``; falls back to ``(0, 1)`` when the
        line cannot be found or the walk is too short.
    """
    line_id = endpoint['line_id']
    if str(line_id) not in lines_data and line_id not in lines_data:
        return (0.0, 1.0)
    pixels = lines_data.get(str(line_id), lines_data.get(line_id, {})).get('pixel_coords', [])
    if not pixels:
        return (0.0, 1.0)

    # Locate the endpoint inside pixel_coords (tolerance of 2 pixels).
    ep_idx = -1
    for i, p in enumerate(pixels):
        if abs(p['x'] - endpoint['x']) < 2 and abs(p['y'] - endpoint['y']) < 2:
            ep_idx = i
            break
    if ep_idx < 0:
        return (0.0, 1.0)

    coord_set = {(p['x'], p['y']) for p in pixels}
    x0, y0 = endpoint['x'], endpoint['y']
    current = (x0, y0)
    visited = {current}
    path = []

    # 8-direction order (matches breakpoint_v3_debug.py).
    directions = [(-1, -1), (-1, 0), (-1, 1),
                  (0, -1), (0, 1),
                  (1, -1), (1, 0), (1, 1)]

    num_steps = 30
    steps = 0
    while steps < num_steps:
        found_next = False
        for dx, dy in directions:
            nx, ny = current[0] + dx, current[1] + dy
            if (nx, ny) in coord_set and (nx, ny) not in visited:
                path.append((nx, ny))
                visited.add((nx, ny))
                current = (nx, ny)
                found_next = True
                break
        if not found_next:
            break
        steps += 1

    if len(path) < 2:
        return (0.0, 1.0)

    avg_dx = sum(p[0] - x0 for p in path) / len(path)
    avg_dy = sum(p[1] - y0 for p in path) / len(path)
    length = math.sqrt(avg_dx ** 2 + avg_dy ** 2)
    if length < 0.1:
        return (0.0, 1.0)

    return (avg_dx / length, avg_dy / length)


def _backtrace_skeleton_points(endpoint, lines_data, num_steps=30):
    """Walk up to ``num_steps`` skeleton pixels from ``endpoint``.

    Mirrors ``breakpoint_v3_debug.py``'s ``get_endpoint_direction_code``
    backtrace exactly: greedy step into an unvisited 8-neighbour pixel at
    each iteration, never crossing previously visited pixels.

    Parameters
    ----------
    endpoint : dict
        Endpoint record with ``'x'``, ``'y'`` and ``'line_id'``.
    lines_data : dict
        Mapping ``line_id -> line_record`` containing ``pixel_coords``.
    num_steps : int, optional
        Maximum number of steps to walk. Default ``30``.

    Returns
    -------
    list of tuple
        Visited pixels ``(x, y)`` in walk order (the starting pixel is
        excluded). Empty list if the line cannot be found.
    """
    line_id = endpoint['line_id']
    pixels = lines_data.get(str(line_id), lines_data.get(line_id, {})).get('pixel_coords', [])
    if not pixels:
        return []

    x0, y0 = endpoint['x'], endpoint['y']
    coord_set = {(p['x'], p['y']) for p in pixels}

    current = (x0, y0)
    visited = {current}
    path = []
    directions = [(-1, -1), (-1, 0), (-1, 1),
                  (0, -1), (0, 1),
                  (1, -1), (1, 0), (1, 1)]
    steps = 0
    while steps < num_steps:
        found_next = False
        for dx, dy in directions:
            nx, ny = current[0] + dx, current[1] + dy
            if (nx, ny) in coord_set and (nx, ny) not in visited:
                path.append((nx, ny))
                visited.add((nx, ny))
                current = (nx, ny)
                found_next = True
                break
        if not found_next:
            break
        steps += 1
    return path


def connect_with_cubic_spline(ep1, ep2, vec1=None, vec2=None, num_points=100,
                              lines_data=None, backtrack_steps=0):
    """Connect two endpoints with a cubic Bezier spline (back-extension algorithm).

    This mirrors the "back-extension intersection" algorithm from
    ``breakpoint_v3_debug.py``. The control points are placed along the
    back-extension of the input tangents; the offset adapts to the
    intersection distance, so close-to-parallel pairs naturally become
    nearly straight.

    Parameters
    ----------
    ep1, ep2 : dict or tuple
        Endpoint coordinates. Accepts ``{'x', 'y'}`` dicts or ``(x, y)``
        tuples.
    vec1, vec2 : tuple[float, float] or None, optional
        Tangent unit vectors at each endpoint. Falls back to ``(0, 1)``.
    num_points : int, optional
        Number of sample points along the curve. Default ``100``.
    lines_data : dict or None, optional
        Reserved for API compatibility; not used here.
    backtrack_steps : int, optional
        Reserved for API compatibility; not used here.

    Returns
    -------
    list of dict
        Sampled curve points as ``{'x', 'y'}`` dicts.
    """
    import numpy as np

    # Accept either dict or tuple endpoints.
    def _xy(ep):
        if isinstance(ep, dict):
            return float(ep['x']), float(ep['y'])
        return float(ep[0]), float(ep[1])

    x0, y0 = _xy(ep1)
    x1, y1 = _xy(ep2)

    dist = math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)
    P0 = np.array([x0, y0])
    P3 = np.array([x1, y1])

    # Normalise input vectors.
    def _norm_vec(v, fb=None):
        if v is None or (hasattr(v, '__len__') and len(v) == 0):
            return np.array([0.0, 1.0])
        try:
            arr = np.array([float(v[0]), float(v[1])])
        except Exception:
            return np.array([0.0, 1.0])
        n = np.linalg.norm(arr)
        if n < 1e-10:
            return np.array([0.0, 1.0])
        return arr / n

    vec1n = _norm_vec(vec1)
    vec2n = _norm_vec(vec2)

    back_vec1 = -vec1n
    back_vec2 = -vec2n

    # Solve the back-extension intersection.
    A = np.array([[back_vec1[0], -back_vec2[0]],
                  [back_vec1[1], -back_vec2[1]]])
    diff = P3 - P0

    intersection = None
    t1_val, t2_val = None, None

    try:
        det = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
        if abs(det) > 1e-10:
            t = np.linalg.solve(A, diff)
            t1_val, t2_val = float(t[0]), float(t[1])
            # The intersection must be in front of both endpoints and within
            # a sensible distance of the chord.
            if t1_val > 0 and t2_val > 0 and t1_val < dist * 3 and t2_val < dist * 3:
                intersection = P0 + t1_val * back_vec1
    except Exception:
        pass

    # Adaptive tension based on whether the intersection is valid.
    if intersection is not None:
        d1 = min(t1_val * 0.4, dist * 0.35)
        d2 = min(t2_val * 0.4, dist * 0.35)
    else:
        d1 = dist * 0.15
        d2 = dist * 0.15

    # Control points along the back-extensions.
    P1 = P0 + d1 * back_vec1
    P2 = P3 + d2 * back_vec2

    # Sample the cubic Bezier curve.
    pts = []
    for i in range(num_points):
        t = i / (num_points - 1)
        mt = 1 - t
        mt2 = mt * mt
        mt3 = mt2 * mt
        t2_ = t * t
        t3 = t2_ * t
        bx = mt3 * P0[0] + 3 * mt2 * t * P1[0] + 3 * mt * t2_ * P2[0] + t3 * P3[0]
        by = mt3 * P0[1] + 3 * mt2 * t * P1[1] + 3 * mt * t2_ * P2[1] + t3 * P3[1]
        pts.append({'x': int(round(bx)), 'y': int(round(by))})
    return pts


def _linear_samples(seq, num_points):
    """Resample a polyline by linear interpolation.

    Parameters
    ----------
    seq : list of tuple
        Sequence of ``(x, y)`` points to be resampled.
    num_points : int
        Number of output samples.

    Returns
    -------
    list of dict
        Resampled points as ``{'x', 'y'}`` dicts.
    """
    xs = np.array([p[0] for p in seq], dtype=float)
    ys = np.array([p[1] for p in seq], dtype=float)
    s = np.linspace(0, 1, len(seq))
    s_samples = np.linspace(0, 1, num_points)
    x_samples = np.interp(s_samples, s, xs)
    y_samples = np.interp(s_samples, s, ys)
    return [{'x': int(round(x)), 'y': int(round(y))} for x, y in zip(x_samples, y_samples)]


def connect_with_visible_bezier(ep1, ep2, vec1=None, vec2=None, num_points=150):
    """Connect two endpoints with a more visibly curved cubic Bezier.

    Same "back-extension intersection" idea as
    ``breakpoint_v3_debug.py`` but the control points also receive a
    perpendicular offset, so the resulting curve has a clear arc instead of
    collapsing onto the chord.

    Parameters
    ----------
    ep1, ep2 : dict or tuple
        Endpoint coordinates. Accepts ``{'x', 'y'}`` dicts or ``(x, y)``
        tuples.
    vec1, vec2 : tuple[float, float] or None, optional
        Tangent unit vectors at each endpoint. ``None`` falls back to the
        chord direction.
    num_points : int, optional
        Number of sample points along the curve. Default ``150``.

    Returns
    -------
    list of dict
        Sampled curve points as ``{'x', 'y'}`` dicts.
    """
    # Accept either dict or tuple.
    def _xy(ep):
        if isinstance(ep, dict):
            return float(ep['x']), float(ep['y'])
        return float(ep[0]), float(ep[1])

    x0, y0 = _xy(ep1)
    x1, y1 = _xy(ep2)

    dist = np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)
    if dist < 1e-6:
        return [{'x': int(x0), 'y': int(y0)}, {'x': int(x1), 'y': int(y1)}]

    P0 = np.array([x0, y0])
    P3 = np.array([x1, y1])

    # Chord / perpendicular unit vectors.
    chord = P3 - P0
    chord_dir = chord / dist
    perp_dir = np.array([-chord_dir[1], chord_dir[0]])  # 90deg clockwise

    def _norm_vec(v):
        if v is None:
            return chord_dir
        try:
            arr = np.array([float(v[0]), float(v[1])])
        except Exception:
            return chord_dir
        n = np.linalg.norm(arr)
        if n < 1e-10:
            return chord_dir
        return arr / n

    vec1n = _norm_vec(vec1)
    vec2n = _norm_vec(vec2)
    back_vec1 = -vec1n
    back_vec2 = -vec2n

    # Back-extension intersection (same algorithm as breakpoint_v3).
    A = np.array([[back_vec1[0], -back_vec2[0]],
                  [back_vec1[1], -back_vec2[1]]])
    diff = P3 - P0

    intersection = None
    t1_val = t2_val = None
    try:
        det = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
        if abs(det) > 1e-10:
            t = np.linalg.solve(A, diff)
            t1_val, t2_val = float(t[0]), float(t[1])
            if t1_val > 0 and t2_val > 0 and t1_val < dist * 3 and t2_val < dist * 3:
                intersection = P0 + t1_val * back_vec1
    except Exception:
        pass

    # Tension along the chord.
    if intersection is not None:
        tension1 = min(t1_val * 0.55, dist * 0.55)
        tension2 = min(t2_val * 0.55, dist * 0.55)
    else:
        tension1 = dist * 0.35
        tension2 = dist * 0.35

    # Decompose back_vec into chord / perpendicular components.
    def _decompose(back_vec):
        c_comp = float(np.dot(back_vec, chord_dir))
        p_comp = float(np.dot(back_vec, perp_dir))
        return c_comp, p_comp

    c1, p1 = _decompose(back_vec1)
    c2, p2 = _decompose(back_vec2)

    P1 = P0 + tension1 * (c1 * chord_dir + p1 * perp_dir)
    P2 = P3 + tension2 * (c2 * chord_dir + p2 * perp_dir)

    # Sample the cubic Bezier curve.
    pts = []
    for i in range(num_points):
        t = i / (num_points - 1)
        mt = 1 - t
        mt2 = mt * mt
        mt3 = mt2 * mt
        t2_ = t * t
        t3 = t2_ * t
        bx = mt3 * P0[0] + 3 * mt2 * t * P1[0] + 3 * mt * t2_ * P2[0] + t3 * P3[0]
        by = mt3 * P0[1] + 3 * mt2 * t * P1[1] + 3 * mt * t2_ * P2[1] + t3 * P3[1]
        pts.append({'x': int(round(bx)), 'y': int(round(by))})
    return pts


def connect_with_cubic_bezier(ep1, ep2, vec1=None, vec2=None, num_points=100):
    """Connect two endpoints with a cubic Bezier curve using tangent hints.

    Parameters
    ----------
    ep1, ep2 : dict
        Endpoint dicts with keys ``'x'`` and ``'y'``.
    vec1, vec2 : tuple[float, float] or None, optional
        Tangent unit vectors at each endpoint. ``None`` falls back to
        ``(0, 1)``.
    num_points : int, optional
        Number of sample points along the curve. Default ``100``.

    Returns
    -------
    list of dict
        Sampled curve points as ``{'x', 'y'}`` dicts.
    """
    x0, y0 = float(ep1['x']), float(ep1['y'])
    x1, y1 = float(ep2['x']), float(ep2['y'])

    dist = np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)

    P0 = np.array([x0, y0])
    P3 = np.array([x1, y1])

    # Normalise vec1.
    if vec1 is None or vec1 == (0, 0) or vec1[0] is None:
        vec1 = np.array([0.0, 1.0])
    else:
        v = np.array([float(vec1[0]), float(vec1[1])])
        v_len = np.linalg.norm(v)
        if v_len > 1e-10:
            vec1 = v / v_len
        else:
            vec1 = np.array([0.0, 1.0])

    # Normalise vec2.
    if vec2 is None or vec2 == (0, 0) or vec2[0] is None:
        vec2 = np.array([0.0, 1.0])
    else:
        v = np.array([float(vec2[0]), float(vec2[1])])
        v_len = np.linalg.norm(v)
        if v_len > 1e-10:
            vec2 = v / v_len
        else:
            vec2 = np.array([0.0, 1.0])

    # Compute the back-extension intersection.
    back_vec1 = -vec1
    back_vec2 = -vec2

    A = np.array([[back_vec1[0], -back_vec2[0]],
                  [back_vec1[1], -back_vec2[1]]])
    diff = P3 - P0

    det = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]

    intersection = None
    t1_val, t2_val = None, None

    if abs(det) > 1e-10:
        try:
            t = np.linalg.solve(A, diff)
            t1_val, t2_val = t[0], t[1]
            if 0 < t1_val < dist * 3 and 0 < t2_val < dist * 3:
                intersection = P0 + t1_val * back_vec1
        except Exception:
            pass

    # Compute control points.
    if intersection is not None:
        d1 = min(t1_val * 0.4, dist * 0.35)
        d2 = min(t2_val * 0.4, dist * 0.35)
    else:
        d1 = dist * 0.15
        d2 = dist * 0.15

    P1 = P0 + d1 * back_vec1
    P2 = P3 + d2 * back_vec2

    # Sample the Bezier curve.
    points = []
    for i in range(num_points + 1):
        t = i / num_points
        t2 = t * t
        t3 = t2 * t
        mt = 1 - t
        mt2 = mt * mt
        mt3 = mt2 * mt

        x = mt3 * P0[0] + 3 * mt2 * t * P1[0] + 3 * mt * t2 * P2[0] + t3 * P3[0]
        y = mt3 * P0[1] + 3 * mt2 * t * P1[1] + 3 * mt * t2 * P2[1] + t3 * P3[1]
        points.append({'x': round(x), 'y': round(y)})

    return points


def calculate_angle_cost(ep1, ep2, lines_data):
    """Compute the angle cost between two endpoints.

    The cost is the minimum of four combinations of forward / backward
    tangent directions at each endpoint. Lower cost means the connection
    direction aligns with the contour tangent at both endpoints.

    Parameters
    ----------
    ep1, ep2 : dict
        Endpoint dicts with ``'x'``, ``'y'`` and ``'line_id'``.
    lines_data : dict
        Mapping ``line_id -> line_record`` containing ``pixel_coords``.

    Returns
    -------
    float
        Angle cost in radians (0 means perfectly aligned).
    """
    dir1 = get_trace_direction(ep1, lines_data)
    dir2 = get_trace_direction(ep2, lines_data)

    vec_x = ep2['x'] - ep1['x']
    vec_y = ep2['y'] - ep1['y']
    norm = math.sqrt(vec_x ** 2 + vec_y ** 2)
    if norm < 0.1:
        return 0
    vec = (vec_x / norm, vec_y / norm)

    angles = []
    cos1 = dir1[0] * vec[0] + dir1[1] * vec[1]
    angles.append(abs(math.acos(max(-1, min(1, cos1)))))
    cos1b = -dir1[0] * vec[0] - dir1[1] * vec[1]
    angles.append(abs(math.acos(max(-1, min(1, cos1b)))))
    cos2 = dir2[0] * vec[0] + dir2[1] * vec[1]
    angles.append(abs(math.acos(max(-1, min(1, cos2)))))
    cos2b = -dir2[0] * vec[0] - dir2[1] * vec[1]
    angles.append(abs(math.acos(max(-1, min(1, cos2b)))))

    return min(angles[0] + angles[2], angles[0] + angles[3],
               angles[1] + angles[2], angles[1] + angles[3])


def _segments_intersect(p1, p2, p3, p4):
    """Test whether two segments intersect in their interiors.

    Parameters
    ----------
    p1, p2 : tuple[float, float]
        Endpoints of the first segment.
    p3, p4 : tuple[float, float]
        Endpoints of the second segment.

    Returns
    -------
    bool
        ``True`` if the two segments cross each other; co-linear overlap
        and shared endpoints are not considered an intersection.
    """
    def orient(pa, pb, pc):
        return (pb[0] - pa[0]) * (pc[1] - pa[1]) - (pb[1] - pa[1]) * (pc[0] - pa[0])

    # Bounding-box early-out.
    if (max(p1[0], p2[0]) < min(p3[0], p4[0]) or
            max(p3[0], p4[0]) < min(p1[0], p2[0]) or
            max(p1[1], p2[1]) < min(p3[1], p4[1]) or
            max(p3[1], p4[1]) < min(p1[1], p2[1])):
        return False
    d1 = orient(p3, p4, p1)
    d2 = orient(p3, p4, p2)
    d3 = orient(p1, p2, p3)
    d4 = orient(p1, p2, p4)
    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
            ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True
    return False


def calculate_intersection_cost(connections, lines_data):
    """Compute the intersection cost (paper Section 4.1.2, cost term 1).

    Two kinds of crossings are penalised:
        1. Two reconstructed curves crossing each other.
        2. A reconstructed curve crossing an original skeleton line that is
           not one of its two endpoint lines.

    Parameters
    ----------
    connections : list of tuple
        Each entry is ``(ep1, ep2)`` where ``ep1``/``ep2`` are endpoint
        dicts.
    lines_data : dict
        Mapping ``line_id -> line_record`` containing ``pixel_coords``.

    Returns
    -------
    float
        Raw intersection cost (multiplied by 50 by
        :func:`calculate_total_cost`). Higher means worse.
    """
    if not connections:
        return 0.0

    # Collect the original skeleton pixels.
    skel_pixels = set()
    for line_id, line_info in lines_data.items():
        for p in line_info.get('pixel_coords', []):
            skel_pixels.add((int(p['x']), int(p['y'])))

    # For each connection, build the curve segments and remember endpoints.
    curve_segs_list = []   # [ [(p0, p1), (p1, p2), ...], ... ]
    curve_endpoints = []   # [(ep1_xy, ep2_xy), ...]

    for ep1, ep2 in connections:
        pts = connect_with_cubic_spline(
            ep1, ep2, None, None,
            num_points=150, lines_data=lines_data, backtrack_steps=10,
        )
        segs = [((int(pts[i]['x']), int(pts[i]['y'])),
                 (int(pts[i + 1]['x']), int(pts[i + 1]['y'])))
                for i in range(len(pts) - 1)]
        curve_segs_list.append(segs)
        curve_endpoints.append((
            (int(ep1['x']), int(ep1['y'])),
            (int(ep2['x']), int(ep2['y']))
        ))

    cost = 0.0

    # 1. Curve-curve crossings.
    n = len(connections)
    for i in range(n):
        ep_i = curve_endpoints[i]
        for j in range(i + 1, n):
            ep_j = curve_endpoints[j]
            cross = False
            for s1 in curve_segs_list[i]:
                for s2 in curve_segs_list[j]:
                    # Skip segments that touch a shared endpoint.
                    if (s1[0] in {ep_j[0], ep_j[1]} and s1[1] in {ep_j[0], ep_j[1]}) or \
                            (s2[0] in {ep_i[0], ep_i[1]} and s2[1] in {ep_i[0], ep_i[1]}):
                        continue
                    if _segments_intersect(s1[0], s1[1], s2[0], s2[1]):
                        cross = True
                        break
                if cross:
                    break
            if cross:
                # Raw curve-curve cost (calculate_total_cost multiplies by 50).
                cost += 2.0

    # 2. Curve-skeleton crossings.
    for i, segs in enumerate(curve_segs_list):
        ep1_xy, ep2_xy = curve_endpoints[i]
        ep1_line_id = connections[i][0].get('line_id')
        ep2_line_id = connections[i][1].get('line_id')

        # Pixels of the two endpoint lines (excluded from crossing count).
        ep_line_pixels = set()
        for lid in (ep1_line_id, ep2_line_id):
            if lid is not None and str(lid) in lines_data:
                for p in lines_data[str(lid)].get('pixel_coords', []):
                    ep_line_pixels.add((int(p['x']), int(p['y'])))

        for s in segs:
            # Skip the endpoint neighbourhood (allow the curve to rejoin
            # the contour).
            d1 = math.hypot(s[0][0] - ep1_xy[0], s[0][1] - ep1_xy[1])
            d2 = math.hypot(s[0][0] - ep2_xy[0], s[0][1] - ep2_xy[1])
            d3 = math.hypot(s[1][0] - ep1_xy[0], s[1][1] - ep1_xy[1])
            d4 = math.hypot(s[1][0] - ep2_xy[0], s[1][1] - ep2_xy[1])
            if min(d1, d2, d3, d4) <= 3:
                continue

            mid = ((s[0][0] + s[1][0]) // 2, (s[0][1] + s[1][1]) // 2)
            for pt in (s[0], s[1], mid):
                if pt in ep_line_pixels:
                    continue
                if pt in skel_pixels:
                    # Raw curve-skeleton cost (calculate_total_cost multiplies by 50).
                    cost += 3.0
                    break

    return cost


# ============================================================
# Endpoint direction coding + candidate graph (mirrors breakpoint_v4_debug.py rules)
# ============================================================

# Direction-match compatibility: each key code lists the set of codes
# that are geometrically compatible with it. For a valid match both
# ``code1 in DIRECTION_MATCH_RULES[code2]`` and the reverse must hold.
DIRECTION_MATCH_RULES = {
    0: {3, 4, 5},   # up      <-> down-right / down / down-left
    1: {3, 5, 7},   # up-right<-> down-right / down-left / up-left
    2: {6},         # right   <-> left
    3: {0, 1, 5, 7},# down-right<-> up / up-right / down-left / up-left
    4: {0},         # down    <-> up
    5: {0, 1, 3, 7},# down-left<-> up / up-right / down-right / up-left
    6: {2},         # left    <-> right
    7: {1, 3, 5},   # up-left <-> up-right / down-right / down-left
}

# Human-readable names for the 8 direction codes (bilingual for traceability).
CODE_NAMES = {
    0: 'up', 1: 'up-right', 2: 'right', 3: 'down-right',
    4: 'down', 5: 'down-left', 6: 'left', 7: 'up-left',
}


def _get_direction_code(x0, y0, x1, y1):
    """Convert a 2D vector into one of the 8 octant direction codes.

    Identical to ``breakpoint_v4_debug.py``'s ``get_direction_code``:
    0=up, 1=up-right, 2=right, 3=down-right, 4=down, 5=down-left,
    6=left, 7=up-left.

    Parameters
    ----------
    x0, y0 : int
        Start pixel coordinate.
    x1, y1 : int
        End pixel coordinate.

    Returns
    -------
    int
        Direction code in ``[0, 7]``.
    """
    dx = x1 - x0
    dy = y1 - y0
    if dx < 0 and dy < 0: return 7
    if dx == 0 and dy < 0: return 0
    if dx > 0 and dy < 0: return 1
    if dx < 0 and dy == 0: return 6
    if dx > 0 and dy == 0: return 2
    if dx < 0 and dy > 0: return 5
    if dx == 0 and dy > 0: return 4
    return 3


def get_endpoint_direction_code(coord_set, start_point, num_steps=30, snap_radius=3):
    """Walk along the skeleton from ``start_point`` and return direction info.

    Improvements over ``breakpoint_v4_debug.py``'s version:
        1. Snaps to the nearest skeleton pixel within ``snap_radius`` when
           the exact start is missing from ``coord_set``.
        2. Falls back to the single-step direction when only one pixel is
           reachable (instead of failing).

    Parameters
    ----------
    coord_set : set[tuple[int, int]]
        Skeleton pixel coordinates of the line.
    start_point : tuple[int, int]
        Starting pixel ``(x, y)``.
    num_steps : int, optional
        Maximum walk length in pixels. Default ``30``.
    snap_radius : int, optional
        Tolerance used to snap ``start_point`` to the nearest skeleton
        pixel. Default ``3``.

    Returns
    -------
    tuple
        ``(local_dir, global_dir, main_dir, tangent_vec)``. Any element may
        be ``None`` if the walk produced too few pixels or no neighbour
        was found inside ``snap_radius``.
    """
    x0, y0 = start_point

    # Snap to the closest skeleton pixel when the exact point is missing.
    if (x0, y0) not in coord_set:
        best_d2 = None
        best_pt = None
        for px, py in coord_set:
            d2 = (px - x0) ** 2 + (py - y0) ** 2
            if best_d2 is None or d2 < best_d2:
                best_d2 = d2
                best_pt = (px, py)
                if d2 == 0:
                    break
        if best_pt is None or best_d2 > snap_radius ** 2:
            return None, None, None, None
        x0, y0 = best_pt

    path = []
    current = (x0, y0)
    visited = {current}
    directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    steps = 0
    while steps < num_steps:
        found_next = False
        for dx, dy in directions:
            nx, ny = current[0] + dx, current[1] + dy
            if (nx, ny) in coord_set and (nx, ny) not in visited:
                path.append((nx, ny))
                visited.add((nx, ny))
                current = (nx, ny)
                found_next = True
                break
        if not found_next:
            break
        steps += 1

    # Accept even a single-step walk.
    if len(path) < 1:
        return None, None, None, None

    if len(path) >= 1:
        local_dir = _get_direction_code(x0, y0, path[0][0], path[0][1])
    else:
        local_dir = None

    if len(path) > 20:
        global_path = path[20:]
        gx0, gy0 = global_path[0][0], global_path[0][1]
        global_dir = _get_direction_code(gx0, gy0, global_path[-1][0], global_path[-1][1])
    else:
        # Short path: use the entire walk to estimate the global direction.
        if len(path) >= 1:
            global_dir = _get_direction_code(path[0][0], path[0][1],
                                             path[-1][0], path[-1][1])
        else:
            global_dir = None

    if len(path) >= 2:
        main_dir = _get_direction_code(x0, y0, path[-1][0], path[-1][1])
    else:
        # Single-step walk: reuse local_dir as main_dir for consistency.
        main_dir = local_dir

    avg_dx = sum(p[0] - x0 for p in path) / len(path)
    avg_dy = sum(p[1] - y0 for p in path) / len(path)
    length = math.sqrt(avg_dx ** 2 + avg_dy ** 2)
    tangent_vec = (avg_dx / length, avg_dy / length) if length >= 1 else None

    return local_dir, global_dir, main_dir, tangent_vec


def build_direction_candidates(endpoints, lines_data, max_distance=150):
    """Phase 1: build the candidate pair graph using direction compatibility.

    Every pair of endpoints within ``max_distance`` is scored:
        - ``0`` - mutual direction compatibility, the strongest match.
        - ``1`` - one-sided compatibility (used as a backup).
        - ``2`` - no direction info on either side (very weak).

    Parameters
    ----------
    endpoints : list of dict
        Endpoints to consider (must contain ``'x'``, ``'y'`` and
        ``'line_id'``).
    lines_data : dict
        Mapping ``line_id -> line_record`` containing ``pixel_coords``.
    max_distance : float, optional
        Maximum distance (pixels) for two endpoints to be considered.
        Default ``150``.

    Returns
    -------
    ep_with_code : list of dict
        Input endpoints enriched with ``'code'``, ``'tangent_vec'``,
        ``'local_dir'``, ``'global_dir'`` and ``'coord_set'`` fields, plus
        an ``'idx'`` matching their position in the input list.
    candidates : dict[int, list[tuple]]
        For each endpoint ``i``: ``[(partner_idx, distance, code_score)]``
        sorted by ``(code_score, distance)`` ascending.
    """
    ep_with_code = []
    for i, ep in enumerate(endpoints):
        lid = ep['line_id']
        line_info = lines_data.get(str(lid))
        if line_info and 'pixel_coords' in line_info:
            coord_set = set((int(p['x']), int(p['y'])) for p in line_info['pixel_coords'])
        else:
            coord_set = set()
        local_dir, global_dir, main_dir, tangent_vec = get_endpoint_direction_code(
            coord_set, (int(ep['x']), int(ep['y'])), num_steps=30
        )
        ep_with_code.append({
            **ep,
            'idx': i,
            'code': main_dir,
            'local_dir': local_dir,
            'global_dir': global_dir,
            'tangent_vec': tangent_vec,
            'coord_set': coord_set,
        })

    candidates = {i: [] for i in range(len(ep_with_code))}

    for i in range(len(ep_with_code)):
        for j in range(i + 1, len(ep_with_code)):
            if ep_with_code[i]['line_id'] == ep_with_code[j]['line_id']:
                continue  # Same line: two endpoints cannot connect to each other.
            dx = ep_with_code[i]['x'] - ep_with_code[j]['x']
            dy = ep_with_code[i]['y'] - ep_with_code[j]['y']
            dist = math.hypot(dx, dy)
            if dist > max_distance:
                continue
            code_i = ep_with_code[i]['code']
            code_j = ep_with_code[j]['code']

            if code_i is not None and code_j is not None:
                # Both endpoints have a direction code: require mutual compatibility.
                compat_i = code_j in DIRECTION_MATCH_RULES.get(code_i, set())
                compat_j = code_i in DIRECTION_MATCH_RULES.get(code_j, set())
                if not (compat_i and compat_j):
                    continue  # Direction-incompatible -> excluded.
                candidates[i].append((j, dist, 0))
                candidates[j].append((i, dist, 0))
            elif code_i is None and code_j is None:
                # No direction info on either side -> weak distance-only match.
                candidates[i].append((j, dist, 2))
                candidates[j].append((i, dist, 2))
            else:
                # One side missing: weak candidate.
                candidates[i].append((j, dist, 1))
                candidates[j].append((i, dist, 1))

    # Sort by (code_score, distance) - lower code_score is better.
    for k in candidates:
        candidates[k].sort(key=lambda c: (c[2], c[1]))

    return ep_with_code, candidates


def post_check_no_intersection(connections, lines_data,
                                ep_endpoint_dist=5,
                                skeleton_neighborhood=3):
    """Phase 3: detect crossings after the connections have been produced.

    Two checks:
        A. Any pair of reconstructed curves crossing each other (shared
           endpoints are ignored).
        B. Each reconstructed curve crossing a skeleton line that is not
           one of its two endpoint lines (curves may briefly rejoin the
           contour within ``ep_endpoint_dist`` pixels of their endpoints).

    Parameters
    ----------
    connections : list of tuple
        Each entry is ``(ep1, ep2)`` with endpoint dicts.
    lines_data : dict
        Mapping ``line_id -> line_record`` containing ``pixel_coords``.
    ep_endpoint_dist : int, optional
        Pixel radius around each endpoint excluded from the skeleton-cross
        test. Default ``5``.
    skeleton_neighborhood : int, optional
        Currently unused (kept for API compatibility).

    Returns
    -------
    invalid_set : set of tuple
        Pairs ``(idx1, idx2)`` whose curves cross each other.
    skel_cross : list of int
        Connection indices whose curve crosses the skeleton.
    """
    if len(connections) < 2:
        return set(), []

    # Build the curve segments for each connection.
    def curve_segs(ep1, ep2):
        pts = connect_with_cubic_spline(
            ep1, ep2, None, None,
            num_points=120, lines_data=lines_data, backtrack_steps=10,
        )
        segs = []
        for k in range(len(pts) - 1):
            segs.append(((int(pts[k]['x']), int(pts[k]['y'])),
                         (int(pts[k + 1]['x']), int(pts[k + 1]['y']))))
        return segs, [(int(ep1['x']), int(ep1['y'])), (int(ep2['x']), int(ep2['y']))]

    all_segs = []
    all_eps = []
    for ep1, ep2 in connections:
        s, e = curve_segs(ep1, ep2)
        all_segs.append(s)
        all_eps.append(e)

    invalid_set = set()

    # Curve-curve crossing detection.
    for i in range(len(connections)):
        for j in range(i + 1, len(connections)):
            shared = (set(all_eps[i]) & set(all_eps[j]))
            cross_found = False
            for s1 in all_segs[i]:
                if s1[0] in shared and s1[1] in shared:
                    continue
                for s2 in all_segs[j]:
                    if s2[0] in shared and s2[1] in shared:
                        continue
                    if _segments_intersect(s1[0], s1[1], s2[0], s2[1]):
                        invalid_set.add((i, j))
                        cross_found = True
                        break
                if cross_found:
                    break

    # Curve-skeleton crossing detection.
    skel_pixels = set()
    for line_id, line_info in lines_data.items():
        for p in line_info.get('pixel_coords', []):
            skel_pixels.add((int(p['x']), int(p['y'])))

    skel_cross = []
    for i, segs in enumerate(all_segs):
        ep1_xy, ep2_xy = all_eps[i]
        lid1 = connections[i][0].get('line_id')
        lid2 = connections[i][1].get('line_id')
        ep_line_pixels = set()
        for lid in (lid1, lid2):
            if lid is not None and str(lid) in lines_data:
                for p in lines_data[str(lid)].get('pixel_coords', []):
                    ep_line_pixels.add((int(p['x']), int(p['y'])))

        bad = False
        for s in segs:
            d1 = math.hypot(s[0][0] - ep1_xy[0], s[0][1] - ep1_xy[1])
            d2 = math.hypot(s[0][0] - ep2_xy[0], s[0][1] - ep2_xy[1])
            d3 = math.hypot(s[1][0] - ep1_xy[0], s[1][1] - ep1_xy[1])
            d4 = math.hypot(s[1][0] - ep2_xy[0], s[1][1] - ep2_xy[1])
            if min(d1, d2, d3, d4) <= ep_endpoint_dist:
                continue
            for pt in (s[0], s[1]):
                if pt in skel_pixels and pt not in ep_line_pixels:
                    bad = True
                    break
            if bad:
                break
        if bad:
            skel_cross.append(i)

    return invalid_set, skel_cross


def calculate_total_cost(connections, lines_data):
    """Compute the total cost (paper Section 4.1.2).

    Formula
    -------
    ``total = angle_cost * 1.0 + intersection_cost * 50.0 + distance_cost * 1.0``

    - ``angle_cost``       - sum of per-connection angle misalignment.
    - ``intersection_cost``- weighted count of curve-curve / curve-skeleton
                             crossings (50x to make any crossing prohibitive).
    - ``distance_cost``    - linear penalty for distances > 30 px.

    Parameters
    ----------
    connections : list of tuple
        Each entry is ``(ep1, ep2)``.
    lines_data : dict
        Mapping ``line_id -> line_record`` containing ``pixel_coords``.

    Returns
    -------
    float
        Total cost (lower is better); ``float('inf')`` for empty input.
    """
    if not connections:
        return float('inf')

    angle_cost = 0.0
    distance_cost = 0.0
    for ep1, ep2 in connections:
        angle_cost += calculate_angle_cost(ep1, ep2, lines_data)
        dist = calculate_distance(ep1, ep2)
        # Distance penalty starts after 30 px.
        if dist > 30:
            distance_cost += (dist - 30) * 0.01

    intersection_cost = calculate_intersection_cost(connections, lines_data)

    return angle_cost * 1.0 + intersection_cost * 50.0 + distance_cost * 1.0


def simulated_annealing(endpoints, lines_data,
                        T0=1000, T_end=1, alpha=0.95, beta=1,
                        max_no_improve=100, record_history=True,
                        initial_seed_connections=None,
                        neighbor_k=6, max_perm_distance=120,
                        candidates=None):
    """Simulated annealing for the breakpoint-connection assignment.

    The perturbation strategies are:
        - 70%: single-endpoint swap - replace one endpoint of an existing
          connection with a partner from its candidate list.
        - 30%: double-end disconnect/reconnect - break a connection and
          re-bind either endpoint to a different candidate.

    Parameters
    ----------
    endpoints : list of dict
        Endpoint dicts with ``'x'``, ``'y'``, ``'line_id'``,
        ``'endpoint_type'``.
    lines_data : dict
        Mapping ``line_id -> line_record`` containing ``pixel_coords``.
    T0 : float, optional
        Initial temperature. Default ``1000``.
    T_end : float, optional
        Final temperature (annealing stops below this). Default ``1``.
    alpha : float, optional
        Cooling rate per outer iteration. Default ``0.95``.
    beta : int, optional
        Inner-loop length factor; inner iterations = ``beta * n``. Default ``1``.
    max_no_improve : int, optional
        Annealing stops after this many inner iterations without finding a
        new best. Default ``100``.
    record_history : bool, optional
        Reserved for future use. Default ``True``.
    initial_seed_connections : list or None, optional
        Optional manual seed. Currently unused.
    neighbor_k : int, optional
        Reserved for future use. Default ``6``.
    max_perm_distance : float, optional
        Reserved for future use. Default ``120``.
    candidates : dict or None, optional
        ``candidates[ep_idx] = [(partner_idx, distance, code_score), ...]``
        from :func:`build_direction_candidates``. When provided, all
        perturbations are constrained to that graph; otherwise the
        algorithm falls back to the legacy any-endpoint pairing.

    Returns
    -------
    list of tuple
        Best-discovered connections as ``(ep1_dict, ep2_dict)`` pairs.
    """
    n = len(endpoints)
    if n < 2:
        return []

    if n % 2 != 0:
        print(f"Warning: Odd number of endpoints ({n}), removing last one")
        endpoints = endpoints[:n - 1]
        n = len(endpoints)

    endpoints_copy = [(ep['x'], ep['y'], ep['line_id'], ep['endpoint_type']) for ep in endpoints]

    # Global candidate graph used for SA perturbation.
    candidates_global = candidates if candidates is not None else {}

    current_connections = []

    # --- Smart initial seed ---
    # Strategy A: mutual best partner.
    if candidates_global:
        assigned = set()
        best_partner = {}
        for i, cands in candidates_global.items():
            sorted_cands = sorted(cands, key=lambda c: (c[2], c[1]))
            if sorted_cands:
                best_partner[i] = sorted_cands[0][0]
            else:
                best_partner[i] = None
        # Mutual best: i.best == j AND j.best == i.
        for i in range(n):
            if i in assigned:
                continue
            j = best_partner.get(i)
            if j is None:
                continue
            if best_partner.get(j) != i:
                continue
            if j in assigned:
                continue
            current_connections.append((endpoints_copy[i], endpoints_copy[j]))
            assigned.add(i)
            assigned.add(j)
        # Orphan rescue: pair any remaining endpoints by mutual direction
        # compatibility + closest distance.
        remaining = [k for k in range(n) if k not in assigned]
        remaining_codes = {}
        for k in remaining:
            try:
                ep = endpoints[k]
                lid = ep['line_id']
                line_info = lines_data.get(str(lid))
                if line_info and 'pixel_coords' in line_info:
                    coord_set = set((int(p['x']), int(p['y']))
                                    for p in line_info['pixel_coords'])
                else:
                    coord_set = set()
                _, _, code, _ = get_endpoint_direction_code(
                    coord_set, (int(ep['x']), int(ep['y'])), num_steps=30)
                remaining_codes[k] = code
            except Exception:
                remaining_codes[k] = None

        for ii in range(len(remaining)):
            if remaining[ii] in assigned:
                continue
            a = remaining[ii]
            code_a = remaining_codes[a]
            best_neighbor = None
            best_dist = None
            for jj in range(ii + 1, len(remaining)):
                if remaining[jj] in assigned:
                    continue
                b = remaining[jj]
                dx = endpoints_copy[a][0] - endpoints_copy[b][0]
                dy = endpoints_copy[a][1] - endpoints_copy[b][1]
                d = math.hypot(dx, dy)
                if d > 150:
                    continue
                code_b = remaining_codes[b]
                if code_a is not None and code_b is not None:
                    compat_ab = code_b in DIRECTION_MATCH_RULES.get(code_a, set())
                    compat_ba = code_a in DIRECTION_MATCH_RULES.get(code_b, set())
                    if not (compat_ab and compat_ba):
                        continue
                if best_neighbor is None or d < best_dist:
                    best_neighbor = b
                    best_dist = d
            if best_neighbor is not None:
                current_connections.append((endpoints_copy[a], endpoints_copy[best_neighbor]))
                assigned.add(a)
                assigned.add(best_neighbor)
    else:
        # Legacy behaviour: hard pair in input order.
        remaining = list(range(n))
        for i in range(n // 2):
            if len(remaining) < 2:
                break
            current_connections.append((endpoints_copy[remaining[0]],
                                        endpoints_copy[remaining[1]]))
            remaining = remaining[2:]

    def make_endpoint(d):
        return {'x': d[0], 'y': d[1], 'line_id': d[2], 'endpoint_type': d[3]}

    def get_connections_tuples(conns):
        return [(make_endpoint(c[0]), make_endpoint(c[1])) for c in conns]

    current_cost = calculate_total_cost(get_connections_tuples(current_connections), lines_data)
    best_connections = current_connections.copy()
    best_cost = current_cost

    T = T0
    iteration = 0
    no_improve_count = 0

    print("=" * 60)
    print("Simulated Annealing - Contour Connection")
    print("=" * 60)
    print(f"Endpoints to connect: {n}")
    print(f"Initial temperature: {T0}")
    print(f"Cooling rate: {alpha}")
    print(f"End temperature: {T_end}")
    print()

    # Standard annealing loop (paper Section 4.1.2):
    #   while T > T_end and no_improve < max_no_improve:
    #       for _ in range(L):  do perturbation   # inner loop L = beta * n
    #       T *= alpha                            # outer cooling
    L = max(1, int(beta * n))
    T = T0
    iteration = 0
    no_improve_count = 0
    while T > T_end and no_improve_count < max_no_improve:
        for _inner in range(L):
            iteration += 1

            new_connections = current_connections.copy()

            if len(new_connections) < 2:
                break

            # Candidate-constrained perturbation.
            if not candidates_global:
                perturbed = False
            else:
                idx_a = random.randrange(len(new_connections))
                ep_a = new_connections[idx_a]
                # Map both endpoints of ep_a back to global indices.
                used_in_conn = set()
                for _ep in ep_a[0:2]:
                    for _j, _t in enumerate(endpoints_copy):
                        if _ep[0] == _t[0] and _ep[1] == _t[1]:
                            used_in_conn.add(_j)
                            break
                if not used_in_conn:
                    perturbed = False
                else:
                    # 70% single-endpoint swap, 30% double-end reconnect.
                    use_disconnect = (random.random() < 0.30)
                    if use_disconnect:
                        # Double-end disconnect / reconnect: let either side
                        # pick a new partner from the candidate graph.
                        ep_idx_a, ep_idx_b = None, None
                        for _ep in ep_a[0:2]:
                            for _j, _t in enumerate(endpoints_copy):
                                if _ep[0] == _t[0] and _ep[1] == _t[1]:
                                    if ep_idx_a is None:
                                        ep_idx_a = _j
                                    else:
                                        ep_idx_b = _j
                                    break
                        all_used = set()
                        for _c in new_connections:
                            for _ep in _c[0:2]:
                                for _j, _t in enumerate(endpoints_copy):
                                    if _ep[0] == _t[0] and _ep[1] == _t[1]:
                                        all_used.add(_j)
                                        break
                        free = [k for k in range(n) if k not in all_used]
                        free.extend([ep_idx_a, ep_idx_b])
                        cand_list_a = candidates_global.get(ep_idx_a, [])
                        cand_list_b = candidates_global.get(ep_idx_b, [])
                        cand_combined = cand_list_a + cand_list_b
                        alt = [c for c in cand_combined
                               if c[0] in free and c[0] != ep_idx_a and c[0] != ep_idx_b]
                        if not alt:
                            perturbed = False
                        else:
                            alt.sort(key=lambda c: (c[2], c[1]))
                            chosen = random.choice(alt[:5]) if len(alt) > 5 else random.choice(alt)
                            new_partner_idx = chosen[0]
                            # Replace the old connection; bind the endpoint
                            # that originally listed the chosen candidate.
                            if chosen in cand_list_a:
                                new_connections.pop(idx_a)
                                new_connections.append((endpoints_copy[ep_idx_a],
                                                        endpoints_copy[new_partner_idx]))
                            else:
                                new_connections.pop(idx_a)
                                new_connections.append((endpoints_copy[ep_idx_b],
                                                        endpoints_copy[new_partner_idx]))
                            perturbed = True
                    else:
                        ep_idx_a = random.choice(list(used_in_conn))
                        cand_list = candidates_global.get(ep_idx_a, [])
                        if not cand_list:
                            perturbed = False
                        else:
                            chosen_t = endpoints_copy[ep_idx_a]
                            cur_partner = None
                            if ep_a[0][0] == chosen_t[0] and ep_a[0][1] == chosen_t[1]:
                                cur_partner = ep_a[1]
                            elif ep_a[1][0] == chosen_t[0] and ep_a[1][1] == chosen_t[1]:
                                cur_partner = ep_a[0]
                            cur_partner_idx = -1
                            if cur_partner is not None:
                                for _j, _t in enumerate(endpoints_copy):
                                    if cur_partner[0] == _t[0] and cur_partner[1] == _t[1]:
                                        cur_partner_idx = _j
                                        break
                            alt_candidates = [c for c in cand_list if c[0] != cur_partner_idx]
                            if not alt_candidates:
                                perturbed = False
                            else:
                                new_partner_idx = random.choice(alt_candidates)[0]
                                new_partner_t = endpoints_copy[new_partner_idx]
                                occupied = False
                                for _k_idx, (_ea, _eb) in enumerate(new_connections):
                                    if _k_idx == idx_a:
                                        continue
                                    if ((_ea[0], _ea[1]) == (new_partner_t[0], new_partner_t[1]) or
                                            (_eb[0], _eb[1]) == (new_partner_t[0], new_partner_t[1])):
                                        occupied = True
                                        break
                                if occupied:
                                    perturbed = False
                                else:
                                    if cur_partner is None:
                                        # Was not paired; cannot form one here.
                                        perturbed = False
                                    else:
                                        # Replace the chosen endpoint's side.
                                        if ep_a[0][0] == chosen_t[0] and ep_a[0][1] == chosen_t[1]:
                                            new_connections[idx_a] = (chosen_t, new_partner_t)
                                        else:
                                            new_connections[idx_a] = (new_partner_t, chosen_t)
                                        perturbed = True

            new_cost = calculate_total_cost(get_connections_tuples(new_connections), lines_data)

            delta = new_cost - current_cost
            if delta < 0 or random.random() < math.exp(-delta / T):
                current_connections = new_connections
                current_cost = new_cost

                if current_cost < best_cost:
                    best_cost = current_cost
                    best_connections = current_connections.copy()
                    no_improve_count = 0
                else:
                    no_improve_count += 1

        T *= alpha

        if iteration % 1000 == 0:
            print(f"Iter {iteration}: T={T:.2f}, Cost={current_cost:.4f}, Best={best_cost:.4f}")

    print()
    print(f"Finished at iteration {iteration}")
    print(f"Best cost: {best_cost:.4f}")
    print(f"Total connections: {len(best_connections)}")

    return get_connections_tuples(best_connections)


def visualize_and_save_results(skeleton_data, connections, lines_data, width, height, output_dir,
                                ep_with_code=None):
    """Render the SA result and save connection data.

    The figure contains:
        [1] Original skeleton (light grey background).
        [2] Bezier connection curves overlaid on the skeleton, with a
            legend (skeleton / Bezier / endpoint 1 / endpoint 2).
        [3] Per-connection details text.
        [4] Direction codes of all non-boundary endpoints (red = connected,
            grey = orphan; arrows show the code direction).

    Parameters
    ----------
    skeleton_data : dict
        Skeleton payload (background).
    connections : list of tuple
        ``(ep1, ep2)`` pairs as produced by :func:`simulated_annealing`.
    lines_data : dict
        Mapping ``line_id -> line_record`` used for direction backtraces.
    width : int
        Image width in pixels.
    height : int
        Image height in pixels.
    output_dir : str
        Folder in which ``annealing_result.png``, ``annealing_clean.png``,
        ``annealing_debug.png``, ``annealing_direction_codes.png``,
        ``annealing_connections.json`` and ``sa_result.json`` are written.
    ep_with_code : list of dict or None, optional
        Endpoints enriched with ``'code'``, ``'local_dir'``, ``'global_dir'``.
        When provided, the direction-code panel uses them for labels.
    """
    from io import BytesIO
    import matplotlib
    matplotlib.use('Agg')

    # 1. Build the skeleton background image.
    skel_img = np.ones((height, width), dtype=np.uint8) * 255
    for line_id, line_info in skeleton_data['lines'].items():
        for p in line_info['pixel_coords']:
            x, y = int(p['x']), int(p['y'])
            if 0 <= y < height and 0 <= x < width:
                skel_img[y, x] = 120  # light grey

    # 2. Overlay the Bezier curves on top of the skeleton.
    skel_rgb = cv2.cvtColor(skel_img, cv2.COLOR_GRAY2RGB)

    spline_pixels_total = 0
    connection_results = {
        'metadata': {
            'image_size': {'width': width, 'height': height},
            'total_connections': len(connections)
        },
        'connections': []
    }

    # Build (x,y) -> code lookup (with (y,x) double entry as a safety net).
    code_lookup = {}
    if ep_with_code is not None:
        for ep in ep_with_code:
            code_lookup[(int(ep['x']), int(ep['y']))] = ep.get('code')
            code_lookup[(int(ep['y']), int(ep['x']))] = ep.get('code')

    for i, (ep1, ep2) in enumerate(connections):
        vec1 = get_trace_direction(ep1, lines_data)
        vec2 = get_trace_direction(ep2, lines_data)
        # Use connect_with_visible_bezier (larger tension) for visible curvature.
        dist_for_pts = calculate_distance(ep1, ep2)
        num_pts = max(100, int(dist_for_pts * 5))
        spline_points = connect_with_visible_bezier(
            ep1, ep2, vec1, vec2, num_points=num_pts,
        )

        # Per-pixel drawing (rather than cv2.line) preserves the curve shape.
        for sp in spline_points:
            sx, sy = int(sp['x']), int(sp['y'])
            if 0 <= sy < height and 0 <= sx < width:
                skel_rgb[sy, sx] = (0, 220, 0)
                spline_pixels_total += 1

        # Endpoint circles (red / blue) with a white border.
        cv2.circle(skel_rgb, (ep1['x'], ep1['y']), 7, (255, 30, 30), -1)
        cv2.circle(skel_rgb, (ep2['x'], ep2['y']), 7, (30, 30, 255), -1)
        cv2.circle(skel_rgb, (ep1['x'], ep1['y']), 7, (255, 255, 255), 2)
        cv2.circle(skel_rgb, (ep2['x'], ep2['y']), 7, (255, 255, 255), 2)
        # Direction-code label next to each endpoint.
        code1 = code_lookup.get((int(ep1['x']), int(ep1['y'])))
        code2 = code_lookup.get((int(ep2['x']), int(ep2['y'])))
        if code1 is not None:
            label1 = f"c{code1}"
            cv2.putText(skel_rgb, label1, (ep1['x'] + 10, ep1['y'] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 30, 30), 2, cv2.LINE_AA)
        if code2 is not None:
            label2 = f"c{code2}"
            cv2.putText(skel_rgb, label2, (ep2['x'] + 10, ep2['y'] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 30, 255), 2, cv2.LINE_AA)

        dist = calculate_distance(ep1, ep2)
        connection_results['connections'].append({
            'id': i + 1,
            'endpoint1': {'x': ep1['x'], 'y': ep1['y'],
                          'line_id': ep1['line_id'], 'type': ep1['endpoint_type']},
            'endpoint2': {'x': ep2['x'], 'y': ep2['y'],
                          'line_id': ep2['line_id'], 'type': ep2['endpoint_type']},
            'direction_vec1': {'dx': round(vec1[0], 4), 'dy': round(vec1[1], 4)},
            'direction_vec2': {'dx': round(vec2[0], 4), 'dy': round(vec2[1], 4)},
            'distance': round(dist, 2),
            'pixel_coords': spline_points
        })

    print(f"Skeleton pixels: {np.sum(skel_img < 255)}")
    print(f"Bezier pixels drawn: {spline_pixels_total}")

    # 3. Save the coloured result image with unicode-path fallback.
    img_path = os.path.join(output_dir, 'annealing_result.png')
    try:
        ok = cv2.imwrite(img_path, skel_rgb)
    except Exception:
        ok = False
    if not ok or not os.path.exists(img_path):
        success, buf = cv2.imencode('.png', skel_rgb)
        if success:
            try:
                from pathlib import Path
                Path(img_path).write_bytes(buf.tobytes())
            except Exception as _e:
                print(f"Warning: {img_path} save failed ({_e}), trying alt")
                try:
                    import tempfile, shutil
                    tmpf = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
                    tmpf.write(buf.tobytes())
                    tmpf.close()
                    shutil.copyfile(tmpf.name, img_path)
                    os.remove(tmpf.name)
                except Exception as _e2:
                    print(f"ERROR: cannot save {img_path}: {_e2}")
            print(f"Saved (via imencode): {img_path}")
        else:
            print(f"ERROR: cannot save {img_path}")
    else:
        print(f"Saved: {img_path}")

    # 4. Compose the breakpoint_v3-style overview figure.
    fig = plt.figure(figsize=(24, 14))
    gs = fig.add_gridspec(2, 3, height_ratios=[3, 2])
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    ax_dir = fig.add_subplot(gs[1, :])

    # Figure 1: original skeleton.
    ax = axes[0]
    ax.imshow(cv2.cvtColor(skel_img, cv2.COLOR_GRAY2RGB))
    ax.set_title('[1] Original Skeleton (%d px)' % np.sum(skel_img < 255), fontsize=12)
    ax.axis('off')

    # Figure 2: connection overlay + legend.
    ax = axes[1]
    ax.imshow(skel_rgb)
    ax.set_title('[2] Cubic Spline Connections (%d pairs)' % len(connections), fontsize=12)
    ax.axis('off')
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_elements = [
        Patch(facecolor='#787878', label='Skeleton'),
        Patch(facecolor='green', label='Cubic Spline Connection'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10,
               markeredgecolor='white', label='Endpoint 1'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10,
               markeredgecolor='white', label='Endpoint 2'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10)

    # Figure 3: per-connection details.
    ax = axes[2]
    ax.set_facecolor('#f5f5f5')
    conn_text = "Connection Details (%d pairs)\n" % len(connections)
    conn_text += "=" * 55 + "\n"
    for i, conn in enumerate(connection_results['connections']):
        ep1 = conn['endpoint1']
        ep2 = conn['endpoint2']
        v1 = conn['direction_vec1']
        v2 = conn['direction_vec2']
        conn_text += (
            "  %d. L%s(%d,%d)\n"
            "       vec1=(%+6.4f,%+6.4f)\n"
            "     <-> L%s(%d,%d)\n"
            "       vec2=(%+6.4f,%+6.4f)\n"
            "     dist=%.1f px, curve_pts=%d\n"
            % (i + 1,
               ep1['line_id'], ep1['x'], ep1['y'],
               v1['dx'], v1['dy'],
               ep2['line_id'], ep2['x'], ep2['y'],
               v2['dx'], v2['dy'],
               conn['distance'], len(conn['pixel_coords']))
        )
        conn_text += "-" * 55 + "\n"
    ax.text(0.02, 0.98, conn_text, fontsize=9, family='monospace',
            verticalalignment='top', transform=ax.transAxes)
    ax.axis('off')

    # Figure 4: direction codes for every non-boundary endpoint.
    if ep_with_code is not None and len(ep_with_code) > 0:
        ax_dir.imshow(skel_rgb)
        ARROW_DIRS = {
            0: (0, -1), 1: (1, -1), 2: (1, 0), 3: (1, 1),
            4: (0, 1), 5: (-1, 1), 6: (-1, 0), 7: (-1, -1),
        }
        connected_keys = set()
        for ep1, ep2 in connections:
            connected_keys.add((int(ep1['x']), int(ep1['y'])))
            connected_keys.add((int(ep2['x']), int(ep2['y'])))
        for ep in ep_with_code:
            x, y = int(ep['x']), int(ep['y'])
            code = ep.get('code')
            lid = ep['line_id']
            is_connected = (x, y) in connected_keys
            color = 'red' if is_connected else '#888888'
            ax_dir.plot(x, y, 'o', color=color, markersize=11,
                        markeredgecolor='white', markeredgewidth=1.5)
            ax_dir.text(x + 9, y - 9, f"L{lid}",
                        color=color, fontsize=7, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.15',
                                  facecolor='white',
                                  edgecolor=color, linewidth=0.6))
            if code is not None and code in ARROW_DIRS:
                dx, dy = ARROW_DIRS[code]
                ax_dir.annotate('', xy=(x + dx * 14, y + dy * 14),
                                xytext=(x, y),
                                arrowprops=dict(arrowstyle='->', color=color, lw=1.6))
                ax_dir.text(x + 9, y + 8, f"c{code}",
                            color=color, fontsize=8, fontweight='bold',
                            bbox=dict(boxstyle='round,pad=0.2',
                                      facecolor='yellow',
                                      edgecolor=color, linewidth=0.8))
            else:
                ax_dir.text(x + 9, y + 8, "c?",
                            color='#aaaaaa', fontsize=8, fontweight='bold',
                            bbox=dict(boxstyle='round,pad=0.2',
                                      facecolor='lightyellow',
                                      edgecolor='#aaaaaa', linewidth=0.8))
        n_conn = sum(1 for ep in ep_with_code
                     if (int(ep['x']), int(ep['y'])) in connected_keys)
        ax_dir.set_title(
            f'[3] Direction Codes (red=connected {n_conn}/{len(ep_with_code)}, grey=orphan) - '
            f'arrow = code direction',
            fontsize=12)
        ax_dir.set_xlim(0, width), ax_dir.set_ylim(height, 0)
        ax_dir.set_xticks([]), ax_dir.set_yticks([])
        dir_legend = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='red',
                   markersize=10, markeredgecolor='white', label='Connected'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#888888',
                   markersize=10, markeredgecolor='white', label='Orphan'),
            Line2D([0], [0], marker='>', color='black', markersize=8, label='Direction'),
            Patch(facecolor='yellow', edgecolor='black', label='Code label'),
        ]
        ax_dir.legend(handles=dir_legend, loc='lower right', fontsize=9)
    else:
        ax_dir.text(0.5, 0.5, '(no direction code data)',
                    ha='center', va='center', transform=ax_dir.transAxes)
        ax_dir.axis('off')

    fig.suptitle('Simulated Annealing + Backtrack Cubic Spline Connections', fontsize=16, y=1.01)
    plt.tight_layout()

    # Save a hi-res direction-code-only figure (no connection lines).
    if ep_with_code is not None and len(ep_with_code) > 0:
        try:
            fig2, ax2 = plt.subplots(figsize=(14, 10))
            ax2.imshow(skel_rgb)
            ARROW_DIRS = {
                0: (0, -1), 1: (1, -1), 2: (1, 0), 3: (1, 1),
                4: (0, 1), 5: (-1, 1), 6: (-1, 0), 7: (-1, -1),
            }
            CODE_NAMES = {
                0: 'up', 1: 'UR', 2: 'R', 3: 'DR',
                4: 'down', 5: 'DL', 6: 'L', 7: 'UL',
            }
            connected_keys = set()
            for ep1, ep2 in connections:
                connected_keys.add((int(ep1['x']), int(ep1['y'])))
                connected_keys.add((int(ep2['x']), int(ep2['y'])))
            for ep in ep_with_code:
                x, y = int(ep['x']), int(ep['y'])
                code = ep.get('code')
                lid = ep['line_id']
                is_connected = (x, y) in connected_keys
                color = 'red' if is_connected else '#666666'
                ax2.plot(x, y, 'o', color=color, markersize=8,
                         markeredgecolor='white', markeredgewidth=1.5)
                if code is not None and code in ARROW_DIRS:
                    dx, dy = ARROW_DIRS[code]
                    ax2.annotate('', xy=(x + dx * 14, y + dy * 14),
                                 xytext=(x, y),
                                 arrowprops=dict(arrowstyle='->', color=color, lw=1.8))
                    label_text = f"L{lid}/{CODE_NAMES.get(code, '?')}(c{code})"
                else:
                    label_text = f"L{lid}/?(c?)"
                ax2.text(x + 9, y + 5, label_text, color=color,
                         fontsize=7, fontweight='bold',
                         bbox=dict(boxstyle='round,pad=0.18',
                                   facecolor='lightyellow' if code is not None else '#eeeeee',
                                   edgecolor=color, linewidth=0.8))
            n_conn = sum(1 for ep in ep_with_code
                         if (int(ep['x']), int(ep['y'])) in connected_keys)
            ax2.set_title(
                f'Direction Codes Detail (red=connected {n_conn}/{len(ep_with_code)}, '
                f'grey=orphan)\nc0=up c1=UR c2=R c3=DR c4=down c5=DL c6=L c7=UL',
                fontsize=13)
            ax2.set_xlim(0, width), ax2.set_ylim(height, 0)
            ax2.set_xticks([]), ax2.set_yticks([])
            dir_path = os.path.join(output_dir, 'annealing_direction_codes.png')
            buf_dir = BytesIO()
            plt.savefig(buf_dir, format='png', dpi=120, bbox_inches='tight')
            plt.close(fig2)
            buf_dir.seek(0)
            from pathlib import Path
            Path(dir_path).write_bytes(buf_dir.read())
            print(f"Saved: {dir_path}")
        except Exception as _e:
            print(f"Warning: direction codes PNG save failed ({_e}), skipping")
            try:
                plt.close(fig2)
            except Exception:
                pass

    from io import BytesIO
    try:
        debug_path = os.path.join(output_dir, 'annealing_debug.png')
        buf_img = BytesIO()
        plt.savefig(buf_img, format='png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        buf_img.seek(0)
        from pathlib import Path
        Path(debug_path).write_bytes(buf_img.read())
        print(f"Saved: {debug_path}")
    except Exception as _e:
        print(f"Warning: debug PNG save failed ({_e}), skipping")
        try:
            plt.close(fig)
        except Exception:
            pass

    # 4.5 White-background, black-line "clean" view.
    clean_img = np.ones((height, width), dtype=np.uint8) * 255
    for line_id, line_info in skeleton_data['lines'].items():
        for p in line_info['pixel_coords']:
            x, y = int(p['x']), int(p['y'])
            if 0 <= y < height and 0 <= x < width:
                clean_img[y, x] = 0
    for i, (ep1, ep2) in enumerate(connections):
        vec1 = get_trace_direction(ep1, lines_data)
        vec2 = get_trace_direction(ep2, lines_data)
        dist_for_pts = calculate_distance(ep1, ep2)
        num_pts = max(100, int(dist_for_pts * 5))
        spline_points = connect_with_visible_bezier(
            ep1, ep2, vec1, vec2, num_points=num_pts,
        )
        for sp in spline_points:
            sx, sy = int(sp['x']), int(sp['y'])
            if 0 <= sy < height and 0 <= sx < width:
                clean_img[sy, sx] = 0

    clean_path = os.path.join(output_dir, 'annealing_clean.png')
    try:
        ok = cv2.imwrite(clean_path, clean_img)
    except Exception:
        ok = False
    if not ok or not os.path.exists(clean_path):
        success, buf = cv2.imencode('.png', clean_img)
        if success:
            try:
                from pathlib import Path
                Path(clean_path).write_bytes(buf.tobytes())
            except Exception as _e:
                print(f"Warning: {clean_path} save failed ({_e})")
        else:
            print(f"ERROR: cannot save {clean_path}")
    print(f"Saved: {clean_path}")

    # 5. Save connection JSON.
    json_path = os.path.join(output_dir, 'annealing_connections.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(connection_results, f, ensure_ascii=False, indent=2)
    print(f"Saved: {json_path}")

    # 5.5 Also emit a breakpoint_v4_result-compatible JSON for downstream
    # tools (merge_skeleton_and_connections.py / extract_connected_lines.py).
    # Field mapping:
    #   conn_id    <- id
    #   line1/2    <- endpoint1/2.line_id
    #   ep1/2      <- endpoint1/2 (x, y)
    #   code1/2    <- direction code from ep_with_code
    #   is_direct  <- True (SA selected pair)
    #   curve_pixels <- pixel_coords
    v4_compatible = {
        'metadata': {
            'image_size': {'width': width, 'height': height},
            'method': 'simulated_annealing'
        },
        'total_connections': len(connections),
        'connections': []
    }
    for conn in connection_results['connections']:
        ep1_xy = (int(conn['endpoint1']['x']), int(conn['endpoint1']['y']))
        ep2_xy = (int(conn['endpoint2']['x']), int(conn['endpoint2']['y']))
        v4_compatible['connections'].append({
            'conn_id': conn['id'],
            'line1': conn['endpoint1']['line_id'],
            'line2': conn['endpoint2']['line_id'],
            'ep1': {'x': ep1_xy[0], 'y': ep1_xy[1]},
            'ep2': {'x': ep2_xy[0], 'y': ep2_xy[1]},
            'code1': code_lookup.get(ep1_xy),
            'code2': code_lookup.get(ep2_xy),
            'distance': conn['distance'],
            'is_direct': True,
            'curve_pixels': [
                {'x': int(p['x']), 'y': int(p['y'])}
                for p in conn['pixel_coords']
            ],
        })
    v4_json_path = os.path.join(output_dir, 'sa_result.json')
    with open(v4_json_path, 'w', encoding='utf-8') as f:
        json.dump(v4_compatible, f, ensure_ascii=False, indent=2)
    print(f"Saved: {v4_json_path}")

    return connection_results


def main(data_dir=None, endpoints_filename='connected_lines_v4.json',
         merged_filename='merged_skeleton_data_v4.json'):
    """End-to-end SA pipeline.

    Parameters
    ----------
    data_dir : str or None, optional
        Folder containing the input JSON files. ``None`` falls back to
        ``<script_dir>/data/test_2025`` (backwards compatibility).
    endpoints_filename : str, optional
        Filename of the connected-lines JSON
        (default ``'connected_lines_v4.json'``).
    merged_filename : str, optional
        Filename of the merged-skeleton JSON
        (default ``'merged_skeleton_data_v4.json'``).
    """
    if data_dir is None:
        # Default test_2025 (backwards compatibility).
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 'data', 'test_2025')
    endpoints_path = os.path.join(data_dir, endpoints_filename)
    skeleton_path = os.path.join(data_dir, merged_filename)
    output_dir = data_dir

    # Load connected-lines data.
    endpoints_data = load_json(endpoints_path)
    endpoints_lines = endpoints_data['lines']
    width = endpoints_data['metadata']['image_size']['width']
    height = endpoints_data['metadata']['image_size']['height']

    # Load merged-skeleton data (full pixel coordinates).
    skeleton_data = load_json(skeleton_path)
    ms_lines = skeleton_data['lines']

    # Map connected_lines line_ids to merged_skeleton line_ids.
    # ``connected_lines.json`` only carries truncated pixel_coords (10 px);
    # the merged skeleton has the full pixel set, so we match by whether
    # the endpoints are present in a line's pixel set.
    cl_id_to_ms_id = {}
    for cl_id, L in endpoints_lines.items():
        fp = (L['first_point']['x'], L['first_point']['y'])
        sp = (L['second_point']['x'], L['second_point']['y'])
        cl_pixels = {(int(p['x']), int(p['y']))
                     for p in L.get('pixel_coords', [])}
        found = None
        # Prefer: both endpoints are in the same merged-skeleton line.
        for ms_id, M in ms_lines.items():
            coords = M.get('pixel_coords', [])
            ms_pixel_set = {(int(p['x']), int(p['y'])) for p in coords}
            if fp in ms_pixel_set and sp in ms_pixel_set:
                found = ms_id
                break
        if found is None:
            # Fallback: one endpoint + best pixel overlap.
            best_score = 0
            best_ms = None
            for ms_id, M in ms_lines.items():
                coords = M.get('pixel_coords', [])
                ms_pixel_set = {(int(p['x']), int(p['y'])) for p in coords}
                if fp in ms_pixel_set or sp in ms_pixel_set:
                    score = sum(1 for p in cl_pixels if p in ms_pixel_set)
                    if score > best_score:
                        best_score = score
                        best_ms = ms_id
            found = best_ms
        cl_id_to_ms_id[cl_id] = found if found else cl_id

    # Enrich ``endpoints_lines`` with the full pixel coordinates.
    n_enriched = 0
    for cl_id, L in endpoints_lines.items():
        ms_id = cl_id_to_ms_id.get(cl_id)
        if ms_id and ms_id in ms_lines:
            full_pixels = ms_lines[ms_id].get('pixel_coords', [])
            if full_pixels:
                L['pixel_coords'] = full_pixels
                n_enriched += 1
        L['ms_id'] = ms_id

    print(f"Image size: {width} x {height}")
    print(f"CL->MS line ID mapping done, {n_enriched}/{len(endpoints_lines)} lines enriched with full pixels")

    # Find non-boundary endpoints.
    endpoints = find_non_boundary_endpoints(endpoints_lines, width, height)
    print(f"Non-boundary endpoints: {len(endpoints)}")
    print()
    for i, ep in enumerate(endpoints):
        print(f"  {i}: ({ep['x']}, {ep['y']}) - Line {ep['line_id']} {ep['endpoint_type']}")
    print()

    # Reload skeleton and assign pixel_coords to every endpoint.
    skeleton_data = load_json(skeleton_path)
    lines_data = skeleton_data['lines']

    for ep in endpoints:
        line_id = ep['line_id']
        if line_id in endpoints_lines:
            ep['pixel_coords'] = endpoints_lines[line_id].get('pixel_coords', [])
        elif line_id in lines_data:
            ep['pixel_coords'] = lines_data[line_id]['pixel_coords']
        else:
            ep['pixel_coords'] = []

    # Phase 1: direction-coded candidate graph (max 150 px, v4 rules).
    print("=" * 60)
    print("Phase 1: Building direction-based candidate graph (max 150 px)")
    print("=" * 60)
    ep_with_code, candidates = build_direction_candidates(endpoints, endpoints_lines, max_distance=150)
    for i, ep in enumerate(ep_with_code):
        n_cands = len(candidates[i])
        cands_str = ', '.join(f'#{c[0]}(d={c[1]:.1f},s={c[2]})' for c in candidates[i][:5])
        if len(candidates[i]) > 5:
            cands_str += f' ...(+{len(candidates[i])-5})'
        code_name = CODE_NAMES.get(ep['code'], '?')
        print(f"  EP{i} L{ep['line_id']} ({ep['x']},{ep['y']}) code={code_name}: {n_cands} candidates [{cands_str}]")
    total_pairs = sum(len(v) for v in candidates.values()) // 2
    print(f"  Total candidate pairs: {total_pairs}")
    print()

    # Phase 2: simulated annealing (perturbations constrained to the candidate graph).
    print("=" * 60)
    print("Phase 2: Simulated Annealing on candidate graph")
    print("=" * 60)
    random.seed(42)
    connections = simulated_annealing(
        endpoints, endpoints_lines,
        T0=1000, T_end=1, alpha=0.95, beta=1,
        max_no_improve=100,
        candidates=candidates,
    )
    print()

    # Phase 3: post-connection intersection validation (hard rule).
    print("=" * 60)
    print("Phase 3: Post-connection intersection validation")
    print("=" * 60)
    if connections:
        invalid_pairs, skel_cross = post_check_no_intersection(connections, endpoints_lines)
        invalid_conns = set()
        for i, j in invalid_pairs:
            invalid_conns.add(i)
            invalid_conns.add(j)
        invalid_conns.update(skel_cross)
        valid_connections = [c for k, c in enumerate(connections) if k not in invalid_conns]
        print(f"  Original connections: {len(connections)}")
        if invalid_pairs:
            print(f"  Crossed pairs (curve x curve): {sorted(invalid_pairs)}")
        if skel_cross:
            print(f"  Curves crossing skeleton: {skel_cross}")
        print(f"  Invalid connections removed: {len(invalid_conns)}")
        print(f"  Valid connections kept: {len(valid_connections)}")
        connections = valid_connections
    print()

    print()
    print("Best connections:")
    for i, (ep1, ep2) in enumerate(connections):
        dist = calculate_distance(ep1, ep2)
        print(f"  {i + 1}: ({ep1['x']},{ep1['y']}) <-> ({ep2['x']},{ep2['y']}) dist={dist:.1f}")

    # Render and save:
    # - skeleton_data is used only as background;
    # - endpoints_lines is used for direction backtrace + cubic spline.
    connection_results = visualize_and_save_results(
        skeleton_data, connections, endpoints_lines, width, height, output_dir,
        ep_with_code=ep_with_code
    )

    return connection_results


if __name__ == '__main__':
    import sys
    # CLI: python simulated_annealing.py [data_dir] [endpoints_filename] [merged_filename]
    args = sys.argv[1:]
    data_dir_arg = args[0] if len(args) > 0 else None
    ep_fn = args[1] if len(args) > 1 else 'connected_lines_v4.json'
    ms_fn = args[2] if len(args) > 2 else 'merged_skeleton_data_v4.json'
    result = main(data_dir=data_dir_arg,
                  endpoints_filename=ep_fn, merged_filename=ms_fn)