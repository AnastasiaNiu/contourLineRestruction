# -*- coding: utf-8 -*-
"""
Breakpoint Detection and Connection Algorithm v4 - Debug Version
(Preserves all v3 logic + adds edge-filter before/after comparison visualization)

Changes relative to v3:
1. During endpoint extraction, keep ALL raw endpoints first (including None
   and on-edge endpoints), stored as ``all_raw_endpoints``.
2. Then apply the edge filter to obtain ``filtered_endpoints``.
3. Output two comparison sub-figures:
   - Sub-figure A: all raw endpoints (with global index + line id + endpoint
     role + on-edge status).
   - Sub-figure B: filtered endpoints (with global index).
4. Output two zoomed comparison sub-figures (A-zoom and B-zoom) for clarity.
5. Save two endpoint JSON files:
   - ``breakpoint_v4_all_endpoints.json``     - all raw endpoints (unfiltered).
   - ``breakpoint_v4_filtered_endpoints.json`` - filtered endpoints.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import csv
from io import BytesIO


# ============================================================
# Helper Functions
# ============================================================
def imwrite_unicode(path, img):
    """Save an image to a path that may contain non-ASCII (e.g. Chinese) characters.

    Parameters
    ----------
    path : str
        Destination file path. The extension determines the image format and
        encoding parameters (``.png`` uses lossless compression, ``.jpg`` /
        ``.jpeg`` uses quality 95).
    img : numpy.ndarray
        Image array in the format expected by ``cv2.imencode``.

    Returns
    -------
    bool
        ``True`` on success, ``False`` if encoding/writing failed.
    """
    try:
        ext = os.path.splitext(path)[1].lower()
        if ext == '.png':
            encode_param = [int(cv2.IMWRITE_PNG_COMPRESSION), 0]
        elif ext in ['.jpg', '.jpeg']:
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 95]
        else:
            encode_param = []
        result, encoded = cv2.imencode(ext, img, encode_param)
        if result:
            encoded.tofile(path)
            return True
        return False
    except Exception as e:
        print("    Warning: imwrite_unicode failed: %s" % e)
        return False


def is_on_edge(point, img_width, img_height, edge_margin=5):
    """Test whether a pixel lies within ``edge_margin`` pixels of any image border.

    Parameters
    ----------
    point : tuple[int, int]
        ``(x, y)`` pixel coordinate.
    img_width : int
        Image width in pixels.
    img_height : int
        Image height in pixels.
    edge_margin : int, optional
        Border thickness (in pixels) considered "on the edge". Default ``5``.

    Returns
    -------
    bool
        ``True`` if the point lies within ``edge_margin`` of the left, right,
        top or bottom border.
    """
    x, y = point
    return (x < edge_margin or x >= img_width - edge_margin or
            y < edge_margin or y >= img_height - edge_margin)


def bresenham_line(x0, y0, x1, y1):
    """Generate the integer pixel coordinates along a Bresenham line.

    Parameters
    ----------
    x0, y0 : int
        Start pixel coordinates.
    x1, y1 : int
        End pixel coordinates (inclusive).

    Returns
    -------
    list[tuple[int, int]]
        Sequence of ``(x, y)`` pixels from ``(x0, y0)`` to ``(x1, y1)``,
        inclusive of both endpoints.
    """
    points = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    while True:
        points.append((x0, y0))
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy
    return points


def line_intersection(p1, v1, p2, v2):
    """Find the intersection of two parametric rays starting at ``p1`` and ``p2``.

    Parameters
    ----------
    p1 : tuple[float, float]
        Origin of the first ray.
    v1 : tuple[float, float]
        Direction vector of the first ray.
    p2 : tuple[float, float]
        Origin of the second ray.
    v2 : tuple[float, float]
        Direction vector of the second ray.

    Returns
    -------
    tuple[float, float] or None
        ``(ix, iy)`` intersection point when both parameter ranges
        ``(-5, 25)`` are satisfied and the lines are not parallel.
        Returns ``None`` for parallel or out-of-range cases.
    """
    det = v1[0] * (-v2[1]) - v1[1] * (-v2[0])
    if abs(det) < 1e-10:
        return None
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    t = (dx * (-v2[1]) - dy * (-v2[0])) / det
    s = (v1[0] * dy - v1[1] * dx) / det
    if -5 < t < 25 and -5 < s < 25:
        ix = p1[0] + t * v1[0]
        iy = p1[1] + t * v1[1]
        return (ix, iy)
    return None


def compute_back_extension_intersection(ep1_pos, ep1_vec, ep2_pos, ep2_vec):
    """Compute the intersection of the back-extensions of two endpoint tangents.

    The back-extension of an endpoint runs opposite to its forward direction,
    modelling the line that "would have continued" beyond the endpoint. A
    well-formed match should produce a small, in-front intersection distance.

    Parameters
    ----------
    ep1_pos : tuple[float, float]
        First endpoint position ``(x, y)``.
    ep1_vec : tuple[float, float]
        First endpoint tangent direction vector (unit length expected).
    ep2_pos : tuple[float, float]
        Second endpoint position ``(x, y)``.
    ep2_vec : tuple[float, float]
        Second endpoint tangent direction vector (unit length expected).

    Returns
    -------
    intersection : tuple[float, float] or None
        Intersection point, or ``None`` if the back-extensions are parallel /
        out of range / could not be solved.
    t1 : float or None
        Parameter along the first back-extension.
    t2 : float or None
        Parameter along the second back-extension.
    valid : bool
        ``True`` when ``t1`` and ``t2`` fall inside the valid range
        ``(-5, 100)`` and the linear system was solved.
    """
    back_vec1 = (-ep1_vec[0], -ep1_vec[1])
    back_vec2 = (-ep2_vec[0], -ep2_vec[1])

    A = np.array([[back_vec1[0], -back_vec2[0]],
                  [back_vec1[1], -back_vec2[1]]])
    diff = np.array([ep2_pos[0] - ep1_pos[0], ep2_pos[1] - ep1_pos[1]])

    det = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
    if abs(det) < 1e-10:
        return None, None, None, False

    try:
        t = np.linalg.solve(A, diff)
    except Exception:
        return None, None, None, False

    t1, t2 = t[0], t[1]

    if t1 < -5 or t1 > 100 or t2 < -5 or t2 > 100:
        return None, t1, t2, False

    ix = ep1_pos[0] + t1 * back_vec1[0]
    iy = ep1_pos[1] + t1 * back_vec1[1]
    intersection = (ix, iy)

    return intersection, t1, t2, True


def evaluate_match_quality(ep1_data, ep2_data, dist, use_back_extension=True):
    """Score a candidate endpoint match combining distance and back-extension.

    A weighted sum of a normalized distance score and a back-extension score
    is returned when back-extension evaluation is enabled and valid; otherwise
    the distance-only score is returned.

    Parameters
    ----------
    ep1_data : dict
        Must contain key ``'ep'`` with ``(x, y)`` and optionally
        ``'direction_vec'`` with a unit tangent.
    ep2_data : dict
        Same contract as ``ep1_data``.
    dist : float
        Euclidean distance between the two endpoints in pixels.
    use_back_extension : bool, optional
        If ``True`` (default), augment the distance score with the
        back-extension intersection score.

    Returns
    -------
    score : float
        Combined match quality score (lower is better).
    details : dict
        Diagnostic dictionary with keys ``'dist'``, ``'dist_score'``,
        and (when available) ``'intersection'``, ``'t1'``, ``'t2'``,
        ``'back_ext_score'``, ``'combined_score'``.
    """
    dist_score = dist / 50.0

    if not use_back_extension:
        return dist_score, {'dist': dist, 'dist_score': dist_score}

    vec1 = ep1_data.get('direction_vec', (0, 0))
    vec2 = ep2_data.get('direction_vec', (0, 0))

    if vec1 == (0, 0) or vec2 == (0, 0):
        return dist_score, {'dist': dist, 'dist_score': dist_score}

    int_pt, t1, t2, valid = compute_back_extension_intersection(
        (ep1_data['ep'][0], ep1_data['ep'][1]),
        vec1,
        (ep2_data['ep'][0], ep2_data['ep'][1]),
        vec2
    )

    if valid and int_pt is not None:
        back_ext_score = (t1 + t2) / 100.0
        alpha = 0.4
        beta = 0.6
        combined_score = alpha * dist_score + beta * back_ext_score
        details = {
            'dist': dist,
            'dist_score': dist_score,
            'intersection': int_pt,
            't1': t1,
            't2': t2,
            'back_ext_score': back_ext_score,
            'combined_score': combined_score
        }
        return combined_score, details
    else:
        return dist_score, {'dist': dist, 'dist_score': dist_score}


def get_endpoint_direction_vector(ep_data):
    """Convert a direction code into its canonical 2D unit vector.

    Parameters
    ----------
    ep_data : dict
        Endpoint record; only the ``'code'`` field is read.

    Returns
    -------
    tuple[float, float]
        Direction unit vector ``(dx, dy)``, or ``(0, 0)`` if the code is
        ``None`` / unknown.
    """
    code = ep_data.get('code')
    if code is None:
        return (0, 0)
    code_vectors = {
        0: (0, 1),
        1: (1, -1),
        2: (1, 0),
        3: (1, 1),
        4: (0, -1),
        5: (-1, 1),
        6: (-1, 0),
        7: (-1, -1),
    }
    return code_vectors.get(code, (0, 0))


def segments_intersect(p1, p2, p3, p4):
    """Test whether two line segments intersect and return the intersection point.

    Parameters
    ----------
    p1, p2 : tuple[float, float]
        Endpoints of the first segment.
    p3, p4 : tuple[float, float]
        Endpoints of the second segment.

    Returns
    -------
    intersection : tuple[float, float] or None
        ``(x, y)`` of the intersection when the segments cross in their
        interiors; otherwise ``None``.
    intersects : bool
        ``True`` if a strict interior intersection was found.
    """
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4

    def bbox(p_a, p_b):
        return (min(p_a[0], p_b[0]), min(p_a[1], p_b[1]),
                max(p_a[0], p_b[0]), max(p_a[1], p_b[1]))

    bb1 = bbox(p1, p2)
    bb2 = bbox(p3, p4)

    if bb1[0] > bb2[2] or bb2[0] > bb1[2]:
        return None, False
    if bb1[1] > bb2[3] or bb2[1] > bb1[3]:
        return None, False

    def orient(pa, pb, pc):
        return (pb[0] - pa[0]) * (pc[1] - pa[1]) - (pb[1] - pa[1]) * (pc[0] - pa[0])

    d1 = orient(p3, p4, p1)
    d2 = orient(p3, p4, p2)
    d3 = orient(p1, p2, p3)
    d4 = orient(p1, p2, p4)

    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        denom = (p1[0] - p2[0]) * (p3[1] - p4[1]) - (p1[1] - p2[1]) * (p3[0] - p4[0])
        if abs(denom) > 1e-10:
            t = ((p1[0] - p3[0]) * (p3[1] - p4[1]) - (p1[1] - p3[1]) * (p3[0] - p4[0])) / denom
            ix = p1[0] + t * (p2[0] - p1[0])
            iy = p1[1] + t * (p2[1] - p1[1])
            return (ix, iy), True

    def on_segment(pa, pb, pc):
        min_x, max_x = min(pa[0], pb[0]), max(pa[0], pb[0])
        min_y, max_y = min(pa[1], pb[1]), max(pa[1], pb[1])
        return (min_x < pc[0] < max_x or min_x <= pc[0] <= max_x) and \
               (min_y < pc[1] < max_y or min_y <= pc[1] <= max_y)

    endpoints_same = (p1 == p3 or p1 == p4 or p2 == p3 or p2 == p4)
    if endpoints_same:
        return None, False

    return None, False


def check_cross_connections(matches, endpoint_list, max_rematch_dist=60, log_lines=None):
    """Remove connections whose paths cross each other and rematch their endpoints.

    The procedure iterates over every pair of current connections, flags
    segment-cross / path-overlap crossings, drops the offending matches,
    and then re-tries pairing the now-free endpoints using the standard
    direction-compatibility rules.

    Parameters
    ----------
    matches : list[dict]
        Current candidate connections. Each dict must contain ``'ep1'``,
        ``'ep2'``, ``'line1'``, ``'line2'``.
    endpoint_list : list[dict]
        Detailed endpoint records (must contain ``'lid'``, ``'ep'``,
        ``'vec'``, ``'code'`` and ``'global_idx'``).
    max_rematch_dist : float, optional
        Maximum allowed rematch distance (pixels). Default ``60``.
    log_lines : list[str] or None, optional
        If provided, human-readable log lines are appended to it.

    Returns
    -------
    new_matches : list[dict]
        Updated list of connections after cross-removal and rematching.
    cross_info : list[dict]
        Records describing each detected crossing (useful for visualization).
    """
    if len(matches) < 2:
        return matches, []

    connections = []
    for m in matches:
        ep1 = tuple(m['ep1'])
        ep2 = tuple(m['ep2'])
        line1 = m['line1']
        line2 = m['line2']
        pts = bresenham_line(ep1[0], ep1[1], ep2[0], ep2[1])
        connections.append({
            'idx': len(connections),
            'match': m,
            'ep1': ep1, 'ep2': ep2,
            'line1': line1, 'line2': line2,
            'points': set(pts),
            'segments': [(pts[i], pts[i + 1]) for i in range(len(pts) - 1)]
        })

    cross_info = []
    for i in range(len(connections)):
        for j in range(i + 1, len(connections)):
            conn1, conn2 = connections[i], connections[j]

            if (conn1['ep1'] == conn2['ep1'] or conn1['ep1'] == conn2['ep2'] or
                conn1['ep2'] == conn2['ep1'] or conn1['ep2'] == conn2['ep2']):
                continue

            for seg1 in conn1['segments']:
                for seg2 in conn2['segments']:
                    pt, intersects = segments_intersect(seg1[0], seg1[1], seg2[0], seg2[1])
                    if intersects:
                        cross_info.append({
                            'idx1': i, 'idx2': j,
                            'line1_pair': (conn1['line1'], conn1['line2']),
                            'line2_pair': (conn2['line1'], conn2['line2']),
                            'ep1': (conn1['ep1'], conn1['ep2']),
                            'ep2': (conn2['ep1'], conn2['ep2']),
                            'intersection': pt,
                            'type': 'segment_cross'
                        })
                        break

            if not any(c['idx1'] == i and c['idx2'] == j for c in cross_info):
                common_points = conn1['points'] & conn2['points']
                shared_endpoints = {conn1['ep1'], conn1['ep2'], conn2['ep1'], conn2['ep2']}
                mid_common = common_points - shared_endpoints
                if mid_common:
                    pt = list(mid_common)[0]
                    cross_info.append({
                        'idx1': i, 'idx2': j,
                        'line1_pair': (conn1['line1'], conn1['line2']),
                        'line2_pair': (conn2['line1'], conn2['line2']),
                        'ep1': (conn1['ep1'], conn1['ep2']),
                        'ep2': (conn2['ep1'], conn2['ep2']),
                        'intersection': pt,
                        'type': 'path_overlap'
                    })

    if not cross_info:
        return matches, []

    if log_lines is not None:
        log_lines.append("")
        log_lines.append("=" * 100)
        log_lines.append("[Cross Check] %d crossing connection pairs found!" % len(cross_info))
        log_lines.append("=" * 100)
        for ci, cross in enumerate(cross_info):
            cross_type = cross.get('type', 'unknown')
            log_lines.append("  Cross %d: conn %d (L%d-L%d) vs conn %d (L%d-L%d) at (%s) [%s]" % (
                ci + 1,
                cross['idx1'], cross['line1_pair'][0], cross['line1_pair'][1],
                cross['idx2'], cross['line2_pair'][0], cross['line2_pair'][1],
                "(%.1f, %.1f)" % (cross['intersection'][0], cross['intersection'][1])
                if isinstance(cross['intersection'], tuple) else str(cross['intersection']),
                cross_type
            ))

    endpoints_to_rematch = set()
    for cross in cross_info:
        conn1 = connections[cross['idx1']]
        conn2 = connections[cross['idx2']]
        endpoints_to_rematch.add((conn1['line1'], conn1['ep1'][0], conn1['ep1'][1]))
        endpoints_to_rematch.add((conn1['line2'], conn1['ep2'][0], conn1['ep2'][1]))
        endpoints_to_rematch.add((conn2['line1'], conn2['ep1'][0], conn2['ep1'][1]))
        endpoints_to_rematch.add((conn2['line2'], conn2['ep2'][0], conn2['ep2'][1]))

    if log_lines is not None:
        log_lines.append("")
        log_lines.append("Endpoints to rematch: %d" % len(endpoints_to_rematch))
        for ep_key in endpoints_to_rematch:
            log_lines.append("  Line%d (%d,%d)" % ep_key)

    ep_map = {}
    for ep in endpoint_list:
        key = (ep['lid'], ep['ep'][0], ep['ep'][1])
        ep_map[key] = ep

    indices_to_remove = set()
    for cross in cross_info:
        indices_to_remove.add(cross['idx1'])
        indices_to_remove.add(cross['idx2'])

    new_matches = [m for i, m in enumerate(matches) if i not in indices_to_remove]
    used_endpoints = set()
    for m in new_matches:
        ep1 = tuple(m['ep1'])
        ep2 = tuple(m['ep2'])
        used_endpoints.add((m['line1'], ep1[0], ep1[1]))
        used_endpoints.add((m['line2'], ep2[0], ep2[1]))

    if log_lines is not None:
        log_lines.append("")
        log_lines.append("Removed %d crossing connections, kept %d" %
                         (len(indices_to_remove), len(new_matches)))

    rematched_count = 0
    remaining_eps = []

    for ep_key in endpoints_to_rematch:
        if ep_key not in used_endpoints:
            if ep_key in ep_map:
                remaining_eps.append(ep_map[ep_key])

    for ep1_data in remaining_eps[:]:
        ep1_key = (ep1_data['lid'], ep1_data['ep'][0], ep1_data['ep'][1])
        if ep1_key in used_endpoints:
            continue
        if ep1_data['code'] is None:
            continue

        code1 = ep1_data['code']
        best = None
        best_dist = float('inf')
        best_ep2 = None

        for ep2_data in remaining_eps:
            ep2_key = (ep2_data['lid'], ep2_data['ep'][0], ep2_data['ep'][1])
            if ep1_key == ep2_key:
                continue
            if ep1_data['lid'] == ep2_data['lid']:
                continue
            if ep2_key in used_endpoints:
                continue
            if ep2_data['code'] is None:
                continue

            code2 = ep2_data['code']
            dist = np.sqrt((ep1_data['ep'][0] - ep2_data['ep'][0]) ** 2 +
                           (ep1_data['ep'][1] - ep2_data['ep'][1]) ** 2)

            if dist > max_rematch_dist:
                continue

            compat1 = code1 in DIRECTION_MATCH_RULES.get(code2, set())
            compat2 = code2 in DIRECTION_MATCH_RULES.get(code1, set())

            if compat1 and compat2 and dist < best_dist:
                best_dist = dist
                best = code2
                best_ep2 = ep2_data

        if best_ep2 is not None:
            ep2_key = (best_ep2['lid'], best_ep2['ep'][0], best_ep2['ep'][1])
            used_endpoints.add(ep1_key)
            used_endpoints.add(ep2_key)
            remaining_eps.remove(best_ep2)

            new_matches.append({
                'line1': ep1_data['lid'], 'ep1': ep1_data['ep'],
                'vec1': ep1_data['vec'], 'code1': ep1_data['code'],
                'line2': best_ep2['lid'], 'ep2': best_ep2['ep'],
                'vec2': best_ep2['vec'], 'code2': best,
                'distance': best_dist, 'is_direct': False,
                'rematched': True
            })
            rematched_count += 1

            if log_lines is not None:
                log_lines.append(
                    "  Rematch: Line%d(%d,%d,code=%s) <-> Line%d(%d,%d,code=%s) dist=%.1f" % (
                        ep1_data['lid'], ep1_data['ep'][0], ep1_data['ep'][1], CODE_NAMES.get(code1, '?'),
                        best_ep2['lid'], best_ep2['ep'][0], best_ep2['ep'][1], CODE_NAMES.get(best, '?'),
                        best_dist
                    ))

    if log_lines is not None:
        log_lines.append("")
        log_lines.append("Rematch complete, %d new connections added" % rematched_count)
        log_lines.append("Final connection count: %d" % len(new_matches))

    return new_matches, cross_info


# ============================================================
# Direction Code System
# ============================================================
def get_direction_code(x0, y0, x1, y1):
    """Convert a 2D vector into one of the 8 octant direction codes.

    The 8 codes are ordered clockwise starting from "up" (code 0):
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
    if dx < 0 and dy < 0:
        return 7
    elif dx == 0 and dy < 0:
        return 0
    elif dx > 0 and dy < 0:
        return 1
    elif dx < 0 and dy == 0:
        return 6
    elif dx > 0 and dy == 0:
        return 2
    elif dx < 0 and dy > 0:
        return 5
    elif dx == 0 and dy > 0:
        return 4
    else:
        return 3


# Direction match compatibility: each key code lists the set of codes that
# are geometrically compatible with it. For a valid match both
# ``code1 in DIRECTION_MATCH_RULES[code2]`` and the reverse must hold.
DIRECTION_MATCH_RULES = {
    0: {3, 4, 5},
    1: {3, 5, 7},
    2: {6},
    3: {0, 1, 5, 7},
    4: {0},
    5: {0, 1, 3, 7},
    6: {2},
    7: {1, 3, 5},
}

# Human-readable names for the 8 direction codes.
CODE_NAMES = {
    0: 'up',
    1: 'up-right',
    2: 'right',
    3: 'down-right',
    4: 'down',
    5: 'down-left',
    6: 'left',
    7: 'up-left',
}


# ============================================================
# Endpoint Extraction (v4 refactor: keep all raw endpoints)
# ============================================================
def extract_endpoints_with_flags(line_data, img_width, img_height, edge_margin=5):
    """Detect the two endpoints of a single skeleton line and flag on-edge ones.

    A pixel is an endpoint if exactly one of its 8-neighbors is also part of
    the line. ``truncate_edge_extensions`` guarantees that on-edge pixels are
    8-connected to the interior, so this rule is sufficient for edge pixels
    too; no extra "isolated edge pixel" handling is required.

    Parameters
    ----------
    line_data : dict
        Skeleton line record containing ``'pixel_coords'`` as a list of
        ``(x, y)`` tuples / lists.
    img_width : int
        Image width in pixels.
    img_height : int
        Image height in pixels.
    edge_margin : int, optional
        Border thickness (pixels) treated as "edge". Default ``5``.

    Returns
    -------
    dict or None
        ``None`` when the line has fewer than 2 pixels. Otherwise a dict with
        keys ``ep1``, ``ep2``, ``ep1_on_edge``, ``ep2_on_edge``,
        ``ep1_neighbors``, ``ep2_neighbors``. ``ep2`` is ``None`` when only
        one endpoint is detected.
    """
    coords = line_data['pixel_coords']
    if len(coords) < 2:
        return None
    coord_set = set((x, y) for x, y in coords)
    endpoints = []
    edge_flags = []
    neighbor_counts = []

    for x, y in coords:
        neighbors = 0
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                if (x + dx, y + dy) in coord_set:
                    neighbors += 1
        if neighbors == 1:
            endpoints.append((x, y))
            edge_flags.append(is_on_edge((x, y), img_width, img_height, edge_margin))
            neighbor_counts.append(neighbors)

    if len(endpoints) >= 2:
        return {
            'ep1': endpoints[0],
            'ep2': endpoints[1],
            'ep1_on_edge': edge_flags[0],
            'ep2_on_edge': edge_flags[1],
            'ep1_neighbors': neighbor_counts[0],
            'ep2_neighbors': neighbor_counts[1],
        }
    elif len(endpoints) == 1:
        return {
            'ep1': endpoints[0],
            'ep2': None,
            'ep1_on_edge': edge_flags[0],
            'ep2_on_edge': False,
            'ep1_neighbors': neighbor_counts[0],
            'ep2_neighbors': 0,
        }
    return None


def get_edge_id(point, img_width, img_height, edge_margin=5):
    """Identify which image border a point lies closest to.

    Parameters
    ----------
    point : tuple[int, int]
        ``(x, y)`` pixel coordinate.
    img_width : int
        Image width in pixels.
    img_height : int
        Image height in pixels.
    edge_margin : int, optional
        Border thickness (pixels). Default ``5``.

    Returns
    -------
    int
        Border id: ``0``=left, ``1``=top, ``2``=right, ``3``=bottom,
        ``-1``=not on any border.
    """
    x, y = point
    if x < edge_margin:
        return 0
    elif y < edge_margin:
        return 1
    elif x >= img_width - edge_margin:
        return 2
    elif y >= img_height - edge_margin:
        return 3
    else:
        return -1


def edge_distance(point, img_width, img_height):
    """Distance from a point to its nearest image border (in pixels).

    Used as an "interior-ness" score: the larger the value, the deeper the
    point sits inside the image.

    Parameters
    ----------
    point : tuple[int, int]
        ``(x, y)`` pixel coordinate.
    img_width : int
        Image width in pixels.
    img_height : int
        Image height in pixels.

    Returns
    -------
    int
        Distance in pixels (minimum ``0``).
    """
    x, y = point
    return min(x, y, img_width - 1 - x, img_height - 1 - y)


def skeleton_distance(coord_set, start, end, max_steps=10):
    """Shortest BFS distance (in pixels) between two points along a skeleton.

    The search uses 8-connectivity and is bounded by ``max_steps``; the
    function returns ``None`` if either endpoint is missing from the set or
    no path of length ``<= max_steps`` exists.

    Parameters
    ----------
    coord_set : set[tuple[int, int]]
        The set of all skeleton pixel coordinates.
    start : tuple[int, int]
        BFS origin pixel.
    end : tuple[int, int]
        Target pixel.
    max_steps : int, optional
        Maximum BFS depth in pixels. Default ``10``.

    Returns
    -------
    int or None
        Number of steps from ``start`` to ``end`` along the skeleton, or
        ``None`` if unreachable / out of range.
    """
    if start == end:
        return 0
    if start not in coord_set or end not in coord_set:
        return None

    visited = {start}
    current_layer = [start]
    dist = 0
    while current_layer and dist < max_steps:
        dist += 1
        next_layer = []
        for pt in current_layer:
            x, y = pt
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = x + dx, y + dy
                    if (nx, ny) == end:
                        return dist
                    if (nx, ny) in coord_set and (nx, ny) not in visited:
                        visited.add((nx, ny))
                        next_layer.append((nx, ny))
        current_layer = next_layer
    return None


def merge_edge_endpoints_on_same_line(coords_list, endpoints_with_flags,
                                      img_width, img_height,
                                      edge_margin=5, max_skeleton_distance=4):
    """Collapse two close-by edge endpoints into one when they are line artifacts.

    When both endpoints of a line lie on the image border and their
    skeleton-connectivity distance is below ``max_skeleton_distance``, they
    are interpreted as a single endpoint whose line happens to follow the
    border; the more "interior" endpoint (more interior neighbors and a
    larger interior score) is kept.

    Parameters
    ----------
    coords_list : list[list[int] | tuple[int, int]]
        All skeleton pixel coordinates of the line.
    endpoints_with_flags : list[tuple[tuple[int, int], bool, int]]
        List of ``(point, on_edge, neighbor_count)`` tuples, length 1 or 2.
    img_width : int
        Image width in pixels.
    img_height : int
        Image height in pixels.
    edge_margin : int, optional
        Border thickness (pixels). Default ``5``.
    max_skeleton_distance : int, optional
        Maximum BFS distance for the merge decision. Default ``4``.

    Returns
    -------
    endpoints : list[dict]
        Either a single-element list with the kept endpoint, or the original
        endpoint records.
    merge_note : str
        Diagnostic description of the merge (``''`` if nothing was merged).
    """
    if not endpoints_with_flags:
        return [], ''

    coord_set = set((int(c[0]), int(c[1])) for c in coords_list)

    # Package the endpoints.
    eps = []
    for ep, on_edge, nb_count in endpoints_with_flags:
        edge_id = get_edge_id(ep, img_width, img_height, edge_margin)
        eps.append({
            'point': ep,
            'on_edge': on_edge,
            'neighbors': nb_count,
            'edge_id': edge_id,
            'interior_score': edge_distance(ep, img_width, img_height),
        })

    # Compute the number of interior neighbors for each endpoint - used to
    # choose which one is the "true" endpoint when merging.
    for ep in eps:
        x, y = ep['point']
        interior_neighbor_count = 0
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if (nx, ny) in coord_set:
                    # Whether the neighbor counts as "interior" depends on
                    # which border this endpoint belongs to.
                    if ep['edge_id'] == 0:    # left border: interior is x >= edge_margin
                        if nx >= edge_margin:
                            interior_neighbor_count += 1
                    elif ep['edge_id'] == 2:  # right border: interior is x < w-edge_margin
                        if nx < img_width - edge_margin:
                            interior_neighbor_count += 1
                    elif ep['edge_id'] == 1:  # top border: interior is y >= edge_margin
                        if ny >= edge_margin:
                            interior_neighbor_count += 1
                    elif ep['edge_id'] == 3:  # bottom border: interior is y < h-edge_margin
                        if ny < img_height - edge_margin:
                            interior_neighbor_count += 1
                    else:
                        interior_neighbor_count += 1
        ep['interior_neighbors'] = interior_neighbor_count

    # Decision: merge if both endpoints are on-edge and close on the skeleton.
    if len(eps) == 2 and eps[0]['on_edge'] and eps[1]['on_edge']:
        d = skeleton_distance(coord_set, eps[0]['point'], eps[1]['point'],
                              max_steps=max_skeleton_distance + 2)
        if d is not None and d <= max_skeleton_distance:
            best = max(eps, key=lambda e: (e['interior_neighbors'], e['interior_score']))
            other = eps[1] if best is eps[0] else eps[0]
            merge_note = ('merged edge-pair: %s <-> %s, skel_dist=%d <= %d, '
                          'kept %s (interior_neighbors=%d)') % (
                eps[0]['point'], eps[1]['point'], d, max_skeleton_distance,
                best['point'], best['interior_neighbors']
            )
            return [best], merge_note

    return eps, ''


# ============================================================
# Direction Code and Tangent Vector
# ============================================================
def get_endpoint_direction_code(coord_set, start_point, num_steps=30):
    """Compute the local / global / main direction code and tangent of an endpoint.

    Starting from ``start_point``, the function walks along the skeleton for
    up to ``num_steps`` pixels and analyses the resulting path to derive:

    - ``local_dir``  - direction code of the very first step (1-2 px away).
    - ``global_dir`` - direction code of the segment beyond 20 px (smoothed).
    - ``main_dir``   - direction code from start to last reached pixel.
    - ``tangent_vec``- unit tangent (average direction) of the whole path.

    Parameters
    ----------
    coord_set : set[tuple[int, int]]
        The set of all skeleton pixel coordinates of the line.
    start_point : tuple[int, int]
        Starting pixel ``(x, y)``.
    num_steps : int, optional
        Maximum walk length in pixels. Default 30.

    Returns
    -------
    local_dir : int or None
    global_dir : int or None
    main_dir : int or None
    tangent_vec : tuple[float, float] or None
        Any of the four may be ``None`` if the walk produced too few pixels.
    """
    x0, y0 = start_point
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

    if len(path) < 2:
        return None, None, None, None

    if len(path) >= 1:
        local_dir = get_direction_code(x0, y0, path[0][0], path[0][1])
    else:
        local_dir = None

    if len(path) > 20:
        global_path = path[20:]
        gx0, gy0 = global_path[0][0], global_path[0][1]
        global_dir = get_direction_code(gx0, gy0, global_path[-1][0], global_path[-1][1])
    else:
        global_dir = None

    if len(path) >= 2:
        main_dir = get_direction_code(x0, y0, path[-1][0], path[-1][1])
    else:
        main_dir = None

    avg_dx = sum(p[0] - start_point[0] for p in path) / len(path)
    avg_dy = sum(p[1] - start_point[1] for p in path) / len(path)
    length = np.sqrt(avg_dx ** 2 + avg_dy ** 2)
    tangent_vec = (avg_dx / length, avg_dy / length) if length >= 1 else None

    return local_dir, global_dir, main_dir, tangent_vec


def check_vectors_intersect(ep1, vec1, ep2, vec2, extend_length=20):
    """Check whether two back-extended tangent rays intersect in front of the endpoints.

    Parameters
    ----------
    ep1 : tuple[float, float]
        First endpoint position.
    vec1 : tuple[float, float] or None
        First endpoint forward tangent (unit vector).
    ep2 : tuple[float, float]
        Second endpoint position.
    vec2 : tuple[float, float] or None
        Second endpoint forward tangent (unit vector).
    extend_length : float, optional
        Length of the back-extension ray. Default ``20``.

    Returns
    -------
    bool
        ``True`` if the back-extensions intersect within the parameter
        ranges accepted by ``line_intersection``.
    """
    if vec1 is None or vec2 is None:
        return False
    p1_extended = (ep1[0] - extend_length * vec1[0], ep1[1] - extend_length * vec1[1])
    p2_extended = (ep2[0] - extend_length * vec2[0], ep2[1] - extend_length * vec2[1])
    intersection = line_intersection(p1_extended, vec1, p2_extended, vec2)
    return intersection is not None


# ============================================================
# Matching Algorithm (identical to v3)
# ============================================================
def _build_skeleton_pixel_map(skeleton_data):
    """Build a dictionary ``{(x, y): set(line_ids)}`` for fast skeleton lookups.

    Built once per run and reused by ``check_connection_cross_skeleton``.

    Parameters
    ----------
    skeleton_data : dict or None
        Skeleton payload with a ``'lines'`` key. Each line must have
        ``'pixel_coords'`` either as ``[[x, y], ...]`` or ``[{'x': ..., 'y': ...}, ...]``.

    Returns
    -------
    dict[tuple[int, int], set[int]]
        Mapping from pixel to the set of line ids that contain it. Empty if
        ``skeleton_data`` is ``None`` or malformed.
    """
    if not skeleton_data or 'lines' not in skeleton_data:
        return {}
    pix_map = {}
    for lid_str, line_info in skeleton_data['lines'].items():
        try:
            lid = int(lid_str)
        except (TypeError, ValueError):
            continue
        for pt in line_info.get('pixel_coords', []):
            try:
                if isinstance(pt, dict):
                    x, y = int(pt['x']), int(pt['y'])
                else:
                    x, y = int(pt[0]), int(pt[1])
            except (TypeError, ValueError, IndexError, KeyError):
                continue
            key = (x, y)
            if key not in pix_map:
                pix_map[key] = set()
            pix_map[key].add(lid)
    return pix_map


def check_connection_cross_skeleton(ep1_pt, ep2_pt, line1_id, line2_id,
                                    skeleton_pixel_map, endpoint_radius=2):
    """Detect whether a candidate connection crosses any unrelated skeleton line.

    The Bresenham pixels between ``ep1_pt`` and ``ep2_pt`` are sampled,
    excluding the two endpoint regions (``endpoint_radius`` pixels each)
    and the two source lines themselves.

    Parameters
    ----------
    ep1_pt : tuple[int, int]
        First endpoint ``(x, y)``.
    ep2_pt : tuple[int, int]
        Second endpoint ``(x, y)``.
    line1_id : int
        Line id of the first endpoint (excluded from intersection check).
    line2_id : int
        Line id of the second endpoint (excluded from intersection check).
    skeleton_pixel_map : dict[tuple[int, int], set[int]]
        Pixel-to-line map produced by ``_build_skeleton_pixel_map``.
    endpoint_radius : int, optional
        Number of pixels around each endpoint to skip. Default ``2``.

    Returns
    -------
    crossed : bool
        ``True`` if the connection crosses at least one other line.
    cross_line_id : int or None
        Id of the first line found to cross the connection.
    cross_pt : tuple[int, int] or None
        Pixel where the crossing was detected.
    """
    if not skeleton_pixel_map:
        return False, None, None
    x1, y1 = int(ep1_pt[0]), int(ep1_pt[1])
    x2, y2 = int(ep2_pt[0]), int(ep2_pt[1])
    conn_pixels = bresenham_line(x1, y1, x2, y2)
    if not conn_pixels:
        return False, None, None
    if len(conn_pixels) > 2 * endpoint_radius:
        check_pixels = conn_pixels[endpoint_radius:-endpoint_radius]
    else:
        check_pixels = conn_pixels
    excluded = {line1_id, line2_id}
    for px, py in check_pixels:
        lids = skeleton_pixel_map.get((px, py))
        if not lids:
            continue
        other = lids - excluded
        if other:
            return True, next(iter(other)), (px, py)
    return False, None, None


def match_endpoints_v4(endpoints_for_matching, skeleton_data,
                       img_width, img_height, log_lines,
                       direct_threshold=13.5, max_distance=55, search_range=45,
                       cross_check_skeleton=True):
    """Match breakpoint endpoints in three progressive phases.

    Phase 1 - direct connection: pairs with distance < ``direct_threshold``.
    Phase 2 - direction-coded matching with progressively widening search
              radii (8, 12, 16, ..., ``search_range``).
    Phase 3 - relaxed matching with back-extension scoring up to
              ``max_distance + 10`` pixels.

    Parameters
    ----------
    endpoints_for_matching : list[tuple[int, tuple[int, int], tuple[float, float] | None, bool]]
        Sequence of ``(line_id, (x, y), tangent_vec, is_ep1)``.
    skeleton_data : dict
        Skeleton payload used for both direction analysis and crossing checks.
    img_width : int
        Image width in pixels.
    img_height : int
        Image height in pixels.
    log_lines : list[str]
        Append-only buffer receiving human-readable progress lines.
    direct_threshold : float, optional
        Maximum distance for the direct-connection phase. Default ``13.5``.
    max_distance : float, optional
        Base maximum distance for the relaxed phase. Default ``55``.
    search_range : float, optional
        Maximum distance for the direction-coded phase. Default ``45``.
    cross_check_skeleton : bool, optional
        If ``True``, candidates that cross an unrelated skeleton line are
        skipped. Default ``True``.

    Returns
    -------
    list[dict]
        All accepted matches. Each dict contains keys ``line1``, ``ep1``,
        ``vec1``, ``code1``, ``line2``, ``ep2``, ``vec2``, ``code2``,
        ``distance`` and ``is_direct``.
    """
    matches = []
    used_endpoints = set()

    # Pre-build skeleton pixel map once and reuse across all phases.
    skeleton_pixel_map = _build_skeleton_pixel_map(skeleton_data) if cross_check_skeleton else {}
    cross_reject_stats = {'phase1': 0, 'phase2': 0, 'phase3': 0}

    endpoint_list = []
    for i, (lid, ep, vec, is_ep1) in enumerate(endpoints_for_matching):
        line_info = skeleton_data['lines'].get(str(lid))
        if line_info and 'pixel_coords' in line_info:
            coord_set = set((int(coord[0]), int(coord[1])) for coord in line_info['pixel_coords'])
        else:
            coord_set = set()

        local_dir, global_dir, main_dir, tangent_vec = get_endpoint_direction_code(
            coord_set, ep, num_steps=30)

        endpoint_list.append({
            'global_idx': i + 1,
            'lid': lid,
            'ep': ep,
            'vec': tangent_vec,
            'code': main_dir,
            'local_dir': local_dir,
            'global_dir': global_dir,
            'direction_vec': tangent_vec,
            'is_ep1': is_ep1,
            'coord_set': coord_set,
        })

    log_lines.append("=" * 100)
    log_lines.append("Endpoint detection result (%d endpoints)" % len(endpoint_list))
    log_lines.append("=" * 100)
    log_lines.append("%-6s %-6s %-8s %-10s %-10s %-10s %-10s %-10s %s" % (
        "idx", "line", "ep", "coord(x,y)",
        "local(1-2px)", "global(>20px)", "overall_dir", "compat_code", "note"))
    log_lines.append("-" * 100)
    for ep in endpoint_list:
        code_str = str(ep['code']) if ep['code'] is not None else 'None'
        compat = ",".join(str(c) for c in DIRECTION_MATCH_RULES.get(ep['code'], set())) \
            if ep['code'] is not None else "N/A"
        log_lines.append("%-6d %-6d %-8s (%4d,%4d) %-10s %-10s %-10s compat:{%s}" % (
            ep['global_idx'], ep['lid'],
            'ep1' if ep['is_ep1'] else 'ep2',
            ep['ep'][0], ep['ep'][1],
            CODE_NAMES.get(ep['local_dir'], '?') if ep['local_dir'] is not None else 'None',
            CODE_NAMES.get(ep['global_dir'], '?') if ep['global_dir'] is not None else 'None',
            CODE_NAMES.get(ep['code'], '?') if ep['code'] is not None else 'None',
            compat
        ))
    log_lines.append("")

    # ----- Phase 1: direct connections -----
    log_lines.append("=" * 100)
    log_lines.append("[Phase 1] Direct connection (distance < %.0f px)" % direct_threshold)
    log_lines.append("=" * 100)

    for i, ep1_data in enumerate(endpoint_list):
        ep1_key = (ep1_data['lid'], ep1_data['ep'][0], ep1_data['ep'][1])
        if ep1_key in used_endpoints:
            continue
        for j, ep2_data in enumerate(endpoint_list):
            if i >= j:
                continue
            if ep1_data['lid'] == ep2_data['lid']:
                continue
            ep2_key = (ep2_data['lid'], ep2_data['ep'][0], ep2_data['ep'][1])
            if ep2_key in used_endpoints:
                continue
            dist = np.sqrt((ep1_data['ep'][0] - ep2_data['ep'][0]) ** 2 +
                           (ep1_data['ep'][1] - ep2_data['ep'][1]) ** 2)
            if dist < direct_threshold:
                # Skeleton-crossing rejection.
                if cross_check_skeleton:
                    crossed, cross_lid, cross_pt = check_connection_cross_skeleton(
                        ep1_data['ep'], ep2_data['ep'],
                        ep1_data['lid'], ep2_data['lid'],
                        skeleton_pixel_map)
                    if crossed:
                        cross_reject_stats['phase1'] += 1
                        log_lines.append(
                            "  Direct-cross skip: EP%d(Line%d) <-> EP%d(Line%d) dist=%.2f "
                            "crosses L%d @(%d,%d)" % (
                                ep1_data['global_idx'], ep1_data['lid'],
                                ep2_data['global_idx'], ep2_data['lid'],
                                dist, cross_lid, cross_pt[0], cross_pt[1]))
                        continue

                used_endpoints.add(ep1_key)
                used_endpoints.add(ep2_key)
                log_lines.append("  Direct: EP%d(Line%d,%s,%d,%d) <-> EP%d(Line%d,%s,%d,%d) dist=%.2f" % (
                    ep1_data['global_idx'], ep1_data['lid'],
                    'ep1' if ep1_data['is_ep1'] else 'ep2',
                    ep1_data['ep'][0], ep1_data['ep'][1],
                    ep2_data['global_idx'], ep2_data['lid'],
                    'ep1' if ep2_data['is_ep1'] else 'ep2',
                    ep2_data['ep'][0], ep2_data['ep'][1],
                    dist
                ))
                matches.append({
                    'line1': ep1_data['lid'], 'ep1': ep1_data['ep'],
                    'vec1': ep1_data['vec'], 'code1': ep1_data['code'],
                    'line2': ep2_data['lid'], 'ep2': ep2_data['ep'],
                    'vec2': ep2_data['vec'], 'code2': ep2_data['code'],
                    'distance': dist, 'is_direct': True
                })

    log_lines.append("  Direct connections: %d pairs" % len(matches))
    log_lines.append("")

    # ----- Phase 2: direction-coded matching with progressive radii -----
    log_lines.append("=" * 100)
    log_lines.append("[Phase 2] Direction-coded matching (progressively widening search range)")
    log_lines.append("=" * 100)

    search_ranges = [8, 12, 16, 20, 25, 30, 35, 40, 45]
    available = [ep for ep in endpoint_list
                 if (ep['lid'], ep['ep'][0], ep['ep'][1]) not in used_endpoints]

    for search_range in search_ranges:
        current_available = [ep for ep in available
                             if (ep['lid'], ep['ep'][0], ep['ep'][1]) not in used_endpoints]
        if not current_available:
            break

        log_lines.append("")
        log_lines.append("--- search range: %d px ---" % search_range)
        this_round_matches = []

        for i, ep1_data in enumerate(current_available):
            ep1_key = (ep1_data['lid'], ep1_data['ep'][0], ep1_data['ep'][1])
            if ep1_key in used_endpoints:
                continue
            code1 = ep1_data['code']
            if code1 is None:
                continue

            candidates = []
            for j, ep2_data in enumerate(current_available):
                if i >= j:
                    continue
                if ep1_data['lid'] == ep2_data['lid']:
                    continue
                ep2_key = (ep2_data['lid'], ep2_data['ep'][0], ep2_data['ep'][1])
                if ep2_key in used_endpoints:
                    continue
                code2 = ep2_data['code']
                if code2 is None:
                    continue

                dist = np.sqrt((ep1_data['ep'][0] - ep2_data['ep'][0]) ** 2 +
                               (ep1_data['ep'][1] - ep2_data['ep'][1]) ** 2)

                if dist < direct_threshold or dist > search_range:
                    continue

                compat1 = code1 in DIRECTION_MATCH_RULES.get(code2, set())
                compat2 = code2 in DIRECTION_MATCH_RULES.get(code1, set())

                if not (compat1 and compat2):
                    log_lines.append(
                        "  Cand-incompat: EP%d(Line%d,%d,%d,code=%s) <-> EP%d(Line%d,%d,%d,code=%s) "
                        "dist=%.2f reason: code%d compat={%s}, code%d compat={%s}" % (
                            ep1_data['global_idx'], ep1_data['lid'],
                            ep1_data['ep'][0], ep1_data['ep'][1], CODE_NAMES.get(code1, '?'),
                            ep2_data['global_idx'], ep2_data['lid'],
                            ep2_data['ep'][0], ep2_data['ep'][1], CODE_NAMES.get(code2, '?'),
                            dist,
                            code2, ','.join(str(c) for c in DIRECTION_MATCH_RULES.get(code2, set())),
                            code1, ','.join(str(c) for c in DIRECTION_MATCH_RULES.get(code1, set())),
                        ))
                    continue

                dx = abs(ep1_data['ep'][0] - ep2_data['ep'][0])
                dy = abs(ep1_data['ep'][1] - ep2_data['ep'][1])

                constraint_ok = True
                constraint_reason = ""
                if code1 in [2, 6] and code2 in [2, 6]:
                    if dy > dx * 0.8:
                        constraint_ok = False
                        constraint_reason = "horizontal endpoints: Y diff too large (%d>%d*0.8)" % (dy, dx)
                elif code1 in [0, 4] and code2 in [0, 4]:
                    if dx > dy * 0.8:
                        constraint_ok = False
                        constraint_reason = "vertical endpoints: X diff too large (%d>%d*0.8)" % (dx, dy)
                # Diagonal endpoints: do NOT constrain dx/dy ratio.
                # Reason: when two diagonals form a V shape, the endpoint-to-endpoint
                # segment direction does not necessarily match each endpoint's diagonal.
                # Example: c=7 <-> c=1 can be connected horizontally (e.g.
                # EP33(528,370) <-> EP37(544,369)).
                elif code1 in [1, 3, 5, 7] or code2 in [1, 3, 5, 7]:
                    pass  # only direction compatibility + distance, no dx/dy constraint

                if not constraint_ok:
                    log_lines.append(
                        "  Cand-coord: EP%d(Line%d,%d,%d,code=%s) <-> EP%d(Line%d,%d,%d,code=%s) "
                        "dist=%.2f reason: %s" % (
                            ep1_data['global_idx'], ep1_data['lid'],
                            ep1_data['ep'][0], ep1_data['ep'][1], CODE_NAMES.get(code1, '?'),
                            ep2_data['global_idx'], ep2_data['lid'],
                            ep2_data['ep'][0], ep2_data['ep'][1], CODE_NAMES.get(code2, '?'),
                            dist, constraint_reason
                        ))
                    continue

                candidates.append({
                    'ep1': ep1_data, 'ep2': ep2_data,
                    'dist': dist,
                    'code1': code1, 'code2': code2,
                })

            if not candidates:
                continue

            for cand in candidates:
                score, details = evaluate_match_quality(
                    cand['ep1'], cand['ep2'], cand['dist'],
                    use_back_extension=True
                )
                cand['score'] = score
                cand['details'] = details

            candidates.sort(key=lambda c: c['score'])

            # Try each candidate in score order; skip ones that cross skeleton.
            best = None
            for cand in candidates:
                if cross_check_skeleton:
                    crossed, cross_lid, cross_pt = check_connection_cross_skeleton(
                        cand['ep1']['ep'], cand['ep2']['ep'],
                        cand['ep1']['lid'], cand['ep2']['lid'],
                        skeleton_pixel_map)
                    if crossed:
                        cross_reject_stats['phase2'] += 1
                        log_lines.append(
                            "  Cand-cross skip: EP%d(Line%d) <-> EP%d(Line%d) "
                            "score=%.3f crosses L%d @(%d,%d)" % (
                                cand['ep1']['global_idx'], cand['ep1']['lid'],
                                cand['ep2']['global_idx'], cand['ep2']['lid'],
                                cand['score'], cross_lid,
                                cross_pt[0], cross_pt[1]))
                        continue
                best = cand
                break

            if best is None:
                log_lines.append(
                    "  EP%d(Line%d) no acceptable candidate "
                    "(%d direction-coded candidates all cross skeleton)" % (
                        ep1_data['global_idx'], ep1_data['lid'], len(candidates)))
                continue

            ep2_data = best['ep2']
            ep2_key = (ep2_data['lid'], ep2_data['ep'][0], ep2_data['ep'][1])

            this_round_matches.append(best)
            used_endpoints.add(ep1_key)
            used_endpoints.add(ep2_key)

            if 'intersection' in best['details']:
                int_pt = best['details']['intersection']
                log_lines.append(
                    "  ** Connect: EP%d(Line%d,%d,%d,code=%s) <-> EP%d(Line%d,%d,%d,code=%s) "
                    "dist=%.1f, score=%.3f, back_ext_t=(%.1f+%.1f=%.1f)" % (
                        ep1_data['global_idx'], ep1_data['lid'],
                        ep1_data['ep'][0], ep1_data['ep'][1], CODE_NAMES.get(code1, '?'),
                        ep2_data['global_idx'], ep2_data['lid'],
                        ep2_data['ep'][0], ep2_data['ep'][1], CODE_NAMES.get(best['code2'], '?'),
                        best['dist'], best['score'],
                        best['details']['t1'], best['details']['t2'],
                        best['details']['t1'] + best['details']['t2']
                    ))
            else:
                log_lines.append(
                    "  ** Connect: EP%d(Line%d,%d,%d,code=%s) <-> EP%d(Line%d,%d,%d,code=%s) "
                    "dist=%.1f, score=%.3f (no back_ext)" % (
                        ep1_data['global_idx'], ep1_data['lid'],
                        ep1_data['ep'][0], ep1_data['ep'][1], CODE_NAMES.get(code1, '?'),
                        ep2_data['global_idx'], ep2_data['lid'],
                        ep2_data['ep'][0], ep2_data['ep'][1], CODE_NAMES.get(best['code2'], '?'),
                        best['dist'], best['score']
                    ))

            matches.append({
                'line1': ep1_data['lid'], 'ep1': ep1_data['ep'],
                'vec1': ep1_data['vec'], 'code1': ep1_data['code'],
                'line2': ep2_data['lid'], 'ep2': ep2_data['ep'],
                'vec2': ep2_data['vec'], 'code2': best['code2'],
                'distance': best['dist'], 'is_direct': False
            })

        log_lines.append("  This round: %d new pairs" % len(this_round_matches))

        available = [ep for ep in available
                     if (ep['lid'], ep['ep'][0], ep['ep'][1]) not in used_endpoints]

    # ----- Phase 3: relaxed matching -----
    log_lines.append("")
    log_lines.append("=" * 100)
    log_lines.append("[Phase 3] Relaxed matching (loosened constraints + back-extension scoring)")
    log_lines.append("=" * 100)

    remaining = [ep for ep in endpoint_list
                 if (ep['lid'], ep['ep'][0], ep['ep'][1]) not in used_endpoints]

    if remaining:
        log_lines.append("  Remaining unmatched endpoints: %d" % len(remaining))
        extended_max = max_distance + 10

        for ep1_data in remaining[:]:
            ep1_key = (ep1_data['lid'], ep1_data['ep'][0], ep1_data['ep'][1])
            if ep1_key in used_endpoints:
                continue
            code1 = ep1_data['code']
            if code1 is None:
                continue

            # Collect and score all candidates (keep all of them so we can
            # fall back to the next-best when one is rejected for crossing).
            candidates = []

            for ep2_data in remaining:
                if ep1_data['global_idx'] >= ep2_data['global_idx']:
                    continue
                if ep1_data['lid'] == ep2_data['lid']:
                    continue
                ep2_key = (ep2_data['lid'], ep2_data['ep'][0], ep2_data['ep'][1])
                if ep2_key in used_endpoints:
                    continue
                code2 = ep2_data['code']
                if code2 is None:
                    continue

                dist = np.sqrt((ep1_data['ep'][0] - ep2_data['ep'][0]) ** 2 +
                               (ep1_data['ep'][1] - ep2_data['ep'][1]) ** 2)
                if dist > extended_max:
                    continue

                compat1 = code1 in DIRECTION_MATCH_RULES.get(code2, set())
                if not compat1:
                    continue

                score, details = evaluate_match_quality(
                    ep1_data, ep2_data, dist,
                    use_back_extension=True
                )

                candidates.append({
                    'ep2_data': ep2_data,
                    'code2': code2,
                    'dist': dist,
                    'score': score,
                    'details': details,
                })

            # Sort candidates by score ascending (lower is better).
            candidates.sort(key=lambda c: c['score'])

            # Try each candidate in order; skip ones that cross skeleton.
            best = None
            best_score = float('inf')
            best_details = None

            for cand in candidates:
                if cross_check_skeleton:
                    crossed, cross_lid, cross_pt = check_connection_cross_skeleton(
                        ep1_data['ep'], cand['ep2_data']['ep'],
                        ep1_data['lid'], cand['ep2_data']['lid'],
                        skeleton_pixel_map)
                    if crossed:
                        cross_reject_stats['phase3'] += 1
                        log_lines.append(
                            "  Relax-cross skip: EP%d(Line%d) <-> EP%d(Line%d) "
                            "score=%.3f crosses L%d @(%d,%d)" % (
                                ep1_data['global_idx'], ep1_data['lid'],
                                cand['ep2_data']['global_idx'], cand['ep2_data']['lid'],
                                cand['score'], cross_lid, cross_pt[0], cross_pt[1]))
                        continue
                best = (cand['ep2_data'], cand['code2'], cand['dist'])
                best_score = cand['score']
                best_details = cand['details']
                break

            if best is None and candidates:
                log_lines.append(
                    "  EP%d(Line%d) no acceptable candidate "
                    "(%d candidates in relaxed phase all cross skeleton)" % (
                        ep1_data['global_idx'], ep1_data['lid'], len(candidates)))

            if best:
                ep2_data, code2, dist = best
                ep2_key = (ep2_data['lid'], ep2_data['ep'][0], ep2_data['ep'][1])
                used_endpoints.add(ep1_key)
                used_endpoints.add(ep2_key)

                if best_details and 'intersection' in best_details:
                    int_pt = best_details['intersection']
                    log_lines.append(
                        "  Relax-relaxed: EP%d(Line%d,%d,%d,code=%s) <-> EP%d(Line%d,%d,%d,code=%s) "
                        "dist=%.1f, score=%.3f, back_ext_t=(%.1f+%.1f=%.1f)" % (
                            ep1_data['global_idx'], ep1_data['lid'],
                            ep1_data['ep'][0], ep1_data['ep'][1], CODE_NAMES.get(code1, '?'),
                            ep2_data['global_idx'], ep2_data['lid'],
                            ep2_data['ep'][0], ep2_data['ep'][1], CODE_NAMES.get(code2, '?'),
                            dist, best_score,
                            best_details['t1'], best_details['t2'],
                            best_details['t1'] + best_details['t2']
                        ))
                else:
                    log_lines.append(
                        "  Relax-relaxed: EP%d(Line%d,%d,%d,code=%s) <-> EP%d(Line%d,%d,%d,code=%s) "
                        "dist=%.1f, score=%.3f" % (
                            ep1_data['global_idx'], ep1_data['lid'],
                            ep1_data['ep'][0], ep1_data['ep'][1], CODE_NAMES.get(code1, '?'),
                            ep2_data['global_idx'], ep2_data['lid'],
                            ep2_data['ep'][0], ep2_data['ep'][1], CODE_NAMES.get(code2, '?'),
                            dist, best_score
                        ))

                matches.append({
                    'line1': ep1_data['lid'], 'ep1': ep1_data['ep'],
                    'vec1': ep1_data['vec'], 'code1': ep1_data['code'],
                    'line2': ep2_data['lid'], 'ep2': ep2_data['ep'],
                    'vec2': ep2_data['vec'], 'code2': code2,
                    'distance': dist, 'is_direct': False
                })

    log_lines.append("")
    log_lines.append("=" * 100)
    log_lines.append("Matching complete: %d connection pairs in total" % len(matches))
    log_lines.append("=" * 100)
    log_lines.append("")

    # Cross-skeleton rejection summary.
    total_rejected = cross_reject_stats['phase1'] + cross_reject_stats['phase2'] + cross_reject_stats['phase3']
    if cross_check_skeleton and total_rejected > 0:
        log_lines.append(
            "[Skeleton-cross skip stats] phase1=%d, phase2=%d, phase3=%d, total=%d" % (
                cross_reject_stats['phase1'], cross_reject_stats['phase2'],
                cross_reject_stats['phase3'], total_rejected))
        log_lines.append("")

    still_free = [ep for ep in endpoint_list
                  if (ep['lid'], ep['ep'][0], ep['ep'][1]) not in used_endpoints]
    log_lines.append("Unmatched endpoints (free endpoints, %d in total):" % len(still_free))
    for ep in still_free:
        log_lines.append("  EP%d: Line%d %s (%d,%d) code=%s local=%s global=%s" % (
            ep['global_idx'], ep['lid'],
            'ep1' if ep['is_ep1'] else 'ep2',
            ep['ep'][0], ep['ep'][1],
            CODE_NAMES.get(ep['code'], '?') if ep['code'] is not None else 'None',
            CODE_NAMES.get(ep['local_dir'], '?') if ep['local_dir'] is not None else 'None',
            CODE_NAMES.get(ep['global_dir'], '?') if ep['global_dir'] is not None else 'None',
        ))

    return matches


def get_line_direction(coord_set, start_point, num_steps=10):
    """Compute a unit direction vector along a skeleton line.

    Walks from ``start_point`` along the skeleton for at most ``num_steps``
    pixels and returns the unit vector from the start to the last reached
    pixel.

    Parameters
    ----------
    coord_set : set[tuple[int, int]]
        Set of all skeleton pixel coordinates for the line.
    start_point : tuple[int, int]
        Starting pixel ``(x, y)``.
    num_steps : int, optional
        Maximum walk length in pixels. Default ``10``.

    Returns
    -------
    tuple[float, float] or None
        Unit direction vector, or ``None`` if the walk produced fewer than
        2 pixels or the displacement length is below 1 pixel.
    """
    x, y = start_point
    path = []
    current = (x, y)
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
    if len(path) < 2:
        return None
    total_dx = path[-1][0] - start_point[0]
    total_dy = path[-1][1] - start_point[1]
    length = np.sqrt(total_dx ** 2 + total_dy ** 2)
    if length < 1:
        return None
    return (total_dx / length, total_dy / length)


def connect_with_cubic_bezier(ep1, ep2, vec1=None, vec2=None, num_points=100):
    """Generate a cubic-Bezier curve between two endpoints respecting their tangents.

    The two control points are placed along the back-extensions of the input
    tangents; if the back-extensions do not intersect inside a reasonable
    range, a default 15% offset is used.

    Parameters
    ----------
    ep1 : tuple[float, float]
        First endpoint ``(x, y)``.
    ep2 : tuple[float, float]
        Second endpoint ``(x, y)``.
    vec1 : tuple[float, float] or None, optional
        Forward unit tangent at ``ep1``. Falls back to ``(0, 1)`` when
        ``None`` / zero.
    vec2 : tuple[float, float] or None, optional
        Forward unit tangent at ``ep2``. Falls back to ``(0, 1)`` when
        ``None`` / zero.
    num_points : int, optional
        Number of sample points along the curve. Default ``100``.

    Returns
    -------
    list[tuple[int, int]]
        Sequence of ``(x, y)`` integer pixel coordinates sampled along the
        Bezier curve, including both endpoints.
    """
    x0, y0 = float(ep1[0]), float(ep1[1])
    x1, y1 = float(ep2[0]), float(ep2[1])

    dist = np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)

    P0 = np.array([x0, y0])
    P3 = np.array([x1, y1])

    if vec1 is None or vec1 == (0, 0) or vec1[0] is None:
        vec1 = np.array([0.0, 1.0])
    else:
        v = np.array([float(vec1[0]), float(vec1[1])])
        v_len = np.linalg.norm(v)
        if v_len > 1e-10:
            vec1 = v / v_len
        else:
            vec1 = np.array([0.0, 1.0])

    if vec2 is None or vec2 == (0, 0) or vec2[0] is None:
        vec2 = np.array([0.0, 1.0])
    else:
        v = np.array([float(vec2[0]), float(vec2[1])])
        v_len = np.linalg.norm(v)
        if v_len > 1e-10:
            vec2 = v / v_len
        else:
            vec2 = np.array([0.0, 1.0])

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

    if intersection is not None:
        d1 = min(t1_val * 0.4, dist * 0.35)
        d2 = min(t2_val * 0.4, dist * 0.35)
    else:
        d1 = dist * 0.15
        d2 = dist * 0.15

    P1 = P0 + d1 * back_vec1
    P2 = P3 + d2 * back_vec2

    def bezier_point(t, P0, P1, P2, P3):
        t2 = t * t
        t3 = t2 * t
        mt = 1 - t
        mt2 = mt * mt
        mt3 = mt2 * mt
        return mt3 * P0 + 3 * mt2 * t * P1 + 3 * mt * t2 * P2 + t3 * P3

    points = []
    for i in range(num_points):
        t = i / (num_points - 1)
        pt = bezier_point(t, P0, P1, P2, P3)
        points.append((round(pt[0]), round(pt[1])))

    return points


def connect_with_cubic_spline(ep1, ep2, vec1=None, vec2=None, num_points=100):
    """Alias of :func:`connect_with_cubic_bezier`.

    Parameters
    ----------
    ep1, ep2 : tuple[float, float]
        Endpoint coordinates.
    vec1, vec2 : tuple[float, float] or None, optional
        Tangent unit vectors.
    num_points : int, optional
        Number of sample points. Default ``100``.

    Returns
    -------
    list[tuple[int, int]]
        Curve sample points.
    """
    return connect_with_cubic_bezier(ep1, ep2, vec1, vec2, num_points)


def connect_with_spline(ep1, ep2, coord_set1, coord_set2, num_steps=10):
    """Connect two endpoints with a default Bezier curve (no tangents).

    The ``coord_set`` parameters are accepted for backward API compatibility
    and currently unused; tangents default to ``None``.

    Parameters
    ----------
    ep1, ep2 : tuple[float, float]
        Endpoint coordinates.
    coord_set1, coord_set2 : set[tuple[int, int]]
        Skeleton coordinate sets (kept for API compatibility; not used).
    num_steps : int, optional
        Legacy parameter; the curve length is computed as ``max(50, num_steps * 5)``.

    Returns
    -------
    list[tuple[int, int]]
        Curve sample points.
    """
    return connect_with_cubic_spline(ep1, ep2, vec1=None, vec2=None, num_points=max(50, num_steps * 5))


# ============================================================
# Visualization (v4 specific)
# ============================================================
def visualize_v4_results(skeleton_data, all_raw_endpoints, filtered_endpoints,
                         matches, img_width, img_height, output_dir):
    """Render the v4 2x3 debug figure and save it as ``breakpoint_v4_debug.png``.

    Sub-figures:
      [1] Raw skeleton + all raw endpoints (green = interior, red = edge)
      [2] Edge-filtered endpoints used for matching (with L# ep1/ep2 labels)
      [3] Direction codes of each filtered endpoint
      [4] Connection result (green = direct, red = cubic-spline)
      [5] Per-connection details (text)
      [6] Direction match rules (text)

    Parameters
    ----------
    skeleton_data : dict
        Skeleton payload.
    all_raw_endpoints : list[dict]
        All raw endpoints before filtering (each dict has at least ``x``,
        ``y``, ``on_edge``).
    filtered_endpoints : list[dict]
        Endpoints kept after the edge filter (each dict has ``line_id``,
        ``ep_name``, ``x``, ``y``).
    matches : list[dict]
        Final matches (each dict has ``ep1``, ``ep2``, ``vec1``, ``vec2``,
        ``distance``, ``is_direct``).
    img_width : int
        Image width in pixels.
    img_height : int
        Image height in pixels.
    output_dir : str
        Folder in which ``breakpoint_v4_debug.png`` is written.
    """
    fig, axes = plt.subplots(2, 3, figsize=(30, 15))

    def draw_skeleton(vis):
        for line_id, line_data in skeleton_data['lines'].items():
            for x, y in line_data['pixel_coords']:
                if 0 <= y < img_height and 0 <= x < img_width:
                    vis[y, x] = [180, 180, 180]

    # ----- Sub-figure 1: raw skeleton + all endpoints -----
    vis1 = np.zeros((img_height, img_width, 3), dtype=np.uint8)
    vis1[:] = [255, 255, 255]
    draw_skeleton(vis1)

    for idx, r in enumerate(all_raw_endpoints, 1):
        if not r.get('present'):
            continue
        x, y = r['x'], r['y']
        if r['on_edge']:
            color = (0, 0, 255)  # red - on-edge endpoint (filtered out)
        else:
            color = (0, 200, 0)  # green - interior endpoint (used for matching)
        cv2.circle(vis1, (x, y), 6, color, -1)
        cv2.circle(vis1, (x, y), 6, (0, 0, 0), 1)
        cv2.putText(vis1, str(idx), (x + 7, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

    axes[0, 0].imshow(cv2.cvtColor(vis1, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('[1] Raw Skeleton + Endpoints (%d total, green=interior, red=edge)' % len(all_raw_endpoints),
                         fontsize=11)
    axes[0, 0].axis('off')

    # ----- Sub-figure 2: filtered endpoints -----
    vis2 = np.zeros((img_height, img_width, 3), dtype=np.uint8)
    vis2[:] = [255, 255, 255]
    draw_skeleton(vis2)

    raw_lookup = {(r['line_id'], r['ep_name']): i + 1 for i, r in enumerate(all_raw_endpoints)}

    for r in filtered_endpoints:
        x, y = r['x'], r['y']
        cv2.circle(vis2, (x, y), 7, (0, 200, 0), -1)
        cv2.circle(vis2, (x, y), 7, (0, 0, 0), 1)
        g = raw_lookup.get((r['line_id'], r['ep_name']), None)
        ep_digit = r['ep_name'].replace('ep', '')
        if g:
            label = "G%d L%d*%s" % (g, r['line_id'], ep_digit)
        else:
            label = "L%d*%s" % (r['line_id'], ep_digit)
        cv2.putText(vis2, label, (x + 8, y + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 1)

    axes[0, 1].imshow(cv2.cvtColor(vis2, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title('[2] Filtered Endpoints (used for matching, n=%d)' % len(filtered_endpoints),
                         fontsize=11)
    axes[0, 1].axis('off')

    # ----- Sub-figure 3: direction codes -----
    vis3 = np.zeros((img_height, img_width, 3), dtype=np.uint8)
    vis3[:] = [255, 255, 255]
    draw_skeleton(vis3)

    for r in filtered_endpoints:
        x, y = r['x'], r['y']
        line_info = skeleton_data['lines'].get(str(r['line_id']))
        if line_info:
            coord_set = set((int(c[0]), int(c[1])) for c in line_info['pixel_coords'])
        else:
            coord_set = set()
        local_dir, global_dir, main_dir, _ = get_endpoint_direction_code(
            coord_set, (x, y), num_steps=30)

        cv2.circle(vis3, (x, y), 7, (0, 200, 0), -1)
        cv2.circle(vis3, (x, y), 7, (0, 0, 0), 1)
        cv2.putText(vis3, 'C%d' % main_dir if main_dir is not None else 'C?',
                    (x + 8, y + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

    axes[0, 2].imshow(cv2.cvtColor(vis3, cv2.COLOR_BGR2RGB))
    axes[0, 2].set_title('[3] Direction Codes (C0=up, C1=up-right, C2=right, C3=down-right, C4=down, C5=down-left, C6=left, C7=up-left)',
                         fontsize=11)
    axes[0, 2].axis('off')

    # ----- Sub-figure 4: connection result -----
    vis4 = np.ones((img_height, img_width, 3), dtype=np.uint8) * 255
    draw_skeleton(vis4)
    for match in matches:
        ep1, ep2 = match['ep1'], match['ep2']
        # green = direct connection (dist < 13.5), red = breakpoint cubic spline
        color = (0, 200, 0) if match.get('is_direct') else (0, 0, 255)
        vec1 = match.get('vec1')
        vec2 = match.get('vec2')
        dist = np.sqrt((ep2[0] - ep1[0]) ** 2 + (ep2[1] - ep1[1]) ** 2)
        num_points = max(100, int(dist * 5))
        pts = connect_with_cubic_spline(ep1, ep2, vec1, vec2, num_points=num_points)
        for x, y in pts:
            if 0 <= y < img_height and 0 <= x < img_width:
                vis4[y, x] = color
        cv2.circle(vis4, ep1, 4, (255, 0, 0), -1)
        cv2.circle(vis4, ep2, 4, (0, 0, 255), -1)
    axes[1, 0].imshow(cv2.cvtColor(vis4, cv2.COLOR_BGR2RGB))
    n_direct = sum(1 for m in matches if m.get('is_direct'))
    n_spline = len(matches) - n_direct
    axes[1, 0].set_title('[4] Connected Skeleton (%d pairs: %d direct/green + %d spline/red)' %
                         (len(matches), n_direct, n_spline), fontsize=11)
    axes[1, 0].axis('off')

    # ----- Sub-figure 5: per-connection details -----
    axes[1, 1].axis('off')
    conn_text = "Connection Details (%d pairs)\n" % len(matches)
    conn_text += "=" * 60 + "\n"
    conn_text += "%-3s %-7s %-13s %-4s %-13s %-5s %-7s %s\n" % (
        "#", "Line1", "EP1(x,y)", "c1", "EP2(x,y)", "c2", "dist", "type")
    conn_text += "-" * 70 + "\n"
    for i, match in enumerate(matches):
        conn_text += "%-3d L%-6d (%-3d,%-3d)  c%-2d  (%-3d,%-3d)  c%-2d  %-6.1f %s\n" % (
            i + 1,
            match['line1'], match['ep1'][0], match['ep1'][1], match.get('code1', -1),
            match['ep2'][0], match['ep2'][1], match.get('code2', -1),
            match['distance'],
            '[direct]' if match.get('is_direct') else '[spline]'
        )
    axes[1, 1].text(0.03, 0.97, conn_text, fontsize=8, family='monospace',
                    verticalalignment='top', transform=axes[1, 1].transAxes)

    # ----- Sub-figure 6: direction match rules -----
    axes[1, 2].axis('off')
    rules_text = "Direction Match Rules\n" + "=" * 50 + "\n"
    for code, compat in sorted(DIRECTION_MATCH_RULES.items()):
        rules_text += "Code %d (%s)\n  compat: {%s}\n" % (
            code, CODE_NAMES[code],
            ','.join(str(c) for c in sorted(compat)))
    rules_text += "\nLegend:\n"
    rules_text += "  [1] Raw endpoints: green=interior, red=edge(filtered)\n"
    rules_text += "  [2] Filtered endpoints: used for matching\n"
    rules_text += "  [3] Direction codes\n"
    rules_text += "  [4] Connection result: green=direct, red=breakpoint\n"
    axes[1, 2].text(0.03, 0.97, rules_text, fontsize=8, family='monospace',
                    verticalalignment='top', transform=axes[1, 2].transAxes)

    plt.suptitle('v4 Edge Filter: %d raw endpoints -> %d filtered -> %d matches' %
                 (len(all_raw_endpoints), len(filtered_endpoints), len(matches)),
                 fontsize=14, y=0.998)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
    buffer.seek(0)
    out_path = os.path.join(output_dir, 'breakpoint_v4_debug.png')
    try:
        arr = np.frombuffer(buffer.getvalue(), dtype=np.uint8)
        img_arr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        ext = os.path.splitext(out_path)[1]
        result, encoded = cv2.imencode(ext, img_arr)
        if result:
            encoded.tofile(out_path)
            print("    Saved: breakpoint_v4_debug.png (via cv2.imencode)")
        else:
            raise IOError("imencode failed")
    except Exception as e:
        print("    cv2.imencode failed (%s), trying fallback..." % e)
        with open(out_path, 'wb') as f:
            f.write(buffer.getvalue())
        print("    Saved: breakpoint_v4_debug.png (via fallback)")
    plt.close()


def visualize_edge_filter_comparison(skeleton_data, all_raw_records, all_merged_records,
                                    filtered_records,
                                    matches, img_width, img_height, output_dir,
                                    merge_stats=None):
    """Render a 6-panel comparison of raw / merged / filtered endpoints.

    Sub-figures:
      A (top-left)    : Raw v3 endpoints (green = interior, red = edge).
      B (top-right)   : After edge-pseudo merge (gray X = merged-off pseudo endpoints).
      C (mid-left)    : Final filtered endpoints (merge + on-edge filter).
      D (mid-right)   : Connected skeleton (green = direct, red = spline).
      E (bottom-left) : Zoom of A bottom-left region (2x).
      F (bottom-right): Zoom of C bottom-left region (2x).

    Parameters
    ----------
    skeleton_data : dict
        Skeleton payload.
    all_raw_records : list[dict]
        Raw v3 endpoint records.
    all_merged_records : list[dict]
        Records after the edge-pseudo merge step.
    filtered_records : list[dict]
        Records after applying the on-edge filter (final, used for matching).
    matches : list[dict]
        Final connection matches.
    img_width : int
        Image width in pixels.
    img_height : int
        Image height in pixels.
    output_dir : str
        Folder in which ``breakpoint_v4_filter_compare.png`` is written.
    merge_stats : dict or None, optional
        Optional merge statistics shown on sub-figure A's title.
    """
    raw_keys = set((r['line_id'], r['ep_name']) for r in all_raw_records)
    merged_keys = set((r['line_id'], r['ep_name']) for r in all_merged_records)
    raw_kept_in_merged = set()
    for r in all_merged_records:
        raw_kept_in_merged.add((r['line_id'], r['ep_name']))

    pseudo_pseudo_endpoints = []  # pseudo endpoints that were merged away
    pseudo_kept = set()
    raw_set = set((r['line_id'], r['ep_name'], r['x'], r['y']) for r in all_raw_records)
    merged_set = set((r['line_id'], r['ep_name']) for r in all_merged_records)
    for r in all_raw_records:
        key = (r['line_id'], r['ep_name'])
        if key not in merged_set:
            pseudo_pseudo_endpoints.append(r)

    fig, axes = plt.subplots(3, 2, figsize=(28, 28))

    def draw_skeleton(vis):
        for line_id, line_data in skeleton_data['lines'].items():
            for x, y in line_data['pixel_coords']:
                if 0 <= y < img_height and 0 <= x < img_width:
                    vis[y, x] = [180, 180, 180]

    # ----- Sub-figure A: raw v3 endpoints -----
    vis1 = np.zeros((img_height, img_width, 3), dtype=np.uint8)
    vis1[:] = [255, 255, 255]
    draw_skeleton(vis1)

    for r in all_raw_records:
        if r.get('missing', False):
            continue
        x, y = r['x'], r['y']
        if r['on_edge']:
            color = (0, 0, 255)  # red - on-edge endpoint
        else:
            color = (0, 200, 0)  # green - interior endpoint
        cv2.circle(vis1, (x, y), 7, color, -1)
        cv2.circle(vis1, (x, y), 7, (0, 0, 0), 1)
        label = "G%d" % r['global_idx']
        cv2.putText(vis1, label, (x + 8, y + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1)

    axes[0, 0].imshow(cv2.cvtColor(vis1, cv2.COLOR_BGR2RGB))
    title_a = '[A] Raw v3 Endpoints (red=edge, green=interior, n=%d)' % len(all_raw_records)
    if merge_stats:
        title_a += '\n  -> %d edge-pseudo endpoints will be merged away' % len(pseudo_pseudo_endpoints)
    axes[0, 0].set_title(title_a, fontsize=11)
    axes[0, 0].axis('off')

    # ----- Sub-figure B: after edge-pseudo merge -----
    vis2 = np.zeros((img_height, img_width, 3), dtype=np.uint8)
    vis2[:] = [255, 255, 255]
    draw_skeleton(vis2)

    for r in all_raw_records:
        if r.get('missing', False):
            continue
        x, y = r['x'], r['y']
        in_merged = (r['line_id'], r['ep_name']) in merged_set
        if not in_merged:
            cv2.line(vis2, (x - 6, y - 6), (x + 6, y + 6), (128, 128, 128), 2)
            cv2.line(vis2, (x - 6, y + 6), (x + 6, y - 6), (128, 128, 128), 2)
        else:
            if r['on_edge']:
                color = (0, 0, 255)
            else:
                color = (0, 200, 0)
            cv2.circle(vis2, (x, y), 7, color, -1)
            cv2.circle(vis2, (x, y), 7, (0, 0, 0), 1)

    for r in pseudo_pseudo_endpoints:
        x, y = r['x'], r['y']
        cv2.putText(vis2, "pseudo", (x + 10, y + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (128, 128, 128), 1)

    for r in all_merged_records:
        x, y = r['x'], r['y']
        if (r['line_id'], r['ep_name']) in raw_set:
            pass
        for rr in all_raw_records:
            if rr['line_id'] == r['line_id'] and rr['ep_name'] == r['ep_name']:
                label = "G%d" % rr['global_idx']
                cv2.putText(vis2, label, (x + 8, y + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1)
                break

    axes[0, 1].imshow(cv2.cvtColor(vis2, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title('[B] After Edge-Pseudo Merge (n=%d, gray X = merged-off pseudo endpoints)' % len(all_merged_records), fontsize=11)
    axes[0, 1].axis('off')

    # ----- Sub-figure C: final filtered endpoints -----
    vis3 = np.zeros((img_height, img_width, 3), dtype=np.uint8)
    vis3[:] = [255, 255, 255]
    draw_skeleton(vis3)

    for r in filtered_records:
        x, y = r['x'], r['y']
        cv2.circle(vis3, (x, y), 7, (0, 200, 0), -1)
        cv2.circle(vis3, (x, y), 7, (0, 0, 0), 1)
        global_idx = None
        for rr in all_raw_records:
            if rr['line_id'] == r['line_id'] and rr['ep_name'] == r['ep_name']:
                global_idx = rr['global_idx']
                break
        ep_digit = r['ep_name'].replace('ep', '')
        if global_idx:
            label = "G%s L%s*%s" % (global_idx, r['line_id'], ep_digit)
        else:
            label = "L%s*%s" % (r['line_id'], ep_digit)
        cv2.putText(vis3, label, (x + 8, y + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1)

    axes[1, 0].imshow(cv2.cvtColor(vis3, cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title('[C] Final Filtered Endpoints (after merge + on_edge filter, n=%d)' % len(filtered_records), fontsize=11)
    axes[1, 0].axis('off')

    # ----- Sub-figure D: connection result -----
    vis4 = np.ones((img_height, img_width, 3), dtype=np.uint8) * 255
    draw_skeleton(vis4)
    for match in matches:
        ep1, ep2 = match['ep1'], match['ep2']
        color = (0, 200, 0) if match.get('is_direct') else (0, 0, 255)
        vec1 = match.get('vec1')
        vec2 = match.get('vec2')
        dist = np.sqrt((ep2[0] - ep1[0]) ** 2 + (ep2[1] - ep1[1]) ** 2)
        num_points = max(100, int(dist * 5))
        pts = connect_with_cubic_spline(ep1, ep2, vec1, vec2, num_points=num_points)
        for x, y in pts:
            if 0 <= y < img_height and 0 <= x < img_width:
                vis4[y, x] = color
        cv2.circle(vis4, ep1, 4, (255, 0, 0), -1)
        cv2.circle(vis4, ep2, 4, (0, 0, 255), -1)
    axes[1, 1].imshow(cv2.cvtColor(vis4, cv2.COLOR_BGR2RGB))
    axes[1, 1].set_title('[D] Connected Skeleton (%d pairs, green=direct, red=spline)' % len(matches), fontsize=11)
    axes[1, 1].axis('off')

    # ----- Sub-figure E: zoom of A bottom-left (2x) -----
    zoom_x1, zoom_y1 = 0, 250
    zoom_x2, zoom_y2 = 200, 564

    crop_zoom_a = np.zeros(((zoom_y2 - zoom_y1) * 2, (zoom_x2 - zoom_x1) * 2, 3), dtype=np.uint8)
    crop_zoom_a[:] = [255, 255, 255]
    for line_id, line_data in skeleton_data['lines'].items():
        for x, y in line_data['pixel_coords']:
            if zoom_y1 <= y < zoom_y2 and zoom_x1 <= x < zoom_x2:
                nx = (x - zoom_x1) * 2
                ny = (y - zoom_y1) * 2
                for ddx in [-1, 0, 1]:
                    for ddy in [-1, 0, 1]:
                        nx2 = nx + ddx
                        ny2 = ny + ddy
                        cw, ch = (zoom_x2 - zoom_x1) * 2, (zoom_y2 - zoom_y1) * 2
                        if 0 <= nx2 < cw and 0 <= ny2 < ch:
                            crop_zoom_a[ny2, nx2] = [180, 180, 180]
    for r in all_raw_records:
        if r.get('missing', False):
            continue
        x, y = r['x'], r['y']
        if zoom_y1 <= y < zoom_y2 and zoom_x1 <= x < zoom_x2:
            nx = (x - zoom_x1) * 2
            ny = (y - zoom_y1) * 2
            if r['on_edge']:
                color = (0, 0, 255)
            else:
                color = (0, 200, 0)
            cv2.circle(crop_zoom_a, (nx, ny), 10, color, -1)
            cv2.circle(crop_zoom_a, (nx, ny), 10, (0, 0, 0), 2)
            label = "G%d" % r['global_idx']
            cv2.putText(crop_zoom_a, label, (nx + 10, ny + 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    axes[2, 0].imshow(cv2.cvtColor(crop_zoom_a, cv2.COLOR_BGR2RGB))
    axes[2, 0].set_title('[E] ZOOM of [A] bottom-left (raw v3 endpoints, %d points)' % len(all_raw_records), fontsize=11)
    axes[2, 0].axis('off')

    # ----- Sub-figure F: zoom of C bottom-left (2x) -----
    crop_zoom_c = np.zeros(((zoom_y2 - zoom_y1) * 2, (zoom_x2 - zoom_x1) * 2, 3), dtype=np.uint8)
    crop_zoom_c[:] = [255, 255, 255]
    for line_id, line_data in skeleton_data['lines'].items():
        for x, y in line_data['pixel_coords']:
            if zoom_y1 <= y < zoom_y2 and zoom_x1 <= x < zoom_x2:
                nx = (x - zoom_x1) * 2
                ny = (y - zoom_y1) * 2
                for ddx in [-1, 0, 1]:
                    for ddy in [-1, 0, 1]:
                        nx2 = nx + ddx
                        ny2 = ny + ddy
                        cw, ch = (zoom_x2 - zoom_x1) * 2, (zoom_y2 - zoom_y1) * 2
                        if 0 <= nx2 < cw and 0 <= ny2 < ch:
                            crop_zoom_c[ny2, nx2] = [180, 180, 180]
    for r in filtered_records:
        x, y = r['x'], r['y']
        if zoom_y1 <= y < zoom_y2 and zoom_x1 <= x < zoom_x2:
            nx = (x - zoom_x1) * 2
            ny = (y - zoom_y1) * 2
            cv2.circle(crop_zoom_c, (nx, ny), 10, (0, 200, 0), -1)
            cv2.circle(crop_zoom_c, (nx, ny), 10, (0, 0, 0), 2)
            global_idx = None
            for rr in all_raw_records:
                if rr['line_id'] == r['line_id'] and rr['ep_name'] == r['ep_name']:
                    global_idx = rr['global_idx']
                    break
            ep_digit = r['ep_name'].replace('ep', '')
            if global_idx:
                label = "G%s L%s*%s" % (global_idx, r['line_id'], ep_digit)
            else:
                label = "L%s*%s" % (r['line_id'], ep_digit)
            cv2.putText(crop_zoom_c, label, (nx + 10, ny + 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    axes[2, 1].imshow(cv2.cvtColor(crop_zoom_c, cv2.COLOR_BGR2RGB))
    axes[2, 1].set_title('[F] ZOOM of [C] bottom-left (final filtered, %d points)' % len(filtered_records), fontsize=11)
    axes[2, 1].axis('off')

    plt.suptitle('v4 Edge Filter + Pseudo Merge Comparison  (raw=%d, merged=%d, filtered=%d, matches=%d)' %
                 (len(all_raw_records), len(all_merged_records), len(filtered_records), len(matches)),
                 fontsize=14, y=0.998)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    buffer.seek(0)
    out_path = os.path.join(output_dir, 'breakpoint_v4_filter_compare.png')
    try:
        arr = np.frombuffer(buffer.getvalue(), dtype=np.uint8)
        img_arr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        ext = os.path.splitext(out_path)[1]
        result, encoded = cv2.imencode(ext, img_arr)
        if result:
            encoded.tofile(out_path)
            print("    Saved: breakpoint_v4_filter_compare.png (via cv2.imencode)")
        else:
            raise IOError("imencode failed")
    except Exception as e:
        print("    cv2.imencode failed (%s), trying fallback..." % e)
        with open(out_path, 'wb') as f:
            f.write(buffer.getvalue())
        print("    Saved: breakpoint_v4_filter_compare.png (via fallback)")
    plt.close()


# ============================================================
# Main Entry
# ============================================================
def main(data_dir=None):
    """End-to-end v4 pipeline: load skeleton, extract endpoints, match, connect, render.

    Parameters
    ----------
    data_dir : str or None, optional
        Folder containing ``skeleton_data.json``. When ``None`` the script
        defaults to ``<script_dir>/data/test_2025``. Output files are written
        to the same folder.
    """
    print("=" * 60)
    print("Breakpoint Connection v4 - Edge Filter")
    print("=" * 60)

    # Default data_dir for backward compatibility.
    if data_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(script_dir, 'data', 'test_2025')
    output_dir = data_dir

    log_lines = []
    log_lines.append("Breakpoint v4 - Edge Filter Comparison LOG")
    log_lines.append("=" * 80)

    # 1. Load skeleton data.
    print("\n[1] Loading skeleton data...")
    skeleton_path = os.path.join(data_dir, 'skeleton_data.json')
    with open(skeleton_path, 'r', encoding='utf-8') as f:
        skeleton_data = json.load(f)

    img_width = skeleton_data['metadata']['image_size']['width']
    img_height = skeleton_data['metadata']['image_size']['height']
    print("    Image size: %d x %d" % (img_width, img_height))
    print("    Total lines: %d" % len(skeleton_data['lines']))

    # 2. Extract endpoints and apply the edge filter.
    # `truncate` guarantees that skeleton lines keep at most one edge pixel
    # that is 8-connected to an interior pixel. The pipeline therefore needs
    # to:
    #   (a) extract each line's ep1/ep2 (standard: neighbors==1)
    #   (b) apply the edge filter: any endpoint closer than `edge_margin`
    #       pixels from the border is treated as on-edge and excluded.
    print("\n[2] Extracting endpoints and applying edge filter...")
    edge_margin = 5

    all_raw_endpoints = []  # raw endpoints (with on_edge flag)
    all_filtered_endpoints = []  # filtered endpoints (only non-edge ones, used for matching)

    for line_id, line_data in skeleton_data['lines'].items():
        ep_info = extract_endpoints_with_flags(line_data, img_width, img_height, edge_margin)
        coord_set = set((int(x), int(y)) for x, y in line_data['pixel_coords'])

        if ep_info is None:
            continue

        # ----- Collect raw endpoints -----
        for ep_name in ['ep1', 'ep2']:
            ep_pos = ep_info[ep_name]
            ep_on_edge = ep_info[ep_name + '_on_edge']
            vec = None
            if ep_pos is not None:
                vec = get_line_direction(coord_set, ep_pos, num_steps=30)
            all_raw_endpoints.append({
                'line_id': int(line_id),
                'ep_name': ep_name,
                'x': int(ep_pos[0]) if ep_pos else None,
                'y': int(ep_pos[1]) if ep_pos else None,
                'on_edge': bool(ep_on_edge),
                'present': ep_pos is not None,
                'vec': vec,
            })

            # ----- Edge filter: on-edge endpoints are excluded from matching -----
            if ep_pos is not None and not ep_on_edge:
                all_filtered_endpoints.append({
                    'line_id': int(line_id),
                    'ep_name': ep_name,
                    'x': int(ep_pos[0]),
                    'y': int(ep_pos[1]),
                    'on_edge': False,
                    'vec': vec,
                })

    print("    Lines processed: %d" % len(skeleton_data['lines']))
    print("    Raw endpoints (incl. on_edge): %d" % len(all_raw_endpoints))
    n_on_edge = sum(1 for e in all_raw_endpoints if e.get('on_edge'))
    print("    On-edge endpoints (filtered out): %d" % n_on_edge)
    print("    After edge filter (final, used for matching): %d" % len(all_filtered_endpoints))

    # 3. Prepare matching data (only filtered_endpoints are used).
    print("\n[3] Preparing matching data (from filtered endpoints only)...")
    endpoints_for_matching = []
    for r in all_filtered_endpoints:
        endpoints_for_matching.append((
            r['line_id'],
            (r['x'], r['y']),
            r['vec'],
            True if r['ep_name'] == 'ep1' else False
        ))

    print("    Total endpoints for matching: %d" % len(endpoints_for_matching))

    # 4. Run the v4 matching algorithm (identical core to v3).
    print("\n[4] Matching endpoints (paper v4 algorithm)...")
    matches = match_endpoints_v4(
        endpoints_for_matching,
        skeleton_data,
        img_width, img_height,
        log_lines,
        direct_threshold=13.5,
        max_distance=55,
        search_range=45
    )
    print("    Total matches: %d" % len(matches))

    # 4.1 Cross-connection check.
    print("\n[4.5] Checking for cross connections...")
    endpoint_list = []
    for i, (lid, ep, vec, is_ep1) in enumerate(endpoints_for_matching):
        line_info = skeleton_data['lines'].get(str(lid))
        if line_info and 'pixel_coords' in line_info:
            coord_set = set((int(c[0]), int(c[1])) for c in line_info['pixel_coords'])
        else:
            coord_set = set()
        local_dir, global_dir, main_dir, _ = get_endpoint_direction_code(coord_set, ep, num_steps=30)
        endpoint_list.append({
            'global_idx': i + 1,
            'lid': lid,
            'ep': ep,
            'vec': vec,
            'code': main_dir,
            'local_dir': local_dir,
            'global_dir': global_dir,
            'is_ep1': is_ep1,
        })

    matches, cross_info = check_cross_connections(matches, endpoint_list, max_rematch_dist=60, log_lines=log_lines)
    print("    Cross connections found: %d" % len(cross_info))
    print("    Matches after cross-check: %d" % len(matches))

    iteration = 1
    while cross_info:
        iteration += 1
        log_lines.append("")
        log_lines.append("=" * 100)
        log_lines.append("[Cross-check round %d]" % iteration)
        log_lines.append("=" * 100)
        matches, cross_info = check_cross_connections(matches, endpoint_list, max_rematch_dist=60, log_lines=log_lines)
        print("    Round %d: Cross connections found: %d, Matches: %d" % (iteration, len(cross_info), len(matches)))
        if iteration > 10:
            log_lines.append("Maximum iterations reached, stopping cross check.")
            break

    print("    Final matches: %d" % len(matches))

    # 5. Connect endpoints using cubic-Bezier splines.
    print("\n[5] Connecting with Cubic Spline...")
    connected_skeleton = np.zeros((img_height, img_width), dtype=np.uint8)

    for line_id, line_data in skeleton_data['lines'].items():
        for x, y in line_data['pixel_coords']:
            if 0 <= y < img_height and 0 <= x < img_width:
                connected_skeleton[y, x] = 255

    connection_records = []
    all_curve_pixels = []

    for i, match in enumerate(matches):
        ep1, ep2 = match['ep1'], match['ep2']
        line1, line2 = match['line1'], match['line2']

        vec1 = match.get('vec1')
        vec2 = match.get('vec2')

        dist = np.sqrt((ep2[0] - ep1[0]) ** 2 + (ep2[1] - ep1[1]) ** 2)
        num_points = max(100, int(dist * 5))

        spline_points = connect_with_cubic_spline(ep1, ep2, vec1, vec2, num_points=num_points)
        for x, y in spline_points:
            if 0 <= y < img_height and 0 <= x < img_width:
                connected_skeleton[y, x] = 255

        curve_pixel_coords = [{'x': int(x), 'y': int(y)} for x, y in spline_points
                              if 0 <= y < img_height and 0 <= x < img_width]

        connection_records.append({
            'conn_id': i + 1,
            'line1': line1,
            'line2': line2,
            'ep1': {'x': ep1[0], 'y': ep1[1]},
            'ep2': {'x': ep2[0], 'y': ep2[1]},
            'code1': match.get('code1'),
            'code2': match.get('code2'),
            'distance': match['distance'],
            'is_direct': match.get('is_direct', False),
            'curve_pixels': curve_pixel_coords
        })

        print("    Connection %d: Line%d(%s) <-> Line%d(%s), dist=%.1f, code=%s->%s" % (
            i + 1, line1, ep1, line2, ep2,
            match['distance'], match.get('code1'), match.get('code2')))

    # 6. Save results.
    print("\n[6] Saving results...")
    result_data = {
        'metadata': {
            'image_size': {'width': img_width, 'height': img_height},
            'method': 'v4_edge_filter_comparison'
        },
        'total_raw_endpoints': len(all_raw_endpoints),
        'total_filtered_endpoints': len(all_filtered_endpoints),
        'total_connections': len(matches),
        'connections': connection_records
    }
    with open(os.path.join(output_dir, 'breakpoint_v4_result.json'), 'w', encoding='utf-8') as f:
        json.dump(result_data, f, indent=2, ensure_ascii=False)
    print("    Saved: breakpoint_v4_result.json")

    # Save curve pixels.
    curve_coords_data = {
        'metadata': {
            'image_size': {'width': img_width, 'height': img_height},
            'total_connections': len(matches)
        },
        'connections': [{
            'conn_id': conn['conn_id'],
            'line1': conn['line1'],
            'line2': conn['line2'],
            'ep1': conn['ep1'],
            'ep2': conn['ep2'],
            'curve_pixels': conn['curve_pixels']
        } for conn in connection_records]
    }
    with open(os.path.join(output_dir, 'connection_curve_coords.json'), 'w', encoding='utf-8') as f:
        json.dump(curve_coords_data, f, indent=2, ensure_ascii=False)
    print("    Saved: connection_curve_coords.json")

    connected_skeleton_inverted = 255 - connected_skeleton
    cv2.imwrite(os.path.join(output_dir, 'connected_skeleton_v4.png'), connected_skeleton_inverted)
    if not imwrite_unicode(os.path.join(output_dir, 'connected_skeleton_v4.png'), connected_skeleton_inverted):
        try:
            np.save(os.path.join(output_dir, 'connected_skeleton_v4.npy'), connected_skeleton_inverted)
        except Exception:
            pass
    print("    Saved: connected_skeleton_v4.png")

    # 7. Generate merged lines CSV.
    print("\n[7] Generating merged lines coordinates CSV...")
    merged_lines = []

    line_connections = {}
    for match in matches:
        line1, line2 = match['line1'], match['line2']
        ep1, ep2 = match['ep1'], match['ep2']
        vec1 = match.get('vec1')
        vec2 = match.get('vec2')

        if line1 not in line_connections:
            line_connections[line1] = []
        if line2 not in line_connections:
            line_connections[line2] = []

        line_connections[line1].append((line2, ep1, vec1))
        line_connections[line2].append((line1, ep2, vec2))

    processed_lines = set()
    for start_line in line_connections.keys():
        if start_line in processed_lines:
            continue

        component_lines = []
        queue = [start_line]
        visited = set()

        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)

            ep_at_current = None
            vec_at_current = None

            if current in line_connections:
                for connected_line, ep, vec in line_connections[current]:
                    if current == start_line:
                        ep_at_current = ep
                        vec_at_current = vec
                    if connected_line not in visited:
                        queue.append(connected_line)

            component_lines.append((current, ep_at_current, vec_at_current))

        for line_id, _, _ in component_lines:
            processed_lines.add(line_id)

        if len(component_lines) == 1:
            line_id = component_lines[0][0]
            line_data = skeleton_data['lines'].get(str(line_id))
            if line_data:
                coords = [(int(c[0]), int(c[1])) for c in line_data['pixel_coords']]
                if coords:
                    merged_lines.append((len(merged_lines) + 1, coords))
            continue

        ordered_lines = []
        remaining = list(component_lines)
        current = remaining.pop(0)
        ordered_lines.append(current)

        while remaining:
            current_line_id = ordered_lines[-1][0]
            found = False
            for i, (lid, ep, vec) in enumerate(remaining):
                if current_line_id in [conn[0] for conn in line_connections.get(lid, [])]:
                    ordered_lines.append(remaining.pop(i))
                    found = True
                    break
            if not found:
                ordered_lines.append(remaining.pop(0))

        merged_coords = []
        for line_idx, (line_id, ep, vec) in enumerate(ordered_lines):
            line_data = skeleton_data['lines'].get(str(line_id))
            if not line_data:
                continue

            coords = [(int(c[0]), int(c[1])) for c in line_data['pixel_coords']]

            if line_idx == 0:
                merged_coords.extend(coords)
            else:
                prev_line_id = ordered_lines[line_idx - 1][0]
                prev_ep = ordered_lines[line_idx - 1][1]
                prev_vec = ordered_lines[line_idx - 1][2]

                prev_ep_conn = None
                current_ep_conn = None
                for conn_line, conn_ep, conn_vec in line_connections.get(prev_line_id, []):
                    if conn_line == line_id:
                        prev_ep_conn = conn_ep
                        current_ep_conn = ep
                        break

                if prev_ep_conn and current_ep_conn:
                    dist = np.sqrt((current_ep_conn[0] - prev_ep_conn[0]) ** 2 +
                                   (current_ep_conn[1] - prev_ep_conn[1]) ** 2)
                    num_points = max(100, int(dist * 5))
                    spline_points = connect_with_cubic_spline(
                        prev_ep_conn, current_ep_conn, prev_vec, vec, num_points=num_points)

                    for pt in spline_points[1:]:
                        merged_coords.append((int(pt[0]), int(pt[1])))

                merged_coords.extend(coords)

        seen = set()
        unique_coords = []
        for coord in merged_coords:
            if coord not in seen:
                seen.add(coord)
                unique_coords.append(coord)

        if unique_coords:
            merged_lines.append((len(merged_lines) + 1, unique_coords))

    # Append skeleton lines that were not part of any connection.
    for lid_str, line_data in skeleton_data['lines'].items():
        lid = int(lid_str)
        if lid not in line_connections:
            coords = [(int(c[0]), int(c[1])) for c in line_data['pixel_coords']]
            if coords:
                merged_lines.append((len(merged_lines) + 1, coords))

    csv_path = os.path.join(output_dir, 'Breakpoint_v4_match_lines_coor.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['line_id', 'x', 'y'])
        for line_id, coords in sorted(merged_lines, key=lambda x: x[0]):
            for x, y in coords:
                writer.writerow([line_id, x, y])

    print("    Saved: Breakpoint_v4_match_lines_coor.csv")

    # 8. Save the log.
    log_path = os.path.join(output_dir, 'breakpoint_v4_debug_log.txt')
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(log_lines))
    print("    Saved: breakpoint_v4_debug_log.txt")

    # 9. Save all raw and filtered endpoints to JSON.
    print("\n[9] Saving raw and filtered endpoints JSON...")
    all_raw_export = []
    for idx, r in enumerate(all_raw_endpoints, 1):
        line_info = skeleton_data['lines'].get(str(r['line_id']))
        if line_info and r['present']:
            coord_set = set((int(c[0]), int(c[1])) for c in line_info['pixel_coords'])
        else:
            coord_set = set()
        local_dir, global_dir, main_dir, _ = (None, None, None, None)
        if r['present']:
            local_dir, global_dir, main_dir, _ = get_endpoint_direction_code(
                coord_set, (r['x'], r['y']), num_steps=30)
        all_raw_export.append({
            'global_idx': idx,
            'line_id': r['line_id'],
            'ep_name': r['ep_name'],
            'x': r['x'],
            'y': r['y'],
            'on_edge': r['on_edge'],
            'present': r['present'],
            'local_dir': local_dir,
            'global_dir': global_dir,
            'main_dir': main_dir,
            'code': main_dir,
            'code_name': CODE_NAMES.get(main_dir, 'unknown') if main_dir is not None else 'unknown',
            'compatible_codes': list(DIRECTION_MATCH_RULES.get(main_dir, set())) if main_dir is not None else [],
            'kept_after_filter': (r['present'] and not r['on_edge']),
        })

    with open(os.path.join(output_dir, 'breakpoint_v4_all_endpoints.json'), 'w', encoding='utf-8') as f:
        json.dump(all_raw_export, f, indent=2, ensure_ascii=False)
    print("    Saved: breakpoint_v4_all_endpoints.json (raw endpoints, n=%d)" % len(all_raw_export))

    # Save filtered endpoints.
    raw_key_to_idx = {(r['line_id'], r['ep_name']): i + 1 for i, r in enumerate(all_raw_endpoints)}
    filtered_export = []
    for r in all_filtered_endpoints:
        line_info = skeleton_data['lines'].get(str(r['line_id']))
        coord_set = set((int(c[0]), int(c[1])) for c in line_info['pixel_coords']) \
            if line_info else set()
        local_dir, global_dir, main_dir, _ = get_endpoint_direction_code(
            coord_set, (r['x'], r['y']), num_steps=30)
        filtered_export.append({
            'global_idx': raw_key_to_idx.get((r['line_id'], r['ep_name']), None),
            'line_id': r['line_id'],
            'ep_name': r['ep_name'],
            'x': r['x'],
            'y': r['y'],
            'on_edge': r['on_edge'],
            'local_dir': local_dir,
            'global_dir': global_dir,
            'main_dir': main_dir,
            'code': main_dir,
            'code_name': CODE_NAMES.get(main_dir, 'unknown') if main_dir is not None else 'unknown',
            'compatible_codes': list(DIRECTION_MATCH_RULES.get(main_dir, set())) if main_dir is not None else [],
        })

    with open(os.path.join(output_dir, 'breakpoint_v4_filtered_endpoints.json'), 'w', encoding='utf-8') as f:
        json.dump(filtered_export, f, indent=2, ensure_ascii=False)
    print("    Saved: breakpoint_v4_filtered_endpoints.json (filtered, n=%d)" % len(filtered_export))

    # 10. Render the visualization.
    print("\n[10] Generating visualization...")
    visualize_v4_results(
        skeleton_data, all_raw_endpoints, all_filtered_endpoints,
        matches, img_width, img_height, output_dir
    )

    # 11. Verification.
    print("\n[11] Verification...")
    num_original, _, _, _ = cv2.connectedComponentsWithStats(connected_skeleton, connectivity=8)
    print("    Components after: %d" % (num_original - 1))
    print("    Total pixels: %d" % int(np.sum(connected_skeleton > 0)))

    print("\n" + "=" * 60)
    print("Done! %d connections made." % len(matches))
    print("=" * 60)
    print("\nOutput files:")
    print("  - breakpoint_v4_all_endpoints.json      (all raw endpoints)")
    print("  - breakpoint_v4_filtered_endpoints.json (filtered endpoints)")
    print("  - breakpoint_v4_result.json             (connection result)")
    print("  - connected_skeleton_v4.png             (skeleton image - black/white)")
    print("  - breakpoint_v4_debug.png               (debug visualization)")


if __name__ == '__main__':
    import sys
    # CLI: python breakpoint_v4_debug.py [data_dir]
    data_dir_arg = sys.argv[1] if len(sys.argv) > 1 else None
    main(data_dir=data_dir_arg)