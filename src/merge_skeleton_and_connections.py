# -*- coding: utf-8 -*-
"""
Merge the original skeleton with the connection curves and re-run the
connectivity analysis.

The output has the same shape as ``skeleton_data.json`` (one entry per line
with ``pixel_coords``), but the lines are recomputed from the union of the
skeleton pixels and every connection curve. The result therefore reflects
both the original line geometry and the bridging curves that connect breaks.

Typical call from ``pipeline.py``:
    main(data_dir=..., result_filename='breakpoint_v4_result.json',
         output_filename='merged_skeleton_data.json',
         mask_filename='merged_skeleton_mask.png')
"""

import cv2
import numpy as np
import json
import os
from collections import defaultdict


def imwrite_unicode(path, img):
    """
    Write ``img`` to ``path`` even when the path contains non-ASCII characters.

    Parameters
    ----------
    path : str
        Destination path (extension determines the encoding).
    img : numpy.ndarray
        Image to write.

    Returns
    -------
    bool
        ``True`` on success, ``False`` otherwise.
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
    except:
        return False


def bresenham_line(x0, y0, x1, y1):
    """
    Discrete line via Bresenham's algorithm.

    Parameters
    ----------
    x0, y0 : int
        Starting pixel coordinate.
    x1, y1 : int
        Ending pixel coordinate.

    Returns
    -------
    list of tuple
        Integer pixel coordinates on the line, including both endpoints.
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


def get_neighbor_pixels(mask, x, y):
    """
    Return the 8-neighbouring skeleton pixels of ``(x, y)``.

    Parameters
    ----------
    mask : numpy.ndarray
        Binary mask.
    x, y : int
        Query pixel coordinate.

    Returns
    -------
    list of tuple
        ``(x, y)`` coordinates of neighbours whose mask value is > 0.
    """
    neighbors = []
    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            if dx == 0 and dy == 0:
                continue
            ny, nx = y + dy, x + dx
            if 0 <= ny < mask.shape[0] and 0 <= nx < mask.shape[1]:
                if mask[ny, nx] > 0:
                    neighbors.append((nx, ny))
    return neighbors


def find_all_endpoints(mask):
    """
    Find every endpoint pixel in ``mask`` (pixels with only 0 or 1 neighbour).

    Parameters
    ----------
    mask : numpy.ndarray
        Binary mask.

    Returns
    -------
    list of tuple
        ``(x, y)`` of every endpoint pixel.
    """
    h, w = mask.shape
    endpoints = []

    for y in range(h):
        for x in range(w):
            if mask[y, x] > 0:
                neighbors = get_neighbor_pixels(mask, x, y)
                if len(neighbors) == 1:  # Endpoint has exactly 1 neighbour.
                    endpoints.append((x, y))

    return endpoints


def trace_line_from_pixel(mask, start, visited):
    """
    Iteratively trace every pixel reachable from ``start`` via 8-connectivity.

    Parameters
    ----------
    mask : numpy.ndarray
        Binary mask.
    start : tuple
        ``(x, y)`` of the starting pixel.
    visited : numpy.ndarray
        Boolean visited map, modified in-place.

    Returns
    -------
    list of tuple
        All pixels visited during this trace.
    """
    h, w = mask.shape
    line_pixels = []
    stack = [start]

    while stack:
        x, y = stack.pop()
        if visited[y, x]:
            continue
        visited[y, x] = True
        line_pixels.append((x, y))

        # Look at the 8 neighbours.
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and not visited[ny, nx]:
                    if mask[ny, nx] > 0:
                        stack.append((nx, ny))

    return line_pixels


def extract_lines_from_mask(mask, min_length=5):
    """
    Extract every connected line segment from a binary mask.

    Parameters
    ----------
    mask : numpy.ndarray
        Binary mask.
    min_length : int, optional
        Minimum number of pixels required to keep a segment (default 5).

    Returns
    -------
    dict
        ``{line_id: {'pixel_coords': [{'x', 'y'}, ...], 'length': int}}``.
    """
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    lines = {}

    line_id = 1
    for y in range(h):
        for x in range(w):
            if mask[y, x] > 0 and not visited[y, x]:
                # Trace the entire line.
                line_pixels = trace_line_from_pixel(mask, (x, y), visited)

                if len(line_pixels) >= min_length:
                    lines[line_id] = {
                        'pixel_coords': [{'x': p[0], 'y': p[1]} for p in line_pixels],
                        'length': len(line_pixels)
                    }
                    line_id += 1

    return lines


def merge_original_and_connections(skeleton_json, result_json, output_json, output_img=None,
                                     extra_connections_json=None):
    """
    Merge the original skeleton with one (or two) sets of connection curves
    and re-extract line segments.

    Parameters
    ----------
    skeleton_json : str
        Path to the original ``skeleton_data.json``.
    result_json : str
        Path to the connection-result JSON
        (e.g. ``breakpoint_v4_result.json``).
    output_json : str
        Path where the merged JSON will be written.
    output_img : str or None, optional
        Path where the merged mask image will be written. ``None`` skips the
        PNG export.
    extra_connections_json : str or None, optional
        Optional path to a second connection result (e.g. SA output) that
        should also be merged in.

    Returns
    -------
    dict
        The merged output payload (same structure that is also written to
        ``output_json``).
    """

    # 1. Load the original skeleton.
    print("Loading skeleton data...")
    with open(skeleton_json, 'r', encoding='utf-8') as f:
        skeleton_data = json.load(f)

    img_height = skeleton_data['metadata']['image_size']['height']
    img_width = skeleton_data['metadata']['image_size']['width']
    print(f"  Image size: {img_width} x {img_height}")
    print(f"  Original lines: {len(skeleton_data['lines'])}")

    # 2. Load the connection result.
    print("Loading connection results...")
    with open(result_json, 'r', encoding='utf-8') as f:
        result_data = json.load(f)

    print(f"  Connections: {len(result_data['connections'])}")

    # 2b. Load an optional second connection result.
    extra_data = None
    if extra_connections_json and os.path.isfile(extra_connections_json):
        print("Loading extra connection results...")
        with open(extra_connections_json, 'r', encoding='utf-8') as f:
            extra_data = json.load(f)
        print(f"  Extra connections: {len(extra_data['connections'])}")

    # 3. Build the merged binary mask.
    print("Creating merged mask...")

    # Debug: check the data structure.
    sample_conn = result_data['connections'][0]
    sample_pt = sample_conn['curve_pixels'][0]
    print(f"    DEBUG: sample_pt type = {type(sample_pt)}, value = {sample_pt}")

    merged_mask = np.zeros((img_height, img_width), dtype=np.uint8)

    # Helper: parse either ``{"x": ..., "y": ...}`` dicts or ``[x, y]`` lists.
    def parse_point(pt):
        if isinstance(pt, dict):
            return pt['x'], pt['y']
        elif isinstance(pt, (list, tuple)):
            return pt[0], pt[1]
        else:
            raise ValueError(f"Unknown point format: {pt}")

    # Paint the original skeleton (value 1).
    for line_id, line_data in skeleton_data['lines'].items():
        for pt in line_data['pixel_coords']:
            x, y = parse_point(pt)
            if 0 <= y < img_height and 0 <= x < img_width:
                merged_mask[y, x] = 255

    # Paint the connection curves (value 255, overwrites original pixels).
    for i, conn in enumerate(result_data['connections']):
        curve_pixels = conn['curve_pixels']
        if i < 3:
            print(f"    DEBUG conn {i}: curve_pixels type = {type(curve_pixels)}, "
                  f"len = {len(curve_pixels)}, "
                  f"first elem type = {type(curve_pixels[0])}")
        for j, pt in enumerate(curve_pixels):
            x, y = parse_point(pt)
            if 0 <= y < img_height and 0 <= x < img_width:
                merged_mask[y, x] = 255

    # Paint the extra connection curves too, if present.
    if extra_data is not None:
        print("  Drawing extra connections...")
        for i, conn in enumerate(extra_data['connections']):
            curve_pixels = conn['curve_pixels']
            for pt in curve_pixels:
                x, y = parse_point(pt)
                if 0 <= y < img_height and 0 <= x < img_width:
                    merged_mask[y, x] = 255

    original_pixels = np.sum(merged_mask == 255)
    total_connections = len(result_data['connections'])
    if extra_data is not None:
        total_connections += len(extra_data['connections'])
    print(f"  Total skeleton pixels: {original_pixels}")

    # 4. Re-extract the line segments from the merged mask.
    print("Extracting lines from merged mask...")
    new_lines = extract_lines_from_mask(merged_mask, min_length=5)
    print(f"  Extracted lines: {len(new_lines)}")

    # 5. Save the merged mask as an image (white background, black lines).
    if output_img:
        output_mask = 255 - merged_mask
        imwrite_unicode(output_img, output_mask)
        print(f"  Saved mask: {output_img}")

    # 6. Build the output payload.
    output_data = {
        'metadata': {
            'image_size': {'width': img_width, 'height': img_height},
            'original_lines': len(skeleton_data['lines']),
            'connections': total_connections,
            'method': 'merged_skeleton_v3'
        },
        'lines': new_lines
    }

    # 7. Write the JSON.
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print(f"  Saved JSON: {output_json}")

    # 8. Summary statistics.
    total_pixels = sum(line['length'] for line in new_lines.values())
    avg_length = total_pixels / len(new_lines) if new_lines else 0
    print(f"\nStatistics:")
    print(f"  Total lines: {len(new_lines)}")
    print(f"  Total pixels: {total_pixels}")
    print(f"  Average line length: {avg_length:.1f}")

    return output_data


def main(data_dir=None, result_filename='breakpoint_v4_result.json',
         output_filename='merged_skeleton_data.json',
         mask_filename='merged_skeleton_mask.png',
         extra_connections_json=None):
    """
    CLI entry point: merge skeleton and connections under ``data_dir``.

    Parameters
    ----------
    data_dir : str or None, optional
        Directory containing ``skeleton_data.json``. ``None`` falls back to
        the legacy ``data/test_2025`` directory.
    result_filename : str, optional
        Filename of the breakpoint-matching result JSON
        (default ``'breakpoint_v4_result.json'``).
    output_filename : str, optional
        Filename of the merged JSON output
        (default ``'merged_skeleton_data.json'``).
    mask_filename : str, optional
        Filename of the merged mask PNG
        (default ``'merged_skeleton_mask.png'``).
    extra_connections_json : str or None, optional
        Optional path to a second connection result that should also be
        merged in.
    """
    print("=" * 60)
    print("Merge Skeleton + Connections & Re-analyse")
    print("=" * 60)

    if data_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(script_dir, 'data', 'test_2025')

    skeleton_json = os.path.join(data_dir, 'skeleton_data.json')
    result_json = os.path.join(data_dir, result_filename)
    output_json = os.path.join(data_dir, output_filename)
    output_img = os.path.join(data_dir, mask_filename)

    # Run the merge and re-analysis.
    merge_original_and_connections(skeleton_json, result_json, output_json, output_img,
                                   extra_connections_json=extra_connections_json)

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    import sys
    # CLI: python merge_skeleton_and_connections.py [data_dir] [result_filename]
    #                                       [output_filename] [mask_filename]
    #                                       [extra_connections]
    args = sys.argv[1:]
    data_dir_arg = args[0] if len(args) > 0 else None
    result_fn_arg = args[1] if len(args) > 1 else 'breakpoint_v4_result.json'
    output_fn_arg = args[2] if len(args) > 2 else 'merged_skeleton_data.json'
    mask_fn_arg = args[3] if len(args) > 3 else 'merged_skeleton_mask.png'
    extra_arg = args[4] if len(args) > 4 else None
    main(data_dir=data_dir_arg, result_filename=result_fn_arg,
         output_filename=output_fn_arg, mask_filename=mask_fn_arg,
         extra_connections_json=extra_arg)