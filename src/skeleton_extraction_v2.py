# -*- coding: utf-8 -*-
"""
Contour-line skeleton extraction - connected-component version.

Each connected line is extracted independently: a per-component skeleton is
computed, numbered, and stored with its pixel coordinates.

Pipeline (per component):
    1. 1-pixel dilation to glue small breaks.
    2. Zhang-Suen thinning via :func:`cv2.ximgproc.thining`.
    3. Optional "edge-extension truncation" that compresses a chain of
       edge-touching pixels into a single edge pixel while keeping the
       connection to the line's interior intact.

Outputs (under ``output_dir``):
    - ``skeleton_figure6.png``: white-background, black-line combined skeleton.
    - ``skeleton_data.json``: per-line skeleton length, centroid, and pixel
      coordinates (sorted by row then column).
    - ``skeletons/line_XXXX.png``: individual per-line skeleton images.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot as pylab


def imread_unicode(path):
    """
    Read an image even when its path contains non-ASCII characters.

    Parameters
    ----------
    path : str
        Image path.

    Returns
    -------
    numpy.ndarray or None
        BGR image, or ``None`` if decoding fails.
    """
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def find_connected_components(binary):
    """
    Locate all connected components (each independent line) in a binary mask.

    Parameters
    ----------
    binary : numpy.ndarray
        8-bit single-channel binary mask (white = foreground).

    Returns
    -------
    num_labels : int
        Number of labels (background included).
    labels : numpy.ndarray
        Integer label map, same shape as ``binary``.
    stats : numpy.ndarray
        Per-label statistics (``cv2.CC_STAT_*``).
    centroids : numpy.ndarray
        Per-label centroid ``(x, y)``.
    """
    # Connected-component stats over 8-connectivity.
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    return num_labels, labels, stats, centroids


def truncate_edge_extensions(skeleton, edge_margin=5):
    """
    Trim the spurious edge extensions that Zhang-Suen thinning leaves behind.

    Near the image border the thinning algorithm produces a run of N edge
    pixels (e.g. 6 horizontal pixels along the top edge) because of the
    asymmetric border handling. The v3 endpoint detector interprets each of
    these as a separate endpoint, which is incorrect. This helper compresses
    each "edge run" into a single edge pixel while preserving all the
    interior pixels that genuinely belong to the line.

    Strategy
    --------
    - ``cluster`` = 8-connected skeleton pixels within ``edge_margin``.
    - "along-edge" pixels (those lying strictly on the outermost row/column)
      are mostly deleted.
    - "internal-direction" pixels (inside ``edge_margin`` but not on the
      outermost row/column) are always kept because they really are part of
      the line's interior.
    - Among the along-edge pixels of each cluster, exactly one is kept: the
      one that is 8-adjacent to an internal-direction pixel, or (if no such
      pixel exists) the one closest to the inside.

    Parameters
    ----------
    skeleton : numpy.ndarray
        Binary skeleton image (0/255) to be modified in-place.
    edge_margin : int, optional
        Pixel distance from the image border considered "edge" (default 5).

    Returns
    -------
    tuple (skeleton, n_deleted)
        The (modified) skeleton and the number of pixels that were deleted.
    """
    h, w = skeleton.shape
    to_delete = set()

    # Collect edge pixels within edge_margin, bucketed by side of the image.
    edge_pixels_by_side = {'top': set(), 'bottom': set(), 'left': set(), 'right': set()}
    for y in range(h):
        for x in range(w):
            if skeleton[y, x] == 0:
                continue
            if y < edge_margin:
                edge_pixels_by_side['top'].add((x, y))
            elif y >= h - edge_margin:
                edge_pixels_by_side['bottom'].add((x, y))
            elif x < edge_margin:
                edge_pixels_by_side['left'].add((x, y))
            elif x >= w - edge_margin:
                edge_pixels_by_side['right'].add((x, y))

    # Cluster pixels per edge via 8-connectivity.
    for edge_name, pixel_set in edge_pixels_by_side.items():
        if not pixel_set:
            continue

        visited = set()
        for px, py in pixel_set:
            if (px, py) in visited:
                continue
            cluster = []
            queue = [(px, py)]
            visited.add((px, py))
            while queue:
                cx, cy = queue.pop()
                cluster.append((cx, cy))
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        if dx == 0 and dy == 0:
                            continue
                        nx, ny = cx + dx, cy + dy
                        if (nx, ny) in pixel_set and (nx, ny) not in visited:
                            visited.add((nx, ny))
                            queue.append((nx, ny))

            # ── Classify the cluster ──
            # along_edge: pixels on the outermost row/column -> largely deleted
            # internal_dir: pixels inside edge_margin but not on the outermost
            #              row/column -> real interior -> always kept
            along_edge = []
            internal_dir = []
            for cx, cy in cluster:
                is_strict_edge = False
                if edge_name == 'top' and cy == 0:
                    is_strict_edge = True
                elif edge_name == 'bottom' and cy == h - 1:
                    is_strict_edge = True
                elif edge_name == 'left' and cx == 0:
                    is_strict_edge = True
                elif edge_name == 'right' and cx == w - 1:
                    is_strict_edge = True

                if is_strict_edge:
                    along_edge.append((cx, cy))
                else:
                    internal_dir.append((cx, cy))

            # internal_dir pixels are always kept.
            # Keep exactly one along_edge pixel: the one 8-adjacent to an
            # internal_dir pixel, so that the endpoint stays on the image
            # border instead of being pulled inward.
            kept_along = None
            if internal_dir:
                internal_set = set(internal_dir)
                for cx, cy in along_edge:
                    has_internal_neighbor = False
                    for dy in [-1, 0, 1]:
                        for dx in [-1, 0, 1]:
                            if dx == 0 and dy == 0:
                                continue
                            nx, ny = cx + dx, cy + dy
                            if (nx, ny) in internal_set:
                                has_internal_neighbor = True
                                break
                        if has_internal_neighbor:
                            break
                    if has_internal_neighbor:
                        kept_along = (cx, cy)
                        break
            else:
                # Cluster lies entirely on the edge (rare): keep the pixel
                # closest to the inside.
                if not along_edge:
                    continue
                if edge_name == 'top':
                    kept_along = max(along_edge, key=lambda p: p[1])
                elif edge_name == 'bottom':
                    kept_along = min(along_edge, key=lambda p: p[1])
                elif edge_name == 'left':
                    kept_along = max(along_edge, key=lambda p: p[0])
                elif edge_name == 'right':
                    kept_along = min(along_edge, key=lambda p: p[0])

            # Delete every along_edge pixel except the one we want to keep.
            for p in along_edge:
                if p != kept_along:
                    to_delete.add(p)

    for x, y in to_delete:
        skeleton[y, x] = 0

    return skeleton, len(to_delete)


def extract_skeleton_per_component(binary_dilated, labels, num_labels, edge_margin=5, truncate_edge=True):
    """
    Thin every connected component independently.

    For each non-background label in ``labels`` the dilated mask is isolated
    and thinned. If the thinned result has fewer than 10 % of the original
    pixels, the original mask is used instead (i.e. thinning is rejected).

    Parameters
    ----------
    binary_dilated : numpy.ndarray
        Binary mask after the 1-pixel dilation step.
    labels : numpy.ndarray
        Integer label map returned by
        :func:`cv2.connectedComponentsWithStats`.
    num_labels : int
        Number of labels (including background).
    edge_margin : int, optional
        Edge-distance threshold passed to :func:`truncate_edge_extensions`
        (default 5).
    truncate_edge : bool, optional
        Whether to apply the edge-extension truncation (default ``True``).
        When ``False``, the original (pre-truncation) behaviour is preserved.

    Returns
    -------
    tuple (skeletons, truncate_stats)
        ``skeletons`` is a dict ``{label: skeleton_image}`` containing one
        skeleton per component. ``truncate_stats`` is a dictionary with
        diagnostic counters (total components processed, edge-pixel counts
        before/after, and the number of pixels deleted by the truncation).
    """
    import cv2
    h, w = binary_dilated.shape
    skeletons = {}
    truncate_stats = {
        'total_lines': 0,
        'edge_pixels_before': 0,
        'edge_pixels_after': 0,
        'pixels_deleted': 0,
    }

    for label in range(1, num_labels):  # Skip background (label 0).
        # Mask for the current component.
        component_mask = (labels == label).astype(np.uint8) * 255

        # Skip tiny components.
        area = np.count_nonzero(component_mask)
        if area < 10:
            continue

        truncate_stats['total_lines'] += 1

        # Thin the mask without an edge guard so the algorithm produces a
        # natural skeleton.
        try:
            skeleton = cv2.ximgproc.thinning(component_mask)

            # Diagnostic: count edge pixels before truncation.
            skel_edge_before = (
                np.count_nonzero(skeleton[:edge_margin, :]) +
                np.count_nonzero(skeleton[-edge_margin:, :]) +
                np.count_nonzero(skeleton[:, :edge_margin]) +
                np.count_nonzero(skeleton[:, -edge_margin:])
            )
            truncate_stats['edge_pixels_before'] += skel_edge_before

            # Edge-extension truncation.
            if truncate_edge:
                skeleton, deleted_count = truncate_edge_extensions(skeleton, edge_margin)
                truncate_stats['pixels_deleted'] += deleted_count

            # Diagnostic: count edge pixels after truncation.
            skel_edge_after = (
                np.count_nonzero(skeleton[:edge_margin, :]) +
                np.count_nonzero(skeleton[-edge_margin:, :]) +
                np.count_nonzero(skeleton[:, :edge_margin]) +
                np.count_nonzero(skeleton[:, -edge_margin:])
            )
            truncate_stats['edge_pixels_after'] += skel_edge_after

            # If thinning destroyed too much, fall back to the dilated mask.
            skeleton_area = np.count_nonzero(skeleton)
            if skeleton_area < area * 0.1:  # Less than 10 % -> reject.
                skeletons[label] = component_mask
            else:
                skeletons[label] = skeleton
        except:
            # If anything goes wrong, keep the dilated mask.
            skeletons[label] = component_mask

    return skeletons, truncate_stats


def create_skeleton_image(skeletons, h, w):
    """
    Compose a single image from every per-line skeleton.

    Parameters
    ----------
    skeletons : dict
        ``{label: skeleton_image}`` as produced by
        :func:`extract_skeleton_per_component`.
    h : int
        Output image height.
    w : int
        Output image width.

    Returns
    -------
    numpy.ndarray
        White-background, black-line image of shape ``(h, w)``.
    """
    result = np.ones((h, w), dtype=np.uint8) * 255  # White background.

    for label, skeleton in skeletons.items():
        # Black skeleton.
        result[skeleton == 255] = 0

    return result


def visualize_with_labels(binary, binary_dilated, labels, num_labels, stats, skeletons):
    """
    Render a 2 x 3 debugging figure plus a per-line overview.

    Parameters
    ----------
    binary : numpy.ndarray
        Original (non-dilated) binary mask.
    binary_dilated : numpy.ndarray
        Binary mask after the 1-pixel dilation.
    labels : numpy.ndarray
        Integer label map.
    num_labels : int
        Number of labels (background included).
    stats : numpy.ndarray
        Per-label statistics.
    skeletons : dict
        ``{label: skeleton_image}``.

    Returns
    -------
    list of tuple
        ``(line_id, skeleton_length)`` sorted by length, descending.
    """
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    # 1. Original binary mask.
    axes[0, 0].imshow(binary, cmap='gray')
    axes[0, 0].set_title(f'[1] Original Binary (white: {np.count_nonzero(binary)})', fontsize=10)
    axes[0, 0].axis('off')

    # 2. Dilated binary mask.
    axes[0, 1].imshow(binary_dilated, cmap='gray')
    axes[0, 1].set_title(f'[2] After 1px Dilation (white: {np.count_nonzero(binary_dilated)})', fontsize=10)
    axes[0, 1].axis('off')

    # 3. Connected components (pseudo-coloured).
    colored = np.zeros((*binary.shape, 3), dtype=np.uint8)
    np.random.seed(42)

    for label in range(1, num_labels):
        color = tuple(int(c) for c in np.random.randint(50, 255, 3))
        colored[labels == label] = color

    axes[0, 2].imshow(colored)
    axes[0, 2].set_title(f'[3] Connected Components ({num_labels - 1} lines)', fontsize=10)
    axes[0, 2].axis('off')

    # 4. Combined skeleton.
    h, w = binary.shape
    combined_skeleton = create_skeleton_image(skeletons, h, w)
    axes[1, 0].imshow(combined_skeleton, cmap='gray')
    axes[1, 0].set_title(f'[4] Combined Skeleton ({len(skeletons)} lines)', fontsize=10)
    axes[1, 0].axis('off')

    # 5. Per-line skeleton length statistics.
    skeleton_lengths = []
    line_ids = []
    for label, skeleton in skeletons.items():
        length = np.count_nonzero(skeleton)
        skeleton_lengths.append(length)
        line_ids.append(label)

    sorted_pairs = sorted(zip(line_ids, skeleton_lengths), key=lambda x: x[1], reverse=True)

    # Show the top-5 longest skeletons in a separate figure.
    top_n = min(5, len(sorted_pairs))
    if top_n > 0:
        fig2, axes2 = plt.subplots(1, top_n, figsize=(4 * top_n, 4))
        if top_n == 1:
            axes2 = [axes2]
        for i, (lid, length) in enumerate(sorted_pairs[:top_n]):
            skeleton = skeletons[lid]
            axes2[i].imshow(skeleton, cmap='gray')
            axes2[i].set_title(f'Line {lid}\nLength: {length}px', fontsize=10)
            axes2[i].axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 'data', 'test_2026', 'skeleton_top_lines.png'),
                    dpi=150, bbox_inches='tight')
        print("Top lines saved to: skeleton_top_lines.png")

    # 6. Bar chart of the top-20 longest skeletons.
    top_n = min(20, len(sorted_pairs))

    y_pos = np.arange(top_n)
    lengths = [p[1] for p in sorted_pairs[:top_n]]
    ids = [p[0] for p in sorted_pairs[:top_n]]

    axes[1, 1].barh(y_pos, lengths, color='steelblue')
    axes[1, 1].set_yticks(y_pos)
    axes[1, 1].set_yticklabels([f'Line {i}' for i in ids])
    axes[1, 1].set_xlabel('Skeleton Length (pixels)')
    axes[1, 1].set_title('[5] Top 20 Lines by Skeleton Length', fontsize=10)
    axes[1, 1].invert_yaxis()

    # 7. Side-by-side before/after dilation.
    axes[1, 2].imshow(np.hstack([binary, binary_dilated]), cmap='gray')
    axes[1, 2].set_title('[6] Before vs After Dilation', fontsize=10)
    axes[1, 2].axis('off')
    axes[1, 2].axvline(x=binary.shape[1]/2, color='red', linewidth=2)
    axes[1, 2].text(binary.shape[1]/4, 10, 'Before', color='white', fontsize=10, ha='center')
    axes[1, 2].text(binary.shape[1]*3/4, 10, 'After', color='white', fontsize=10, ha='center')

    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'data', 'test_2026', 'skeleton_result.png'),
                dpi=150, bbox_inches='tight')
    print("Result saved to: skeleton_result.png")
    pylab.show()

    return sorted_pairs


def main(image_path=None, output_dir=None):
    """
    Main entry point: extract per-line skeletons and write them to disk.

    Parameters
    ----------
    image_path : str or None, optional
        Path to the input image. ``None`` falls back to a legacy hard-coded
        path used in earlier tests.
    output_dir : str or None, optional
        Directory in which to write outputs. ``None`` falls back to the
        legacy ``data/test_2026`` directory.
    """
    # Defaults (backwards-compatible with old hard-coded paths).
    if image_path is None:
        image_path = r"I:\E\STME\data\26-Kong\break_figure_2.jpg"
    if output_dir is None:
        _default_base = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                      'data', 'test_2026')
        output_dir = _default_base

    import os
    import json

    skeletons_dir = os.path.join(output_dir, 'skeletons')
    skeleton_img_path = os.path.join(output_dir, 'skeleton_figure6.png')
    json_path = os.path.join(output_dir, 'skeleton_data.json')

    print("=" * 60)
    print("Contour Skeleton Extraction - Connected Components Version")
    print("=" * 60)

    # 1. Load image.
    print("\n[1] Loading image...")
    img = imread_unicode(image_path)
    if img is None:
        print(f"Error: Cannot load image")
        return
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print(f"    Image size: {img.shape[1]} x {img.shape[0]}")

    # 2. Binarise with threshold 150.
    print("\n[2] Binarising with threshold 150...")
    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    print(f"    White pixels (before dilation): {np.count_nonzero(binary)}")

    # 2.5 Apply a 1-pixel dilation to glue small breaks.
    print("\n[2.5] Applying 1-pixel dilation to connect breaks...")
    kernel = np.ones((3, 3), np.uint8)
    binary_dilated = cv2.dilate(binary, kernel, iterations=1)
    print(f"    White pixels (after dilation): {np.count_nonzero(binary_dilated)}")

    # 3. Connected components (run on the dilated mask).
    print("\n[3] Finding connected components (on dilated image)...")
    num_labels, labels, stats, centroids = find_connected_components(binary_dilated)
    print(f"    Total lines detected: {num_labels - 1}")

    # 5. Per-component skeleton extraction.
    print("\n[5] Extracting connected regions for each component...")
    skeletons, truncate_stats = extract_skeleton_per_component(
        binary_dilated, labels, num_labels, edge_margin=5, truncate_edge=True)
    print(f"    Lines with data: {len(skeletons)}")
    print(f"    Truncate stats: total_lines={truncate_stats['total_lines']}, "
          f"edge pixels before={truncate_stats['edge_pixels_before']}, "
          f"after={truncate_stats['edge_pixels_after']}, "
          f"deleted={truncate_stats['pixels_deleted']}")

    # 6. Save the combined skeleton image.
    print("\n[6] Saving skeleton image...")
    h, w = binary.shape
    combined_skeleton = create_skeleton_image(skeletons, h, w)

    cv2.imwrite(skeleton_img_path, combined_skeleton)
    print(f"    Saved to: {skeleton_img_path}")

    # 7. Save per-line skeleton images and their pixel coordinates.
    print("\n[7] Saving individual line skeletons and coordinates...")
    os.makedirs(skeletons_dir, exist_ok=True)

    line_data = {}

    for label, skeleton in skeletons.items():
        line_path = os.path.join(skeletons_dir, f"line_{label:04d}.png")
        cv2.imwrite(line_path, skeleton)

        coords = np.where(skeleton == 255)
        pixel_coords = [(int(x), int(y)) for y, x in zip(coords[0], coords[1])]
        pixel_coords_sorted = sorted(pixel_coords, key=lambda p: (p[1], p[0]))

        line_data[label] = {
            'image_path': line_path,
            'skeleton_length': np.count_nonzero(skeleton),
            'pixel_coords': pixel_coords_sorted,
            'centroid': tuple(centroids[label]),
            'original_area': stats[label, cv2.CC_STAT_AREA]
        }

    print(f"    Saved {len(line_data)} line images to: {skeletons_dir}")

    # 8. Save a JSON summary.
    print("\n[8] Saving JSON data...")
    json_data = {
        'metadata': {
            'total_lines': len(skeletons),
            'threshold': 150,
            'image_size': {'width': int(img.shape[1]), 'height': int(img.shape[0])}
        },
        'lines': {}
    }

    for label in sorted(line_data.keys()):
        info = line_data[label]
        json_data['lines'][str(label)] = {
            'skeleton_length': int(info['skeleton_length']),
            'original_area': int(info['original_area']),
            'centroid': {'x': float(info['centroid'][0]), 'y': float(info['centroid'][1])},
            'pixel_count': len(info['pixel_coords']),
            'image_path': info['image_path'],
            'pixel_coords': [[int(x), int(y)] for x, y in info['pixel_coords']]
        }

    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    print(f"    JSON data saved to: {json_path}")

    print("\n" + "=" * 60)
    print("Done! Skeleton data ready for breakpoint detection.")
    print("=" * 60)
    print("\nOutput files:")
    print("  - skeleton_figure6.png: Combined skeleton")
    print("  - skeleton_data.json: Full data (lines + pixel coords)")
    print("  - skeletons/line_XXXX.png: Individual line images")


if __name__ == '__main__':
    import sys
    # CLI: python skeleton_extraction_v2.py [image_path] [output_dir]
    img_arg = sys.argv[1] if len(sys.argv) > 1 else None
    out_arg = sys.argv[2] if len(sys.argv) > 2 else None
    main(image_path=img_arg, output_dir=out_arg)