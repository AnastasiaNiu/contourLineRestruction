# -*- coding: utf-8 -*-
"""
Full pipeline orchestrator for contour-line extraction.

Input:  an image + an output directory.

Steps:
    1. skeleton_extraction_v2.py             -> skeleton_data.json
    2. breakpoint_v4_debug.py                -> breakpoint_v4_result.json (v3 algorithm result)
    3. merge_skeleton_and_connections.py     -> merged_skeleton_data_v4.json
    4. extract_connected_lines.py            -> connected_lines_v4.json (v4 result)
    5. simulated_annealing.py                -> sa_result.json + annealing_connections.json
    6. merge_skeleton_and_connections.py     -> merged_skeleton_data.json
                                                (v4 + SA merged binary mask, used for
                                                visualisation only)
    7. merge_sa_into_lines.py                -> connected_lines_sa_merged.json
                                                (merge v4 segments by SA connections
                                                -> 23 lines, used as downstream input)
    8. reference_line_reconstruction.py      -> interactive GUI; connects the non-boundary
                                                endpoints that SA could not resolve
                                                (skips the GUI automatically when there
                                                are no pending endpoints)

Output directory layout:
    output_dir/
        breakpoint_v4_debug.png                <- step 2 (kept for inspection)
        annealing_debug.png                     <- step 5 (kept)
        annealing_clean.png                     <- step 5 (kept)
        reference_reconstruction_clean.png      <- step 8 (only if it actually ran)
        reconstruction_result.png               <- final image: step 8 -> ref_recon_clean,
                                                  otherwise annealing_clean
        temp/                                   <- all intermediate JSON / PNG / masks
            skeleton_data.json
            breakpoint_v4_result.json
            breakpoint_v4_filtered_endpoints.json
            breakpoint_v4_filter_compare.png
            connected_skeleton_v4.png
            merged_skeleton_data_v4.json
            merged_skeleton_mask_v4.png
            connected_lines_v4.json
            sa_result.json
            annealing_connections.json
            annealing_result.png
            annealing_direction_codes.png
            merged_skeleton_data.json
            merged_skeleton_mask.png
            connected_lines_sa_merged.json
            reference_reconstruction_result.json
            reference_reconstruction_vis.png
            reference_lines_overview.png
            _focus_tmp.png

Usage:
    python pipeline.py --image <path> --output-dir <dir>
    python pipeline.py --image <path> --output-dir <dir> --steps 2-5,7   # skip 1, 6, 8
    python pipeline.py --image <path> --output-dir <dir> --skip-skeleton # reuse existing
                                                                   # skeleton_data.json
"""

import argparse
import json
import os
import sys
import shutil
import time
import traceback

# ─── Project root ───────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(PROJECT_ROOT, 'src')

# The implementation files live under ``src/``. Prepend that directory to
# ``sys.path`` so the ``from xxx import main`` style imports below resolve
# to ``src/xxx.py`` regardless of the working directory from which
# ``pipeline.py`` is invoked.
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)


def banner(title):
    """
    Print a 70-character banner around ``title``.

    Parameters
    ----------
    title : str
        Text to display inside the banner.
    """
    print()
    print("=" * 70)
    print("  " + title)
    print("=" * 70)


def step(n, name):
    """
    Print a banner for step ``n`` titled ``name``.

    Parameters
    ----------
    n : int or None
        Step number, or ``None`` to omit the number prefix.
    name : str
        Step description.
    """
    banner(f"[STEP {n}] {name}")


def safe_remove(path):
    """
    Delete the file at ``path`` if it exists.

    Parameters
    ----------
    path : str
        File path to remove. No error if it does not exist.
    """
    if os.path.isfile(path):
        os.remove(path)


def safe_copy(src, dst):
    """
    Copy ``src`` to ``dst`` if ``src`` exists.

    Parameters
    ----------
    src : str
        Source file path.
    dst : str
        Destination file path.

    Returns
    -------
    bool
        ``True`` if a copy was performed, ``False`` otherwise.
    """
    if os.path.isfile(src):
        shutil.copy2(src, dst)
        return True
    return False


# ─── Step wrappers ──────────────────────────────────────────────────────
def step1_skeleton(image_path, output_dir):
    """
    Run step 1: skeleton extraction.

    Parameters
    ----------
    image_path : str
        Path to the input image.
    output_dir : str
        Directory in which ``skeleton_extraction_v2`` writes its outputs.
    """
    step(1, "Skeleton Extraction (skeleton_extraction_v2.py)")
    t0 = time.time()
    from skeleton_extraction_v2 import main as sk_main
    sk_main(image_path=image_path, output_dir=output_dir)
    print(f"\n[STEP 1 DONE] {time.time() - t0:.1f}s")


def step2_breakpoint_v4(data_dir):
    """
    Run step 2: v4 breakpoint matching and connection.

    Parameters
    ----------
    data_dir : str
        Directory holding ``skeleton_data.json`` (and where ``breakpoint_v4``
        will write its results).
    """
    step(2, "Breakpoint v4 Detection (breakpoint_v4_debug.py)")
    t0 = time.time()
    from breakpoint_v4_debug import main as bp_main
    bp_main(data_dir=data_dir)
    print(f"\n[STEP 2 DONE] {time.time() - t0:.1f}s")


def step3_merge(data_dir, result_filename='breakpoint_v4_result.json',
                merged_filename='merged_skeleton_data.json',
                mask_filename='merged_skeleton_mask.png',
                extra_connections=None,
                tag=''):
    """
    Run step 3 (or step 6 in the second call): merge skeleton + connection curves.

    Parameters
    ----------
    data_dir : str
        Directory containing the skeleton JSON and breakpoint result JSON.
    result_filename : str, optional
        Filename of the breakpoint-matching result (default
        ``'breakpoint_v4_result.json'``).
    merged_filename : str, optional
        Filename for the merged skeleton output (default
        ``'merged_skeleton_data.json'``).
    mask_filename : str, optional
        Filename for the visualisation mask (default
        ``'merged_skeleton_mask.png'``).
    extra_connections : str or None, optional
        Optional path to a second set of connections (e.g. SA output) that
        should also be merged. Pass ``None`` to skip.
    tag : str, optional
        Free-form suffix appended to the banner title (e.g. ``'v4-only'``,
        ``'v4+SA'``). An empty string means "step 3 banner".
    """
    step(3 if not tag else None, f"Merge Skeleton + Connections{(' [' + tag + ']') if tag else ''}")
    t0 = time.time()
    from merge_skeleton_and_connections import main as merge_main
    merge_main(data_dir=data_dir,
               result_filename=result_filename,
               output_filename=merged_filename,
               mask_filename=mask_filename,
               extra_connections_json=extra_connections)
    print(f"\n[STEP 3 DONE] {time.time() - t0:.1f}s")


def step4_extract(data_dir, merged_filename='merged_skeleton_data.json',
                  output_filename='connected_lines.json'):
    """
    Run step 4: extract connected line segments from the merged skeleton.

    Parameters
    ----------
    data_dir : str
        Directory holding the merged skeleton JSON.
    merged_filename : str, optional
        Filename of the merged skeleton JSON (default
        ``'merged_skeleton_data.json'``).
    output_filename : str, optional
        Filename for the connected-line JSON (default
        ``'connected_lines.json'``).
    """
    step(4, f"Extract Connected Lines ({output_filename})")
    t0 = time.time()
    from extract_connected_lines import main as ext_main
    ext_main(data_dir=data_dir,
             merged_filename=merged_filename,
             output_filename=output_filename)
    print(f"\n[STEP 4 DONE] {time.time() - t0:.1f}s")


def step5_simulated_annealing(data_dir):
    """
    Run step 5: simulated-annealing connection.

    Parameters
    ----------
    data_dir : str
        Directory holding the v4 connected-line JSON and the merged skeleton
        JSON; SA writes ``sa_result.json`` and ``annealing_connections.json``
        here.
    """
    step(5, "Simulated Annealing Connection (simulated_annealing.py)")
    t0 = time.time()
    from simulated_annealing import main as sa_main
    sa_main(data_dir=data_dir)
    print(f"\n[STEP 5 DONE] {time.time() - t0:.1f}s")


def step7_merge_sa(data_dir,
                   v4_filename='connected_lines_v4.json',
                   sa_filename='annealing_connections.json',
                   output_filename='connected_lines_sa_merged.json',
                   mask_filename='merged_lines_sa_mask.png'):
    """
    Run step 7: merge v4 line segments into longer lines using SA connections.

    For each SA connection the two endpoints it links are stitched into a
    single new line so that the v4 topology is preserved while the SA bridges
    become part of the geometry. A colour-coded visualisation of the merged
    result is written next to the JSON.

    Parameters
    ----------
    data_dir : str
        Directory holding ``connected_lines_v4.json`` and
        ``annealing_connections.json``.
    v4_filename : str, optional
        Filename of the v4 connected-line JSON (default
        ``'connected_lines_v4.json'``).
    sa_filename : str, optional
        Filename of the SA connection JSON (default
        ``'annealing_connections.json'``).
    output_filename : str, optional
        Filename for the merged-line output (default
        ``'connected_lines_sa_merged.json'``).
    mask_filename : str, optional
        Filename for the visualisation PNG (default
        ``'merged_lines_sa_mask.png'``).
    """
    step(7, "Merge SA Connections into Lines (merge_sa_into_lines.py)")
    t0 = time.time()
    from merge_sa_into_lines import merge_lines_by_sa
    v4_path = os.path.join(data_dir, v4_filename)
    sa_path = os.path.join(data_dir, sa_filename)
    out_path = os.path.join(data_dir, output_filename)
    out = merge_lines_by_sa(v4_path, sa_path, out_path)

    # Optional: render a v4 + SA merged visualisation (acts as a sanity-check
    # image alongside the JSON).
    try:
        import cv2
        import numpy as np
        meta = out['metadata']
        w, h = meta['image_size']['width'], meta['image_size']['height']
        canvas = np.full((h, w, 3), 255, dtype=np.uint8)

        # Background: the original 48 skeleton lines (light grey).
        sk_path = os.path.join(data_dir, 'skeleton_data.json')
        if os.path.isfile(sk_path):
            with open(sk_path, 'r', encoding='utf-8') as f:
                sk = json.load(f)
            for _, ldata in sk['lines'].items():
                pts = [(p[0], p[1]) for p in ldata['pixel_coords']]
                if pts:
                    arr = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
                    cv2.polylines(canvas, [arr], False, (210, 210, 210), 1)

        # SA connection curves (thin red).
        if os.path.isfile(sa_path):
            with open(sa_path, 'r', encoding='utf-8') as f:
                sa = json.load(f)
            for c in sa.get('connections', []):
                pts = [(p['x'], p['y']) for p in c['pixel_coords']]
                if pts:
                    arr = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
                    cv2.polylines(canvas, [arr], False, (0, 0, 220), 2)

        # Merged line segments (palette + numerical id).
        palette = [
            (220, 20, 60), (0, 128, 0), (0, 0, 220), (255, 140, 0),
            (148, 0, 211), (0, 180, 180), (255, 20, 147), (85, 107, 47),
        ]
        for idx, (k, v) in enumerate(sorted(out['lines'].items(),
                                             key=lambda kv: int(kv[0]))):
            color = palette[idx % len(palette)]
            pts = [(p['x'], p['y']) for p in v['pixel_coords']]
            if not pts:
                continue
            arr = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(canvas, [arr], False, color, 2)
            mid = pts[len(pts) // 2]
            cv2.putText(canvas, str(k), (int(mid[0]), int(mid[1])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 3, cv2.LINE_AA)
            cv2.putText(canvas, str(k), (int(mid[0]), int(mid[1])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)

        result, buf = cv2.imencode('.png', canvas)
        buf.tofile(os.path.join(data_dir, mask_filename))
    except Exception as e:
        print(f"  [warn] visualisation generation failed: {e}")

    print(f"\n[STEP 7 DONE] {time.time() - t0:.1f}s")


def step8_reference_line_reconstruction(data_dir, bezier_degree=4):
    """
    Run step 8: interactive reference-line reconstruction (Tkinter GUI).

    Only launches the GUI when, after step 7, there are still non-boundary
    endpoints that have not been connected. Otherwise it prints a message
    and returns.

    Parameters
    ----------
    data_dir : str
        Directory holding ``connected_lines_sa_merged.json``.
    bezier_degree : int, optional
        Degree of the Bezier curve fitted to each line (default 4).

    Returns
    -------
    bool or None
        ``True`` if the user clicked "Finish and save" and produced
        ``reference_reconstruction_clean.png``; ``None`` if step 8 was skipped
        for any other reason (no endpoints, missing input, etc.).
    """
    step(8, "Reference Line Reconstruction (interactive, Tkinter)")
    t0 = time.time()

    from reference_line_reconstruction import (
        load_lines, fit_bezier, collect_endpoints,
    )

    json_path = os.path.join(data_dir, 'connected_lines_sa_merged.json')
    if not os.path.isfile(json_path):
        print(f"  [skip] missing {json_path}, skipping Step 8 (run Step 7 first)")
        return

    lines, img_size = load_lines(json_path)
    for ln in lines:
        fit_bezier(ln, degree=bezier_degree)

    endpoints = collect_endpoints(lines)
    print(f"  Remaining non-boundary endpoints: {len(endpoints)} "
          f"(across {len(lines)} lines)")

    if not endpoints:
        print(f"  [OK] no pending non-boundary endpoints; SA resolved everything, skipping Step 8")
        return

    print(f"  -> launching interactive UI; complete reference lines manually...")
    from reference_line_reconstruction import ReconstructionApp
    ReconstructionApp(lines, endpoints, img_size, data_dir, bezier_degree)

    # When the user clicks "Finish and save" inside the GUI,
    # reference_reconstruction_clean.png has been written to disk.
    clean_path = os.path.join(data_dir, 'reference_reconstruction_clean.png')
    saved = os.path.isfile(clean_path)
    if not saved:
        print(f"  [warn] {clean_path} not detected, reference-line result may be unsaved")

    print(f"\n[STEP 8 DONE] {time.time() - t0:.1f}s")
    return saved


# ─── Pipeline main flow ─────────────────────────────────────────────────
TEMP_SUBDIR = 'temp'

# After each step the named PNGs are copied from temp/ to the output root.
# The user-facing result is reduced to those few files.
# All remaining files (intermediate JSON, intermediate PNG, masks, ...) stay in temp/.
KEPT_PNGS_AT_ROOT = {
    2: ['breakpoint_v4_debug.png'],
    5: ['annealing_debug.png', 'annealing_clean.png'],
    8: ['reference_reconstruction_clean.png'],
}


def _promote_files(temp_dir: str, root_dir: str, filenames: list):
    """
    Copy each file in ``filenames`` from ``temp_dir`` to ``root_dir``.

    Parameters
    ----------
    temp_dir : str
        Source directory (typically ``output_dir/temp``).
    root_dir : str
        Destination directory (typically ``output_dir``).
    filenames : list of str
        Basenames of the files to copy.
    """
    for fn in filenames:
        src = os.path.join(temp_dir, fn)
        dst = os.path.join(root_dir, fn)
        if os.path.isfile(src):
            try:
                shutil.copy2(src, dst)
                print(f"  [promote] temp/{fn}  ->  {os.path.basename(dst)}")
            except Exception as e:
                print(f"  [warn] promote failed for {fn}: {e}")
        else:
            print(f"  [skip-promote] temp/{fn} does not exist")


def _setup_dirs(output_dir):
    """
    Create ``output_dir`` and ``output_dir/temp``.

    Parameters
    ----------
    output_dir : str
        Top-level output directory.

    Returns
    -------
    str
        Path to the created ``temp`` subdirectory.
    """
    os.makedirs(output_dir, exist_ok=True)
    temp_dir = os.path.join(output_dir, TEMP_SUBDIR)
    os.makedirs(temp_dir, exist_ok=True)
    return temp_dir


def run_pipeline(image_path, output_dir, steps=None, skip_skeleton=False):
    """
    End-to-end pipeline execution.

    Parameters
    ----------
    image_path : str
        Path to the input image. Required when step 1 is requested.
    output_dir : str
        Top-level output directory (only the kept PNGs land here; everything
        else goes under ``output_dir/temp``).
    steps : set of int or None, optional
        Set of step numbers to execute (``1..8``). ``None`` means "all steps".
    skip_skeleton : bool, optional
        If ``True``, drop step 1 and assume ``output_dir/temp/skeleton_data.json``
        already exists.

    Output directory layout
    ------------------------
    output_dir/
        breakpoint_v4_debug.png                <- step 2
        annealing_debug.png                     <- step 5
        annealing_clean.png                     <- step 5
        reference_reconstruction_clean.png      <- step 8 (only if it ran)
        reconstruction_result.png               <- final image: step 8 ->
                                                   ref_recon_clean, otherwise
                                                   annealing_clean
        temp/                                   <- all intermediate JSON /
                                                   intermediate PNG / masks
    """
    temp_dir = _setup_dirs(output_dir)

    if steps is None:
        steps = set(range(1, 9))
    if skip_skeleton:
        steps.discard(1)

    print(f"Pipeline configuration:")
    print(f"  Image:      {image_path}")
    print(f"  Output dir: {output_dir}")
    print(f"  Temp dir:   {temp_dir}")
    print(f"  Steps:      {sorted(steps)}")

    if not os.path.isfile(image_path):
        if 1 in steps:
            raise FileNotFoundError(f"Input image not found: {image_path}")

    # ── Step 1: skeleton extraction ──
    if 1 in steps:
        step1_skeleton(image_path, temp_dir)

    # Mandatory intermediate file check.
    skeleton_json = os.path.join(temp_dir, 'skeleton_data.json')
    if 2 in steps and not os.path.isfile(skeleton_json):
        raise FileNotFoundError(f"Missing {skeleton_json}; please run Step 1 first (--skip-skeleton=False)")

    # ── Step 2: v4 breakpoint matching and connection ──
    if 2 in steps:
        step2_breakpoint_v4(temp_dir)
        _promote_files(temp_dir, output_dir, KEPT_PNGS_AT_ROOT.get(2, []))

    # ── Step 3: merge skeleton + v4 connections (intermediate visualisation only) ──
    if 3 in steps:
        step3_merge(temp_dir,
                   result_filename='breakpoint_v4_result.json',
                   merged_filename='merged_skeleton_data_v4.json',
                   mask_filename='merged_skeleton_mask_v4.png',
                   extra_connections=None,
                   tag='v4-only')

    # ── Step 4: extract connected lines (used as SA endpoint-matching input) ──
    if 4 in steps:
        step4_extract(temp_dir,
                      merged_filename='merged_skeleton_data_v4.json',
                      output_filename='connected_lines_v4.json')

    # ── Step 5: simulated annealing (writes sa_result.json; does not overwrite
    #             breakpoint_v4_result.json) ──
    if 5 in steps:
        connected_required = os.path.join(temp_dir, 'connected_lines_v4.json')
        merged_required = os.path.join(temp_dir, 'merged_skeleton_data_v4.json')
        if not os.path.isfile(connected_required):
            raise FileNotFoundError(f"Missing {connected_required}; please run Step 4 first")
        if not os.path.isfile(merged_required):
            raise FileNotFoundError(f"Missing {merged_required}; please run Step 3 first")
        step5_simulated_annealing(temp_dir)
        _promote_files(temp_dir, output_dir, KEPT_PNGS_AT_ROOT.get(5, []))

    # ── Step 6: one-shot merge of skeleton + v4 + SA (final result) ──
    if 6 in steps:
        v4_result = os.path.join(temp_dir, 'breakpoint_v4_result.json')
        sa_result = os.path.join(temp_dir, 'sa_result.json')
        if not os.path.isfile(v4_result):
            raise FileNotFoundError(f"Missing {v4_result}; please run Step 2 first")
        step3_merge(temp_dir,
                    result_filename='breakpoint_v4_result.json',
                    merged_filename='merged_skeleton_data.json',
                    mask_filename='merged_skeleton_mask.png',
                    extra_connections=sa_result if os.path.isfile(sa_result) else None,
                    tag='v4+SA')

    # ── Step 7: stitch v4 line segments into a new topology using SA (recommended
    #             downstream input) ──
    if 7 in steps:
        v4_required = os.path.join(temp_dir, 'connected_lines_v4.json')
        sa_required = os.path.join(temp_dir, 'annealing_connections.json')
        if not os.path.isfile(v4_required):
            raise FileNotFoundError(f"Missing {v4_required}; please run Step 4 first")
        if not os.path.isfile(sa_required):
            raise FileNotFoundError(f"Missing {sa_required}; please run Step 5 first")
        step7_merge_sa(temp_dir)

    # ── Step 8: interactive reference-line reconstruction (Tkinter GUI,
    #             launches only when there are still non-boundary endpoints) ──
    ref_recon_saved = False
    if 8 in steps:
        sa_merged = os.path.join(temp_dir, 'connected_lines_sa_merged.json')
        if not os.path.isfile(sa_merged):
            raise FileNotFoundError(f"Missing {sa_merged}; please run Step 7 first")
        ref_recon_saved = step8_reference_line_reconstruction(temp_dir)
        if ref_recon_saved:
            _promote_files(temp_dir, output_dir, KEPT_PNGS_AT_ROOT.get(8, []))

    # ── Final product: reconstruction_result.png ──
    # - Use ref_recon_clean when step 8 ran and saved.
    # - Otherwise fall back to annealing_clean.
    final_src = None
    if ref_recon_saved:
        final_src = os.path.join(temp_dir, 'reference_reconstruction_clean.png')
    else:
        final_src = os.path.join(temp_dir, 'annealing_clean.png')

    final_dst = os.path.join(output_dir, 'reconstruction_result.png')
    if os.path.isfile(final_src):
        try:
            shutil.copy2(final_src, final_dst)
            print(f"[FINAL] reconstruction_result.png <- {os.path.basename(final_src)}")
        except Exception as e:
            print(f"[warn] reconstruction_result.png copy failed: {e}")
    else:
        print(f"[FINAL] {final_src} does not exist, skipping reconstruction_result.png")

    banner("Pipeline Finished")


def parse_args():
    """
    Build the CLI argument parser.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    p = argparse.ArgumentParser(
        description='Full contour-line extraction pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument('--image', required=True, help='Path to the input image')
    p.add_argument('--output-dir', required=True, help='Output directory path')
    p.add_argument('--steps', default='1-8',
                   help='Steps to run, e.g. "1-8", "2-5", "1,3,5,7"')
    p.add_argument('--skip-skeleton', action='store_true',
                   help='Skip Step 1 (reuse an existing skeleton_data.json)')
    return p.parse_args()


def parse_steps(spec):
    """
    Parse a step-spec string like ``'1-7'``, ``'2-5'`` or ``'1,3,5'`` into a set.

    Parameters
    ----------
    spec : str
        Step specification. Ranges use ``-``; individual steps use ``,``.

    Returns
    -------
    set of int
        Set of step numbers to execute.
    """
    steps = set()
    for part in spec.split(','):
        part = part.strip()
        if not part:
            continue
        if '-' in part:
            a, b = part.split('-')
            steps.update(range(int(a), int(b) + 1))
        else:
            steps.add(int(part))
    return steps


def main():
    """CLI entry point."""
    args = parse_args()
    steps = parse_steps(args.steps)

    try:
        run_pipeline(args.image, args.output_dir, steps, args.skip_skeleton)
    except Exception as e:
        print(f"\n[ERROR] Pipeline failed: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()