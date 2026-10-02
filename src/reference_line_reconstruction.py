# -*- coding: utf-8 -*-
"""
Reference-line-assisted contour-line reconstruction - Tkinter interactive
version.

Implements Section 4.12 of the paper, "Contour-line reconstruction based on
neighbouring geometric features":

    Step 1: uniformly sub-sample every contour into ``n`` Bezier control
            points.
    Step 2: in the GUI:
                1) select a reference line from the dropdown; the side (same /
                   opposite) and distance of every non-boundary endpoint is
                   computed automatically.
                2) tick the two endpoints to bind.
                3) click "Confirm this round" to reconstruct.
                4) repeat until every endpoint is bound, then "Finish & save".
    Step 3: project both endpoints onto the reference line, then use the
            projection offsets to generate a new curve parallel to the
            reference line; splice it back into the original lines.

Usage:
    python reference_line_reconstruction.py --data-dir <dir>
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk

# Project-local Bezier helper.
from BezierCurve import BezierCurve as B


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class LineSegment:
    """
    A single contour-line segment.

    Attributes
    ----------
    line_id : str
        Identifier of this line in the input JSON.
    pixel_coords : numpy.ndarray
        ``(N, 2)`` integer pixel coordinates.
    first_point : tuple
        ``(x, y)`` of the first endpoint.
    second_point : tuple
        ``(x, y)`` of the second endpoint.
    first_is_boundary : bool
        ``True`` when the first endpoint lies on the image border.
    second_is_boundary : bool
        ``True`` when the second endpoint lies on the image border.
    source_line_ids : list of str
        Original v4 line IDs that were merged into this segment (empty when
        the segment was not merged).
    bezier_degree : int
        Degree of the Bezier curve used to approximate this line.
    control_points : numpy.ndarray or None
        Bezier control points, populated by :func:`fit_bezier`.
    """
    line_id: str
    pixel_coords: np.ndarray
    first_point: Tuple[int, int]
    second_point: Tuple[int, int]
    first_is_boundary: bool
    second_is_boundary: bool
    source_line_ids: List[str] = field(default_factory=list)
    bezier_degree: int = 4
    control_points: Optional[np.ndarray] = None


@dataclass
class Endpoint:
    """
    An endpoint of a contour line that may need to be reconstructed.

    Attributes
    ----------
    pid : str
        Globally unique identifier for this endpoint.
    line_id : str
        Identifier of the line this endpoint belongs to.
    end : str
        Either ``'first'`` or ``'second'`` (which endpoint of the line).
    point : tuple
        ``(x, y)`` coordinate.
    is_used : bool
        ``True`` once the endpoint has been bound to another endpoint.
    used_by_ref : str or None
        ``line_id`` of the reference line that drove this binding.
    side : int or None
        ``+1`` same side as the reference tangent, ``-1`` opposite side,
        ``0`` indeterminate. Populated lazily while a reference line is
        selected.
    distance : float or None
        Distance to the reference line, in pixels.
    """
    pid: str
    line_id: str
    end: str
    point: Tuple[int, int]
    is_used: bool = False
    used_by_ref: Optional[str] = None
    side: Optional[int] = None
    distance: Optional[float] = None


# ─────────────────────────────────────────────────────────────────────────────
# Data loading + Bezier fitting
# ─────────────────────────────────────────────────────────────────────────────
def load_lines(json_path: str) -> Tuple[List[LineSegment], Tuple[int, int]]:
    """
    Read the connected-line JSON into a list of :class:`LineSegment` objects.

    Parameters
    ----------
    json_path : str
        Path to ``connected_lines_sa_merged.json`` (or any compatible file).

    Returns
    -------
    tuple
        ``(lines, (image_width, image_height))``.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    w = data["metadata"]["image_size"]["width"]
    h = data["metadata"]["image_size"]["height"]
    lines: List[LineSegment] = []
    for lid, raw in data["lines"].items():
        coords = np.array(
            [(int(p["x"]), int(p["y"])) for p in raw["pixel_coords"]],
            dtype=np.int32,
        )
        lines.append(LineSegment(
            line_id=str(lid),
            pixel_coords=coords,
            first_point=(int(raw["first_point"]["x"]), int(raw["first_point"]["y"])),
            second_point=(int(raw["second_point"]["x"]), int(raw["second_point"]["y"])),
            first_is_boundary=bool(raw["first_is_boundary"]),
            second_is_boundary=bool(raw["second_is_boundary"]),
            source_line_ids=[str(s) for s in raw.get("source_line_ids", [])],
        ))
    return lines, (w, h)


def fit_bezier(line: LineSegment, degree: int = 4) -> np.ndarray:
    """
    Uniformly sub-sample ``line.pixel_coords`` into ``degree + 1`` Bezier
    control points and store them on ``line.control_points``.

    Parameters
    ----------
    line : LineSegment
        Target line; modified in-place.
    degree : int, optional
        Bezier degree (default 4, i.e. 5 control points).

    Returns
    -------
    numpy.ndarray
        The newly computed control points.
    """
    pts = line.pixel_coords
    if len(pts) < degree + 1:
        line.control_points = pts.astype(np.float64).copy()
        return line.control_points
    idx = np.linspace(0, len(pts) - 1, degree + 1, dtype=int)
    cp = pts[idx].astype(np.float64)
    line.control_points = cp
    return cp


def collect_endpoints(lines: List[LineSegment]) -> List[Endpoint]:
    """
    Collect every **non-boundary** endpoint in the input ``lines``.

    Parameters
    ----------
    lines : list of LineSegment
        Input lines.

    Returns
    -------
    list of Endpoint
        One :class:`Endpoint` per non-boundary endpoint, sorted by line id.
    """
    eps: List[Endpoint] = []
    counter = 1
    for ln in sorted(lines, key=lambda x: int(x.line_id)):
        if not ln.first_is_boundary:
            eps.append(Endpoint(
                pid=f"p{counter}", line_id=ln.line_id, end="first",
                point=ln.first_point,
            ))
            counter += 1
        if not ln.second_is_boundary:
            eps.append(Endpoint(
                pid=f"p{counter}", line_id=ln.line_id, end="second",
                point=ln.second_point,
            ))
            counter += 1
    return eps


# ─────────────────────────────────────────────────────────────────────────────
# Geometric operators
# ─────────────────────────────────────────────────────────────────────────────
def project_onto_bezier(point: np.ndarray, cps: np.ndarray,
                         num_samples: int = 200):
    """
    Project ``point`` onto the Bezier curve defined by ``cps``.

    Parameters
    ----------
    point : numpy.ndarray
        ``(2,)`` query point.
    cps : numpy.ndarray
        ``(degree + 1, 2)`` Bezier control points.
    num_samples : int, optional
        Number of samples used to discretise the curve (default 200).

    Returns
    -------
    tuple
        ``(projected_point, distance, index)`` where ``index`` is the index
        of the closest sampled curve point.
    """
    pts = np.array([B.bezier_curve(t, cps) for t in
                     np.linspace(0, 1, num_samples)], dtype=np.float64)
    diffs = pts - point.astype(np.float64)
    dists = np.linalg.norm(diffs, axis=1)
    idx = int(np.argmin(dists))
    return pts[idx], float(dists[idx]), idx


def tangent_at_index(cps: np.ndarray, index: int,
                      num_samples: int = 200) -> np.ndarray:
    """
    Approximate the unit tangent of the Bezier curve at sample ``index``.

    Uses central differences when possible and falls back to forward /
    backward differences at the endpoints.

    Parameters
    ----------
    cps : numpy.ndarray
        ``(degree + 1, 2)`` Bezier control points.
    index : int
        Index of the sampled curve point.
    num_samples : int, optional
        Number of samples used to discretise the curve (default 200).

    Returns
    -------
    numpy.ndarray
        Normalised ``(2,)`` tangent vector. Falls back to ``[1, 0]`` when
        the difference is degenerate.
    """
    pts = np.array([B.bezier_curve(t, cps) for t in
                     np.linspace(0, 1, num_samples)], dtype=np.float64)
    if 0 < index < len(pts) - 1:
        d = pts[index + 1] - pts[index - 1]
    elif index == 0:
        d = pts[1] - pts[0]
    else:
        d = pts[-1] - pts[-2]
    n = np.linalg.norm(d)
    return d / n if n > 1e-9 else np.array([1.0, 0.0])


def classify_side(break_point: Tuple[int, int], cps: np.ndarray) -> int:
    """
    Classify ``break_point`` as being on the same side (``+1``), the opposite
    side (``-1``) or on the curve (``0``) relative to the Bezier curve.

    The "same side" decision uses a 2-D cross product between the
    ``break_point - projection`` vector and the local tangent.

    Parameters
    ----------
    break_point : tuple
        ``(x, y)`` of the candidate point.
    cps : numpy.ndarray
        ``(degree + 1, 2)`` Bezier control points.

    Returns
    -------
    int
        ``+1``, ``-1`` or ``0``.
    """
    p = np.array(break_point, dtype=np.float64)
    Q, _, idx = project_onto_bezier(p, cps)
    QP = p - Q
    QI = tangent_at_index(cps, idx)
    cross = QP[0] * QI[1] - QP[1] * QI[0]
    if abs(cross) < 1e-6:
        return 0
    return 1 if cross > 0 else -1


# ─────────────────────────────────────────────────────────────────────────────
# Step 3: connect two endpoints with a curve that follows the reference line
# ─────────────────────────────────────────────────────────────────────────────
def connect_two_endpoints(
    p1_endpoint: Endpoint,
    p2_endpoint: Endpoint,
    p1_line: LineSegment,
    p2_line: LineSegment,
    ref_line: LineSegment,
) -> np.ndarray:
    """
    Connect two endpoints with a curve that mirrors the geometry of
    ``ref_line``.

    Steps:
        1. Find the projections ``Q1`` and ``Q2`` of ``P1`` and ``P2`` onto
           ``ref_line`` and their sample indices.
        2. Order the projections by ``t`` so that we walk along ``ref_line``.
        3. Sample a dense curve along ``ref_line`` between the two
           projections; add a linearly interpolated offset
           ``(P - Q)`` so that the new curve starts at ``P1`` and ends at
           ``P2``.
        4. Snap the first and last samples exactly to ``P1`` / ``P2`` to
           avoid floating-point drift.

    The result is a single-pixel-thick dense polyline that, when ``ref_line``
    is a straight line, is straight; when ``ref_line`` is curved, mirrors
    that curvature.

    Parameters
    ----------
    p1_endpoint, p2_endpoint : Endpoint
        The two endpoints to connect.
    p1_line, p2_line : LineSegment
        Their respective parent lines (currently unused, but accepted for
        future extension).
    ref_line : LineSegment
        The reference line that determines the geometry.

    Returns
    -------
    numpy.ndarray
        ``(N, 2)`` array of dense polyline coordinates.
    """
    P1 = np.array(p1_endpoint.point, dtype=np.float64)
    P2 = np.array(p2_endpoint.point, dtype=np.float64)

    # Q1, Q2: closest points on ref_line + their sample indices.
    Q1, _, t1 = project_onto_bezier(P1, ref_line.control_points)
    Q2, _, t2 = project_onto_bezier(P2, ref_line.control_points)
    t_lo, t_hi = (t1, t2) if t1 <= t2 else (t2, t1)
    # Normalise the endpoint sample indices back to [0, 1].
    t_lo_s = t_lo / 199.0
    t_hi_s = t_hi / 199.0
    # If the order is reversed, swap everything accordingly.
    if t1 > t2:
        Q_lo, Q_hi, P_lo, P_hi = Q2, Q1, P2, P1
    else:
        Q_lo, Q_hi, P_lo, P_hi = Q1, Q2, P1, P2
    offset_lo = P_lo - Q_lo
    offset_hi = P_hi - Q_hi

    d_pp = float(np.linalg.norm(P2 - P1))

    # Sample densely along ref_line and add a linearly interpolated offset.
    n_samples = max(20, int(d_pp * 1.2) + 1)
    ts = np.linspace(t_lo_s, t_hi_s, n_samples)
    curve_pts = []
    for t in ts:
        ref_pt = B.bezier_curve(t, ref_line.control_points)
        s = (t - t_lo_s) / max(t_hi_s - t_lo_s, 1e-9)
        offset = offset_lo * (1 - s) + offset_hi * s
        curve_pts.append(ref_pt + offset)
    curve = np.array(curve_pts, dtype=np.float64)

    # Snap endpoints exactly to P1 / P2 (s = 0 / 1 should already match).
    curve[0] = P1
    curve[-1] = P2
    return curve


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation (writes PNGs to be loaded by Tkinter)
# ─────────────────────────────────────────────────────────────────────────────
PALETTE = [
    (220, 20, 60), (0, 128, 0), (0, 0, 220), (255, 140, 0),
    (148, 0, 211), (0, 180, 180), (255, 20, 147), (85, 107, 47),
    (255, 0, 255), (0, 200, 0), (50, 50, 200), (200, 80, 0),
    (120, 0, 120), (0, 100, 100), (180, 0, 90), (90, 60, 30),
]


def _imwrite_unicode(path: str, img: np.ndarray) -> bool:
    """
    Write ``img`` to ``path`` even when the path contains non-ASCII chars.

    Parameters
    ----------
    path : str
        Destination file path.
    img : numpy.ndarray
        Image to encode.

    Returns
    -------
    bool
        ``True`` on success, ``False`` otherwise.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".png":
        param = [int(cv2.IMWRITE_PNG_COMPRESSION), 0]
    elif ext in (".jpg", ".jpeg"):
        param = [int(cv2.IMWRITE_JPEG_QUALITY), 95]
    else:
        param = []
    result, encoded = cv2.imencode(ext, img, param)
    if result:
        encoded.tofile(path)
        return True
    return False


def visualize_overview(lines, endpoints, img_size, output_path,
                        reconstructed=None):
    """
    Render the overview: grey lines, blue endpoints, red reconstructed curves.

    Parameters
    ----------
    lines : list of LineSegment
        All lines to draw in light grey.
    endpoints : list of Endpoint
        Endpoints to mark.
    img_size : tuple
        ``(width, height)`` of the canvas.
    output_path : str
        PNG path to write.
    reconstructed : list of array-like or None, optional
        Reconstructed curve segments (each an iterable of ``(x, y)``).
    """
    w, h = img_size
    canvas = np.full((h, w, 3), 255, dtype=np.uint8)

    for ln in lines:
        pts = ln.pixel_coords.reshape(-1, 1, 2)
        cv2.polylines(canvas, [pts], False, (200, 200, 200), 1)
        mid = ln.pixel_coords[len(ln.pixel_coords) // 2]
        text = ln.line_id
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(canvas,
                      (int(mid[0]) - 2, int(mid[1]) - th - 2),
                      (int(mid[0]) + tw + 2, int(mid[1]) + 3),
                      (255, 255, 255), -1)
        cv2.putText(canvas, text, (int(mid[0]), int(mid[1])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 80, 80), 1)

    if reconstructed:
        for seg in reconstructed:
            arr = np.array(seg, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(canvas, [arr], False, (0, 0, 220), 2)

    for ep in endpoints:
        c = (0, 180, 0) if ep.is_used else (0, 0, 220)
        cv2.circle(canvas, ep.point, 5, c, -1)
        cv2.putText(canvas, ep.pid,
                    (ep.point[0] + 8, ep.point[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (40, 40, 40), 1)

    _imwrite_unicode(output_path, canvas)


def visualize_clean(lines, img_size, output_path, reconstructed=None):
    """
    Render a "clean" view: original lines + reconstructed segments, all in
    single-pixel black.

    Parameters
    ----------
    lines : list of LineSegment
        All lines to draw.
    img_size : tuple
        ``(width, height)`` of the canvas.
    output_path : str
        PNG path to write.
    reconstructed : list of array-like or None, optional
        Reconstructed curve segments (each an iterable of ``(x, y)``).
    """
    w, h = img_size
    canvas = np.full((h, w, 3), 255, dtype=np.uint8)

    # Original lines (single pixel).
    for ln in lines:
        pts = ln.pixel_coords.reshape(-1, 1, 2)
        cv2.polylines(canvas, [pts], False, (0, 0, 0), 1)

    # Reconstructed segments (single pixel, drawn the same as the originals).
    if reconstructed:
        for seg in reconstructed:
            arr = np.array(seg, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(canvas, [arr], False, (0, 0, 0), 1)

    _imwrite_unicode(output_path, canvas)


def visualize_focus(ref_line, lines, endpoints, img_size, output_path,
                     reconstructed=None):
    """
    Render the "focus" view: thick black reference line + coloured endpoints.

    Parameters
    ----------
    ref_line : LineSegment
        The selected reference line.
    lines : list of LineSegment
        Background lines (drawn light grey).
    endpoints : list of Endpoint
        Endpoints to colour.
    img_size : tuple
        ``(width, height)`` of the canvas.
    output_path : str
        PNG path to write.
    reconstructed : list of array-like or None, optional
        Reconstructed curve segments (drawn bright red).
    """
    w, h = img_size
    canvas = np.full((h, w, 3), 255, dtype=np.uint8)

    # Background lines.
    for ln in lines:
        pts = ln.pixel_coords.reshape(-1, 1, 2)
        cv2.polylines(canvas, [pts], False, (220, 220, 220), 1)
        mid = ln.pixel_coords[len(ln.pixel_coords) // 2]
        text = ln.line_id
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(canvas,
                      (int(mid[0]) - 2, int(mid[1]) - th - 2),
                      (int(mid[0]) + tw + 2, int(mid[1]) + 3),
                      (255, 255, 255), -1)
        cv2.putText(canvas, text, (int(mid[0]), int(mid[1])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (140, 140, 140), 1)

    # Reference line (thick black).
    pts = ref_line.pixel_coords.reshape(-1, 1, 2)
    cv2.polylines(canvas, [pts], False, (0, 0, 0), 3)
    mid = ref_line.pixel_coords[len(ref_line.pixel_coords) // 2]
    (tw, th), _ = cv2.getTextSize("REF " + ref_line.line_id,
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
    cv2.rectangle(canvas,
                  (int(mid[0]) - 3, int(mid[1]) - th - 3),
                  (int(mid[0]) + tw + 3, int(mid[1]) + 5),
                  (0, 0, 0), -1)
    cv2.putText(canvas, "REF " + ref_line.line_id,
                (int(mid[0]), int(mid[1])),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    # Reconstructed segments (bright red).
    if reconstructed:
        for seg in reconstructed:
            arr = np.array(seg, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(canvas, [arr], False, (0, 0, 220), 2)

    # Endpoints, coloured by status.
    for ep in endpoints:
        if ep.is_used:
            cv2.circle(canvas, ep.point, 6, (160, 160, 160), -1)
            cv2.putText(canvas, ep.pid,
                        (ep.point[0] + 8, ep.point[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1)
            continue
        if ep.side == 1:
            color = (0, 0, 220)
        elif ep.side == -1:
            color = (220, 0, 0)
        else:
            color = (0, 180, 0)
        cv2.circle(canvas, ep.point, 7, color, -1)
        cv2.putText(canvas, ep.pid,
                    (ep.point[0] + 8, ep.point[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    _imwrite_unicode(output_path, canvas)


# ─────────────────────────────────────────────────────────────────────────────
# Tkinter GUI
# ─────────────────────────────────────────────────────────────────────────────
class ReconstructionApp:
    """Tkinter UI that lets the user bind the remaining non-boundary endpoints."""

    def __init__(self, lines, endpoints, img_size, output_dir,
                 bezier_degree: int):
        """
        Build and run the Tkinter UI.

        Parameters
        ----------
        lines : list of LineSegment
            All lines loaded from the input JSON.
        endpoints : list of Endpoint
            Non-boundary endpoints that still need to be reconstructed.
        img_size : tuple
            ``(width, height)`` of the source image.
        output_dir : str
            Directory for the focus / overview / result PNGs and JSON.
        bezier_degree : int
            Bezier degree used for the line approximations.
        """
        self.lines = lines
        self.line_by_id = {ln.line_id: ln for ln in lines}
        self.endpoints = endpoints
        self.img_size = img_size
        self.output_dir = output_dir
        self.bezier_degree = bezier_degree

        self.current_ref: Optional[LineSegment] = None
        self.results: List[Dict] = []
        self.photo: Optional[ImageTk.PhotoImage] = None
        self.check_vars: Dict[str, tk.IntVar] = {}
        self.refined_vars: Dict[str, tk.StringVar] = {}  # Display text.

        self.root = tk.Tk()
        self.root.title("Reference-line reconstruction (paper 4.12)")
        self.root.geometry("960x980")
        self._build_ui()
        self._refresh_overview()
        self.root.mainloop()

    # ──────────────── UI construction ────────────────
    def _build_ui(self):
        """Build the top toolbar, image area, scrollable endpoint list, and bottom action bar."""
        # Top: status bar + reference-line picker.
        top = ttk.Frame(self.root, padding=8)
        top.pack(side=tk.TOP, fill=tk.X)

        self.status_var = tk.StringVar()
        ttk.Label(top, textvariable=self.status_var,
                  font=("Microsoft YaHei", 10, "bold")).pack(side=tk.LEFT)

        ttk.Label(top, text="   Reference line:").pack(side=tk.LEFT)
        self.ref_var = tk.StringVar()
        line_ids = [ln.line_id for ln in sorted(
            self.lines, key=lambda x: int(x.line_id))]
        self.ref_combo = ttk.Combobox(top, textvariable=self.ref_var,
                                       values=line_ids, width=8,
                                       state="readonly")
        self.ref_combo.pack(side=tk.LEFT, padx=4)
        self.ref_combo.bind("<<ComboboxSelected>>",
                             lambda e: self._refresh_focus())
        ttk.Button(top, text="Refresh focus",
                   command=self._refresh_focus).pack(side=tk.LEFT, padx=4)
        ttk.Button(top, text="Overview",
                   command=self._refresh_overview).pack(side=tk.LEFT, padx=4)

        # Image area.
        img_frame = ttk.LabelFrame(self.root, text="Reference-line focus view", padding=4)
        img_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=8)
        self.img_label = ttk.Label(img_frame)
        self.img_label.pack(fill=tk.BOTH, expand=True)

        # Scrollable endpoint list with checkboxes.
        list_outer = ttk.LabelFrame(self.root, text="Candidate endpoints (sorted by distance)",
                                     padding=4)
        list_outer.pack(side=tk.TOP, fill=tk.X, padx=8, pady=4)

        # Header row.
        hdr = ttk.Frame(list_outer)
        hdr.pack(fill=tk.X)
        ttk.Label(hdr, text=" Pick ", width=4, anchor=tk.W).grid(row=0, column=0)
        ttk.Label(hdr, text="pid", width=6, anchor=tk.W).grid(row=0, column=1)
        ttk.Label(hdr, text="Line", width=5, anchor=tk.W).grid(row=0, column=2)
        ttk.Label(hdr, text="End", width=6, anchor=tk.W).grid(row=0, column=3)
        ttk.Label(hdr, text="Coord", width=18, anchor=tk.W).grid(row=0, column=4)

        # Scroll container.
        canvas_list = tk.Canvas(list_outer, height=160, highlightthickness=0)
        scroll = ttk.Scrollbar(list_outer, orient=tk.VERTICAL,
                                command=canvas_list.yview)
        self.list_inner = ttk.Frame(canvas_list)
        self.list_inner.bind("<Configure>",
                              lambda e: canvas_list.configure(
                                  scrollregion=canvas_list.bbox("all")))
        canvas_list.create_window((0, 0), window=self.list_inner, anchor=tk.NW)
        canvas_list.configure(yscrollcommand=scroll.set)
        canvas_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)

        # Quick-selection buttons.
        quick = ttk.Frame(self.root, padding=4)
        quick.pack(side=tk.TOP, fill=tk.X, padx=8)
        ttk.Button(quick, text="Tick the first N unbound endpoints",
                   command=lambda: self._select_n_unused(2)).pack(side=tk.LEFT, padx=2)
        ttk.Button(quick, text="Clear selection",
                   command=self._clear_selection).pack(side=tk.LEFT, padx=2)

        # Bottom action bar.
        bottom = ttk.Frame(self.root, padding=8)
        bottom.pack(side=tk.TOP, fill=tk.X)
        ttk.Button(bottom, text="Connect the 2 selected endpoints",
                   command=self._connect_selected).pack(side=tk.RIGHT)
        ttk.Button(bottom, text="Finish & save",
                   command=self._finish).pack(side=tk.RIGHT, padx=8)
        ttk.Button(bottom, text="Undo last connection",
                   command=self._undo_round).pack(side=tk.RIGHT, padx=8)
        ttk.Button(bottom, text="Clear selection",
                   command=self._clear_selection).pack(side=tk.RIGHT, padx=8)

        self._update_status()

    # ──────────────── Status ────────────────
    def _update_status(self):
        """Refresh the top status label: ``connected / total``."""
        unused = [ep for ep in self.endpoints if not ep.is_used]
        total = len(self.endpoints)
        used = total - len(unused)
        self.status_var.set(f"Connected {used} / {total} endpoints")

    def _find_partner(self, pid: str) -> str:
        """Return the pid bound to ``pid`` in any past round, or ``'-'``."""
        for r in self.results:
            if r["pid_a"] == pid:
                return r["pid_b"]
            if r["pid_b"] == pid:
                return r["pid_a"]
        return "-"

    # ──────────────── List rendering ────────────────
    def _render_endpoint_list(self):
        """Rebuild the scrollable endpoint list widget."""
        for w in self.list_inner.winfo_children():
            w.destroy()
        self.check_vars.clear()

        if self.current_ref is None:
            ttk.Label(self.list_inner,
                      text="(Select a reference line in the dropdown above first; it defines the connection direction)",
                      foreground="gray").pack(anchor=tk.W, padx=4, pady=8)
            return

        unused = [ep for ep in self.endpoints if not ep.is_used]

        for ep in unused:
            var = tk.IntVar(value=0)
            self.check_vars[ep.pid] = var
            row = ttk.Frame(self.list_inner)
            row.pack(fill=tk.X, anchor=tk.W)

            cb = ttk.Checkbutton(row, variable=var)
            cb.grid(row=0, column=0, sticky=tk.W)
            ttk.Label(row, text=ep.pid, width=6,
                     font=("Consolas", 10, "bold")).grid(row=0, column=1, sticky=tk.W)
            ttk.Label(row, text=f"L{ep.line_id}", width=5,
                     font=("Consolas", 10)).grid(row=0, column=2, sticky=tk.W)
            ttk.Label(row, text=ep.end, width=6,
                     font=("Consolas", 10)).grid(row=0, column=3, sticky=tk.W)
            ttk.Label(row, text=f"({ep.point[0]},{ep.point[1]})",
                     foreground="gray",
                     font=("Consolas", 9)).grid(row=0, column=4, sticky=tk.W)

        # Already-connected section.
        used = [ep for ep in self.endpoints if ep.is_used]
        if used:
            sep = ttk.Separator(self.list_inner, orient=tk.HORIZONTAL)
            sep.pack(fill=tk.X, pady=4)
            ttk.Label(self.list_inner,
                      text=f"-- Connected ({len(used)}) --",
                      foreground="gray").pack(anchor=tk.W, padx=4)
            for ep in used:
                row = ttk.Frame(self.list_inner)
                row.pack(fill=tk.X, anchor=tk.W)
                ttk.Label(row, text="  ", width=4).grid(row=0, column=0)
                ttk.Label(row, text=ep.pid, width=6,
                         foreground="gray",
                         font=("Consolas", 10)).grid(row=0, column=1, sticky=tk.W)
                ttk.Label(row, text=f"L{ep.line_id}", width=5,
                         foreground="gray",
                         font=("Consolas", 10)).grid(row=0, column=2, sticky=tk.W)
                ttk.Label(row, text=ep.end, width=6,
                         foreground="gray",
                         font=("Consolas", 10)).grid(row=0, column=3, sticky=tk.W)
                partner = self._find_partner(ep.pid)
                ttk.Label(row, text=f"<-> {partner}",
                         foreground="green",
                         font=("Consolas", 9, "bold")).grid(row=0, column=4,
                                                              sticky=tk.W)

    # ──────────────── Image rendering ────────────────
    def _load_image_into_label(self, png_path: str, max_w: int = 880):
        """
        Load ``png_path`` into the image label, downscaling to ``max_w``.

        Parameters
        ----------
        png_path : str
            PNG file to display.
        max_w : int, optional
            Maximum display width in pixels (default 880).
        """
        img = Image.open(png_path)
        w, h = img.size
        if w > max_w:
            scale = max_w / w
            img = img.resize((int(w * scale), int(h * scale)),
                              Image.LANCZOS)
        self.photo = ImageTk.PhotoImage(img)
        self.img_label.configure(image=self.photo, text="")

    def _refresh_focus(self):
        """Rebuild the focus view with the currently selected reference line."""
        ref_id = self.ref_var.get()
        if not ref_id:
            return
        ref_line = self.line_by_id[ref_id]
        self.current_ref = ref_line

        focus_path = os.path.join(self.output_dir, "_focus_tmp.png")
        reconstructed = [r["curve"] for r in self.results]
        visualize_focus(ref_line, self.lines, self.endpoints,
                         self.img_size, focus_path,
                         reconstructed=reconstructed)
        self._load_image_into_label(focus_path)
        self._render_endpoint_list()

    def _refresh_overview(self):
        """Render the full overview image."""
        overview_path = os.path.join(self.output_dir,
                                      "reference_lines_overview.png")
        reconstructed = [r["curve"] for r in self.results]
        visualize_overview(self.lines, self.endpoints, self.img_size,
                            overview_path, reconstructed=reconstructed)
        self._load_image_into_label(overview_path)

    # ──────────────── Selection shortcuts ────────────────
    def _select_n_unused(self, n: int):
        """
        Tick the first ``n`` unbound endpoints in the list.

        Parameters
        ----------
        n : int
            How many endpoints to tick.
        """
        for ep in self.endpoints:
            if not ep.is_used and ep.pid in self.check_vars:
                self.check_vars[ep.pid].set(1)
                n -= 1
                if n <= 0:
                    break

    def _clear_selection(self):
        """Untick every checkbox."""
        for var in self.check_vars.values():
            var.set(0)

    def _undo_round(self):
        """Undo the most recent connection round."""
        if not self.results:
            messagebox.showinfo("Info", "Nothing to undo")
            return
        last = self.results.pop()
        for ep in self.endpoints:
            if ep.pid in (last["pid_a"], last["pid_b"]):
                ep.is_used = False
                ep.used_by_ref = None
        self._refresh_focus()
        self._update_status()
        messagebox.showinfo("Undone",
                             f"Undid {last['pid_a']} <-> {last['pid_b']} (ref L{last['ref_id']})")

    # ──────────────── Core actions ────────────────
    def _connect_selected(self):
        """Build the connecting curve for the two ticked endpoints."""
        if self.current_ref is None:
            messagebox.showwarning("Info", "Please select a reference line in the dropdown first")
            return
        selected = [ep for ep in self.endpoints
                    if (not ep.is_used and
                        self.check_vars.get(ep.pid, tk.IntVar(value=0)).get())]
        if len(selected) != 2:
            messagebox.showinfo("Info",
                                 f"Exactly 2 endpoints must be ticked (currently {len(selected)})")
            return
        if selected[0].line_id == selected[1].line_id:
            messagebox.showwarning("Info",
                                    "Both endpoints must belong to different original lines (currently on line L"
                                    + selected[0].line_id + ")")
            return

        try:
            curve = connect_two_endpoints(
                selected[0], selected[1],
                self.line_by_id[selected[0].line_id],
                self.line_by_id[selected[1].line_id],
                self.current_ref,
            )
        except Exception as e:
            messagebox.showerror("Connection failed", str(e))
            return

        # Mark both endpoints as used.
        for ep in selected:
            ep.is_used = True
            ep.used_by_ref = self.current_ref.line_id

        self.results.append({
            "pid_a": selected[0].pid,
            "pid_b": selected[1].pid,
            "line_a": selected[0].line_id,
            "line_b": selected[1].line_id,
            "ref_id": self.current_ref.line_id,
            "curve": curve.tolist(),
        })

        # Refresh views.
        self._refresh_focus()
        self._update_status()
        messagebox.showinfo(
            "Done",
            f"Connected {selected[0].pid}(L{selected[0].line_id}) <-> "
            f"{selected[1].pid}(L{selected[1].line_id}), "
            f"reference L{self.current_ref.line_id}\n"
            f"{sum(1 for e in self.endpoints if not e.is_used)} unbound endpoints remaining"
        )

    def _finish(self):
        """Save the result JSON, the overview PNG and the clean PNG, then close."""
        unused = [ep for ep in self.endpoints if not ep.is_used]
        if unused and not messagebox.askyesno(
                "Not fully bound",
                f"{len(unused)} endpoints are still unbound: "
                f"{[ep.pid for ep in unused]}\nSave and exit anyway?"):
            return

        # 1) Save JSON.
        final_json = os.path.join(self.output_dir,
                                   "reference_reconstruction_result.json")
        self._save_json(final_json)

        # 2) Save final overview (grey lines + red reconstructed segments).
        final_vis = os.path.join(self.output_dir,
                                  "reference_reconstruction_vis.png")
        visualize_overview(self.lines, self.endpoints, self.img_size,
                            final_vis,
                            reconstructed=[r["curve"] for r in self.results])

        # 3) Save the clean view (all lines + reconstructed segments in black).
        final_clean = os.path.join(self.output_dir,
                                     "reference_reconstruction_clean.png")
        visualize_clean(self.lines, self.img_size, final_clean,
                         reconstructed=[r["curve"] for r in self.results])

        messagebox.showinfo("Saved",
                             f"Result: {final_json}\n"
                             f"Overview: {final_vis}\n"
                             f"Clean view: {final_clean}")
        self.root.destroy()

    def _save_json(self, path: str):
        """Persist the result payload to ``path``."""
        payload = {
            "metadata": {
                "image_size": {"width": self.img_size[0],
                                "height": self.img_size[1]},
                "bezier_degree": self.bezier_degree,
                "total_endpoints": len(self.endpoints),
                "connected_endpoints": sum(1 for e in self.endpoints if e.is_used),
                "unconnected_pids": [e.pid for e in self.endpoints if not e.is_used],
            },
            "endpoints": [
                {
                    "pid": e.pid, "line_id": e.line_id, "end": e.end,
                    "point": list(e.point), "is_used": e.is_used,
                    "connected_to": self._find_partner(e.pid) if e.is_used else None,
                    "ref_id": e.used_by_ref,
                }
                for e in self.endpoints
            ],
            "connections": [
                {
                    "pid_a": r["pid_a"], "line_a": r["line_a"],
                    "pid_b": r["pid_b"], "line_b": r["line_b"],
                    "reference_id": r["ref_id"],
                    "curve": r["curve"],
                }
                for r in self.results
            ],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    """Build the CLI argument parser."""
    p = argparse.ArgumentParser(
        description="Reference-line reconstruction, interactive Tkinter version (paper 4.12)")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--bezier-degree", type=int, default=4)
    return p.parse_args()


def main():
    """CLI entry point."""
    args = parse_args()

    json_path = os.path.join(args.data_dir, "connected_lines_sa_merged.json")
    if not os.path.isfile(json_path):
        print(f"Missing {json_path}", file=sys.stderr)
        sys.exit(1)

    lines, img_size = load_lines(json_path)
    for ln in lines:
        fit_bezier(ln, degree=args.bezier_degree)

    endpoints = collect_endpoints(lines)
    print(f"[load] {len(lines)} lines, {len(endpoints)} non-boundary endpoints")

    if not endpoints:
        messagebox.showinfo("Info", "No non-boundary endpoints to reconstruct")
        return

    ReconstructionApp(lines, endpoints, img_size,
                      args.data_dir, args.bezier_degree)


if __name__ == "__main__":
    main()