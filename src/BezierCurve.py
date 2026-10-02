# -*- coding: utf-8 -*-
"""
BezierCurve
===========

A collection of static helper methods that operate on planar (2D) Bezier curves.
The class wraps the Bernstein polynomial basis and exposes utilities for:

- Sampling points along a Bezier curve.
- Computing first-order tangent / normal vectors (either via analytic
  derivative, finite difference, or back-tracing over sampled points).
- Projecting arbitrary points onto the curve and querying nearest neighbours.
- Classifying points as being on the same side of the curve or the opposite
  side using a tangent cross-product test.
- Computing geometric properties such as arc length and curvature.
- Plotting curves together with control points, target points and projections.

All methods are ``@staticmethod``s so the class is essentially a namespace; no
state is kept between calls. Throughout this module the term "control points"
means an iterable of 2D points (each point is a sequence ``(x, y)`` or a
length-2 ``numpy.ndarray``).

Notes
-----
The module imports :func:`scipy.special.comb` (Bernstein combinatorial
coefficient), :func:`scipy.integrate.simps` (numerical integration for arc
length) and :func:`scipy.optimize.minimize` (1-D bounded optimization used for
curve projection). Plotting uses :mod:`matplotlib.pyplot`.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simps
from scipy.optimize import minimize
from scipy.special import comb


class BezierCurve:
    """
    Static-method container with utilities for planar Bezier curves.

    The class holds no instance state; every method is a pure function on its
    arguments. The coordinate convention is ``(x, y)`` where ``x`` increases to
    the right and ``y`` increases downward (matching OpenCV / image arrays).
    Several plotting helpers invert the y-axis to follow that convention.
    """

    # ---------------------------------------------------------------------
    # Bernstein polynomial basis and core evaluation
    # ---------------------------------------------------------------------
    @staticmethod
    def bernstein_poly(i, n, t):
        """
        Compute the i-th Bernstein basis polynomial of degree ``n`` evaluated
        at parameter ``t``.

        This is the standard Bernstein basis

            B_{i,n}(t) = C(n, i) * t^(n-i) * (1 - t)^i

        which is used as a blending weight when evaluating a Bezier curve.

        Parameters
        ----------
        i : int
            Basis index, ``0 <= i <= n``.
        n : int
            Degree of the curve. The number of control points is ``n + 1``.
        t : float or numpy.ndarray
            Curve parameter, expected to lie in ``[0, 1]``.

        Returns
        -------
        float or numpy.ndarray
            The Bernstein basis value(s) at ``t``. Same shape as ``t``.
        """
        return comb(n, i) * (t ** (n - i)) * ((1 - t) ** i)

    @staticmethod
    def bezier_curve(t, control_points):
        """
        Evaluate a planar Bezier curve at parameter ``t``.

        The curve is the standard Bernstein expansion over the supplied
        control points. The output is a ``numpy.ndarray`` so that callers can
        treat the result like a 2D vector directly.

        Parameters
        ----------
        t : float
            Curve parameter, expected to lie in ``[0, 1]``.
        control_points : sequence of (x, y)
            Control polygon of the curve. Must contain at least two points.
            Length-1 input is not meaningful for a Bezier curve and will
            yield a zero-length vector.

        Returns
        -------
        numpy.ndarray, shape (2,)
            The 2D point ``B(t)`` on the curve at parameter ``t``.
        """
        n = len(control_points) - 1
        x = y = 0
        for i, point in enumerate(control_points):
            x += BezierCurve.bernstein_poly(i, n, t) * point[0]
            y += BezierCurve.bernstein_poly(i, n, t) * point[1]
        return np.array([x, y])

    @staticmethod
    def plot_bezier_curve(control_points, num_points=600):
        """
        Sample ``num_points`` points along the curve and return them.

        Plotting code is included but currently commented out; this helper is
        effectively a thin wrapper around a vectorised call to
        :func:`bezier_curve`.

        Parameters
        ----------
        control_points : sequence of (x, y)
            Control polygon of the curve.
        num_points : int, optional
            Number of uniformly spaced samples in ``[0, 1]`` (default 600).

        Returns
        -------
        numpy.ndarray, shape (num_points, 2)
            Sampled points on the curve, ordered by increasing ``t``.
        """
        t_values = np.linspace(0, 1, num_points)
        curve_points = np.array([BezierCurve.bezier_curve(t, control_points) for t in t_values])

        # Build a correspondence array between t, control points and samples.
        # correspondence_array = np.column_stack((t_values, control_points, curve_points))

        # Optional visualisation (currently disabled).
        # plt.plot(curve_points[:, 1], curve_points[:, 2], label='Bezier curve')
        # plt.scatter(*zip(*control_points), color='red', label='Control points')
        # plt.legend()
        # plt.show()
        return curve_points

    # ---------------------------------------------------------------------
    # Curve projection: find the closest point on a curve to a target
    # ---------------------------------------------------------------------
    @staticmethod
    def find_projection_on_curve(target_point, control_points):
        """
        Project a 2D point onto the Bezier curve.

        Solves the 1-D optimisation ``min_{t in [0, 1]} || B(t) - target ||``
        using :func:`scipy.optimize.minimize` with the L-BFGS-B algorithm and
        a scalar ``x0 = 0.5`` as the initial guess.

        Parameters
        ----------
        target_point : (x, y) or numpy.ndarray
            The 2D point to project onto the curve.
        control_points : sequence of (x, y)
            Control polygon of the curve.

        Returns
        -------
        numpy.ndarray, shape (2,)
            The point on the curve closest (in Euclidean distance) to
            ``target_point``.
        """
        result = minimize(BezierCurve.distance_function_1, x0=[0.5],
                          args=(control_points, target_point), bounds=[(0, 1)])
        t_projection = result.x[0]
        projection_point_on_curve = BezierCurve.bezier_curve(t_projection, control_points)
        return projection_point_on_curve

    @staticmethod
    def plot_curve_and_projection(control_points, target_point, projection_points):
        """
        Plot the Bezier curve together with its control points, a target point
        and one or more projection points. This is a debugging helper.

        Parameters
        ----------
        control_points : sequence of (x, y)
            Control polygon of the curve.
        target_point : (x, y) or numpy.ndarray
            The target point to display.
        projection_points : iterable of (x, y)
            Projection points (typically output of
            :func:`find_projection_on_curve`) to display in green.
        """
        t_values = np.linspace(0, 1, 100)
        curve_points = np.array([BezierCurve.bezier_curve(t, control_points) for t in t_values])

        plt.plot(curve_points[:, 0], curve_points[:, 1], label='Bezier curve')
        plt.scatter(*zip(*control_points), color='red', label='Control points')
        plt.scatter(*target_point, color='blue', label='Target point')
        for projection_point in projection_points:
            plt.scatter(*projection_point, color='green', label='Projection point')
        plt.legend()
        plt.show()

    @staticmethod
    def find_point_from_projection(projection_point, target_point, distance):
        """
        Step along the line connecting ``projection_point`` -> ``target_point``
        by ``distance`` units.

        The resulting point lies on the ray that starts at the projection and
        points toward the original target. Useful for slightly relaxing the
        projection so that two endpoints no longer coincide exactly.

        Parameters
        ----------
        projection_point : numpy.ndarray, shape (2,)
            The projection result (typically on the curve).
        target_point : numpy.ndarray, shape (2,)
            The original target point.
        distance : float
            Number of pixels (or whatever unit the coordinates use) to step
            along the projection -> target direction.

        Returns
        -------
        numpy.ndarray, shape (2,)
            ``projection_point + (target_point - projection_point) / ||...|| * distance``.
        """
        # 1. Compute the direction vector from projection to target.
        direction_vector = target_point - projection_point
        # 2. Normalise the direction vector.
        normalized_direction = direction_vector / np.linalg.norm(direction_vector)
        # 3. Step along the direction.
        new_point = projection_point + normalized_direction * distance
        return new_point

    # ---------------------------------------------------------------------
    # I/O helpers for sampled curve points
    # ---------------------------------------------------------------------
    @staticmethod
    def save_curve_points(control_points, filename='curve_points.csv'):
        """
        Sample 100 points on the curve and write them to a CSV file.

        Parameters
        ----------
        control_points : sequence of (x, y)
            Control polygon of the curve.
        filename : str, optional
            Output path. Default ``'curve_points.csv'``.
        """
        t_values = np.linspace(0, 1, 100)
        curve_points = np.array([BezierCurve.bezier_curve(t, control_points) for t in t_values])
        np.savetxt(filename, curve_points, delimiter=',')

    @staticmethod
    def save_curve_points_2(all_curve_points, filename='all_curve_points.csv'):
        """
        Save a list of sampled curves to disk using NumPy's ``.npy`` format.

        Parameters
        ----------
        all_curve_points : array-like
            Iterable of arrays. The whole collection is converted to a single
            ``numpy.ndarray`` and written via :func:`numpy.save`.
        filename : str, optional
            Output path. Default ``'all_curve_points.csv'``. (Note: the suffix
            is ``.npy`` in spirit even though the default keeps the legacy
            ``.csv`` name.)
        """
        np.save(filename, np.array(all_curve_points))

    @staticmethod
    def plot_curves_from_file(filename='all_curve_points.csv'):
        """
        Load a previously saved curve collection and plot it.

        Parameters
        ----------
        filename : str, optional
            Path to the file written by :func:`save_curve_points_2` (or any
            compatible array file). Default ``'all_curve_points.csv'``.
        """
        all_curve_points = np.loadtxt(filename, delimiter=',')
        # all_curve_points = all_curve_points.tolist()
        for curve_points in all_curve_points:
            plt.plot(curve_points[:, 1], curve_points[:, 0], label='Bezier lines')

        plt.gca().invert_yaxis()  # Flip y-axis to match image coordinates.
        plt.scatter(*zip(*[(y, x) for x, y in all_curve_points.flatten().tolist()]),
                    s=2, color='red', label='all_control_points')
        plt.legend()
        plt.show()

    # ---------------------------------------------------------------------
    # Derivatives / tangent vectors
    # ---------------------------------------------------------------------
    @staticmethod
    def bezier_curve_derivative(t, control_points):
        """
        First derivative of the Bezier curve at parameter ``t`` (un-normalised).

        Computed analytically as the weighted sum of adjacent control-point
        differences with degree ``n - 1`` Bernstein bases.

        Parameters
        ----------
        t : float
            Curve parameter in ``[0, 1]``.
        control_points : sequence of (x, y)
            Control polygon of the curve.

        Returns
        -------
        tuple (dx, dy)
            The first derivative vector at ``t``. Not normalised.
        """
        n = len(control_points) - 1
        x, y = 0, 0
        for i, point in enumerate(control_points):
            if i > 0:
                basis_derivative = n * (
                    BezierCurve.bernstein_poly(i - 1, n - 1, t)
                    - BezierCurve.bernstein_poly(i, n - 1, t)
                )
                x += point[0] * basis_derivative
                y += point[1] * basis_derivative
        return x, y

    @staticmethod
    def bezier_curve_derivative_for_tan(t, control_points):
        """
        First derivative of the Bezier curve at ``t``, as a single 2D vector.

        This formulation differs slightly from :func:`bezier_curve_derivative`
        in how the binomial coefficients are combined and returns a single
        ``numpy.ndarray`` instead of a tuple of two scalars.

        Parameters
        ----------
        t : float
            Curve parameter in ``[0, 1]``.
        control_points : sequence of (x, y)
            Control polygon of the curve.

        Returns
        -------
        numpy.ndarray, shape (2,)
            The un-normalised first derivative vector at ``t``.
        """
        n = len(control_points) - 1
        result = 0
        for i, point in enumerate(control_points[:-1]):
            result += np.math.comb(n - 1, i) * (n - i) * (1 - t) ** (n - i - 1) * t ** i * (
                control_points[i + 1] - point)
        return result

    # ---------------------------------------------------------------------
    # Side classification: are two points on the same side of the curve?
    # ---------------------------------------------------------------------
    @staticmethod
    def is_points_on_same_side(control_points, point1, point2):
        """
        Check whether two points lie on the same side of the entire curve.

        For every sampled parameter value the tangent at that parameter is
        compared with the vectors to the two candidate points via a 2-D cross
        product. If both cross products are positive for every sample, the
        points are on the same side.

        Parameters
        ----------
        control_points : sequence of (x, y)
            Control polygon of the curve.
        point1 : (x, y) or numpy.ndarray
            First point.
        point2 : (x, y) or numpy.ndarray
            Second point.

        Returns
        -------
        bool
            ``True`` if both points are consistently on the same side of the
            curve; ``False`` as soon as one sample disagrees.
        """
        t_values = np.linspace(0, 1, 1000)

        for t in t_values:
            curve_point = np.array(BezierCurve.bezier_curve(t, control_points))
            derivative_point = np.array(BezierCurve.bezier_curve_derivative(t, control_points))

            vector1 = np.array(point1) - curve_point
            vector2 = np.array(point2) - curve_point
            cross_product = np.cross(derivative_point, vector1) * np.cross(derivative_point, vector2)

            if cross_product <= 0:
                return False

        return True

    @staticmethod
    def classify_points_on_curve(control_points, points):
        """
        Split ``points`` into those on the same side of the curve as the
        tangent direction and those on the opposite side.

        The tangent at every sampled parameter is compared with the vector
        to each candidate point via a 2-D cross product. A positive cross
        product is interpreted as "same side as the tangent".

        Parameters
        ----------
        control_points : sequence of (x, y)
            Control polygon of the curve.
        points : iterable of (x, y)
            Candidate points to classify.

        Returns
        -------
        tuple (same_side, opposite_side)
            Two lists partitioning ``points``. Each entry is the original
            point value (no copy is made).
        """
        points_on_same_side = []
        points_on_opposite_side = []

        for point in points:
            is_same_side = True
            t_values = np.linspace(0, 1, 1000)

            for t in t_values:
                curve_point = np.array(BezierCurve.bezier_curve(t, control_points))
                derivative_point = np.array(BezierCurve.bezier_curve_derivative(t, control_points))

                vector = np.array(point) - curve_point
                cross_product = np.cross(derivative_point, vector)

                if cross_product <= 0:
                    is_same_side = False
                    break

            if is_same_side:
                points_on_same_side.append(point)
            else:
                points_on_opposite_side.append(point)

        return points_on_same_side, points_on_opposite_side

    @staticmethod
    def classify_points_on_curve_2(control_points, points, num_samples=2):
        """
        Same idea as :func:`classify_points_on_curve`, but only evaluates the
        side test at ``num_samples`` evenly spaced positions along the curve
        instead of all 1000. This is much faster at the cost of some accuracy.

        Parameters
        ----------
        control_points : sequence of (x, y)
            Control polygon of the curve.
        points : iterable of (x, y)
            Candidate points to classify.
        num_samples : int, optional
            Number of sampling positions on the curve (default 2).

        Returns
        -------
        tuple (same_side, opposite_side)
            Two lists partitioning ``points``.
        """
        points_on_same_side = []
        points_on_opposite_side = []

        t_values = np.linspace(0, 1, 1000)
        curve_points = np.array([BezierCurve.bezier_curve(t, control_points) for t in t_values])

        # Evenly down-sample the curve and remember the matching t values.
        sampled_curve_points = []
        sampled_t_values = []

        for i in range(0, len(curve_points), len(curve_points) // num_samples):
            t = t_values[i]
            sampled_t_values.append(t)
            sampled_curve_points.append(curve_points[i])

        for point in points:
            is_same_side = True

            for i, curve_point in enumerate(sampled_curve_points):
                t = sampled_t_values[i]
                derivative_point = np.array(BezierCurve.bezier_curve_derivative(t, control_points))

                vector = np.array(point) - curve_point
                cross_product = np.cross(derivative_point, vector)

                if cross_product <= 0:
                    is_same_side = False
                    break

            if is_same_side:
                points_on_same_side.append(point)
            else:
                points_on_opposite_side.append(point)

        return points_on_same_side, points_on_opposite_side

    @staticmethod
    def classify_points_on_curve_final(control_points, points):
        """
        Classify ``points`` using only the local tangent at the curve position
        closest to each point.

        For every candidate point the closest sampled curve point is located
        first; the local tangent at that position is then used to decide on
        which side of the curve the candidate lies.

        Parameters
        ----------
        control_points : sequence of (x, y)
            Control polygon of the curve.
        points : iterable of (x, y)
            Candidate points to classify.

        Returns
        -------
        tuple (same_side, opposite_side)
            Two lists partitioning ``points``.
        """
        points_on_same_side = []
        points_on_opposite_side = []

        t_values = np.linspace(0, 1, 100)
        curve_points = np.array([BezierCurve.bezier_curve(t, control_points) for t in t_values])

        # Sampled curve points and their t values (one sample per index here).
        sampled_curve_points = []
        sampled_t_values = []

        for i in range(0, len(curve_points)):
            t = t_values[i]
            sampled_t_values.append(t)
            sampled_curve_points.append(curve_points[i])

        for point in points:
            is_same_side = True

            # Find the closest curve sample, then read the local tangent
            # from the neighbour with index +1 (or -1 at the end of the curve).
            curve_point, distance, nearest_index = BezierCurve.find_nearest_point_and_distance(point, curve_points)

            if nearest_index + 1 >= len(curve_points):
                derivative_point = curve_points[nearest_index - 1] - curve_points[nearest_index]
                vector = np.array(point) - curve_point
                cross_product = np.cross(vector, derivative_point)
            else:
                derivative_point = curve_points[nearest_index + 1] - curve_points[nearest_index]
                vector = np.array(point) - curve_point
                cross_product = np.cross(derivative_point, vector)

            if cross_product <= 0:
                is_same_side = False

            if is_same_side:
                points_on_same_side.append(point)
            else:
                points_on_opposite_side.append(point)

        return points_on_same_side, points_on_opposite_side

    @staticmethod
    def judge_line_strike(curve_points, point):
        """
        Decide whether a ``point`` lies on the same side as the tangent at
        the closest sampled curve position.

        Parameters
        ----------
        curve_points : numpy.ndarray, shape (N, 2)
            Pre-sampled points on the Bezier curve.
        point : (x, y) or numpy.ndarray
            Candidate point to classify.

        Returns
        -------
        bool
            ``True`` if ``point`` is on the same side as the local tangent
            at the nearest curve sample, ``False`` otherwise.
        """
        is_same_side = True

        # Find the closest curve sample, then use the neighbour with index +1
        # (or -1 at the end of the curve) as the local tangent direction.
        curve_point, distance, nearest_index = BezierCurve.find_nearest_point_and_distance(point, curve_points)

        if nearest_index + 1 >= len(curve_points):
            derivative_point = curve_points[nearest_index] - curve_points[nearest_index - 1]
            vector = np.array(point) - curve_points[nearest_index - 1]
            cross_product = np.cross(derivative_point, vector)
        else:
            derivative_point = curve_points[nearest_index + 1] - curve_points[nearest_index]
            vector = np.array(point) - curve_point
            cross_product = np.cross(derivative_point, vector)

        if cross_product <= 0:
            is_same_side = False
        return is_same_side

    @staticmethod
    def find_nearest_point_and_distance(target_point, curve_points):
        """
        Locate the closest sample to ``target_point`` on a pre-sampled curve.

        Parameters
        ----------
        target_point : (x, y) or numpy.ndarray
            Point to query.
        curve_points : numpy.ndarray, shape (N, 2)
            Pre-sampled curve points.

        Returns
        -------
        tuple (nearest_point, nearest_distance, nearest_index)
            ``nearest_point`` is the closest sample (``curve_points[i]``).
            ``nearest_distance`` is its Euclidean distance to ``target_point``.
            ``nearest_index`` is its index in ``curve_points``.
        """
        distances = np.linalg.norm(curve_points - target_point, axis=1)
        nearest_index = np.argmin(distances)
        nearest_point = curve_points[nearest_index]
        nearest_distance = distances[nearest_index]
        return nearest_point, nearest_distance, nearest_index

    @staticmethod
    def plot_curve_and_classified_points(control_points, same_side_points, opposite_side_points):
        """
        Plot a curve together with points classified as "same side" (green)
        and "opposite side" (blue).

        Parameters
        ----------
        control_points : sequence of (x, y)
            Control polygon of the curve.
        same_side_points : iterable of (x, y)
            Points expected to be on the same side as the tangent.
        opposite_side_points : iterable of (x, y)
            Points expected to be on the opposite side.
        """
        t_values = np.linspace(0, 1, 1000)
        curve_points = np.array([BezierCurve.bezier_curve(t, control_points) for t in t_values])

        plt.plot(curve_points[:, 1], curve_points[:, 0], label="Bezier Curve")
        # plt.scatter(*zip(*control_points), color='red', label="Control Points", zorder=5)
        plt.scatter(*zip(*[(y, x) for x, y in control_points]),
                    color='red', label="Control Points", zorder=5)

        if same_side_points:
            same_side_y, same_side_x = zip(*same_side_points)
            plt.scatter(same_side_x, same_side_y,
                        color='green', label="Same Side Points", zorder=5)

        if opposite_side_points:
            opposite_side_y, opposite_side_x = zip(*opposite_side_points)
            plt.scatter(opposite_side_x, opposite_side_y,
                        color='blue', label="Opposite Side Points", zorder=5)
        plt.gca().invert_yaxis()  # Flip y-axis to match image coordinates.
        plt.title("Bezier Curve and Classified Points")
        plt.axis('equal')
        # plt.legend()
        plt.show()

    # ---------------------------------------------------------------------
    # Distance helpers used by the projection optimisers
    # ---------------------------------------------------------------------
    @staticmethod
    def distance_function_1(point, control_points, target_point):
        """
        Scalar objective for :func:`find_projection_on_curve`.

        Given a 1-D optimisation variable ``point = [t]`` it returns the
        Euclidean distance between ``B(t)`` and ``target_point``. The unusual
        argument order (``point`` first, ``control_points`` second,
        ``target_point`` third) matches the call site in
        :func:`find_projection_on_curve`.

        Parameters
        ----------
        point : array-like, shape (1,)
            The optimisation variable ``[t]``.
        control_points : sequence of (x, y)
            Control polygon of the curve.
        target_point : (x, y) or numpy.ndarray
            The point to project onto the curve.

        Returns
        -------
        float
            Euclidean distance between ``B(t)`` and ``target_point``.
        """
        curve_point = BezierCurve.bezier_curve(point[0], control_points)
        return np.linalg.norm(curve_point - target_point)

    @staticmethod
    def distance_function(t, point, control_points):
        """
        Scalar objective used by :func:`find_nearest_points_on_curve` and
        :func:`distance_to_curve`.

        Parameters
        ----------
        t : float or array-like
            Curve parameter ``t``.
        point : (x, y) or numpy.ndarray
            Point to compare against the curve.
        control_points : sequence of (x, y)
            Control polygon of the curve.

        Returns
        -------
        float
            Euclidean distance between ``B(t)`` and ``point``.
        """
        curve_point = BezierCurve.bezier_curve(t, control_points)
        return np.linalg.norm(curve_point - point)

    @staticmethod
    def find_nearest_points_on_curve(points, control_points):
        """
        For each input point, compute the Euclidean distance to the closest
        point on the curve.

        Parameters
        ----------
        points : iterable of (x, y)
            Points to query.
        control_points : sequence of (x, y)
            Control polygon of the curve.

        Returns
        -------
        list of float
            The minimum distance for each input point, in the same order.
        """
        nearest_distances = []

        for point in points:
            result = minimize(BezierCurve.distance_function, x0=[0.5],
                              args=(point, control_points), bounds=[(0, 1)])
            t_nearest = result.x[0]
            nearest_point_on_curve = BezierCurve.bezier_curve(t_nearest, control_points)
            distance = np.linalg.norm(nearest_point_on_curve - point)
            nearest_distances.append(distance)

        return nearest_distances

    @staticmethod
    def find_two_closest_points(points, control_points):
        """
        Return the two input points that are closest to the curve.

        Parameters
        ----------
        points : iterable of (x, y)
            Points to query.
        control_points : sequence of (x, y)
            Control polygon of the curve.

        Returns
        -------
        list of (x, y)
            Subset of ``points`` (length 2) ordered by their distance to the
            curve, ascending.
        """
        distances = BezierCurve.find_nearest_points_on_curve(points, control_points)

        # Find the two smallest distances.
        closest_indices = np.argsort(distances)[:2]

        # Map those indices back to the original points.
        closest_points = [points[i] for i in closest_indices]

        return closest_points

    @staticmethod
    def distance_to_curve(point, control_points):
        """
        Compute the shortest Euclidean distance from ``point`` to the curve.

        Parameters
        ----------
        point : (x, y) or numpy.ndarray
            Point to query.
        control_points : sequence of (x, y)
            Control polygon of the curve.

        Returns
        -------
        float
            Minimum distance between ``point`` and any point on the curve.
        """
        result = minimize(BezierCurve.distance_function, x0=[0.5],
                          args=(point, control_points), bounds=[(0, 1)])
        t_nearest = result.x[0]
        nearest_point_on_curve = BezierCurve.bezier_curve(t_nearest, control_points)
        distance = np.linalg.norm(nearest_point_on_curve - point)
        return distance

    # ---------------------------------------------------------------------
    # Tangent / normal vectors via finite differences
    # ---------------------------------------------------------------------
    @staticmethod
    def tangent_vector(t, control_points):
        """
        Unit tangent vector at parameter ``t`` via symmetric finite
        differences.

        Parameters
        ----------
        t : float
            Curve parameter in ``[0, 1]``.
        control_points : sequence of (x, y)
            Control polygon of the curve.

        Returns
        -------
        numpy.ndarray, shape (2,)
            The normalised tangent vector at ``t``.
        """
        epsilon = 1e-5
        t1 = min(max(t - epsilon, 0), 1)
        t2 = min(max(t + epsilon, 0), 1)
        p1 = BezierCurve.bezier_curve(t1, control_points)
        p2 = BezierCurve.bezier_curve(t2, control_points)
        tangent_vector = (p2 - p1) / (t2 - t1)
        return tangent_vector / np.linalg.norm(tangent_vector)

    @staticmethod
    def normal_vector(t, control_points):
        """
        Unit normal vector at parameter ``t``.

        Computed as the 90-degree rotation of the tangent (counter-clockwise
        in a standard right-handed coordinate system).

        Parameters
        ----------
        t : float
            Curve parameter in ``[0, 1]``.
        control_points : sequence of (x, y)
            Control polygon of the curve.

        Returns
        -------
        numpy.ndarray, shape (2,)
            The normalised normal vector at ``t``.
        """
        tangent_vector = BezierCurve.tangent_vector(t, control_points)
        normal_vector = np.array([-tangent_vector[1], tangent_vector[0]])
        return normal_vector / np.linalg.norm(normal_vector)

    @staticmethod
    def find_point_along_normal(start_point, control_points, distance, i):
        """
        Step from the curve projection of ``start_point`` along its local
        normal by ``distance`` pixels.

        Parameters
        ----------
        start_point : (x, y) or numpy.ndarray
            The starting point; its nearest projection on the curve is found
            first.
        control_points : sequence of (x, y)
            Control polygon of the curve.
        distance : float
            Step length along the normal (in coordinate units).
        i : int
            ``+1`` to step in the direction opposite to the normal,
            ``-1`` to step along the normal.

        Returns
        -------
        numpy.ndarray, shape (2,)
            ``projection +/- normal * distance``.
        """
        result = minimize(
            lambda t: np.linalg.norm(BezierCurve.bezier_curve(t, control_points) - start_point),
            x0=[0.5], bounds=[(0, 1)]
        )
        t = result.x[0]
        normal_vector = BezierCurve.normal_vector(t, control_points)
        if i == 1:
            new_point = start_point - normal_vector * distance  # Opposite direction.
        if i == -1:
            new_point = start_point + normal_vector * distance
        return new_point

    @staticmethod
    def find_point_along_normal_on_curve(t, control_points, distance):
        """
        Move from ``B(t)`` along its local normal by ``distance``.

        Parameters
        ----------
        t : float
            Curve parameter in ``[0, 1]``.
        control_points : sequence of (x, y)
            Control polygon of the curve.
        distance : float
            Step length along the normal.

        Returns
        -------
        numpy.ndarray, shape (2,)
            ``B(t) + normal * distance``.
        """
        # Tangent and normal vectors.
        tangent_vector = BezierCurve.tangent_vector(t, control_points)
        normal_vector = np.array([-tangent_vector[1], tangent_vector[0]])

        # New point.
        new_point = BezierCurve.bezier_curve(t, control_points) + normal_vector * distance
        return new_point

    # ---------------------------------------------------------------------
    # Projection helpers (index-based, used by classify_points_on_curve_final)
    # ---------------------------------------------------------------------
    @staticmethod
    def find_nearest_curve_point(projection_point, curve_points):
        """
        Index of the curve sample closest to ``projection_point``.

        Parameters
        ----------
        projection_point : (x, y) or numpy.ndarray
            Query point.
        curve_points : numpy.ndarray, shape (N, 2)
            Pre-sampled curve points.

        Returns
        -------
        int
            Index in ``curve_points`` of the closest sample.
        """
        index_nearest = np.argmin(np.linalg.norm(curve_points - projection_point, axis=1))
        return index_nearest

    @staticmethod
    def find_second_nearest_curve_point(projection_point, curve_points):
        """
        Index of the *second*-closest curve sample to ``projection_point``.

        The closest one is masked with ``+inf`` before the second
        :func:`numpy.argmin` call.

        Parameters
        ----------
        projection_point : (x, y) or numpy.ndarray
            Query point.
        curve_points : numpy.ndarray, shape (N, 2)
            Pre-sampled curve points.

        Returns
        -------
        int
            Index in ``curve_points`` of the second-closest sample.
        """
        distances = np.linalg.norm(curve_points - projection_point, axis=1)

        # Mask the minimum so argmin finds the second-smallest distance.
        min_distance_index = np.argmin(distances)
        distances[min_distance_index] = np.inf
        second_min_distance_index = np.argmin(distances)

        return second_min_distance_index

    @staticmethod
    def find_t_from_point(target_point, control_points):
        """
        Solve for the parameter ``t`` of the curve point closest to
        ``target_point``.

        Parameters
        ----------
        target_point : (x, y) or numpy.ndarray
            Query point.
        control_points : sequence of (x, y)
            Control polygon of the curve.

        Returns
        -------
        float
            The optimal parameter ``t`` in ``[0, 1]`` that minimises the
            Euclidean distance between ``B(t)`` and ``target_point``.
        """
        result = minimize(
            lambda t: BezierCurve.distance_function(t, target_point, control_points),
            x0=[0.5], bounds=[(0, 1)]
        )
        t = result.x[0]
        return t

    @staticmethod
    def tangent_vector_1(t, control_points):
        """
        Unit tangent at parameter ``t`` via central differences (h = 1e-5).

        Equivalent to :func:`tangent_vector` but uses an explicit symmetric
        difference instead of clamping ``t ± epsilon`` into ``[0, 1]``.

        Parameters
        ----------
        t : float
            Curve parameter in ``[0, 1]``.
        control_points : sequence of (x, y)
            Control polygon of the curve.

        Returns
        -------
        numpy.ndarray, shape (2,)
            Normalised tangent vector at ``t``.
        """
        h = 1e-5
        tangent_vector = (
            BezierCurve.bezier_curve(t + h, control_points)
            - BezierCurve.bezier_curve(t - h, control_points)
        ) / (2 * h)
        return tangent_vector / np.linalg.norm(tangent_vector)

    @staticmethod
    def perpendicular_vector(vector):
        """
        Return the 90-degree rotation of a 2D vector (counter-clockwise).

        Parameters
        ----------
        vector : sequence of length 2
            Any 2D vector.

        Returns
        -------
        numpy.ndarray, shape (2,)
            ``[-vector[1], vector[0]]`` (note: not normalised).
        """
        return np.array([-vector[1], vector[0]])

    @staticmethod
    def find_point_along_normal_on_curve_2(t, control_points, distance):
        """
        Variant of :func:`find_point_along_normal_on_curve` that uses
        :func:`tangent_vector_1` (central differences) instead of the
        clamped-difference tangent.

        Parameters
        ----------
        t : float
            Curve parameter in ``[0, 1]``.
        control_points : sequence of (x, y)
            Control polygon of the curve.
        distance : float
            Step length along the normal.

        Returns
        -------
        numpy.ndarray, shape (2,)
            ``B(t) + normal * distance``.
        """
        # Tangent and normal vectors.
        tangent_vector = BezierCurve.tangent_vector_1(t, control_points)
        normal_vector = BezierCurve.perpendicular_vector(tangent_vector)

        # New point.
        new_point = BezierCurve.bezier_curve(t, control_points) + normal_vector * distance
        return new_point

    @staticmethod
    def find_projection_on_curve_1(target_point, control_points):
        """
        Project ``target_point`` onto the curve, minimising distance while
        also keeping the connecting line approximately perpendicular to the
        tangent (the angle is logged but only distance enters the objective).

        Parameters
        ----------
        target_point : (x, y) or numpy.ndarray
            Point to project.
        control_points : sequence of (x, y)
            Control polygon of the curve.

        Returns
        -------
        numpy.ndarray, shape (2,)
            The point on the curve closest to ``target_point``.
        """
        def distance_and_perpendicular_angle(t):
            curve_point = BezierCurve.bezier_curve(t, control_points)
            distance = np.linalg.norm(curve_point - target_point)
            tangent_vector = BezierCurve.tangent_vector(t, control_points)
            target_vector = target_point - curve_point
            angle = np.arccos(
                np.dot(target_vector, tangent_vector)
                / (np.linalg.norm(target_vector) * np.linalg.norm(tangent_vector))
            )
            return distance, angle

        result = minimize(lambda t: distance_and_perpendicular_angle(t)[0],
                          x0=[0.5], bounds=[(0, 1)])
        t_projection = result.x[0]
        projection_point_on_curve = BezierCurve.bezier_curve(t_projection, control_points)
        return projection_point_on_curve

    @staticmethod
    def find_projection_on_curve_2(target_point, control_points):
        """
        Project ``target_point`` onto the curve using a constrained
        optimisation where the projection is forced to lie exactly on the
        curve.

        Parameters
        ----------
        target_point : (x, y) or numpy.ndarray
            Point to project.
        control_points : sequence of (x, y)
            Control polygon of the curve.

        Returns
        -------
        numpy.ndarray, shape (2,)
            The constrained projection point on the curve.
        """
        def constraint(t):
            # Equality constraint: projection lies on the curve.
            return BezierCurve.bezier_curve_2(t, control_points) - target_point

        result = minimize(
            lambda t: BezierCurve.distance_function_2(t, target_point, control_points)[0],
            x0=[0.5], bounds=[(0, 1)],
            constraints={'type': 'eq', 'fun': constraint}
        )
        t_projection = result.x[0]
        projection_point_on_curve = BezierCurve.bezier_curve_2(t_projection, control_points)
        return projection_point_on_curve

    @staticmethod
    def distance_function_2(t, target_point, control_points):
        """
        Distance + perpendicular-angle objective used by
        :func:`find_projection_on_curve_2`.

        Parameters
        ----------
        t : float or array-like
            Curve parameter.
        target_point : (x, y) or numpy.ndarray
            Query point.
        control_points : sequence of (x, y)
            Control polygon of the curve.

        Returns
        -------
        tuple (distance, angle)
            ``distance`` is the Euclidean distance from ``B(t)`` to
            ``target_point``; ``angle`` is the angle between the connecting
            vector and the local tangent (in radians).
        """
        curve_point = BezierCurve.bezier_curve_2(t, control_points)
        distance = np.linalg.norm(curve_point - target_point)
        tangent_vector = BezierCurve.tangent_vector_2(t, control_points)
        target_vector = target_point - curve_point
        angle = np.arccos(
            np.dot(target_vector, tangent_vector)
            / (np.linalg.norm(target_vector) * np.linalg.norm(tangent_vector))
        )
        return distance, angle

    @staticmethod
    def bezier_curve_2(t, control_points):
        """
        Vector-form Bezier evaluation. Same mathematical result as
        :func:`bezier_curve` but keeps intermediate arrays in numpy form and
        accumulates with ``np.zeros_like`` to broadcast cleanly with array
        control points.

        Parameters
        ----------
        t : float or array-like
            Curve parameter in ``[0, 1]``.
        control_points : sequence of (x, y) or numpy.ndarray
            Control polygon of the curve.

        Returns
        -------
        numpy.ndarray, shape (2,) (or matching ``control_points[0]``)
            ``B(t)``.
        """
        n = len(control_points) - 1
        point = np.zeros_like(control_points[0])
        for i, ctrl_point in enumerate(control_points):
            point += BezierCurve.bernstein_poly(i, n, t) * ctrl_point
        return point

    @staticmethod
    def tangent_vector_2(t, control_points):
        """
        Unit tangent at ``t`` via central differences, using
        :func:`bezier_curve_2` for evaluation.

        Parameters
        ----------
        t : float
            Curve parameter in ``[0, 1]``.
        control_points : sequence of (x, y)
            Control polygon of the curve.

        Returns
        -------
        numpy.ndarray, shape (2,)
            Normalised tangent vector at ``t``.
        """
        h = 1e-5
        tangent_vector = (
            BezierCurve.bezier_curve_2(t + h, control_points)
            - BezierCurve.bezier_curve_2(t - h, control_points)
        ) / (2 * h)
        return tangent_vector / np.linalg.norm(tangent_vector)

    # ---------------------------------------------------------------------
    # Multi-curve helpers
    # ---------------------------------------------------------------------
    @staticmethod
    def find_nearest_control_points(target_point, control_points_list):
        """
        Given a list of candidate control polygons, return the one whose
        nearest control point to ``target_point`` is closest.

        Parameters
        ----------
        target_point : (x, y) or numpy.ndarray
            Query point.
        control_points_list : iterable of sequences
            Each entry is a control polygon (sequence of (x, y)).

        Returns
        -------
        tuple (nearest_control_points, index)
            The polygon with the smallest minimum control-point distance, and
            its position in the input list. The polygon is returned as a copy
            so that the caller can mutate it without affecting the input.
        """
        nearest_distance = float('inf')
        nearest_control_points = None
        index = -1

        for i, control_points in enumerate(control_points_list):
            distance = np.linalg.norm(target_point - control_points, axis=1).min()

            if distance < nearest_distance:
                nearest_distance = distance
                nearest_control_points = control_points.copy()
                index = i
        return nearest_control_points, index

    @staticmethod
    def plot_for_debug(all_curve_points, closest_points, projection_points, new_points):
        """
        Debug plot showing multiple sampled curves together with three sets of
        test points (closest, projection, new) in distinct colours.

        Parameters
        ----------
        all_curve_points : iterable of numpy.ndarray, each (N, 2)
            Curves to draw in blue.
        closest_points : iterable of (x, y)
            Closest-input points (drawn green).
        projection_points : iterable of (x, y)
            Projection points (drawn red).
        new_points : iterable of (x, y)
            New points (drawn blue).
        """
        for curve_points in all_curve_points:
            plt.plot(curve_points[:, 1], curve_points[:, 0], label='Bezier lines')

        for point in closest_points:
            x, y = point[1], point[0]
            plt.scatter(x, y, color='green', label="closest_points", zorder=5)
        for point in projection_points:
            x, y = point[1], point[0]
            plt.scatter(x, y, color='red', label="projection_points", zorder=5)
        for point in new_points:
            x, y = point[1], point[0]
            plt.scatter(x, y, color='blue', label="new_points", zorder=5)

        plt.gca().invert_yaxis()  # Flip y-axis to match image coordinates.
        plt.title("Bezier Curve and Classified Points")
        plt.legend()
        plt.show()

    @staticmethod
    def find_nearest_point_speederror(points, control_points):
        """
        Vectorised fast approximation of the closest point on the curve.

        Samples the curve at 100 values, computes the full pairwise distance
        matrix in one shot, and returns the closest sample's coordinates and
        distance. This is a rough approximation compared with the L-BFGS-B
        solver used in :func:`find_projection_on_curve` but is much faster.

        Parameters
        ----------
        points : iterable of (x, y)
            Query points.
        control_points : sequence of (x, y)
            Control polygon of the curve.

        Returns
        -------
        tuple (nearest_point, min_distance)
            ``nearest_point`` is the closest sample (a ``list`` of two
            scalars); ``min_distance`` is the corresponding distance.
            The values are taken from the *first* input point, which is
            almost certainly a bug — see the inline comment for context.
        """
        t_values = np.linspace(0, 1, 100)

        # Pairwise distance matrix between every point and every curve sample.
        curve_points = np.array([BezierCurve.bezier_curve(t, control_points) for t in t_values])
        points_matrix = np.array(points)
        distances_matrix = np.linalg.norm(points_matrix[:, np.newaxis, :] - curve_points, axis=-1)

        # Closest sample and its index for each query point.
        min_distances = np.min(distances_matrix, axis=1)
        min_distance_indices = np.argmin(distances_matrix, axis=1)

        # Closest sample and its distance for the first input point.
        nearest_point = points_matrix[min_distance_indices[0]]
        min_distance = min_distances[0]

        return list(nearest_point), min_distance

    @staticmethod
    def find_nearest_point(points, control_points):
        """
        Iterate over ``points`` and return the one whose closest distance to
        the curve is smallest.

        Parameters
        ----------
        points : iterable of (x, y)
            Query points.
        control_points : sequence of (x, y)
            Control polygon of the curve.

        Returns
        -------
        tuple (nearest_point, min_distance)
            The input point with the smallest minimum distance to the curve,
            together with that distance.
        """
        min_distance = float('inf')
        nearest_point = None

        for point in points:
            # Euclidean distance from the point to each sampled curve point.
            t_values = np.linspace(0, 1, 100)
            distances = [np.linalg.norm(point - BezierCurve.bezier_curve(t, control_points)) for t in t_values]

            # Smallest distance for the current point.
            min_distance_value = np.min(distances)

            # Update the running best across all points.
            if min_distance_value < min_distance:
                min_distance = min_distance_value
                nearest_point = point

        return nearest_point, min_distance

    @staticmethod
    def find_nearest_control_point(point, control_points):
        """
        Return the control point closest (in Euclidean distance) to ``point``.

        Parameters
        ----------
        point : (x, y) or numpy.ndarray
            Query point.
        control_points : sequence of (x, y)
            Control polygon.

        Returns
        -------
        numpy.ndarray, shape (2,)
            The closest control point.
        """
        # Euclidean distance to each control point.
        distances = [np.linalg.norm(point - control_point) for control_point in control_points]

        # Index of the minimum.
        nearest_index = np.argmin(distances)

        # Closest control point.
        nearest_control_point = control_points[nearest_index]

        return nearest_control_point

    # ---------------------------------------------------------------------
    # Direction / curvature / length utilities
    # ---------------------------------------------------------------------
    @staticmethod
    def determine_direction(A, B, points, index):
        """
        Walk through ``points[index:]`` and check whether the angle between
        the constant vector ``A -> B`` and the moving vector ``A -> points[i]``
        is monotonically increasing, decreasing, or neither.

        Returns ``1`` as soon as a decrease is detected, ``-1`` as soon as an
        increase is detected, and ``0`` if the angles stay constant.

        Parameters
        ----------
        A : numpy.ndarray, shape (2,)
            The reference start point.
        B : numpy.ndarray, shape (2,)
            The reference end point. Used only to define the direction
            ``A -> B``.
        points : list of numpy.ndarray
            Candidate points to evaluate.
        index : int
            Starting index inside ``points``.

        Returns
        -------
        int
            ``1``  - the angle decreases,
            ``-1`` - the angle increases,
            ``0``  - the angle stays constant over the walk.
        """
        # Direction vector from A to B.
        AB = B - A

        # Angle accumulator.
        angle1 = None

        for i in range(index, len(points) - 1):
            # Angle between AB and A -> points[i].
            vector_AP = points[i] - A
            cos_angle = np.dot(AB, vector_AP) / (np.linalg.norm(AB) * np.linalg.norm(vector_AP))
            current_angle1 = np.arccos(cos_angle)

            if angle1 is not None and current_angle1 < angle1:
                # Angle is shrinking: direction has changed.
                return 1
            if angle1 is not None and current_angle1 > angle1:
                return -1

            angle1 = current_angle1

        # No change: angle stays monotonically non-decreasing.
        return 0

    @staticmethod
    def calculate_curvature(t, control_points, epsilon=1e-6):
        """
        Curvature ``kappa(t)`` of the Bezier curve at parameter ``t``.

        Uses central differences for the first and second derivatives and the
        analytic formula ``kappa = |B' x B''| / |B'|^3``.

        Parameters
        ----------
        t : float
            Curve parameter in ``[0, 1]``.
        control_points : sequence of (x, y)
            Control polygon of the curve.
        epsilon : float, optional
            Finite-difference step (default 1e-6).

        Returns
        -------
        float
            The unsigned curvature at ``t``.
        """
        # First derivative.
        dt = epsilon
        dx_dt = (BezierCurve.bezier_curve(t + dt, control_points)
                 - BezierCurve.bezier_curve(t - dt, control_points)) / (2 * dt)

        # Second derivative.
        d2x_dt2 = (BezierCurve.bezier_curve(t + dt, control_points)
                   - 2 * BezierCurve.bezier_curve(t, control_points)
                   + BezierCurve.bezier_curve(t - dt, control_points)) / (dt ** 2)

        # Curvature.
        curvature = np.linalg.norm(np.cross(dx_dt, d2x_dt2)) / (np.linalg.norm(dx_dt) ** 3)

        return curvature

    @staticmethod
    def calculate_curvature_at_points(t_values, control_points, epsilon=1e-6):
        """
        Compute the curvature of the curve at every value in ``t_values``.

        Parameters
        ----------
        t_values : iterable of float
            Curve parameters (each in ``[0, 1]``).
        control_points : sequence of (x, y)
            Control polygon of the curve.
        epsilon : float, optional
            Finite-difference step passed to :func:`calculate_curvature`.

        Returns
        -------
        list of float
            Curvatures, in the same order as ``t_values``.
        """
        curvatures = []
        for t in t_values:
            curvature_at_point = BezierCurve.calculate_curvature(t, control_points, epsilon)
            curvatures.append(curvature_at_point)
        return curvatures

    @staticmethod
    def bezier_curve_length(control_points, num_points=100):
        """
        Approximate arc length of the curve via Simpson's rule.

        Parameters
        ----------
        control_points : sequence of (x, y)
            Control polygon of the curve.
        num_points : int, optional
            Number of samples used to evaluate the integrand (default 100).

        Returns
        -------
        float
            Approximate arc length of the curve.
        """
        t_values = np.linspace(0, 1, num_points)
        curve_points = np.array([BezierCurve.bezier_curve(t, control_points) for t in t_values])

        # Simpson's rule on ||B'(t)||.
        dx_dt = np.gradient(curve_points[:, 0], t_values)
        dy_dt = np.gradient(curve_points[:, 1], t_values)
        integrand = np.sqrt(dx_dt ** 2 + dy_dt ** 2)

        length = simps(integrand, t_values)
        return length

    @staticmethod
    def find_nearby_points_within_threshold(points, control_points, threshold=5):
        """
        Vectorised version of :func:`find_nearby_points_within_threshold_1`.

        Returns the subset of ``points`` whose minimum distance to the curve
        is within ``threshold`` of the global minimum.

        Parameters
        ----------
        points : iterable of (x, y)
            Candidate points.
        control_points : sequence of (x, y)
            Control polygon of the curve.
        threshold : float, optional
            Allowed deviation from the global minimum distance (default 5).

        Returns
        -------
        tuple (nearby_points, global_min)
            ``nearby_points`` is the filtered subset; ``global_min`` is the
            minimum distance observed across all points.
        """
        t_values = np.linspace(0, 1, 100)

        # Pairwise distance matrix between points and sampled curve positions.
        curve_points = np.array([BezierCurve.bezier_curve(t, control_points) for t in t_values])
        distances_matrix = np.linalg.norm(np.array(points)[:, np.newaxis, :] - curve_points, axis=-1)

        min_distances = np.min(distances_matrix, axis=1)

        # Filter points whose distance is close to the global minimum.
        nearby_points = [point for point, min_distance in zip(points, min_distances)
                         if np.abs(min_distance - min_distances.min()) < threshold]
        return nearby_points, min_distances.min()

    @staticmethod
    def find_nearby_points_within_threshold_1(points, control_points, threshold=5):
        """
        Loop-based variant of :func:`find_nearby_points_within_threshold`. Same
        contract but processes one point at a time.

        Parameters
        ----------
        points : iterable of (x, y)
            Candidate points.
        control_points : sequence of (x, y)
            Control polygon of the curve.
        threshold : float, optional
            Allowed deviation from the running minimum distance (default 5).

        Returns
        -------
        tuple (nearby_points, min_distance)
            ``nearby_points`` is the filtered subset; ``min_distance`` is the
            final minimum distance observed.
        """
        min_distance = float('inf')
        nearby_points = []

        for point in points:
            # Euclidean distance to every sampled curve position.
            t_values = np.linspace(0, 1, 100)
            distances = [np.linalg.norm(point - BezierCurve.bezier_curve(t, control_points)) for t in t_values]

            # Smallest distance for this point.
            min_distance_value = np.min(distances)

            # Keep the point if it is within ``threshold`` of the running minimum.
            distance_difference = np.abs(min_distance_value - min_distance)
            if distance_difference < threshold:
                nearby_points.append(point)

            # Update the running minimum.
            if min_distance_value < min_distance:
                min_distance = min_distance_value

        return nearby_points, min_distance

    @staticmethod
    def tangent_at_point(t, control_points):
        """
        Unit tangent at parameter ``t`` using the analytic derivative
        :func:`bezier_curve_derivative_for_tan`.

        Parameters
        ----------
        t : float
            Curve parameter in ``[0, 1]``.
        control_points : sequence of (x, y)
            Control polygon of the curve.

        Returns
        -------
        numpy.ndarray, shape (2,)
            Normalised tangent vector at ``t``.
        """
        derivative = BezierCurve.bezier_curve_derivative_for_tan(t, control_points)
        return derivative / np.linalg.norm(derivative)

    # ---------------------------------------------------------------------
    # Back-traced tangent / normal (matches the v3 endpoint direction code)
    # ---------------------------------------------------------------------
    @staticmethod
    def _backtrace_curve_path(curve_points, start_index, num_steps=30,
                              direction='forward'):
        """
        Walk through ``curve_points`` starting at ``start_index`` and return
        the visited index sequence.

        The walk rules mirror :func:`breakpoint_v3_debug.get_endpoint_direction_code`:
        at every step the candidates are neighbouring sampled positions
        (``±1``, ``±2`` offsets); when there are several candidates the one
        with the smallest spatial distance from the current sample wins.

        Parameters
        ----------
        curve_points : numpy.ndarray, shape (N, 2)
            Pre-sampled curve points.
        start_index : int
            Index in ``curve_points`` where the walk starts.
        num_steps : int, optional
            Maximum number of steps to take (default 30).
        direction : {'forward', 'backward', 'auto'}
            Which neighbours are eligible at each step:
            - ``'forward'``: only ``+1`` / ``+2`` (default, towards the curve interior).
            - ``'backward'``: only ``-1`` / ``-2``.
            - ``'auto'``: any of ``-2``, ``-1``, ``+1``, ``+2``.

        Returns
        -------
        list of int
            The visited indices, starting with ``start_index``. Length is
            between 1 and ``num_steps + 1``.
        """
        if curve_points is None or len(curve_points) < 2:
            return [int(start_index)]

        if direction == 'forward':
            offsets = (1, 2)
        elif direction == 'backward':
            offsets = (-1, -2)
        else:  # 'auto'
            offsets = (-2, -1, 1, 2)

        path_indices = [int(start_index)]
        visited = {int(start_index)}
        current_idx = int(start_index)
        steps = 0

        while steps < num_steps:
            candidates = []
            for off in offsets:
                cand = current_idx + off
                if 0 <= cand < len(curve_points) and cand not in visited:
                    candidates.append(cand)

            if not candidates:
                break

            # Greedy: pick the spatially closest eligible neighbour.
            best = min(candidates,
                       key=lambda i: np.linalg.norm(curve_points[i] - curve_points[current_idx]))
            path_indices.append(best)
            visited.add(best)
            current_idx = best
            steps += 1

        return path_indices

    @staticmethod
    def tangent_vector_backtrack(t, control_points, num_steps=30, num_curve_samples=600,
                                 direction='forward'):
        """
        Unit tangent at parameter ``t`` obtained by back-tracing sampled
        curve points.

        Replaces the ``epsilon`` finite-difference used in
        :func:`tangent_vector` with a chain of neighbouring samples — the
        same approach as ``breakpoint_v3_debug.get_endpoint_direction_code``.
        The tangent direction is the average displacement of all visited
        samples relative to the start, normalised.

        Parameters
        ----------
        t : float
            Curve parameter in ``[0, 1]``.
        control_points : sequence of (x, y)
            Control polygon of the curve.
        num_steps : int, optional
            Maximum back-trace length (default 30, matching the v3
            endpoint-direction code).
        num_curve_samples : int, optional
            Number of samples used to discretise the curve (default 600).
        direction : {'forward', 'backward', 'auto'}
            Walk direction, see :func:`_backtrace_curve_path`.

        Returns
        -------
        numpy.ndarray, shape (2,) or None
            Normalised tangent vector, or ``None`` if the back-traced path
            is too short to determine a meaningful direction.
        """
        if control_points is None or len(control_points) < 2:
            return None

        # 1) Sample the curve.
        t_values = np.linspace(0, 1, num_curve_samples)
        curve_points = np.array([BezierCurve.bezier_curve(ti, control_points) for ti in t_values])

        # 2) Locate the sample closest to B(t).
        target = np.asarray(BezierCurve.bezier_curve(t, control_points), dtype=float)
        start_index = int(np.argmin(np.linalg.norm(curve_points - target, axis=1)))

        # 3) Back-trace along the curve.
        path_indices = BezierCurve._backtrace_curve_path(
            curve_points, start_index, num_steps=num_steps, direction=direction
        )

        if len(path_indices) < 2:
            return None

        # 4) Average displacement of visited samples relative to the start.
        origin = curve_points[path_indices[0]]
        avg_dx = sum(curve_points[i][0] - origin[0] for i in path_indices) / len(path_indices)
        avg_dy = sum(curve_points[i][1] - origin[1] for i in path_indices) / len(path_indices)
        length = np.sqrt(avg_dx ** 2 + avg_dy ** 2)

        if length < 1e-9:
            return None

        return np.array([avg_dx / length, avg_dy / length])

    @staticmethod
    def normal_vector_backtrack(t, control_points, num_steps=30, num_curve_samples=600,
                                direction='forward'):
        """
        Unit normal at parameter ``t`` via back-traced tangent.

        The normal is computed as the 90-degree rotation of the
        back-traced tangent from :func:`tangent_vector_backtrack`.

        Parameters
        ----------
        t : float
            Curve parameter in ``[0, 1]``.
        control_points : sequence of (x, y)
            Control polygon of the curve.
        num_steps : int, optional
            Maximum back-trace length (default 30).
        num_curve_samples : int, optional
            Number of samples used to discretise the curve (default 600).
        direction : {'forward', 'backward', 'auto'}
            Walk direction, see :func:`_backtrace_curve_path`.

        Returns
        -------
        numpy.ndarray, shape (2,) or None
            Normalised normal vector, or ``None`` if the tangent could not be
            computed.
        """
        tangent_vector = BezierCurve.tangent_vector_backtrack(
            t, control_points,
            num_steps=num_steps, num_curve_samples=num_curve_samples,
            direction=direction
        )
        if tangent_vector is None:
            return None
        normal_vector = np.array([-tangent_vector[1], tangent_vector[0]])
        return normal_vector / np.linalg.norm(normal_vector)

    @staticmethod
    def tangent_at_point_backtrack(t, control_points, num_steps=30, num_curve_samples=600,
                                   direction='forward'):
        """
        Back-traced version of :func:`tangent_at_point`.

        Parameters
        ----------
        t : float
            Curve parameter in ``[0, 1]``.
        control_points : sequence of (x, y)
            Control polygon of the curve.
        num_steps : int, optional
            Maximum back-trace length (default 30).
        num_curve_samples : int, optional
            Number of samples used to discretise the curve (default 600).
        direction : {'forward', 'backward', 'auto'}
            Walk direction, see :func:`_backtrace_curve_path`.

        Returns
        -------
        numpy.ndarray, shape (2,) or None
            Normalised tangent vector, or ``None`` if the back-traced path is
            too short.
        """
        return BezierCurve.tangent_vector_backtrack(
            t, control_points,
            num_steps=num_steps, num_curve_samples=num_curve_samples,
            direction=direction
        )