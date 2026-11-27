"""Footprint-aware A* on the wall boxes used by the MuJoCo maze."""

import heapq

import numpy as np


class MazeGrid:
    """Bounded, four-connected grid; coordinates and wall sizes are in metres.
    """

    def __init__(self, maze, *, resolution, robot_radius, clearance):
        values = np.asarray([resolution, robot_radius, clearance], dtype=float)
        if (not np.isfinite(values).all() or resolution <= 0
                or robot_radius <= 0 or clearance < 0):
            raise ValueError('Grid resolution/radius must be positive and clearance nonnegative')
        self.resolution = float(resolution)
        self.radius = float(robot_radius + clearance)
        self.walls = []
        corners = []
        for wall in maze['walls']:
            center = np.asarray(wall['center_xy'], dtype=float)
            half = .5 * np.asarray(wall['size_xy'], dtype=float)
            yaw = float(wall.get('yaw', 0.))
            if (center.shape != (2,) or half.shape != (2,)
                    or not np.isfinite(np.r_[center, half, yaw]).all() or np.any(half <= 0)):
                raise ValueError('Maze walls need finite center_xy/yaw and positive size_xy')
            rotation = np.array([[np.cos(yaw), -np.sin(yaw)],
                                 [np.sin(yaw), np.cos(yaw)]])
            self.walls.append((center, half, rotation))
            corners.extend(center + (np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]]) * half)
                           @ rotation.T)
        if not self.walls:
            raise ValueError('Maze must contain walls')
        corners = np.asarray(corners)
        self.bounds = np.stack((corners.min(axis=0), corners.max(axis=0)))
        self.origin = self.bounds[0]
        shape = np.floor((self.bounds[1] - self.origin) / self.resolution + 1e-9).astype(int) + 1
        if np.prod(shape, dtype=float) > 2_000_000:
            raise ValueError('Maze grid exceeds two million cells; increase resolution')
        coordinates = np.stack(np.meshgrid(*(self.origin[i] + self.resolution * np.arange(shape[i])
                                             for i in range(2)), indexing='ij'), axis=-1)
        self.distances = self.wall_distance(coordinates)
        self.free = self.distances > self.radius + self.resolution / np.sqrt(2.)

    def wall_distance(self, points):
        """Signed Euclidean distance to the nearest (possibly rotated) wall."""
        points = np.asarray(points, dtype=float)
        distance = np.full(points.shape[:-1], np.inf)
        for center, half, rotation in self.walls:
            outside = np.abs((points - center) @ rotation) - half
            signed = (np.linalg.norm(np.maximum(outside, 0.), axis=-1)
                      + np.minimum(np.max(outside, axis=-1), 0.))
            distance = np.minimum(distance, signed)
        return distance

    def segment_is_free(self, first, second):
        """Conservative continuous segment check, also used for path simplification."""
        first, second = np.asarray(first), np.asarray(second)
        count = max(1, int(np.ceil(2. * np.linalg.norm(second - first) / self.resolution)))
        points = np.linspace(first, second, count + 1)
        return bool(np.all(points >= self.bounds[0]) and np.all(points <= self.bounds[1])
                    and np.all(self.wall_distance(points) > self.radius + self.resolution / 4.))

    def plan(self, start, goal):
        """Return a collision-checked world XY polyline, or raise if no path fits."""
        endpoints = np.asarray([start, goal], dtype=float)
        if endpoints.shape != (2, 2) or not np.isfinite(endpoints).all():
            raise ValueError('A* start and goal must be finite [x, y]')
        indices = np.rint((endpoints - self.origin) / self.resolution).astype(int)
        for point, index in zip(endpoints, indices):
            if (np.any(point < self.bounds[0]) or np.any(point > self.bounds[1])
                    or np.any(index < 0) or np.any(index >= self.free.shape)
                    or not self.free[tuple(index)]
                    or not self.segment_is_free(point, self.origin + index * self.resolution)):
                raise ValueError('A* start or goal is blocked for the configured footprint/clearance')
        start_index, goal_index = map(tuple, indices)
        costs = {start_index: 0.}
        parents = {}

        def heuristic(index):
            return abs(index[0] - goal_index[0]) + abs(index[1] - goal_index[1])

        queue = [(heuristic(start_index), 0., start_index)]
        while queue:
            _, cost, current = heapq.heappop(queue)
            if cost != costs[current]:
                continue
            if current == goal_index:
                break
            for dx, dy in ((1, 0), (0, 1), (-1, 0), (0, -1)):
                neighbor = (current[0] + dx, current[1] + dy)
                if (not 0 <= neighbor[0] < self.free.shape[0]
                        or not 0 <= neighbor[1] < self.free.shape[1] or not self.free[neighbor]):
                    continue
                # Prefer the middle of passages; the Manhattan heuristic remains admissible.
                step = 1. + (self.radius / self.distances[neighbor]) ** 2
                candidate = cost + step
                if candidate < costs.get(neighbor, np.inf):
                    costs[neighbor] = candidate
                    parents[neighbor] = current
                    heapq.heappush(queue, (candidate + heuristic(neighbor), candidate, neighbor))
        else:
            raise ValueError('No maze path fits the configured robot footprint/clearance')
        route = [goal_index]
        while route[-1] != start_index:
            route.append(parents[route[-1]])
        path = self.origin + self.resolution * np.asarray(route[::-1])
        path = np.vstack((endpoints[0], path, endpoints[1]))
        # First remove collinear grid samples, then replace staircases with
        # collision-checked orthogonal legs. This leaves room to stop and pivot
        # at corners instead of taking diagonal shortcuts past wall ends.
        path = path[np.r_[True, np.linalg.norm(np.diff(path, axis=0), axis=1) > 1e-9]]
        if len(path) > 2:
            differences = np.diff(path, axis=0)
            directions = differences / np.linalg.norm(differences, axis=1)[:, None]
            keep = np.r_[True, np.any(np.abs(np.diff(directions, axis=0)) > 1e-7, axis=1), True]
            path = path[keep]
        simplified, index = [path[0]], 0
        while index < len(path) - 1:
            found = False
            for last in range(len(path) - 1, index, -1):
                for elbow in (np.array([path[last, 0], path[index, 1]]),
                              np.array([path[index, 0], path[last, 1]])):
                    if (self.segment_is_free(path[index], elbow)
                            and self.segment_is_free(elbow, path[last])):
                        simplified.extend((elbow, path[last]))
                        index, found = last, True
                        break
                if found:
                    break
            if not found:
                # Off-grid endpoints can require a short non-axis-aligned connector.
                index += 1
                simplified.append(path[index])
        path = np.asarray(simplified)
        path = path[np.r_[True, np.linalg.norm(np.diff(path, axis=0), axis=1) > 1e-9]]
        if len(path) > 2:
            directions = np.diff(path, axis=0)
            directions /= np.linalg.norm(directions, axis=1)[:, None]
            path = path[np.r_[True, np.any(np.abs(np.diff(directions, axis=0)) > 1e-7, axis=1), True]]
        if any(not self.segment_is_free(a, b) for a, b in zip(path[:-1], path[1:])):
            raise ValueError('Planned maze segment has insufficient clearance')
        return path
