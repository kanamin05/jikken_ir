"""Reachability/progress guard for the existing DQN (NumPy only, no ROS).

This is a hybrid controller: Q-values rank actions that reduce obstacle-aware
distance to the goal. It never relaxes the learned footprint or collision mask.
"""
import heapq
import math
import numpy as np


class RouteGuard:
    def __init__(self, policy):
        self.policy = policy
        self.distances = None

    def rebuild(self):
        p = self.policy
        if p.blocked is None:
            raise ValueError('No cloud grid received')
        rows, cols = np.indices(p.shape)
        self.centers = p.origin + np.stack((cols+.5, rows+.5), axis=-1)*p.resolution
        xy = self.centers
        self.free = ((p.data['clearance'] > p.body) & ~p.blocked &
                     (xy[..., 0] >= 0) & (xy[..., 0] <= 3.2) &
                     (xy[..., 1] >= 1.2) & (xy[..., 1] <= 4.2))
        self.distances = np.full(p.shape, np.inf)
        queue = []
        # Same goal region as execution, rather than requiring an exact cell.
        goals = self.free & (np.linalg.norm(xy-p.data['goal'], axis=-1) <= .12)
        for y, x in np.argwhere(goals):
            self.distances[y, x] = 0.
            heapq.heappush(queue, (0., int(y), int(x)))
        h, w = p.shape
        while queue:
            distance, y, x = heapq.heappop(queue)
            if distance != self.distances[y, x]:
                continue
            for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0),
                           (1, 1), (1, -1), (-1, 1), (-1, -1)):
                ny, nx = y+dy, x+dx
                if not (0 <= ny < h and 0 <= nx < w and self.free[ny, nx]):
                    continue
                if dx and dy and not (self.free[y, nx] and self.free[ny, x]):
                    continue  # No diagonal corner cutting.
                candidate = distance+p.resolution*math.hypot(dx, dy)
                if candidate < self.distances[ny, nx]:
                    self.distances[ny, nx] = candidate
                    heapq.heappush(queue, (candidate, ny, nx))

    def distance(self, position):
        p = self.policy
        if not p.known_clear(position, position):
            return float('inf')
        if np.linalg.norm(position-p.data['goal']) <= .12:
            return 0.
        x, y = np.floor((np.asarray(position)-p.origin)/p.resolution).astype(int)
        best = float('inf')
        # Connect the measured position to nearby graph cells with collision checks.
        for ny in range(max(0, y-1), min(p.shape[0], y+2)):
            for nx in range(max(0, x-1), min(p.shape[1], x+2)):
                if np.isfinite(self.distances[ny, nx]):
                    center = self.centers[ny, nx]
                    if p.known_clear(position, center):
                        best = min(best, self.distances[ny, nx]+np.linalg.norm(center-position))
        return float(best)

    def assess(self, position, values):
        self.rebuild()
        p = self.policy
        current = self.distance(position)
        report = {'position': np.asarray(position).tolist(),
                  'goal': p.data['goal'].tolist(), 'body_radius': p.body,
                  'route_distance': current if np.isfinite(current) else None,
                  'raw_action': int(np.argmax(values)), 'actions': []}
        for i, direction in enumerate(p.data['actions']):
            target = position+direction*float(p.data['step_size'])
            reasons = p.segment_rejections(position, target)
            distance = self.distance(target) if not reasons else float('inf')
            progress = bool(np.isfinite(current) and np.isfinite(distance) and
                            distance < current-.02)
            report['actions'].append(dict(action=i, q=float(values[i]),
                target=target.tolist(), rejected_by=reasons,
                route_distance=distance if np.isfinite(distance) else None,
                progress=progress))
        if not p.known_clear(position, position):
            report['status'] = 'start_blocked'
        elif not np.isfinite(current):
            report['status'] = 'no_route'
        else:
            report['status'] = 'reachable'
        allowed = [i for i in np.argsort(-values) if report['actions'][i]['progress']]
        report['selected_action'] = int(allowed[0]) if allowed else None
        if report['status'] == 'reachable' and not allowed:
            report['status'] = 'no_progress_action'
        return report
