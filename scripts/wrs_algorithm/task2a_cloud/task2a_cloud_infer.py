#!/usr/bin/env python3
"""Kanami: point-cloud grid -> existing DQN, inference only, NO MOTION.

Run beside pointcloud_obstacles.py and task2a_policy_numpy.npz in ROS Python.
The learned model is unchanged; perceived shapes differ from training circles.
"""
import os
import math
import numpy as np
from .pointcloud_obstacles import dilate_disk


class CloudDQN(object):
    """Reusable numerical functions, separated from ROS transport."""
    def __init__(self, path):
        with np.load(path, allow_pickle=False) as archive:
            self.data = {k: archive[k].copy() for k in archive.files}
        d = self.data
        if int(d['version']) != 1 or d['actions'].shape != (8, 2):
            raise ValueError('Unexpected policy export format')
        self.body = float(d['robot_radius'])+float(d['margin'])
        self.resolution = float(d['resolution'])
        self.origin = d['origin']
        self.shape = d['free'].shape
        self.blocked = None
        self.observed = None

    def sample(self, grid, points, outside):
        points = np.asarray(points)
        cells = np.floor((points-self.origin)/self.resolution).astype(int)
        x, y = cells[..., 0], cells[..., 1]
        inside = (x >= 0) & (y >= 0) & (x < self.shape[1]) & (y < self.shape[0])
        result = np.full(x.shape, outside, dtype=grid.dtype)
        result[inside] = grid[y[inside], x[inside]]
        return result

    def update(self, raw):
        if raw.shape != self.shape or not np.isin(raw, [-1, 0, 100]).all():
            raise ValueError('Unexpected occupancy grid values/shape')
        self.observed = raw >= 0
        self.raw = np.array(raw, copy=True)
        # Raw cells already describe the object surface: add body radius ONCE.
        self.blocked = dilate_disk(raw == 100, self.body, self.resolution)

    def observation(self, position):
        if self.blocked is None:
            raise ValueError('No cloud grid received')
        axis = (np.arange(30)-14.5)*.10
        dx, dy = np.meshgrid(axis, axis)
        points = position+np.stack([dx, dy], axis=-1)
        static = self.sample(self.data['clearance'], points, -1.) <= self.body
        dynamic = self.sample(self.blocked, points, False)
        # Retain the TRAINING coverage-channel definition (map axes, +Y forward).
        # This is NOT a certificate that all these cells were actually observed.
        seen = ((dy > 0) & (np.hypot(dx, dy) <= 1.5) &
                (np.abs(np.arctan2(dx, dy)) <= np.deg2rad(50)))
        rows, cols = np.where(seen)
        rays = (position+(points[rows, cols]-position)[:, None, :]
                * np.linspace(0, 1, 30)[None, :, None])
        seen[rows, cols] &= ~np.any(self.sample(self.data['clearance'], rays, -1.) < 0, axis=1)
        channels = [static, dynamic, seen]
        observation = np.concatenate([c.ravel() for c in channels]+
            [np.clip((self.data['goal']-position)/3., -1, 1)]).astype(np.float32)
        if observation.shape != (2702,) or not np.isfinite(observation).all():
            raise ValueError('Invalid observation')
        return observation, channels

    def qvalues(self, observation):
        x = observation
        n = int(self.data['n_layers'])
        for i in range(n):
            x = x @ self.data['w%d' % i].T+self.data['b%d' % i]
            if i < n-1:
                x = np.maximum(x, 0)
        if x.shape != (8,) or not np.isfinite(x).all():
            raise ValueError('Invalid Q-values')
        return x

    def known_clear(self, start, target):
        """Known obstacles/static map only; unseen geometry can still collide."""
        return not self.segment_rejections(start, target)

    def segment_rejections(self, start, target):
        """Keep all rejection reasons for reproducible runtime diagnosis."""
        if self.blocked is None:
            raise ValueError('No cloud grid received')
        if not np.isfinite([start, target]).all():
            return ['nonfinite']
        points = np.linspace(start, target,
                             max(2, int(np.ceil(np.linalg.norm(target-start)/.01))+1))
        roi = ((points[:, 0] >= 0) & (points[:, 0] <= 3.2) &
               (points[:, 1] >= 1.2) & (points[:, 1] <= 4.2))
        reasons = []
        if not roi.all():
            reasons.append('roi')
        if np.any(self.sample(self.data['clearance'], points, -1.) <= self.body):
            reasons.append('static')
        if np.any(self.sample(self.blocked, points, True)):
            reasons.append('pointcloud')
        return reasons


def main():
    import rospy
    import tf2_ros
    from nav_msgs.msg import OccupancyGrid
    from std_msgs.msg import Bool, String
    from visualization_msgs.msg import Marker, MarkerArray
    from geometry_msgs.msg import Point
    import json

    rospy.init_node('task2a_cloud_infer')
    default = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'task2a_policy_numpy.npz')
    policy = CloudDQN(os.path.expanduser(rospy.get_param('~weights', default)))
    live = rospy.wait_for_message('/map', OccupancyGrid, timeout=20)

    def validate_grid(msg):
        q = msg.info.origin.orientation
        if (msg.header.frame_id.lstrip('/') != 'map' or
            (msg.info.height, msg.info.width) != policy.shape or
            abs(msg.info.resolution-policy.resolution) > 1e-6 or
            not np.allclose([msg.info.origin.position.x, msg.info.origin.position.y],
                            policy.origin, atol=1e-6) or
            abs(q.x)+abs(q.y)+abs(q.z) > 1e-6 or abs(abs(q.w)-1) > 1e-6):
            raise ValueError('Map metadata differs from training')

    validate_grid(live)
    grid = np.asarray(live.data).reshape(policy.shape)
    if not np.array_equal((grid >= 0) & (grid <= 19), policy.data['free']):
        raise ValueError('Live static map differs from training')
    mapper = '/task2a_pointcloud_obstacles/'
    for key in ('robot_radius', 'margin'):
        actual = float(rospy.get_param(mapper+key, float(policy.data[key])))
        if abs(actual-float(policy.data[key])) > 1e-6:
            raise ValueError('Mapper and policy disagree on '+key)

    buffer = tf2_ros.Buffer(cache_time=rospy.Duration(30))
    listener = tf2_ros.TransformListener(buffer)
    latest, valid = [None], [False]
    def receive(msg):
        latest[0] = msg
    def receive_valid(msg):
        valid[0] = msg.data
    rospy.Subscriber('/task2a_cloud/occupied', OccupancyGrid, receive, queue_size=1)
    rospy.Subscriber('/task2a_cloud/valid', Bool, receive_valid, queue_size=1)
    marker_pub = rospy.Publisher('/task2a_dqn/markers', MarkerArray, queue_size=1, latch=True)
    result_pub = rospy.Publisher('/task2a_dqn/inference', String, queue_size=1, latch=True)
    input_pubs = [rospy.Publisher('/task2a_dqn/input_'+name, OccupancyGrid,
                                 queue_size=1, latch=True)
                  for name in ('static', 'dynamic', 'seen')]

    def clear_markers():
        marker = Marker()
        marker.action = Marker.DELETEALL
        marker_pub.publish(MarkerArray(markers=[marker]))

    rospy.loginfo('Cloud DQN inference only. NO MOTION. Body inflation=%.3f m', policy.body)
    rospy.logwarn('Point-cloud shapes differ from training circles; inference is not a validated driving policy.')
    rate = rospy.Rate(1)
    while not rospy.is_shutdown():
        try:
            msg = latest[0]
            if msg is None or not valid[0]:
                raise ValueError('Waiting for valid point-cloud map')
            age = (rospy.Time.now()-msg.header.stamp).to_sec()
            if msg.header.stamp == rospy.Time(0) or age < -.5 or age > 3:
                raise ValueError('Cloud grid stale: age=%.2f s' % age)
            validate_grid(msg)
            transform = buffer.lookup_transform('map',
                rospy.get_param('~base_frame', 'base_footprint'), msg.header.stamp,
                rospy.Duration(1)).transform
            p, q = transform.translation, transform.rotation
            position = np.array([p.x, p.y])
            yaw = math.atan2(2*(q.w*q.z+q.x*q.y), 1-2*(q.y*q.y+q.z*q.z))
            if not np.isfinite([p.x, p.y, yaw]).all():
                raise ValueError('Invalid robot pose')
            error = math.atan2(math.sin(yaw-math.pi/2), math.cos(yaw-math.pi/2))
            if abs(error) > math.radians(15):
                raise ValueError('Expected robot yaw near +90 degrees; got %.1f' % math.degrees(yaw))
            policy.update(np.asarray(msg.data).reshape(policy.shape))
            observation, channels = policy.observation(position)
            values = policy.qvalues(observation)
            raw = int(np.argmax(values))
            targets = position+policy.data['actions']*float(policy.data['step_size'])
            allowed = [policy.known_clear(position, t) for t in targets]
            chosen = next((int(a) for a in np.argsort(-values) if allowed[a]), None)

            markers = []
            for index, action, color in [(0, raw, (1., 0., 0.)), (1, chosen, (0., 1., 0.))]:
                if action is None:
                    continue
                arrow = Marker()
                arrow.header.frame_id, arrow.header.stamp = 'map', msg.header.stamp
                arrow.ns, arrow.id, arrow.type = 'cloud_dqn', index, Marker.ARROW
                arrow.pose.orientation.w = 1.
                arrow.scale.x, arrow.scale.y, arrow.scale.z = .018, .04, .06
                arrow.color.r, arrow.color.g, arrow.color.b = color
                arrow.color.a = 1.
                arrow.lifetime = rospy.Duration(3)
                z = .16+.05*index
                arrow.points = [Point(x=float(position[0]), y=float(position[1]), z=z),
                    Point(x=float(targets[action, 0]), y=float(targets[action, 1]), z=z)]
                markers.append(arrow)
            clear_markers()
            marker_pub.publish(MarkerArray(markers=markers))
            for publisher, channel in zip(input_pubs, channels):
                local = OccupancyGrid()
                local.header.frame_id, local.header.stamp = 'map', msg.header.stamp
                local.info.resolution = .10
                local.info.width = local.info.height = 30
                local.info.origin.position.x = float(position[0]-1.5)
                local.info.origin.position.y = float(position[1]-1.5)
                local.info.origin.orientation.w = 1.
                local.data = (channel.astype(np.int8)*100).ravel().tolist()
                publisher.publish(local)
            report = dict(valid=True, no_motion=True, position=position.tolist(),
                          yaw_deg=math.degrees(yaw), raw_action=raw,
                          known_clear_action=chosen, known_clear=allowed,
                          targets=targets.tolist(), q_values=values.tolist(),
                          goal=policy.data['goal'].tolist(),
                          dynamic_cells=int(channels[1].sum()))
            result_pub.publish(String(data=json.dumps(report)))
            rospy.loginfo('NO MOTION: xy=(%.3f,%.3f) raw=%d known_clear=%s dynamic_cells=%d Q=%s',
                          position[0], position[1], raw, chosen, int(channels[1].sum()),
                          np.array2string(values, precision=2))
        except Exception as error:
            clear_markers()
            result_pub.publish(String(data=json.dumps(dict(valid=False, no_motion=True, error=str(error)))))
            rospy.logwarn_throttle(5, 'Inference waiting: %s' % error)
        rate.sleep()


if __name__ == '__main__':
    main()
