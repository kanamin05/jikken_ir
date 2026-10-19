import sys
import types
import unittest
from unittest.mock import patch
import numpy as np
from scipy.spatial import cKDTree
from wrs_algorithm.task2a_cloud.task2a_cloud_infer import CloudDQN
from wrs_algorithm.task2a_cloud.task2a_route_guard import RouteGuard
from wrs_algorithm.task2a_cloud.task2a_precise_run import PreciseDQN, clear_observed_floor, widest_free_grid, validate_resume_scene
from wrs_algorithm.task2a_cloud.task2a_cloud_step import run_steps, velocity_command, execute_velocity_step, ProgressWatchdog
from wrs_algorithm.task2a_cloud.task2a_handoff import handoff_reason


def policy(cls=CloudDQN):
    p = cls.__new__(cls)
    p.shape, p.resolution, p.origin, p.body = (61, 65), .05, np.array([0., 1.2]), .05
    actions = np.array([(0, 1), (0, -1), (-1, 0), (1, 0),
                        (-1, 1), (1, 1), (-1, -1), (1, -1)], dtype=float)
    actions /= np.linalg.norm(actions, axis=1)[:, None]
    p.data = dict(clearance=np.ones(p.shape), goal=np.array([1.5, 3.5]),
                  actions=actions, step_size=.15)
    p.blocked = np.zeros(p.shape, dtype=bool)
    return p


class NavigationTests(unittest.TestCase):
    def test_lookahead_retreats_before_greedy_approach_gets_trapped(self):
        # Reduced geometry from the stopped run: goal lies beyond the upper
        # left obstacle. The first step must be allowed to move away from goal.
        p = policy(PreciseDQN)
        p.data['goal'] = np.array([2., 3.5])
        p.tree = cKDTree([[2.39, 2.56], [2.96, 2.50]])
        start = np.array([2.514, 2.336])
        target, info = p.choose_local_finish(start, [])
        self.assertLess(info['progress'], 0.)
        self.assertLess(target[1], start[1])
        self.assertGreater(p.tree.query(target)[0], p.tree.query(start)[0]+.02)
        self.assertTrue(p.known_clear(start, target))
        self.assertLessEqual(np.linalg.norm(target-start), .05)

    def test_timeout_log_commands_restore_previously_working_speed(self):
        # Regression: remaining 13 mm, heading error 3.2 deg previously produced
        # only ~0.008 m/s and 0.056 rad/s, with almost no measured progress.
        body, angular, _ = velocity_command([0., 0.], [0., .013], np.pi/2,
            -np.deg2rad(3.2), 0., tolerance=.005, min_speed=.02,
            heading_gain=3., max_speed=.03, heading_deadband=np.deg2rad(2),
            max_angular_speed=.15)
        self.assertAlmostEqual(np.linalg.norm(body), .02)
        self.assertAlmostEqual(angular, .15)

    def test_stall_watchdog_ignores_submillimetre_noise_and_resets_on_progress(self):
        watch = ProgressWatchdog(0., .15)
        self.assertFalse(watch.stalled(1., .013))
        self.assertFalse(watch.stalled(2.8, .0125))
        self.assertTrue(watch.stalled(3.1, .0125))
        self.assertFalse(watch.stalled(3.2, .010))
        self.assertFalse(watch.stalled(4., .0095))

    def test_local_finish_can_move_without_complete_route(self):
        p = policy(PreciseDQN)
        p.tree = cKDTree([[3., 4.]])
        p.data['clearance'][26, :] = 0.  # Disconnect the goal beyond a wall.
        start = np.array([1.5, 2.])
        with self.assertRaises(ValueError):
            p.plan(start)
        with patch.object(p, 'qvalues', side_effect=AssertionError('DQN called')):
            target, info = p.choose_local_finish(start, [])
        self.assertFalse(info['requires_complete_route'])
        self.assertLessEqual(np.linalg.norm(target-start), .05+1e-9)
        self.assertTrue(p.known_clear(start, target))

    def test_local_finish_still_refuses_contact_and_repeated_targets(self):
        p = policy(PreciseDQN); start = np.array([1.5, 2.])
        p.tree = cKDTree([start])
        with self.assertRaises(ValueError):
            p.choose_local_finish(start, [])
        p.tree = cKDTree([[3., 4.]])
        visited = [start+direction*distance for direction in p.data['actions']
                   for distance in (.05, .025, .015)]
        target, info = p.choose_local_finish(start, visited)
        self.assertTrue(all(np.linalg.norm(target-old) >= .012 for old in visited))
        self.assertTrue(p.known_clear(start, target))
        self.assertEqual(info['mode'], 'local_guarded_detour')

    def test_precise_waypoint_settles_with_zero_rotation(self):
        clock, commands = [0.], []
        handle = types.SimpleNamespace(unregister=lambda: None)
        def twist():
            return types.SimpleNamespace(linear=types.SimpleNamespace(x=0., y=0., z=0.),
                angular=types.SimpleNamespace(x=0., y=0., z=0.))
        publisher = types.SimpleNamespace(get_num_connections=lambda: 1,
            publish=lambda c: commands.append((c.linear.x, c.linear.y, c.angular.z)),
            unregister=lambda: None)
        rospy = types.SimpleNamespace(is_shutdown=lambda: False,
            wait_for_message=lambda *a, **k: types.SimpleNamespace(status_list=[]),
            Subscriber=lambda *a, **k: handle, Publisher=lambda *a, **k: publisher,
            loginfo=lambda *a: None, loginfo_throttle=lambda *a: None,
            Time=types.SimpleNamespace(now=lambda: types.SimpleNamespace(to_sec=lambda: clock[0])))
        status = types.SimpleNamespace(PENDING=0, ACTIVE=1, PREEMPTING=6, RECALLING=7)
        modules = {'rospy': rospy, 'geometry_msgs.msg': types.SimpleNamespace(Twist=twist),
            'actionlib_msgs.msg': types.SimpleNamespace(GoalStatusArray=object, GoalStatus=status)}
        args = types.SimpleNamespace(timeout=5., target_tolerance=.005,
            velocity_reference_odom=0., settle_without_rotation=True, settle_seconds=.6)
        p = types.SimpleNamespace(pose_diagnostic=dict(yaw_odom_deg=np.rad2deg(.07)),
            known_clear=lambda *a: True, data=dict(step_size=.15))
        def sleep(seconds): clock[0] += seconds
        with patch.dict(sys.modules, modules), patch('wrs_algorithm.task2a_cloud.task2a_cloud_step.time.monotonic',
                side_effect=lambda: clock[0]), patch('wrs_algorithm.task2a_cloud.task2a_cloud_step.time.sleep', side_effect=sleep):
            self.assertTrue(execute_velocity_step(args, p, np.zeros(2), np.zeros(2),
                lambda: None, lambda: (np.zeros(2), np.pi/2)))
        self.assertGreaterEqual(clock[0], .6)
        self.assertGreater(len(commands), 6)
        self.assertTrue(all(c == (0., 0., 0.) for c in commands))

    def test_resume_refuses_reset_or_changed_localization(self):
        saved = dict(xy_map=[2.5, 2.4], map_odom_xy=[.1, .1],
                     yaw_map_odom_deg=3., tf_stamp=100.)
        validate_resume_scene(saved, saved, 101.)
        for current, now in [(saved, 99.), (dict(saved, xy_map=[0., 0.]), 101.),
                             (dict(saved, map_odom_xy=[.2, .1]), 101.),
                             (dict(saved, yaw_map_odom_deg=8.), 101.)]:
            with self.assertRaises(ValueError):
                validate_resume_scene(current, saved, now)

    def test_widest_route_chooses_roomier_corridor(self):
        distances = np.zeros((7, 9))
        distances[1, 1:8] = .254
        distances[5, 1:8] = .259
        distances[1:6, 1] = distances[1:6, 7] = .28
        goals = np.zeros_like(distances, dtype=bool); goals[3, 7] = True
        free, threshold = widest_free_grid(distances, np.ones_like(goals),
                                           (3, 1), goals, .252, .262)
        self.assertGreater(threshold, .258)
        self.assertFalse(free[1, 4])
        self.assertTrue(free[5, 4])
        distances[5, 4] = distances[1, 4] = .251
        free, _ = widest_free_grid(distances, np.ones_like(goals),
                                  (3, 1), goals, .252, .262)
        self.assertIsNone(free)

    def test_quiet_heading_feedback_and_lower_speed_limits(self):
        body, angular, _ = velocity_command([0., 0.], [0., 1.], 0., .02, 0.,
            max_speed=.03, heading_deadband=np.deg2rad(2), max_angular_speed=.06)
        self.assertAlmostEqual(np.linalg.norm(body), .03)
        self.assertEqual(angular, 0.)
        _, angular, _ = velocity_command([0., 0.], [0., 1.], 0., .07, 0.,
            heading_gain=3., max_angular_speed=.06)
        self.assertAlmostEqual(angular, -.06)

    def test_geometry_cache_still_checks_freshness_and_reprojects_tf(self):
        class Stamp:
            def __init__(self, value): self.value = value
            def to_sec(self): return self.value
            def to_nsec(self): return int(self.value*1e9)
            def __sub__(self, other): return Stamp(self.value-other.value)
        p = policy(PreciseDQN)
        p.tree, p.memory = None, set()
        p.processed_stamp = p.processed_floor_stamp = p.geometry_key = None
        p.cleared_surface_voxels = 0
        p.observed = np.ones(p.shape, dtype=bool)
        tr = types.SimpleNamespace(x=0., y=0., z=0.)
        q = types.SimpleNamespace(x=0., y=0., z=0., w=1.)
        p.tf_buffer = types.SimpleNamespace(lookup_transform=lambda *a:
            types.SimpleNamespace(transform=types.SimpleNamespace(rotation=q, translation=tr)))
        msg = types.SimpleNamespace(header=types.SimpleNamespace(stamp=Stamp(10), frame_id='map'))
        now = [10.]
        rospy = types.SimpleNamespace(Time=types.SimpleNamespace(now=lambda: Stamp(now[0])),
                                      Duration=lambda seconds: seconds)
        # lookup_transform also uses rospy.Time(0).
        class Clock:
            def __new__(cls, seconds): return Stamp(seconds)
            @staticmethod
            def now(): return Stamp(now[0])
        rospy.Time = Clock
        with patch.dict(sys.modules, rospy=rospy), patch('wrs_algorithm.task2a_cloud.task2a_precise_run.cloud_xyz',
                return_value=np.array([[1.5, 2.5, .1]])):
            p.refresh(msg); tree = p.tree; before = p.surfaces.copy()
            p.refresh(msg)
            self.assertIs(tree, p.tree)
            p.update(np.zeros(p.shape, dtype=np.int8))
            p.refresh(msg)
            self.assertTrue((p.raw == 100).any())
            tr.x = .1
            p.refresh(msg)
            np.testing.assert_allclose(p.surfaces, before+[.1, 0.])
            now[0] = 14.
            with self.assertRaisesRegex(ValueError, 'stale'):
                p.refresh(msg)

    def test_guarded_finish_checks_surfaces_and_requires_route(self):
        p = policy(PreciseDQN)
        start = np.array([1.5, 2.])
        path = np.array([start, [1.5, 2.025], [1.5, 2.05]])
        p.tree = cKDTree([[3., 4.]])
        with patch.object(p, 'qvalues', side_effect=AssertionError('DQN called')):
            target, info = p.choose_guarded_route(start, path)
        self.assertEqual(info['mode'], 'guarded_route_finish')
        self.assertIsNone(info['action'])
        self.assertLessEqual(np.linalg.norm(target-start), .05+1e-9)
        p.tree = cKDTree([[1.5, 2.26]])
        with self.assertRaises(ValueError):
            p.choose_guarded_route(start, path)
        with self.assertRaises(ValueError):
            p.choose_guarded_route(start, None)

    def test_handoff_requires_new_score_room_and_position(self):
        score = dict(score_start=0., score_latest=30., room2_count=1)
        self.assertIsNotNone(handoff_reason([2.68, 2.45], score))
        self.assertIsNone(handoff_reason([2.5, 1.85], score))
        self.assertIsNone(handoff_reason([2.68, 2.45], dict(score, room2_count=0)))
        self.assertIsNone(handoff_reason([2.68, 2.45], dict(score, score_start=30.)))
        self.assertIsNone(handoff_reason([2.68, 2.45], dict(score, score_start=None)))

    def test_explicit_exit_plane_works_without_score(self):
        self.assertIsNone(handoff_reason([2.5, 2.99], {}, exit_y=3.))
        self.assertEqual(handoff_reason([2.5, 3.], {}, exit_y=3.),
                         'passed_obstacle_exit_y')
        self.assertIsNone(handoff_reason([np.nan, 3.], {}, exit_y=3.))

    def test_dqn_first_can_choose_off_route_high_q(self):
        p = policy(PreciseDQN); p.tree = cKDTree([[3., 4.]])
        start = np.array([1.5, 2.])
        path = np.array([start, [1.5, 2.05], [1.5, 2.1], [1.5, 2.15]])
        values = np.zeros(8); values[5] = 10
        target, info = p.choose_dqn_first(start, path, values, [])
        self.assertEqual(info['action'], 5)
        self.assertTrue(p.known_clear(start, target))

    def test_local_dqn_rejects_collision_and_visited_target(self):
        p = policy(PreciseDQN); start = np.array([1.5, 2.])
        p.tree = cKDTree([[1.5, 2.26]])
        values = np.zeros(8); values[0] = 100; values[1] = 90
        visited = [start + [0, -s] for s in (.025, .05, .1, .15)]
        target, info = p.choose_dqn_first(start, None, values, visited)
        self.assertNotIn(info['action'], [0, 1])
        self.assertTrue(p.known_clear(start, target))

    def test_floor_clears_ghost_but_preserves_unseen_and_current_obstacles(self):
        memory = {(500, 1000), (1000, 1000), (1500, 1000)}
        current = np.array([[3., 2.]])
        floor = np.array([[1., 2.], [3., 2.]])
        kept, removed = clear_observed_floor(memory, current, floor)
        self.assertEqual(removed, 1)
        self.assertEqual(kept, {(1000, 1000), (1500, 1000)})

    def test_route_guard_rejects_high_q_retreat(self):
        p = policy()
        values = np.zeros(8); values[1] = 100; values[0] = 2
        report = RouteGuard(p).assess(np.array([1.5, 2.]), values)
        self.assertEqual(report['raw_action'], 1)
        self.assertEqual(report['selected_action'], 0)

    def test_disconnected_goal_is_not_local_clearance(self):
        p = policy(); p.blocked[25, :] = True
        start = np.array([1.5, 2.])
        self.assertTrue(p.known_clear(start, start+[0, .15]))
        self.assertEqual(RouteGuard(p).assess(start, np.zeros(8))['status'], 'no_route')

    def test_blocked_start_reports_surface(self):
        p = policy(); p.blocked[:] = True
        report = RouteGuard(p).assess(np.array([1.5, 2.]), np.zeros(8))
        self.assertEqual(report['status'], 'start_blocked')
        self.assertTrue(all('pointcloud' in a['rejected_by'] for a in report['actions']))

    def test_precise_segment_checks_middle_not_only_endpoints(self):
        p = policy(PreciseDQN); p.tree = cKDTree([[1.5, 2.5]])
        self.assertTrue(p.known_clear([1., 2.5], [1., 2.5]))
        self.assertTrue(p.known_clear([2., 2.5], [2., 2.5]))
        self.assertFalse(p.known_clear([1., 2.5], [2., 2.5]))

    def test_precise_route_goes_around_obstacle(self):
        p = policy(PreciseDQN); p.route_resolution = .05
        p.tree = cKDTree([[1.5, 2.7]])
        start = np.array([1.5, 2.])
        path = p.plan(start)
        self.assertLessEqual(np.linalg.norm(path[-1]-p.data['goal']), .10)
        for a, b in zip(path[:-1], path[1:]):
            self.assertTrue(p.known_clear(a, b))
        target, info = p.choose(start, path, np.arange(8))
        self.assertTrue(p.known_clear(start, target))
        self.assertGreater(info['progress'], 0)

    def test_velocity_feedback_corrects_heading_at_rest(self):
        body, angular, error = velocity_command(np.zeros(2), np.zeros(2), 0., .1, 0.)
        np.testing.assert_array_equal(body, [0., 0.])
        self.assertLess(angular, 0.)

    def test_small_target_overcomes_velocity_deadband_but_stops_at_tolerance(self):
        body, _, _ = velocity_command(np.zeros(2), np.array([.01, 0]), 0, 0, 0,
                                       tolerance=.005, min_speed=.02)
        self.assertAlmostEqual(np.linalg.norm(body), .02)
        body, _, _ = velocity_command(np.zeros(2), np.array([.004, 0]), 0, 0, 0,
                                       tolerance=.005, min_speed=.02)
        np.testing.assert_array_equal(body, [0., 0.])

    def test_step_limit_is_failure(self):
        p = policy()
        rospy = types.SimpleNamespace(is_shutdown=lambda: False, loginfo=lambda *a: None,
            logwarn=lambda *a: None, Time=types.SimpleNamespace(now=lambda: 1))
        args = types.SimpleNamespace(steps=1, execute=True)
        session = (p, lambda: None, lambda: (np.array([1.5, 2.]), 0.), (None,))
        with patch.dict(sys.modules, rospy=rospy), \
             patch('wrs_algorithm.task2a_cloud.task2a_cloud_step.create_session', return_value=session), \
             patch('wrs_algorithm.task2a_cloud.task2a_cloud_step.run_one_step', return_value=True):
            self.assertFalse(run_steps(args))


if __name__ == '__main__':
    unittest.main()
