"""Exercise task orchestration without importing ROS or sending robot commands."""
import ast
import math
from pathlib import Path
import types
import unittest
from unittest.mock import Mock, patch


SOURCE = Path(__file__).resolve().parents[1] / "scripts/wrs_algorithm/algorithm/wrs_main_node.py"


class Stamp(float):
    def __sub__(self, other):
        return types.SimpleNamespace(to_sec=lambda: float(self) - float(other))


class TaskSequenceTest(unittest.TestCase):
    def setUp(self):
        tree = ast.parse(SOURCE.read_text())
        controller = next(node for node in tree.body if isinstance(node, ast.ClassDef))
        keep = {"run", "execute_task1", "execute_task2a", "execute_avoid_blocks",
                "task2a_error_fallback", "move_forward_backward", "task2a_dqn_run"}
        controller.body = [node for node in controller.body
                           if isinstance(node, ast.FunctionDef) and node.name in keep]
        module = ast.parse("")
        module.body = [controller]
        self.params = {"~task2a_dqn_enabled": True, "~task2a_dqn_dry_run": False}
        self.ros = Mock()
        self.ros.get_param.side_effect = lambda name, default=None: self.params.get(name, default)
        self.ros.is_shutdown.return_value = False
        self.ros.Time.now.return_value = Stamp(0)
        self.body = Mock()
        self.gripper = Mock()
        self.base = Mock()
        namespace = {"rospy": self.ros, "whole_body": self.body, "gripper": self.gripper,
                     "omni_base": self.base, "math": math}
        exec(compile(module, str(SOURCE), "exec"), namespace)
        self.ctrl = namespace[controller.name]()
        self.ctrl.change_pose = Mock()
        self.ctrl.execute_task1 = Mock(return_value=True)
        self.ctrl.task2a_dqn_run = Mock(return_value=True)
        self.ctrl.execute_task2b = Mock()
        self.ctrl.task2a_error_fallback = Mock(return_value=False)
        self.ctrl.move_forward_backward = Mock()
        self.ctrl.coordinates = {"positions": {"go_throw_2a": [2.0, 3.5, 90]}}

    @staticmethod
    def pose(x, y, yaw=90):
        return types.SimpleNamespace(
            translation=types.SimpleNamespace(x=x, y=y),
            rotation=types.SimpleNamespace(x=0, y=0, z=math.sin(math.radians(yaw) / 2),
                                           w=math.cos(math.radians(yaw) / 2)))

    def test_full_sequence_waits_for_dqn_and_stows_arm_before_task2b(self):
        events = []
        self.ctrl.execute_task1.side_effect = lambda: events.append("task1") or True
        self.ctrl.task2a_dqn_run.side_effect = lambda: events.append("task2a") or True
        self.body.move_to_go.side_effect = lambda: events.append("stow")
        self.ctrl.execute_task2b.side_effect = lambda: events.append("task2b")
        self.ctrl.run()
        self.assertEqual(events, ["task1", "task2a", "stow", "task2b"])
        self.ctrl.task2a_error_fallback.assert_not_called()

    def test_task1_failure_stops_sequence(self):
        self.ctrl.execute_task1.return_value = False
        self.assertFalse(self.ctrl.run())
        self.ctrl.task2a_dqn_run.assert_not_called()
        self.ctrl.execute_task2b.assert_not_called()

    def test_failed_fallback_stops_before_task2b(self):
        self.ctrl.task2a_dqn_run.return_value = False
        self.assertFalse(self.ctrl.run())
        self.ctrl.task2a_error_fallback.assert_called_once_with()
        self.body.move_to_go.assert_not_called()
        self.ctrl.execute_task2b.assert_not_called()

    def test_dqn_error_moves_forward_then_goal_then_task2b_without_detection(self):
        del self.ctrl.task2a_error_fallback
        self.ctrl.task2a_dqn_run.return_value = False
        self.ctrl.task2a_dqn_detect = Mock(side_effect=AssertionError("Unexpected point cloud detection"))
        self.ctrl.goto_name = Mock(side_effect=AssertionError("Unexpected move_base navigation"))
        self.ctrl.get_relative_coordinate = Mock()
        events = []
        self.ctrl.move_forward_backward.side_effect = lambda **kw: events.append("forward")
        self.ctrl.get_relative_coordinate.side_effect = lambda *a, **kw: (
            events.append("goal") or self.pose(2, 3.5))
        self.ctrl.execute_task2b.side_effect = lambda: events.append("task2b")
        self.ctrl.run()
        self.assertEqual(events, ["forward", "goal", "task2b"])
        self.ctrl.move_forward_backward.assert_called_once_with(duration=5.0, speed=0.1)
        self.base.NAVCLIENT.cancel_goal.assert_called_once_with()
        self.base.NAVCLIENT.send_goal.assert_not_called()
        self.ctrl.task2a_dqn_run.assert_called_once_with()
        self.ctrl.task2a_dqn_detect.assert_not_called()
        self.ctrl.goto_name.assert_not_called()

    def test_unhandled_task2a_exception_also_enters_fallback(self):
        self.ctrl.task2a_dqn_run.side_effect = ValueError("bad map")
        self.ctrl.task2a_error_fallback.return_value = True
        self.ctrl.run()
        self.ctrl.task2a_error_fallback.assert_called_once_with()
        self.ctrl.execute_task2b.assert_called_once_with()

    def test_fallback_converts_map_direction_to_robot_velocity_and_stops(self):
        del self.ctrl.task2a_error_fallback
        self.ctrl.get_relative_coordinate = Mock(side_effect=[self.pose(2, 3.0), self.pose(2, 3.5)])
        self.assertTrue(self.ctrl.task2a_error_fallback())
        movement = self.base.cmd_vel.call_args_list[1][0]
        self.assertAlmostEqual(movement[0], 0.1)
        self.assertAlmostEqual(movement[1], 0.0)
        self.assertAlmostEqual(movement[2], 0.0)
        self.assertEqual(self.base.cmd_vel.call_args[0], (0.0, 0.0, 0.0))
        self.body.move_to_go.assert_called_once_with()

    def test_fallback_with_missing_tf_stops_without_task2b(self):
        del self.ctrl.task2a_error_fallback
        self.ctrl.task2a_dqn_run.return_value = False
        self.ctrl.get_relative_coordinate = Mock(return_value=None)
        self.assertFalse(self.ctrl.run())
        self.ctrl.execute_task2b.assert_not_called()
        self.assertEqual(self.base.cmd_vel.call_args[0], (0.0, 0.0, 0.0))

    def test_fallback_converges_from_different_heading_with_velocity_feedback(self):
        del self.ctrl.task2a_error_fallback
        state = {"x": 0.0, "y": 0.0, "yaw": -45.0, "time": 0.0}
        self.ros.Time.now.side_effect = lambda: Stamp(state["time"])
        self.ctrl.get_relative_coordinate = Mock(side_effect=lambda *a, **kw:
                                                self.pose(state["x"], state["y"], state["yaw"]))

        def step():
            forward, left, rotation = self.base.cmd_vel.call_args[0]
            self.assertLessEqual(math.hypot(forward, left), 0.1 + 1e-9)
            yaw = math.radians(state["yaw"])
            state["x"] += (math.cos(yaw) * forward - math.sin(yaw) * left) * 0.1
            state["y"] += (math.sin(yaw) * forward + math.cos(yaw) * left) * 0.1
            state["yaw"] += rotation * 0.1
            state["time"] += 0.1

        self.ros.Rate.return_value.sleep.side_effect = step
        self.assertTrue(self.ctrl.task2a_error_fallback())
        self.assertLess(math.hypot(state["x"] - 2.0, state["y"] - 3.5), 0.05)
        self.assertLess(abs(state["yaw"] - 90), 5.0)

    def test_fallback_timeout_stops_without_querying_point_cloud(self):
        del self.ctrl.task2a_error_fallback
        self.ros.Time.now.side_effect = [Stamp(0), Stamp(91)]
        self.ctrl.get_relative_coordinate = Mock()
        self.assertFalse(self.ctrl.task2a_error_fallback())
        self.ctrl.get_relative_coordinate.assert_not_called()
        self.assertEqual(self.base.cmd_vel.call_args[0], (0.0, 0.0, 0.0))

    def test_fallback_rejects_invalid_motion_settings(self):
        del self.ctrl.task2a_error_fallback
        for value in [0, -1, float("nan")]:
            self.params["~task2a_fallback_speed"] = value
            self.assertFalse(self.ctrl.task2a_error_fallback())
        self.ctrl.move_forward_backward.assert_not_called()

    def test_shutdown_after_dqn_failure_does_not_trigger_fallback(self):
        self.ctrl.task2a_dqn_run.side_effect = lambda: (
            setattr(self.ros.is_shutdown, "return_value", True) or False)
        self.assertFalse(self.ctrl.run())
        self.ctrl.task2a_error_fallback.assert_not_called()
        self.ctrl.execute_task2b.assert_not_called()

    def test_forward_motion_sends_positive_speed_for_five_seconds_then_stops(self):
        del self.ctrl.move_forward_backward
        self.ros.Time.now.side_effect = [Stamp(0), Stamp(0), Stamp(4.9), Stamp(5)]
        self.ctrl.move_forward_backward(duration=5.0, speed=0.1)
        self.assertEqual([call[0] for call in self.base.cmd_vel.call_args_list],
                         [(0.1, 0.0, 0.0), (0.1, 0.0, 0.0), (0.0, 0.0, 0.0)])

    def test_shutdown_after_task1_stops_sequence(self):
        self.ros.is_shutdown.side_effect = [False, True]
        self.assertFalse(self.ctrl.run())
        self.ctrl.task2a_dqn_run.assert_not_called()
        self.ctrl.execute_task2b.assert_not_called()

    def test_full_sequence_rejects_dry_run_before_task1_moves(self):
        self.params["~task2a_dqn_dry_run"] = True
        self.assertFalse(self.ctrl.run())
        self.ctrl.change_pose.assert_not_called()
        self.ctrl.execute_task1.assert_not_called()

    def test_dqn_only_preserves_standalone_preview(self):
        self.params.update({"~task2a_dqn_only": True, "~task2a_dqn_dry_run": True})
        self.assertTrue(self.ctrl.run())
        self.ctrl.task2a_dqn_run.assert_called_once_with()
        self.ctrl.execute_task1.assert_not_called()
        self.ctrl.execute_task2b.assert_not_called()

    def test_main_dqn_entry_calls_cloud_runner_and_stops_before_handoff(self):
        del self.ctrl.task2a_dqn_run
        with patch("wrs_algorithm.task2a_cloud.runner.run_cloud_task", return_value=True) as cloud:
            self.assertTrue(self.ctrl.task2a_dqn_run())
        cloud.assert_called_once_with(self.ctrl)
        self.gripper.command.assert_called_once_with(0)
        self.base.cmd_vel.assert_called_once_with(0.0, 0.0, 0.0)

    def test_cloud_preview_entry_does_not_send_arm_or_base_commands(self):
        del self.ctrl.task2a_dqn_run
        self.params["~task2a_dqn_dry_run"] = True
        with patch("wrs_algorithm.task2a_cloud.runner.run_cloud_task", return_value=True):
            self.assertTrue(self.ctrl.task2a_dqn_run())
        self.gripper.command.assert_not_called()
        self.base.cmd_vel.assert_not_called()

    def test_legacy_task2a_moves_through_exit(self):
        self.params["~task2a_dqn_enabled"] = False
        self.ctrl.goto_name = Mock(return_value=True)
        self.ctrl.execute_avoid_blocks = Mock(return_value=True)
        self.ctrl.run()
        self.assertEqual([call[0][0] for call in self.ctrl.goto_name.call_args_list],
                         ["standby_2a", "go_throw_2a"])
        self.ctrl.task2a_dqn_run.assert_not_called()
        self.ctrl.execute_task2b.assert_called_once_with()

    def test_legacy_navigation_failure_stops_sequence(self):
        self.params["~task2a_dqn_enabled"] = False
        self.ctrl.goto_name = Mock(return_value=True)
        self.ctrl.execute_avoid_blocks = Mock(return_value=False)
        self.assertFalse(self.ctrl.run())
        self.ctrl.execute_task2b.assert_not_called()

    def test_avoid_blocks_stops_on_navigation_failure(self):
        self.ctrl.get_latest_detection = Mock(return_value=types.SimpleNamespace(bboxes=[]))
        self.ctrl.select_next_waypoint = Mock(return_value=[1, 2, 90])
        self.ctrl.goto_pos = Mock(return_value=False)
        self.assertFalse(self.ctrl.execute_avoid_blocks())
        self.ctrl.goto_pos.assert_called_once_with([1, 2, 90])

    def test_task1_returns_success_after_drawers_and_search(self):
        del self.ctrl.execute_task1
        self.ctrl.DETECT_CNT = 1
        self.ctrl.open_drawer = Mock(return_value=True)
        self.ctrl.goto_name = Mock(return_value=True)
        self.ctrl.get_latest_detection = Mock(return_value=types.SimpleNamespace(bboxes=[]))
        self.ctrl.get_most_graspable_obj = Mock(return_value=None)
        self.assertTrue(self.ctrl.execute_task1())
        self.assertEqual(self.ctrl.open_drawer.call_count, 3)

    def test_drawer_failure_does_not_start_task2a(self):
        del self.ctrl.execute_task1
        self.ctrl.open_drawer = Mock(return_value=False)
        self.assertFalse(self.ctrl.run())
        self.ctrl.task2a_dqn_run.assert_not_called()
        self.ctrl.execute_task2b.assert_not_called()


if __name__ == "__main__":
    unittest.main()
