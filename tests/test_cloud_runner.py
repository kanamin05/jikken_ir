"""Test point-cloud child process ownership and success/failure handoff."""
import hashlib
import json
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

from wrs_algorithm.task2a_cloud import runner


ROOT = Path(__file__).resolve().parents[1]


class CloudRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.params = {"~task2a_dqn_dry_run": False}
        self.ros = Mock()
        self.ros.get_param.side_effect = lambda key, default=None: self.params.get(key, default)
        self.ros.is_shutdown.return_value = False
        self.ros.wait_for_message.return_value = types.SimpleNamespace(data=True)
        self.ros.ROSException = RuntimeError
        self.controller = Mock()
        self.controller.get_path.side_effect = lambda parts: str(ROOT.joinpath(*parts))
        self.controller.goto_name.return_value = True
        self.mapper = Mock()
        self.mapper.poll.return_value = None
        self.navigation = Mock()
        self.navigation.poll.return_value = 0
        self.summary = {"internal_goal_reached": True}
        self.commands = []

        def spawn(command, **kwargs):
            self.commands.append(command)
            if command[2].endswith("pointcloud_obstacles"):
                return self.mapper
            directory = command[command.index("--diagnostics-dir") + 1]
            Path(directory, "summary.json").write_text(json.dumps(self.summary))
            return self.navigation

        self.spawn = Mock(side_effect=spawn)
        self.addCleanup(patch.stopall)
        patch.dict(sys.modules, {"rospy": self.ros,
                                 "std_msgs.msg": types.SimpleNamespace(Bool=object)}).start()
        patch.dict(runner.os.environ, {"ROS_LOG_DIR": self.temp.name}).start()
        patch.object(runner.subprocess, "Popen", self.spawn).start()

    def test_executes_preserved_cloud_modules_and_stops_mapper_before_return(self):
        self.assertTrue(runner.run_cloud_task(self.controller))
        self.assertEqual(self.commands[0][2], "wrs_algorithm.task2a_cloud.pointcloud_obstacles")
        self.assertIn("_self_filter_radius:=0.23", self.commands[0])
        self.assertEqual(self.commands[1][2], "wrs_algorithm.task2a_cloud.task2a_precise_run")
        for flag in ["--execute", "--dqn-first", "--local-finish"]:
            self.assertIn(flag, self.commands[1])
        self.controller.goto_name.assert_called_once_with("standby_2a")
        self.mapper.send_signal.assert_called_once_with(signal.SIGINT)
        self.mapper.wait.assert_called_once_with(timeout=5)

    def test_preview_never_prepares_or_moves_robot(self):
        self.params["~task2a_dqn_dry_run"] = True
        self.summary = {"preview": True}
        self.assertTrue(runner.run_cloud_task(self.controller))
        self.controller.change_pose.assert_not_called()
        self.controller.goto_name.assert_not_called()
        self.assertNotIn("--execute", self.commands[1])

    def test_zero_exit_without_goal_confirmation_is_failure(self):
        self.summary = {"internal_goal_reached": False}
        self.assertFalse(runner.run_cloud_task(self.controller))
        self.mapper.wait.assert_called_once_with(timeout=5)

    def test_navigation_failure_stops_mapper_and_returns_to_fallback(self):
        self.navigation.poll.return_value = 1
        self.summary = {"error": "No collision-checked local finish step"}
        self.assertFalse(runner.run_cloud_task(self.controller))
        self.mapper.wait.assert_called_once_with(timeout=5)
        self.ros.logerr.assert_called_once()

    def test_mapper_failure_never_starts_navigation(self):
        self.mapper.poll.return_value = 1
        self.assertFalse(runner.run_cloud_task(self.controller))
        self.assertEqual(len(self.commands), 1)

    def test_navigation_is_stopped_before_mapper_if_ros_shuts_down(self):
        self.navigation.poll.return_value = None
        events = []
        self.navigation.send_signal.side_effect = lambda sig: events.append("stop navigation")
        self.mapper.send_signal.side_effect = lambda sig: events.append("stop mapper")
        self.ros.is_shutdown.side_effect = [False, False, False, True]
        self.assertFalse(runner.run_cloud_task(self.controller))
        self.assertEqual(events, ["stop navigation", "stop mapper"])

    def test_mapper_that_ignores_sigint_is_terminated(self):
        self.mapper.wait.side_effect = [subprocess.TimeoutExpired("mapper", 5), 0]
        self.assertTrue(runner.run_cloud_task(self.controller))
        self.mapper.terminate.assert_called_once_with()

    def test_unique_diagnostics_do_not_overwrite_previous_run(self):
        self.assertTrue(runner.run_cloud_task(self.controller))
        self.assertTrue(runner.run_cloud_task(self.controller))
        directories = list(Path(self.temp.name).glob("task2a-cloud-*/summary.json"))
        self.assertEqual(len(directories), 2)

    def test_source_algorithms_and_weights_match_preserved_hashes(self):
        package = ROOT / "scripts/wrs_algorithm/task2a_cloud"
        provenance = json.loads((package / "source_manifest.json").read_text())
        manifest = provenance["sha256"]
        modules = [Path(name).stem for name in manifest if name.endswith(".py")]
        for name, expected in manifest.items():
            if name.endswith(".npz"):
                data = (ROOT / "config" / name).read_bytes()
            else:
                source = (package / name).read_text()
                for module in modules:
                    source = source.replace("from ." + module + " import ", "from " + module + " import ")
                source = source.rstrip("\n") + "\n" * provenance["source_trailing_newlines"][name]
                data = source.encode()
            self.assertEqual(hashlib.sha256(data).hexdigest(), expected, name)


if __name__ == "__main__":
    unittest.main()
