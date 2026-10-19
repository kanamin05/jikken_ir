"""Run the preserved point-cloud mapper and DQN between Task1 and Task2b."""
import json
import os
import signal
import subprocess
import sys
import tempfile
import time


def stop_process(process):
    if process.poll() is not None:
        return
    process.send_signal(signal.SIGINT)
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.terminate()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def run_cloud_task(controller):
    import rospy
    from std_msgs.msg import Bool

    dry_run = bool(rospy.get_param("~task2a_dqn_dry_run", True))
    weights = os.path.expanduser(rospy.get_param(
        "~task2a_dqn_weights", controller.get_path(["config", "task2a_policy_numpy.npz"])))
    if not os.path.isfile(weights):
        raise ValueError("Task2a point-cloud DQN weights not found: " + weights)
    if not dry_run:
        controller.change_pose("look_at_near_floor")
        if not controller.goto_name("standby_2a"):
            return False
    if rospy.is_shutdown():
        return False

    log_root = os.path.expanduser(os.environ.get("ROS_LOG_DIR", "~/.ros/log"))
    os.makedirs(log_root, exist_ok=True)
    diagnostics = tempfile.mkdtemp(prefix="task2a-cloud-", dir=log_root)
    rospy.loginfo("Task2a: label-free point-cloud DQN, dqn-first + local-finish, dry_run=%s", dry_run)
    rospy.loginfo("Task2a point-cloud diagnostics: %s", diagnostics)
    environment = dict(os.environ)
    environment.setdefault("OPENBLAS_NUM_THREADS", "1")
    environment.setdefault("OMP_NUM_THREADS", "1")
    processes = []
    try:
        mapper = subprocess.Popen([
            sys.executable, "-m", "wrs_algorithm.task2a_cloud.pointcloud_obstacles",
            "_self_filter_radius:=0.23", "__log:=" + os.path.join(diagnostics, "pointcloud.log")
        ], env=environment)
        processes.append(mapper)
        deadline = time.monotonic() + 30
        while not rospy.is_shutdown():
            if mapper.poll() is not None:
                rospy.logerr("Task2a point-cloud mapper exited before becoming ready")
                return False
            if time.monotonic() >= deadline:
                rospy.logerr("Task2a timed out waiting for a valid point-cloud map")
                return False
            try:
                if rospy.wait_for_message("/task2a_cloud/valid", Bool, timeout=1).data:
                    break
            except rospy.ROSException:
                pass
        if rospy.is_shutdown():
            return False

        command = [sys.executable, "-m", "wrs_algorithm.task2a_cloud.task2a_precise_run",
                   "--dqn-first", "--local-finish", "--weights", weights,
                   "--steps", str(rospy.get_param("~task2a_cloud_steps", 80)),
                   "--timeout", str(rospy.get_param("~task2a_cloud_step_timeout", 25.0)),
                   "--diagnostics-dir", diagnostics,
                   "__log:=" + os.path.join(diagnostics, "navigation.log")]
        if not dry_run:
            command.append("--execute")
        navigation = subprocess.Popen(command, env=environment)
        processes.append(navigation)
        while not rospy.is_shutdown():
            result = navigation.poll()
            if result is not None:
                summary_path = os.path.join(diagnostics, "summary.json")
                summary = {}
                if os.path.isfile(summary_path):
                    with open(summary_path) as source:
                        summary = json.load(source)
                succeeded = summary.get("preview" if dry_run else "internal_goal_reached", False)
                if result == 0 and succeeded:
                    return True
                rospy.logerr("Task2a point-cloud DQN failed: %s (exit=%s); diagnostics=%s",
                             summary.get("error", "goal not confirmed; see navigation log"), result, diagnostics)
                return False
            if mapper.poll() is not None:
                rospy.logerr("Task2a point-cloud mapper exited during navigation")
                return False
            time.sleep(0.1)
        return False
    finally:
        # Stop both children before allowing fallback or Task2b to send commands.
        for process in reversed(processes):
            stop_process(process)
