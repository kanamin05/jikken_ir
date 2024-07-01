#!/bin/bash -eu

unset PYTHONPATH
source /opt/ros/noetic/setup.bash
source /workspace/py2_ws/devel/setup.bash
roslaunch wrs_algorithm start_task.launch &

wait
