#!/bin/bash -eu

unset PYTHONPATH
source /opt/ros/noetic/setup.bash
source /workspace/devel/setup.bash
exec roslaunch wrs_algorithm all_tasks.launch "$@"
