#!/bin/bash -eu

unset PYTHONPATH
source /opt/ros/noetic/setup.bash
source /workspace/devel/setup.bash
roslaunch hsr_perception pcd_to_tf.launch &
roslaunch wrs_detector frcnn_finetuned.launch &
roslaunch wrs_algorithm start_task.launch &

wait
