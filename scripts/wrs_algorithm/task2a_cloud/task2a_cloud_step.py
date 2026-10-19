#!/usr/bin/env python3
"""Kanami: bounded DQN actions through move_base in Gazebo, then exit.

Uses existing exported weights and live point-cloud grid; no detector labels.
Default is one action. --steps N enables a bounded observe/infer/move loop.
Run with --execute to send goals, otherwise prints one candidate only.
"""
import os
import math
import time
import argparse
import json
from collections import deque
import numpy as np
from .task2a_cloud_infer import CloudDQN
from .task2a_route_guard import RouteGuard


class LocalizationChanged(RuntimeError):
    """A local motion must stop and be replanned after a map-frame correction."""


class SegmentBlocked(RuntimeError):
    """Fresh geometry invalidated the segment; stop before any new plan."""


class StepStalled(RuntimeError):
    """Some progress was made, but the short target needs to be replanned."""


class ProgressWatchdog:
    """Detect millimetre-scale nonconvergence without declaring goal success."""
    def __init__(self, now, distance, seconds=2.):
        self.best, self.since, self.seconds = distance, now, seconds

    def stalled(self, now, distance):
        if distance < self.best-.001:
            self.best, self.since = distance, now
        return now-self.since >= self.seconds


def validate_map(policy, msg, check_static=False):
    q = msg.info.origin.orientation
    if (msg.header.frame_id.lstrip('/') != 'map' or
        (msg.info.height, msg.info.width) != policy.shape or
        abs(msg.info.resolution-policy.resolution) > 1e-6 or
        not np.allclose([msg.info.origin.position.x, msg.info.origin.position.y],
                        policy.origin, atol=1e-6) or
        abs(q.x)+abs(q.y)+abs(q.z) > 1e-6 or abs(abs(q.w)-1) > 1e-6):
        raise ValueError('Map metadata differs from training')
    raw = np.asarray(msg.data).reshape(policy.shape)
    if check_static and not np.array_equal((raw >= 0) & (raw <= 19), policy.data['free']):
        raise ValueError('Live static map differs from training')
    return raw


def check_path(policy, start, target, path, max_length):
    """Check every segment, including the actual start and final target."""
    previous = np.asarray(start)
    total = 0.
    for point in list(path)+[target]:
        point = np.asarray(point)
        if not np.isfinite(point).all():
            raise ValueError('Invalid move_base plan coordinate')
        total += np.linalg.norm(point-previous)
        if not policy.known_clear(previous, point):
            raise ValueError('move_base plan enters known blocked cells or leaves ROI')
        previous = point
    if total > max_length:
        raise ValueError('Plan detours too far for a one-step check: %.3f m' % total)


def velocity_command(position, target, map_yaw, odom_yaw, reference_odom_yaw,
                     tolerance=.03, min_speed=0., heading_gain=1.2,
                     max_speed=.05, heading_deadband=0., max_angular_speed=.15):
    """Map-frame position feedback; odom-frame heading feedback. SI units."""
    difference = np.asarray(target)-np.asarray(position)
    angle = math.atan2(math.sin(reference_odom_yaw-odom_yaw),
                       math.cos(reference_odom_yaw-odom_yaw))
    velocity = difference*.7
    norm = np.linalg.norm(velocity)
    if norm > max_speed:
        velocity *= max_speed/norm
    elif 0 < norm < min_speed:
        velocity *= min_speed/norm
    if np.linalg.norm(difference) <= tolerance or abs(angle) > .08:
        velocity[:] = 0.
    c, s = math.cos(map_yaw), math.sin(map_yaw)
    body = np.array([c*velocity[0]+s*velocity[1], -s*velocity[0]+c*velocity[1]])
    angular = 0. if abs(angle) <= heading_deadband else angle*heading_gain
    return body, float(np.clip(angular, -max_angular_speed, max_angular_speed)), angle


def execute_velocity_step(args, policy, start, target, update_cloud, pose):
    """One holonomic step with heading correction; never sends a move_base goal."""
    import rospy
    from geometry_msgs.msg import Twist
    from actionlib_msgs.msg import GoalStatusArray, GoalStatus
    active_states = (GoalStatus.PENDING, GoalStatus.ACTIVE,
                     GoalStatus.PREEMPTING, GoalStatus.RECALLING)
    status = rospy.wait_for_message('/move_base/status', GoalStatusArray, timeout=5)
    if any(item.status in active_states for item in status.status_list):
        raise RuntimeError('move_base has an active goal; velocity mode will not start')
    active = [False]
    def receive_status(msg):
        active[0] = any(item.status in active_states for item in msg.status_list)
    status_sub = rospy.Subscriber('/move_base/status', GoalStatusArray, receive_status, queue_size=1)
    publisher = rospy.Publisher('/hsrb/command_velocity', Twist, queue_size=1)
    until = time.monotonic()+3
    while publisher.get_num_connections() == 0:
        if rospy.is_shutdown() or time.monotonic() > until:
            status_sub.unregister()
            raise RuntimeError('No subscriber for command_velocity')
        time.sleep(.05)
    reference = None
    previous_linear = np.zeros(2)
    previous_angular = 0.
    previous_time = rospy.Time.now().to_sec()
    settled = None
    deadline = time.monotonic()+args.timeout
    try:
        update_cloud()
        current, map_yaw = pose()
        if 'yaw_odom_deg' not in policy.pose_diagnostic:
            raise RuntimeError('Synchronized odom heading unavailable')
        reference = getattr(args, 'velocity_reference_odom',
                            math.radians(policy.pose_diagnostic['yaw_odom_deg']))
        args.velocity_reference_odom = reference
        starting_diagnostic = dict(policy.pose_diagnostic)
        initial_distance = float(np.linalg.norm(current-target))
        watchdog = ProgressWatchdog(time.monotonic(), initial_distance)
        rospy.loginfo('VELOCITY STEP: speed <=%.3f m/s, correction <=%.3f rad/s.',
            getattr(args, 'max_speed', .05), getattr(args, 'max_angular_speed', .15))
        while not rospy.is_shutdown():
            if active[0]:
                raise RuntimeError('Another move_base goal became active')
            if time.monotonic() > deadline:
                raise RuntimeError('Velocity step timed out')
            update_cloud()
            current, map_yaw = pose()
            diagnostic = policy.pose_diagnostic
            if 'map_odom_xy' in diagnostic and 'map_odom_xy' in starting_diagnostic:
                shift = np.linalg.norm(np.asarray(diagnostic['map_odom_xy'])-
                                       starting_diagnostic['map_odom_xy'])
                turn = math.radians(diagnostic['yaw_map_odom_deg']-
                                    starting_diagnostic['yaw_map_odom_deg'])
                turn = math.atan2(math.sin(turn), math.cos(turn))
                if shift > .04 or abs(turn) > math.radians(2):
                    raise LocalizationChanged('map->odom changed during motion: translation=%.3f m rotation=%.2f deg' %
                                              (shift, math.degrees(turn)))
            if 'yaw_odom_deg' not in diagnostic:
                raise RuntimeError('Odom heading unavailable during motion')
            odom_yaw = math.radians(diagnostic['yaw_odom_deg'])
            if not policy.known_clear(current, target):
                raise SegmentBlocked('Updated cloud blocks velocity-step segment')
            if np.linalg.norm(current-start) > float(policy.data['step_size'])+.1:
                raise RuntimeError('Robot left the one-step area: start=%s current=%s diagnostic=%s' %
                                   (start, current, diagnostic))
            body, angular, error = velocity_command(current, target, map_yaw,
                odom_yaw, reference, args.target_tolerance,
                getattr(args, 'min_speed', 0.), getattr(args, 'heading_gain', 1.2),
                getattr(args, 'max_speed', .05), getattr(args, 'heading_deadband', 0.),
                getattr(args, 'max_angular_speed', .15))
            if abs(error) > math.radians(15):
                raise RuntimeError('Odom heading drift exceeded 15 degrees despite correction')
            training_error = math.atan2(math.sin(map_yaw-math.pi/2), math.cos(map_yaw-math.pi/2))
            if abs(training_error) > math.radians(20):
                raise RuntimeError('Map heading left training orientation range')
            now = rospy.Time.now().to_sec()
            dt = max(0., min(now-previous_time, .15))
            previous_time = now
            # Bound acceleration as well as speed; heading correction remains active at rest.
            delta = body-previous_linear
            length = np.linalg.norm(delta)
            if length > .1*dt and length > 0:
                delta *= .1*dt/length
            body = previous_linear+delta
            angular = float(np.clip(angular, previous_angular-.3*dt, previous_angular+.3*dt))
            previous_linear, previous_angular = body, angular
            distance = float(np.linalg.norm(current-target))
            if (getattr(args, 'replan_stalled_steps', False) and
                    watchdog.stalled(time.monotonic(), distance) and distance > args.target_tolerance):
                progress = initial_distance-distance
                if progress >= .01:
                    raise StepStalled('Partial step: progress=%.3f m, remaining=%.3f m; stop and replan' %
                                      (progress, distance))
                raise RuntimeError('Velocity step made no progress for 2 seconds: remaining=%.3f m' % distance)
            if distance <= args.target_tolerance and (
                    getattr(args, 'settle_without_rotation', False) or abs(error) <= math.radians(2)):
                if settled is None:
                    settled = time.monotonic()
                # Stop linear movement immediately once both criteria are met.
                body = np.zeros(2)
                previous_linear = body
                if getattr(args, 'settle_without_rotation', False):
                    # TASK2A tracking fix: residual angular velocity moved the
                    # base after reporting millimetre-level target completion.
                    angular = previous_angular = 0.
                if time.monotonic()-settled >= getattr(args, 'settle_seconds', .3):
                    rospy.loginfo('ONE STEP COMPLETE: final=%s displacement=%.3f m target_error=%.3f m heading_error=%.2f deg.',
                                  current, np.linalg.norm(current-start), distance, math.degrees(error))
                    return True
            else:
                settled = None
            command = Twist()
            command.linear.x, command.linear.y = float(body[0]), float(body[1])
            command.angular.z = angular
            publisher.publish(command)
            rospy.loginfo_throttle(.5, 'VELOCITY DIAG: distance=%.3f m heading_error=%.2f deg cmd=(%.3f,%.3f,%.3f)' %
                                  (distance, math.degrees(error), body[0], body[1], angular))
            time.sleep(.05)
        return False
    finally:
        for _ in range(6):
            publisher.publish(Twist())
            time.sleep(.05)
        status_sub.unregister()
        publisher.unregister()
        rospy.loginfo('Velocity step: zero velocity sent; controller finished.')


def create_session(args, policy_factory=CloudDQN):
    import rospy
    import tf2_ros
    import actionlib
    from nav_msgs.msg import OccupancyGrid
    from nav_msgs.srv import GetPlan, GetPlanRequest
    from std_msgs.msg import Bool
    from geometry_msgs.msg import PoseStamped, Twist
    from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
    from actionlib_msgs.msg import GoalStatus

    policy = policy_factory(os.path.expanduser(args.weights))
    policy.motion_commands = deque(maxlen=200)
    policy.pose_diagnostic = {}
    validate_map(policy, rospy.wait_for_message('/map', OccupancyGrid, timeout=15), True)
    for key in ('robot_radius', 'margin'):
        value = float(rospy.get_param('/task2a_pointcloud_obstacles/'+key, float(policy.data[key])))
        if abs(value-float(policy.data[key])) > 1e-6:
            raise ValueError('Mapper and model disagree on '+key)
    buffer = tf2_ros.Buffer(cache_time=rospy.Duration(30))
    policy.tf_buffer = buffer
    listener = tf2_ros.TransformListener(buffer)
    latest, validity = [None, 0.], [False, 0.]

    def receive(msg):
        latest[:] = [msg, time.monotonic()]

    def receive_valid(msg):
        validity[:] = [msg.data, time.monotonic()]

    def receive_command(msg):
        policy.motion_commands.append((time.monotonic(), float(msg.linear.x),
                                       float(msg.linear.y), float(msg.angular.z)))

    cloud_sub = rospy.Subscriber('/task2a_cloud/occupied', OccupancyGrid, receive, queue_size=1)
    valid_sub = rospy.Subscriber('/task2a_cloud/valid', Bool, receive_valid, queue_size=1)
    command_sub = rospy.Subscriber('/hsrb/command_velocity', Twist, receive_command, queue_size=50)
    last_coarse = [None]

    def update_cloud():
        msg, received = latest[:]
        valid, valid_received = validity[:]
        now = time.monotonic()
        if (msg is None or not valid or now-received > 3 or now-valid_received > 3):
            raise ValueError('Point-cloud map unavailable/invalid or updates stopped')
        age = (rospy.Time.now()-msg.header.stamp).to_sec()
        if msg.header.stamp == rospy.Time(0) or age < -.5 or age > 3:
            raise ValueError('Point-cloud map stale: age=%.2f s' % age)
        # TASK2A tracking fix: validate liveness every cycle, but only rasterize
        # a coarse message once. Precise geometry has its own independent guard.
        key = (msg.header.stamp.to_nsec(), msg.header.seq)
        if key != last_coarse[0]:
            policy.update(validate_map(policy, msg))
            last_coarse[0] = key
        return msg.header.stamp

    def pose():
        stamped = buffer.lookup_transform('map', args.base_frame,
                                          rospy.Time(0), rospy.Duration(.5))
        age = (rospy.Time.now()-stamped.header.stamp).to_sec()
        if age < -.5 or age > 3:
            raise ValueError('Robot TF stale: age=%.2f s' % age)
        p, q = stamped.transform.translation, stamped.transform.rotation
        yaw = math.atan2(2*(q.w*q.z+q.x*q.y), 1-2*(q.y*q.y+q.z*q.z))
        if not np.isfinite([p.x, p.y, yaw]).all():
            raise ValueError('Invalid robot TF')
        policy.pose_diagnostic = dict(yaw_map_deg=math.degrees(yaw),
                                      xy_map=[float(p.x), float(p.y)],
                                      tf_stamp=stamped.header.stamp.to_sec(),
                                      tf_age=age)
        try:
            odom = buffer.lookup_transform('odom', args.base_frame,
                                           stamped.header.stamp, rospy.Duration(.05)).transform
            oq = odom.rotation
            odom_yaw = math.atan2(2*(oq.w*oq.z+oq.x*oq.y), 1-2*(oq.y*oq.y+oq.z*oq.z))
            policy.pose_diagnostic['yaw_odom_deg'] = math.degrees(odom_yaw)
            op = odom.translation
            policy.pose_diagnostic['xy_odom'] = [float(op.x), float(op.y)]
            offset = math.atan2(math.sin(yaw-odom_yaw), math.cos(yaw-odom_yaw))
            policy.pose_diagnostic['yaw_map_odom_deg'] = math.degrees(offset)
            c, s = math.cos(offset), math.sin(offset)
            policy.pose_diagnostic['map_odom_xy'] = [p.x-c*op.x+s*op.y,
                                                     p.y-s*op.x-c*op.y]
        except Exception as error:
            policy.pose_diagnostic['odom_error'] = str(error)
        return np.array([p.x, p.y]), yaw

    return policy, update_cloud, pose, (listener, cloud_sub, valid_sub, command_sub)


def run_one_step(args, session=None):
    import rospy
    import actionlib
    from nav_msgs.srv import GetPlan, GetPlanRequest
    from geometry_msgs.msg import PoseStamped
    from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
    from actionlib_msgs.msg import GoalStatus

    if session is None:
        session = create_session(args)
    policy, update_cloud, pose, resources = session
    ready_until = time.monotonic()+15
    while True:
        if rospy.is_shutdown():
            return False
        try:
            stamp = update_cloud()
            if getattr(args, 'after_stamp', None) is not None and stamp <= args.after_stamp:
                raise ValueError('Waiting for a cloud captured after the previous movement')
            start, yaw = pose()
            break
        except Exception as error:
            if time.monotonic() >= ready_until:
                raise RuntimeError('Initialization timed out: '+str(error))
            rospy.logwarn_throttle(3, 'Waiting: %s' % error)
            time.sleep(.1)
    yaw_error = math.atan2(math.sin(yaw-math.pi/2), math.cos(yaw-math.pi/2))
    if abs(yaw_error) > math.radians(15):
        raise ValueError('Expected yaw near +90 degrees; got %.1f' % math.degrees(yaw))
    if args.hold_heading and args.controller == 'dwa':
        if not hasattr(args, 'locked_yaw'):
            args.locked_yaw = yaw
            rospy.loginfo('HEADING LOCK reference=%.1f deg (initial measured heading).', math.degrees(yaw))
        drift = math.atan2(math.sin(yaw-args.locked_yaw), math.cos(yaw-args.locked_yaw))
        if abs(drift) > .08:
            raise ValueError('Heading drifted from locked reference by %.1f deg' % math.degrees(drift))
    if np.linalg.norm(start-policy.data['goal']) <= .12:
        rospy.loginfo('Already near Task2a goal; no motion sent.')
        return True
    observation, _ = policy.observation(start)
    values = policy.qvalues(observation)
    report = RouteGuard(policy).assess(start, values)
    report.update(yaw_deg=math.degrees(yaw), cloud_stamp=stamp.to_sec(),
                  controller=args.controller, route_guard=args.route_guard)
    if args.diagnostics_dir:
        os.makedirs(args.diagnostics_dir, exist_ok=True)
        prefix = os.path.join(args.diagnostics_dir, 'decision_%d' % time.time_ns())
        np.savez_compressed(prefix+'.npz', raw=policy.raw, position=start,
                            observation=observation, q_values=values)
        with open(prefix+'.json', 'w') as output:
            json.dump(report, output, indent=2, allow_nan=False)
    rospy.loginfo('DQN DIAG: %s', json.dumps(report, allow_nan=False))
    if report['status'] in ('start_blocked', 'no_route'):
        raise RuntimeError('Reachability check: %s; no motion sent. Inspect DQN DIAG.' % report['status'])
    raw = int(np.argmax(values))
    chosen = None
    for candidate in np.argsort(-values):
        target = start+policy.data['actions'][candidate]*float(policy.data['step_size'])
        if policy.known_clear(start, target):
            chosen = int(candidate)
            break
    if chosen is None:
        raise ValueError('All eight actions blocked by known geometry')
    if args.route_guard:
        chosen = report['selected_action']
        if chosen is None:
            raise RuntimeError('No collision-free DQN action reduces route distance')
        if chosen != raw:
            rospy.logwarn('ROUTE GUARD intervened: DQN=%d selected=%d; hybrid control', raw, chosen)
    target = start+policy.data['actions'][chosen]*float(policy.data['step_size'])
    step_size = float(policy.data['step_size'])
    if not (0 < step_size <= .15+1e-6):
        raise ValueError('This script permits step_size at most 0.15 m')
    rospy.loginfo('ONE STEP: start=%s yaw=%.1f deg raw=%d selected=%d target=%s Q=%s',
                  start, math.degrees(yaw), raw, chosen, target, np.array2string(values, precision=2))
    if not args.execute:
        rospy.loginfo('Preview complete; no goal sent. --execute enables ONE goal.')
        return True

    client = actionlib.SimpleActionClient('/move_base', MoveBaseAction)
    if not client.wait_for_server(rospy.Duration(5)):
        raise RuntimeError('move_base unavailable')

    def pose_message(xy):
        msg = PoseStamped()
        msg.header.frame_id, msg.header.stamp = 'map', rospy.Time.now()
        msg.pose.position.x, msg.pose.position.y = float(xy[0]), float(xy[1])
        goal_yaw = args.locked_yaw if args.hold_heading else math.pi/2
        msg.pose.orientation.z = math.sin(goal_yaw/2)
        msg.pose.orientation.w = math.cos(goal_yaw/2)
        return msg

    rospy.wait_for_service('/move_base/make_plan', timeout=5)
    response = rospy.ServiceProxy('/move_base/make_plan', GetPlan)(
        GetPlanRequest(start=pose_message(start), goal=pose_message(target), tolerance=0.))
    if not response.plan.poses:
        raise ValueError('move_base returned an empty plan')
    plan_frame = response.plan.header.frame_id.lstrip('/')
    if plan_frame and plan_frame != 'map':
        raise ValueError('Unexpected plan frame: '+plan_frame)
    # Some ROS1 planners fill PoseStamped headers but leave Path.header empty.
    # Accept that case only when each pose resolves explicitly to map.
    for waypoint in response.plan.poses:
        frame = waypoint.header.frame_id.lstrip('/') or plan_frame
        if frame != 'map':
            raise ValueError('Plan waypoint frame is not confirmed as map: %r' % frame)
    path = [np.array([p.pose.position.x, p.pose.position.y]) for p in response.plan.poses]
    update_cloud()
    current, _ = pose()
    if np.linalg.norm(current-start) > .03:
        raise ValueError('Robot moved while preparing the goal; run again')
    check_path(policy, current, target, path, max_length=2*step_size+.1)
    if args.controller == 'velocity':
        return execute_velocity_step(args, policy, start, target, update_cloud, pose)
    goal = MoveBaseGoal(target_pose=pose_message(target))
    sent = False
    motion_started = time.monotonic()
    starting_diagnostic = dict(policy.pose_diagnostic)
    try:
        rospy.loginfo('Sending ONE move_base goal, maximum wait %.1f wall seconds.', args.timeout)
        client.send_goal(goal)
        sent = True
        deadline = time.monotonic()+args.timeout
        while not rospy.is_shutdown():
            update_cloud()
            current, current_yaw = pose()
            commands = [c for c in list(policy.motion_commands) if c[0] >= motion_started]
            diagnostic = dict(policy.pose_diagnostic)
            diagnostic['start'] = starting_diagnostic
            diagnostic['command_count'] = len(commands)
            diagnostic['max_abs_cmd_angular_z'] = max([abs(c[3]) for c in commands] or [0.])
            diagnostic['latest_cmd_vx_vy_w'] = list(commands[-1][1:]) if commands else None
            rospy.loginfo_throttle(.5, 'MOTION DIAG: %s' % diagnostic)
            if np.linalg.norm(current-start) > step_size+.10:
                raise RuntimeError('Robot traveled beyond the one-step area')
            yaw_error = math.atan2(math.sin(current_yaw-math.pi/2), math.cos(current_yaw-math.pi/2))
            if abs(yaw_error) > math.radians(20):
                raise RuntimeError('Robot yaw deviated: start=%.1f current=%.1f deg; xy=%s' %
                                   (math.degrees(yaw), math.degrees(current_yaw), current))
            if args.hold_heading:
                drift = math.atan2(math.sin(current_yaw-args.locked_yaw),
                                   math.cos(current_yaw-args.locked_yaw))
                if abs(drift) > .08:
                    raise RuntimeError('Heading drifted by %.1f deg while rotation was disabled' %
                                       math.degrees(drift))
            if not policy.known_clear(current, target):
                raise RuntimeError('Updated cloud blocks current position/remaining segment')
            check_path(policy, start, target, path, max_length=2*step_size+.1)
            state = client.get_state()
            if state == GoalStatus.SUCCEEDED:
                if np.linalg.norm(current-target) > args.target_tolerance+.005:
                    raise RuntimeError('move_base succeeded but target error exceeds %.3f m' %
                                       (args.target_tolerance+.005))
                rospy.loginfo('ONE STEP COMPLETE: final=%s displacement=%.3f m target_error=%.3f m.',
                              current, np.linalg.norm(current-start), np.linalg.norm(current-target))
                return True
            if state in (GoalStatus.ABORTED, GoalStatus.REJECTED, GoalStatus.PREEMPTED,
                         GoalStatus.RECALLED, GoalStatus.LOST):
                raise RuntimeError('move_base ended with state=%d: %s' % (state, client.get_goal_status_text()))
            if time.monotonic() >= deadline:
                raise RuntimeError('One-step motion timed out')
            time.sleep(.05)
        return False
    finally:
        if sent:
            commands = [c for c in list(policy.motion_commands) if c[0] >= motion_started]
            rospy.loginfo('FINAL MOTION DIAG: start=%s current=%s command_count=%d max_abs_cmd_angular_z=%.6f recent_vx_vy_w=%s',
                          starting_diagnostic, policy.pose_diagnostic, len(commands),
                          max([abs(c[3]) for c in commands] or [0.]),
                          [tuple(round(v, 5) for v in c[1:]) for c in commands[-12:]])
        if sent and client.get_state() not in (GoalStatus.SUCCEEDED, GoalStatus.ABORTED,
                GoalStatus.REJECTED, GoalStatus.PREEMPTED, GoalStatus.RECALLED):
            client.cancel_goal()
            # Wait briefly for acknowledgement; never cancel another node's goals.
            until = time.monotonic()+2
            while time.monotonic() < until and client.get_state() in (
                    GoalStatus.PENDING, GoalStatus.ACTIVE, GoalStatus.PREEMPTING, GoalStatus.RECALLING):
                time.sleep(.05)
            rospy.logwarn('One-step goal canceled; move_base state=%d', client.get_state())
            if client.get_state() in (GoalStatus.PENDING, GoalStatus.ACTIVE,
                                     GoalStatus.PREEMPTING, GoalStatus.RECALLING):
                raise RuntimeError('Cancellation not acknowledged; do not continue moving')


def run_steps(args):
    """One shared subscriber/TF session; wait for a post-movement cloud each time."""
    import rospy
    session = create_session(args)
    policy, update_cloud, pose, resources = session
    recent = []
    try:
        for index in range(args.steps):
            if rospy.is_shutdown():
                return False
            rospy.loginfo('===== DQN STEP %d/%d =====', index+1, args.steps)
            if not run_one_step(args, session):
                return False
            if not args.execute:
                return True
            current, _ = pose()
            remaining = float(np.linalg.norm(current-policy.data['goal']))
            rospy.loginfo('PROGRESS: step=%d/%d remaining_to_goal=%.3f m',
                          index+1, args.steps, remaining)
            if remaining <= .12:
                rospy.loginfo('TASK2A GOAL REACHED: within 0.12 m. Exiting.')
                return True
            recent.append(tuple(np.round(current/.05).astype(int)))
            if len(recent) >= 8 and len(set(recent[-8:])) <= 3:
                raise RuntimeError('Repeated local loop; no further goal sent')
            args.after_stamp = rospy.Time.now()
        rospy.logwarn('STEP LIMIT REACHED: %d steps complete; goal not reached. Exiting.', args.steps)
        return False
    finally:
        for subscriber in resources[1:]:
            subscriber.unregister()


def main():
    import rospy
    rospy.init_node('task2a_cloud_step')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--controller', choices=['dwa', 'velocity'], default='dwa')
    parser.add_argument('--route-guard', action='store_true',
                        help='Rank route-progress actions by DQN Q-value (hybrid controller)')
    parser.add_argument('--diagnostics-dir', default='diagnostics/runs',
                        help='Save each observed grid, input, Q-values and rejection reasons')
    parser.add_argument('--tune-dwa', action='store_true',
                        help='Temporarily permit backward/sideways motion at <=0.10 m/s; restore on exit')
    parser.add_argument('--hold-heading', action='store_true',
                        help='Disable DWA rotation and preserve initial measured heading (must be near 90 degrees)')
    parser.add_argument('--weights', default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                        'task2a_policy_numpy.npz'))
    parser.add_argument('--base-frame', default='base_footprint')
    parser.add_argument('--timeout', type=float, default=20.)
    parser.add_argument('--steps', type=int, default=1)
    parser.add_argument('--target-tolerance', type=float, default=.03)
    args = parser.parse_args(rospy.myargv()[1:])
    if not (0 < args.timeout <= 30):
        parser.error('--timeout must be >0 and <=30 seconds')
    if not 1 <= args.steps <= 50:
        parser.error('--steps must be 1..50')
    if not .02 <= args.target_tolerance <= .05:
        parser.error('--target-tolerance must be 0.02..0.05 m')
    if args.execute and args.controller == 'dwa' and not args.tune_dwa:
        parser.error('Execution now requires --tune-dwa to set and restore the target tolerance')
    if args.hold_heading and args.controller == 'dwa' and not args.tune_dwa:
        parser.error('--hold-heading requires --tune-dwa')
    if args.controller == 'velocity' and (args.tune_dwa or args.hold_heading):
        parser.error('Velocity mode uses its own heading feedback; omit --tune-dwa and --hold-heading')
    tuning = None
    original = None
    try:
        if args.execute and args.tune_dwa:
            from dynamic_reconfigure.client import Client
            planner = rospy.get_param('/move_base/base_local_planner')
            if planner != 'dwa_local_planner/DWAPlannerROS':
                raise ValueError('--tune-dwa expects DWAPlannerROS, got '+planner)
            tuning = Client('/move_base/DWAPlannerROS', timeout=5)
            settings = dict(min_vel_x=-.10, max_vel_x=.10,
                            min_vel_y=-.10, max_vel_y=.10,
                            min_vel_trans=.02, max_vel_trans=.10,
                            min_vel_theta=.05, max_vel_theta=.30,
                            vx_samples=11, vy_samples=11, vth_samples=11,
                            twirling_scale=1., forward_point_distance=0.,
                            xy_goal_tolerance=args.target_tolerance)
            if args.hold_heading:
                settings.update(min_vel_theta=0., max_vel_theta=0.,
                                vth_samples=1, yaw_goal_tolerance=.10)
            before = tuning.get_configuration(timeout=5)
            if not all(key in before for key in settings):
                raise ValueError('DWA configuration missing required settings')
            original = {key: before[key] for key in settings}
            actual = tuning.update_configuration(settings)
            if any(abs(float(actual[key])-value) > 1e-6 for key, value in settings.items()):
                raise ValueError('DWA did not accept requested settings')
            rospy.loginfo('Temporary DWA settings: backwards/lateral enabled, speed <=0.10 m/s, rotation <=%.2f rad/s.',
                          settings['max_vel_theta'])
            if args.hold_heading:
                rospy.loginfo('HEADING LOCK: angular velocity sampling fixed at zero; yaw deviations still monitored.')
            rospy.loginfo('Temporary xy_goal_tolerance=%.3f m', args.target_tolerance)
        return 0 if run_steps(args) else 1
    except Exception as error:
        rospy.logerr('ONE STEP STOPPED: %s', error)
        return 1
    finally:
        if tuning is not None and original is not None:
            try:
                tuning.update_configuration(original)
                rospy.loginfo('Original DWA settings restored.')
            except Exception as error:
                rospy.logerr('Could not restore DWA settings: %s; original=%s', error, original)


if __name__ == '__main__':
    raise SystemExit(main())
