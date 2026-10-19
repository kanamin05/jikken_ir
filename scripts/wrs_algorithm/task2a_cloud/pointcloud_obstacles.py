#!/usr/bin/env python3
"""Kanami: label-free floor obstacle mapping for ROS1. NO MOTION COMMANDS.

Run in the existing ROS environment. Requires rospy/tf2_ros/numpy, not Torch,
SB3, scipy, or Open3D. Defaults target the supplied flat-floor Task2a world.
"""
import math
import numpy as np


def cloud_xyz(message):
    """Read organized/unorganized XYZ clouds, respecting row padding/endian."""
    endian = '>' if message.is_bigendian else '<'
    fields = {field.name: field for field in message.fields}
    if message.width == 0 or message.height == 0:
        return np.empty((0, 3), dtype=np.float64)
    coordinates = []
    for name in ('x', 'y', 'z'):
        field = fields.get(name)
        if field is None or field.datatype not in (7, 8) or field.count != 1:
            raise ValueError('XYZ fields must be scalar FLOAT32 or FLOAT64')
        dtype = np.dtype(endian+('f4' if field.datatype == 7 else 'f8'))
        values = np.ndarray((message.height, message.width), dtype=dtype,
                            buffer=message.data, offset=field.offset,
                            strides=(message.row_step, message.point_step))
        coordinates.append(values.ravel())
    points = np.column_stack(coordinates).astype(np.float64)
    return points[np.isfinite(points).all(axis=1)]


def xyz_cloud(header, points):
    """Pack XYZ in one array operation instead of Python loops per point."""
    from sensor_msgs.msg import PointCloud2, PointField
    packed = np.asarray(points, dtype='<f4').reshape(-1, 3)
    return PointCloud2(header=header, height=1, width=len(packed),
        fields=[PointField(name=name, offset=4*i, datatype=PointField.FLOAT32, count=1)
                for i, name in enumerate(('x', 'y', 'z'))],
        is_bigendian=False, point_step=12, row_step=12*len(packed),
        data=packed.tobytes(), is_dense=bool(np.isfinite(packed).all()))


def rotation_matrix(quaternion):
    q = np.asarray(quaternion, dtype=float)
    if not np.isfinite(q).all() or np.linalg.norm(q) < 1e-10:
        raise ValueError('Invalid TF quaternion')
    x, y, z, w = q/np.linalg.norm(q)
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
    ])


def fit_floor(points, expected_z=0., band=.08, threshold=.004,
              max_tilt_deg=12., iterations=100):
    """Constrained RANSAC: only near expected floor, upward nearly vertical normal.

    Assumes the supplied world's floor is near map z=0; does not choose an
    arbitrary largest plane (which might be a table).
    """
    candidates = points[np.abs(points[:, 2]-expected_z) <= band]
    if len(candidates) < 150:
        raise ValueError('Too few points near expected floor: %d' % len(candidates))
    rng = np.random.RandomState(42)
    if len(candidates) > 5000:
        candidates = candidates[rng.choice(len(candidates), 5000, replace=False)]
    vertical = math.cos(math.radians(max_tilt_deg))
    best = None
    best_count = 0
    for _ in range(iterations):
        a, b, c = candidates[rng.choice(len(candidates), 3, replace=False)]
        normal = np.cross(b-a, c-a)
        norm = np.linalg.norm(normal)
        if norm < 1e-10:
            continue
        normal /= norm
        if normal[2] < 0:
            normal = -normal
        if normal[2] < vertical:
            continue
        offset = -np.dot(normal, a)
        if abs(-offset/normal[2]-expected_z) > .04:
            continue
        mask = np.abs(candidates @ normal+offset) <= threshold
        count = int(mask.sum())
        if count > best_count:
            best, best_count = mask, count
    if best is None or best_count < max(150, int(.35*len(candidates))):
        raise ValueError('No reliable horizontal floor plane; check camera view/TF')
    inliers = candidates[best]
    center = inliers.mean(axis=0)
    _, _, vt = np.linalg.svd(inliers-center, full_matrices=False)
    normal = vt[-1]
    if normal[2] < 0:
        normal = -normal
    offset = -float(np.dot(normal, center))
    if normal[2] < vertical or abs(-offset/normal[2]-expected_z) > .04:
        raise ValueError('Refined floor plane violates floor constraints')
    residual = np.abs(inliers @ normal+offset)
    return normal, offset, float(np.percentile(residual, 95))


def raster_counts(points, origin, resolution, shape):
    cells = np.floor((points[:, :2]-origin)/resolution).astype(int)
    valid = ((cells[:, 0] >= 0) & (cells[:, 0] < shape[1]) &
             (cells[:, 1] >= 0) & (cells[:, 1] < shape[0]))
    selected = cells[valid]
    return np.bincount(selected[:, 1]*shape[1]+selected[:, 0],
                       minlength=shape[0]*shape[1]).reshape(shape), valid


def dilate_disk(mask, radius, resolution):
    """Disk dilation in NumPy; account for occupied-cell half-diagonal."""
    limit = radius+resolution/math.sqrt(2.)
    count = int(math.ceil(limit/resolution))
    output = np.zeros_like(mask)
    h, w = mask.shape
    for dy in range(-count, count+1):
        for dx in range(-count, count+1):
            if math.hypot(dx, dy)*resolution > limit:
                continue
            src_y0, src_y1 = max(0, -dy), min(h, h-dy)
            src_x0, src_x1 = max(0, -dx), min(w, w-dx)
            if src_y0 >= src_y1 or src_x0 >= src_x1:
                continue
            output[src_y0+dy:src_y1+dy, src_x0+dx:src_x1+dx] |= mask[src_y0:src_y1, src_x0:src_x1]
    return output


def main():
    import rospy
    import tf2_ros
    from sensor_msgs.msg import PointCloud2
    from sensor_msgs import point_cloud2
    from nav_msgs.msg import OccupancyGrid
    from std_msgs.msg import Header, Bool

    rospy.init_node('task2a_pointcloud_obstacles')
    topic = rospy.get_param('~cloud_topic',
                           '/hsrb/head_rgbd_sensor/depth_registered/rectified_points')
    base_frame = rospy.get_param('~base_frame', 'base_footprint')
    min_height = float(rospy.get_param('~min_height', .008))
    max_height = float(rospy.get_param('~max_height', .35))
    robot_radius = float(rospy.get_param('~robot_radius', .30))
    margin = float(rospy.get_param('~margin', .03))
    self_filter_radius = float(rospy.get_param('~self_filter_radius', robot_radius+margin))
    min_points = int(rospy.get_param('~min_points_per_cell', 3))
    max_range = float(rospy.get_param('~max_range', 2.5))
    floor_z = float(rospy.get_param('~floor_z', 0.))
    remember = bool(rospy.get_param('~remember_obstacles', True))
    if not (0 < min_height < max_height and robot_radius > 0 and margin >= 0 and
            min_points >= 1 and max_range > robot_radius+margin and
            0 < self_filter_radius <= robot_radius+margin):
        raise ValueError('Invalid point-cloud parameters')
    map_msg = rospy.wait_for_message('/map', OccupancyGrid, timeout=20)
    if map_msg.header.frame_id.lstrip('/') != 'map':
        raise ValueError('Expected /map in frame map')
    q = map_msg.info.origin.orientation
    if abs(q.x)+abs(q.y)+abs(q.z) > 1e-6 or abs(abs(q.w)-1) > 1e-6:
        raise ValueError('This version expects a map without rotation')
    info = map_msg.info
    shape = (info.height, info.width)
    origin = np.array([info.origin.position.x, info.origin.position.y])
    resolution = info.resolution
    obstacle_memory = np.zeros(shape, dtype=bool)
    floor_memory = np.zeros(shape, dtype=bool)
    buffer = tf2_ros.Buffer(cache_time=rospy.Duration(30))
    listener = tf2_ros.TransformListener(buffer)
    latest = [None]

    def receive(message):
        latest[0] = message

    rospy.Subscriber(topic, PointCloud2, receive, queue_size=1, buff_size=16*1024*1024)
    cloud_pub = rospy.Publisher('/task2a_cloud/obstacles', PointCloud2, queue_size=1)
    floor_pub = rospy.Publisher('/task2a_cloud/floor', PointCloud2, queue_size=1)
    raw_pub = rospy.Publisher('/task2a_cloud/occupied', OccupancyGrid, queue_size=1, latch=True)
    blocked_pub = rospy.Publisher('/task2a_cloud/blocked', OccupancyGrid, queue_size=1, latch=True)
    valid_pub = rospy.Publisher('/task2a_cloud/valid', Bool, queue_size=1, latch=True)
    valid_pub.publish(Bool(data=False))

    def publish_grid(publisher, occupied, observed_floor, stamp):
        result = OccupancyGrid()
        result.header = Header(stamp=stamp, frame_id='map')
        result.info = info
        data = np.full(shape, -1, dtype=np.int8)
        data[observed_floor] = 0
        data[occupied] = 100
        result.data = data.ravel().tolist()
        publisher.publish(result)

    def invalidate():
        valid_pub.publish(Bool(data=False))
        # Clear apparent free space on invalid output; previous obstacles remain.
        stamp = rospy.Time.now()
        empty = np.zeros(shape, dtype=bool)
        publish_grid(raw_pub, obstacle_memory, empty, stamp)
        publish_grid(blocked_pub, dilate_disk(obstacle_memory, robot_radius+margin, resolution), empty, stamp)

    rospy.loginfo('Point-cloud mapper: %s; min_height=%.3f m, inflation=%.3f m. NO MOTION.',
                  topic, min_height, robot_radius+margin)
    rate = rospy.Rate(1.0)
    last_stamp = None
    while not rospy.is_shutdown():
        message = latest[0]
        try:
            if message is None:
                raise ValueError('Waiting for camera point cloud')
            age = (rospy.Time.now()-message.header.stamp).to_sec()
            if message.header.stamp == rospy.Time(0) or age > 3.0 or age < -.5:
                raise ValueError('Point cloud is stale or clock differs: age=%.3f' % age)
            if last_stamp == message.header.stamp:
                rate.sleep()
                continue
            stamp = message.header.stamp
            camera_tf = buffer.lookup_transform('map', message.header.frame_id,
                                                 stamp, rospy.Duration(1.0)).transform
            base_tf = buffer.lookup_transform('map', base_frame,
                                               stamp, rospy.Duration(1.0)).transform
            points = cloud_xyz(message)
            if len(points) < 150:
                raise ValueError('Too few finite depth points')
            q = camera_tf.rotation
            rotation = rotation_matrix([q.x, q.y, q.z, q.w])
            t = camera_tf.translation
            points = points @ rotation.T + np.array([t.x, t.y, t.z])
            p = base_tf.translation
            robot_xy = np.array([p.x, p.y])
            distance = np.linalg.norm(points[:, :2]-robot_xy, axis=1)
            # Only nearby geometry, excluding the current body footprint.
            points = points[(distance > self_filter_radius) & (distance <= max_range)
                            & (points[:, 2] > floor_z-.10) & (points[:, 2] < floor_z+.60)]
            normal, offset, residual = fit_floor(points, expected_z=floor_z)
            heights = points @ normal+offset
            floor_points = points[np.abs(heights) < min_height]
            obstacle_points = points[(heights >= min_height) & (heights <= max_height)]
            floor_counts, _ = raster_counts(floor_points, origin, resolution, shape)
            obstacle_counts, inside = raster_counts(obstacle_points, origin, resolution, shape)
            occupied = obstacle_counts >= min_points
            observed_floor = floor_counts >= min_points
            if remember:
                obstacle_memory |= occupied
                floor_memory |= observed_floor
            else:
                obstacle_memory[:] = occupied
                floor_memory[:] = observed_floor
            blocked = dilate_disk(obstacle_memory, robot_radius+margin, resolution)
            publish_grid(raw_pub, obstacle_memory, floor_memory, stamp)
            publish_grid(blocked_pub, blocked, floor_memory, stamp)
            visible_obstacles = obstacle_points[inside]
            if len(visible_obstacles):
                cells = np.floor((visible_obstacles[:, :2]-origin)/resolution).astype(int)
                keep = occupied[cells[:, 1], cells[:, 0]]
                visible_obstacles = visible_obstacles[keep]
                # Visualization can be subsampled after map construction.
                visible_obstacles = visible_obstacles[::max(1, len(visible_obstacles)//30000)]
            header = Header(stamp=stamp, frame_id='map')
            # A separate, synchronized floor observation can disprove old surface
            # locations. Keep one real sample per 5 mm XY cell (not a fake center).
            if len(floor_points):
                floor_cells = np.floor(floor_points[:, :2]/.005).astype(np.int64)
                # Collision-free integer key, including negative coordinates.
                floor_cells -= floor_cells.min(axis=0)
                keys = floor_cells[:, 0]*(floor_cells[:, 1].max()+1)+floor_cells[:, 1]
                _, unique = np.unique(keys, return_index=True)
                sampled_floor = floor_points[unique]
            else:
                sampled_floor = floor_points
            floor_pub.publish(xyz_cloud(header, sampled_floor))
            cloud_pub.publish(xyz_cloud(header, visible_obstacles))
            valid_pub.publish(Bool(data=True))
            last_stamp = stamp
            rospy.loginfo_throttle(5.0,
                'Floor z=%.4f m, fit residual95=%.4f m, obstacle points=%d, occupied cells=%d' %
                (-offset/normal[2], residual, len(visible_obstacles), int(obstacle_memory.sum())))
        except Exception as error:
            invalidate()
            rospy.logwarn_throttle(3.0, 'Cloud map NOT VALID: %s' % error)
        rate.sleep()


if __name__ == '__main__':
    main()
