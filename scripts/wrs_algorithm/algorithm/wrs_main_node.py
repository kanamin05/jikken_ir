#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
WRS環境内でロボットを動作させるためのメインプログラム
"""

from __future__ import unicode_literals, print_function, division, absolute_import
import json
import os
from select import select
import traceback
from turtle import pos
import rospy
import rospkg
import tf2_ros
from std_msgs.msg import String
from detector_msgs.srv import (
    SetTransformFromBBox, SetTransformFromBBoxRequest,
    GetObjectDetection, GetObjectDetectionRequest)
from wrs_algorithm.util import omni_base, whole_body, gripper
import math

import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo




class WrsMainController(object):
    """
    WRSのシミュレーション環境内でタスクを実行するクラス
    """
    IGNORE_LIST = ["small_marker", "large_marker", "lego_duplo", "spatula", "nine_hole_peg_test"]
    GRASP_TF_NAME = "object_grasping"
    GRASP_BACK_SAFE = {"z": 0.05, "xy": 0.3}
    GRASP_BACK = {"z": 0.05, "xy": 0.1}
    HAND_PALM_OFFSET = 0.05  # hand_palm_linkは指の付け根なので、把持のために少しずらす必要がある
    HAND_PALM_Z_OFFSET = 0.075
    DETECT_CNT = 10
    TROFAST_Y_OFFSET = 0.2

    def __init__(self):
        # 変数の初期化
        self.instruction_list = []
        self.detection_list   = []

        # configファイルの受信
        self.coordinates = self.load_json(self.get_path(["config", "coordinates.json"]))
        self.poses       = self.load_json(self.get_path(["config", "poses.json"]))

        self.sorting = self.load_json(self.get_path(["config", "object_sorting.json"]))

        # ROS通信関連の初期化
        tf_from_bbox_srv_name = "set_tf_from_bbox"
        rospy.wait_for_service(tf_from_bbox_srv_name)
        self.tf_from_bbox_clt = rospy.ServiceProxy(tf_from_bbox_srv_name, SetTransformFromBBox)

        obj_detection_name = "detection/get_object_detection"
        rospy.wait_for_service(obj_detection_name)
        self.detection_clt = rospy.ServiceProxy(obj_detection_name, GetObjectDetection)

        self.tf_buffer   = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.instruction_sub = rospy.Subscriber("/message",    String, self.instruction_cb, queue_size=10)
        self.detection_sub   = rospy.Subscriber("/detect_msg", String, self.detection_cb,   queue_size=10)

    @staticmethod
    def get_path(pathes, package="wrs_algorithm"):
        """
        ROSパッケージ名とファイルまでのパスを指定して、ファイルのパスを取得する
        """
        if not pathes:  # check if the list is empty
            rospy.logerr("Can NOT resolve file path.")
            raise ValueError("You must specify the path to file.")
        pkg_path = rospkg.RosPack().get_path(package)
        path = os.path.join(*pathes)
        return os.path.join(pkg_path, path)

    @staticmethod
    def load_json(path):
        """
        jsonファイルを辞書型で読み込む
        """
        with open(path, "r") as json_file:
            return json.load(json_file)

    def instruction_cb(self, msg):
        """
        指示文を受信する
        """
        rospy.loginfo("instruction received. [%s]", msg.data)
        self.instruction_list.append(msg.data)

    def detection_cb(self, msg):
        """
        検出結果を受信する
        """
        rospy.loginfo("received [Collision detected with %s]", msg.data)
        self.detection_list.append(msg.data)

    def get_relative_coordinate(self, parent, child):
        """
        tfで相対座標を取得する
        """
        try:
            # 4秒待機して各tfが存在すれば相対関係をセット
            trans = self.tf_buffer.lookup_transform(parent, child,rospy.Time.now(),rospy.Duration(4.0))
            return trans.transform
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException):
            log_str = "failed to get transform between [{}] and [{}]\n".format(parent, child)
            log_str += traceback.format_exc()
            rospy.logerr(log_str)
            return None

    def goto_name(self, name):
        """
        waypoint名で指定された場所に移動する
        """
        if name in self.coordinates["positions"].keys():
            pos = self.coordinates["positions"][name]
            rospy.loginfo("go to [%s](%.2f, %.2f, %.2f)", name, pos[0], pos[1], pos[2])
            return omni_base.go_abs(pos[0], pos[1], pos[2])

        rospy.logerr("unknown waypoint name [%s]", name)
        return False

    def goto_pos(self, pos):
        """
        waypoint名で指定された場所に移動する
        """
        rospy.loginfo("go to [raw_pos](%.2f, %.2f, %.2f)", pos[0], pos[1], pos[2])
        return omni_base.go_abs(pos[0], pos[1], pos[2])

    def change_pose(self, name):
        """
        指定された姿勢名に遷移する
        """
        if name in self.poses.keys():
            rospy.loginfo("change pose to [%s]", name)
            return whole_body.move_to_joint_positions(self.poses[name])

        rospy.logerr("unknown pose name [%s]", name)
        return False

    def check_positions(self):
        """
        読み込んだ座標ファイルの座標を巡回する
        """
        whole_body.move_to_go()
        for wp_name in self.coordinates["routes"]["test"]:
            self.goto_name(wp_name)
            rospy.sleep(1)

    def get_latest_detection(self):
        """
        最新の認識結果が到着するまで待つ
        """
        res = self.detection_clt(GetObjectDetectionRequest())
        return res.bboxes

    def get_grasp_coordinate(self, bbox):
        """
        BBox情報から把持座標を取得する
        """
        # BBox情報からtfを生成して、座標を取得
        self.tf_from_bbox_clt.call(
            SetTransformFromBBoxRequest(
                bbox=bbox,
                frame=self.GRASP_TF_NAME
            )
        )

        rospy.sleep(1.0)

        transform = self.get_relative_coordinate(
            "map", self.GRASP_TF_NAME
        )
        if transform is None:
            return None

        return transform.translation
    
    @classmethod
    def get_most_graspable_bbox(cls, obj_list):
        """
        最も把持が行えそうなbboxを一つ返す。
        """
        # objが一つもない場合は、Noneを返す
        obj = cls.get_most_graspable_obj(obj_list)
        if obj is None: return None
        return obj["bbox"]

    @classmethod
    def get_most_graspable_obj(cls, obj_list):
        """
        把持すべきscoreが最も高い物体を返す。
        """
        extracted = []
        extract_str = "detected object list\n"
        ignore_str = ""

        for obj in obj_list:
            info_str = (
                "{:<15}({:.2%}, {:3d}, {:3d}, {:3d}, {:3d})\n"
            ).format(
                obj.label, obj.score, obj.x, obj.y, obj.w, obj.h
            )

            if obj.label in cls.IGNORE_LIST:
                ignore_str += "- ignored  : " + info_str
            else:
                score = cls.calc_score_bbox(obj)
                extracted.append({
                    "bbox": obj,
                    "score": score,
                    "label": obj.label
                })
                extract_str += (
                    "- extracted: {:07.3f} ".format(score) + info_str
                )

        rospy.loginfo(extract_str + ignore_str)

        # つかむべきかのscoreが一番高い物体を返す
        for obj_info in sorted(
            extracted,
            key=lambda x: x["score"],
            reverse=True
        ):
            obj = obj_info["bbox"]

            info_str = (
                "{} ({:.2%}, {:3d}, {:3d}, {:3d}, {:3d})\n"
            ).format(
                obj.label, obj.score, obj.x, obj.y, obj.w, obj.h
            )

            rospy.loginfo("selected bbox: " + info_str)
            return obj_info

        return None

    @classmethod
    def calc_score_bbox(cls, bbox):
        """
        detector_msgs/BBoxのスコアを計算する
        """
        gravity_x = bbox.x + bbox.w / 2
        gravity_y = bbox.y + bbox.h / 2

        xy_diff = (
            abs(320 - gravity_x) / 320
            + abs(360 - gravity_y) / 240
        )

        return 1 / max(xy_diff, 0.001)

    @classmethod
    def get_most_graspable_bboxes_by_label(cls, obj_list, label):
        """
        label名が一致するオブジェクトの中から
        最も把持すべき物体のbboxを返す
        """
        match_objs = [
            obj for obj in obj_list
            if obj.label == label
        ]

        if not match_objs:
            rospy.logwarn(
                "Cannot find a object which labeled with similar name."
            )
            return None

        return cls.get_most_graspable_bbox(match_objs)

    @staticmethod
    def extract_target_obj_and_person(instruction):
        """指示文から対象物と左右を読み取る"""
        import re

        text = instruction.lower().replace("_", " ").replace("-", " ")
        text = re.sub(r"\s+", " ", text)

        # 検出器で使われる物体名
        labels = [
            "cracker_box", "sugar_box", "pudding_box",
            "gelatin_box", "potted_meat_can", "master_chef_can",
            "tuna_fish_can", "chips_can", "mustard_bottle",
            "tomato_soup_can", "banana", "strawberry",
            "apple", "lemon", "peach", "pear", "orange", "plum",
            "mug", "cup", "bowl", "plate", "sponge"
        ]

        # 指示文で使われる別の呼び方
        aliases = {
            "tuna_fish_can": ["tuna can", "tuna"],
            "master_chef_can": ["coffee can", "coffee"],
            "potted_meat_can": ["potted meat"],
            "chips_can": ["chips", "pringles"],
            "tomato_soup_can": ["tomato soup"],
            "cracker_box": ["crackers"],
            "sugar_box": ["sugar"],
            "pudding_box": ["pudding"],
            "gelatin_box": ["gelatin"],
            "mustard_bottle": ["mustard"]
        }

        matched_objects = set()

        for label in labels:
            names = [
                label.replace("_", " ")
            ] + aliases.get(label, [])

            for name in names:
                pattern = r"\b" + re.escape(name) + r"\b"
                if re.search(pattern, text):
                    matched_objects.add(label)
                    break

        matched_people = [
            direction for direction in ("left", "right")
            if re.search(r"\b" + direction + r"\b", text)
        ]

        if len(matched_objects) != 1 or len(matched_people) != 1:
            rospy.logwarn(
                "Cannot interpret instruction: %s", instruction
            )
            return None, None

        return next(iter(matched_objects)), matched_people[0]

    def grasp_from_side(self, pos_x, pos_y, pos_z, yaw, pitch, roll, preliminary="-y"):
        """
        把持の一連の動作を行う

        NOTE: tall_tableに対しての予備動作を生成するときはpreliminary="-y"と設定することになる。
        """
        if preliminary not in [ "+y", "-y", "+x", "-x" ]: raise RuntimeError("unnkown graps preliminary type [{}]".format(preliminary))

        rospy.loginfo("move hand to grasp (%.2f, %.2f, %.2f)", pos_x, pos_y, pos_z)

        grasp_back_safe = {"x": pos_x, "y": pos_y, "z": pos_z + self.GRASP_BACK["z"]}
        grasp_back = {"x": pos_x, "y": pos_y, "z": pos_z + self.GRASP_BACK["z"]}
        grasp_pos = {"x": pos_x, "y": pos_y, "z": pos_z}

        if "+" in preliminary:
            sign = 1
        elif "-" in preliminary:
            sign = -1

        if "x" in preliminary:
            grasp_back_safe["x"] += sign * self.GRASP_BACK_SAFE["xy"]
            grasp_back["x"] += sign * self.GRASP_BACK["xy"]
        elif "y" in preliminary:
            grasp_back_safe["y"] += sign * self.GRASP_BACK_SAFE["xy"]
            grasp_back["y"] += sign * self.GRASP_BACK["xy"]

        gripper.command(1)
        whole_body.move_end_effector_pose(grasp_back_safe["x"], grasp_back_safe["y"], grasp_back_safe["z"], yaw, pitch, roll)
        whole_body.move_end_effector_pose( grasp_back["x"], grasp_back["y"], grasp_back["z"], yaw, pitch, roll)
        whole_body.move_end_effector_pose(
            grasp_pos["x"], grasp_pos["y"], grasp_pos["z"], yaw, pitch, roll)
        gripper.command(0)
        whole_body.move_end_effector_pose(grasp_back_safe["x"], grasp_back_safe["y"], grasp_back_safe["z"], yaw, pitch, roll)

    def grasp_from_front_side(self, grasp_pos):
        """
        正面把持を行う
        ややアームを下に向けている
        """
        grasp_pos.y -= self.HAND_PALM_OFFSET
        rospy.loginfo("grasp_from_front_side (%.2f, %.2f, %.2f)",grasp_pos.x, grasp_pos.y, grasp_pos.z)
        self.grasp_from_side(grasp_pos.x, grasp_pos.y, grasp_pos.z, -90, -100, 0, "-y")

    def grasp_from_upper_side(self, grasp_pos):
        """
        上面から把持を行う
        オブジェクトに寄るときは、y軸から近づく上面からは近づかない
        """
        grasp_pos.z += self.HAND_PALM_Z_OFFSET
        rospy.loginfo("grasp_from_upper_side (%.2f, %.2f, %.2f)",grasp_pos.x, grasp_pos.y, grasp_pos.z)
        self.grasp_from_side(grasp_pos.x, grasp_pos.y, grasp_pos.z, -90, -160, 0, "-y")


    def exec_graspable_method(self, grasp_pos, label=""):
        """
        task1専用:posの位置によって把持方法を判定し実行する。
        """
        method = None
        graspable_y = 1.85  # これ以上奥は把持できない
        desk_y = 1.5
        desk_z = 0.35

        # 把持禁止判定
        if graspable_y < grasp_pos.y and desk_z > grasp_pos.z:
            return False

        if label in ["cup", "frisbee", "bowl"]:
            # bowlの張り付き対策
            method = self.grasp_from_upper_side
        else:
            if desk_y < grasp_pos.y and desk_z > grasp_pos.z:
                # 机の下である場合
                method = self.grasp_from_front_side
            else:
                method = self.grasp_from_upper_side

        method(grasp_pos)
        return True

    def where_to_put(self, label, category=None):
        """Return the configured category and deposit for a detector label.

        Pass category="boss" for an object identified as a boss.
        The first deposit is preferred when a category has multiple deposits.
        Unrecognized labels use the configured unknown-object category.
        This selects a destination; it does not perform drawer opening or
        orientation-specific placement.
        """
        if category is None:
            category = self.sorting["default_category"]
            for name, rule in self.sorting["categories"].items():
                if label in rule["labels"]:
                    category = name
                    break
        destination = self.sorting["categories"][category]["destinations"][0]
        result = dict(self.sorting["destinations"][destination])
        result.update(category=category, destination=destination)
        return result

    def put_in_place(self, place, into_pose):
        # 指定場所に入れ、all_neutral姿勢を取る。
        self.change_pose("look_at_near_floor")
        a = "go_palce" # TODO 不要な変数
        self.goto_name(place)
        self.change_pose("all_neutral")
        self.change_pose(into_pose)
        gripper.command(1)
        rospy.sleep(5.0)
        self.change_pose("all_neutral")

    def pull_out_trofast(self, x, y, z, yaw, pitch, roll):
        # trofastの引き出しを引き出す
        # stair_like_drawer へ移動する
        self.goto_name("stair_like_drawer")
        # grasp_on_table の姿勢になる
        self.change_pose("grasp_on_table")
        a = True  # TODO 不要な変数

        # グリッパを開く
        gripper.command(1)
        # アーム先端を指定された位置と姿勢へ移動させる関数
        whole_body.move_end_effector_pose(x, y + self.TROFAST_Y_OFFSET, z, yaw, pitch, roll)
        # 取っ手にアームを近づける
        whole_body.move_end_effector_pose(x, y, z, yaw, pitch, roll)
        # グリッパを閉じる
        gripper.command(0)
        # 引き出しを引く
        whole_body.move_end_effector_pose(x, y + self.TROFAST_Y_OFFSET, z, yaw,  pitch, roll)
        # グリッパを開く
        gripper.command(1)
        # アームを元の姿勢に戻す
        self.change_pose("all_neutral")

    def push_in_trofast(self, pos_x, pos_y, pos_z, yaw, pitch, roll):
        """
        trofastの引き出しを戻す
        NOTE:サンプル
            self.push_in_trofast(0.178, -0.29, 0.75, -90, 100, 0)
        """
        self.goto_name("stair_like_drawer")
        self.change_pose("grasp_on_table")
        pos_y+=self.HAND_PALM_OFFSET

        # 予備動作-押し込む
        whole_body.move_end_effector_pose(pos_x, pos_y +    self.TROFAST_Y_OFFSET * 1.5, pos_z, yaw, pitch, roll)
        gripper.command(0)
        whole_body.move_end_effector_pose(pos_x, pos_y + self.TROFAST_Y_OFFSET, pos_z, yaw, pitch, roll)
        whole_body.move_end_effector_pose(pos_x, pos_y, pos_z, yaw, pitch, roll)

        self.change_pose("all_neutral")

    def deliver_to_target(self, target_obj, target_person):
        """
        棚で取得したものを人に渡す。
        """
        self.change_pose("look_at_near_floor")
        self.goto_name("shelf")
        self.change_pose("look_at_shelf")

        rospy.loginfo(
            "target_obj: %s  target_person: %s",
            target_obj,
            target_person
        )

        # 同じカメラ姿勢で、指定物体を最大３回探す
        grasp_bbox = None

        for attempt in range(1, 4):
            if rospy.is_shutdown():
                return

            rospy.loginfo(
                "Task2b detection attempt %d/3: target=%s",
                attempt,
                target_obj
            )

            detected_objs = self.get_latest_detection()

            if detected_objs is not None:
                grasp_bbox = self.get_most_graspable_bboxes_by_label(
                    detected_objs.bboxes,
                    target_obj
                )

            if grasp_bbox is not None:
                rospy.loginfo(
                    "Target found on attempt %d",
                    attempt
                )
                break

            rospy.logwarn(
                "Target not found on attempt %d: %s",
                attempt,
                target_obj
            )

            if attempt < 3:
                rospy.sleep(1.0)

        if grasp_bbox is None:
            rospy.logwarn(
                "Target not found after 3 attempts. Task2b aborted."
            )
            return

        # 物体の３次元座標を取得
        grasp_pos = self.get_grasp_coordinate(grasp_bbox)

        if grasp_pos is None:
            rospy.logwarn("Cannot obtain object coordinates")
            return

        # 把持は１回だけ行う
        self.change_pose("grasp_on_shelf")
        self.grasp_from_front_side(grasp_pos)
        self.change_pose("all_neutral")

        # 指示された人へ運ぶ
        person_places = {
            "left": "person_a",
            "right": "person_b"
        }

        self.change_pose("look_at_near_floor")
        self.goto_name(person_places[target_person])
        self.change_pose("deliver_to_human")

        # 人の前で受け渡す
        rospy.sleep(10.0)
        gripper.command(1)
        self.change_pose("all_neutral")

    def execute_avoid_blocks(self):
        # blockを避ける
        for i in range(3):
            detected_objs = self.get_latest_detection()
            bboxes = detected_objs.bboxes
            pos_bboxes = [self.get_grasp_coordinate(bbox) for bbox in bboxes]
            waypoint = self.select_next_waypoint(i, pos_bboxes)
            # TODO メッセージを確認するためコメントアウトを外す
            # rospy.loginfo(waypoint)
            self.goto_pos(waypoint)

    def select_next_waypoint(self, current_stp, pos_bboxes):
        """
        waypoints から近い場所にあるものを除外し、最適なwaypointを返す。
        x座標を原点に近い方からxa,xb,xcに定義する。bboxを判断基準として移動先を決定する(デフォルトは0.45間隔)
        pos_bboxesは get_grasp_coordinate() 済みであること
        """
        interval = 0.45
        pos_xa = 1.7
        pos_xb = pos_xa + interval
        pos_xc = pos_xb + interval

        # xa配列はcurrent_stpに関係している
        waypoints = {"xa": [ [pos_xa, 2.5, 45],[pos_xa, 2.9, 45],[pos_xa, 3.3, 90] ], "xb": [ [pos_xb, 2.5, 90], [pos_xb, 2.9, 90], [pos_xb, 3.3, 90] ],
            "xc": [ [pos_xc, 2.5, 135],   [pos_xc, 2.9, 135],  [pos_xc, 3.3, 90 ]]
        }

        # posがxa,xb,xcのラインに近い場合は候補から削除
        is_to_xa = True
        is_to_xb = True
        is_to_xc = True
        for bbox in pos_bboxes:
            pos_x = bbox.x
            # TODO デバッグ時にコメントアウトを外す
            # rospy.loginfo("detected object obj.x = {:.2f}".format(bbox.x))

            # NOTE Hint:ｙ座標次第で無視してよいオブジェクトもある。
            if pos_x < pos_xa + (interval/2):
                is_to_xa = False
                # rospy.loginfo("is_to_xa=False")
                continue
            elif pos_x < pos_xb + (interval/2):
                is_to_xb = False
                # rospy.loginfo("is_to_xb=False")
                continue
            elif pos_x < pos_xc + (interval/2):
                is_to_xc = False
                # rospy.loginfo("is_to_xc=False")
                continue

        x_line = None   # xa,xb,xcいずれかのリストが入る
        # NOTE 優先的にxcに移動する
        if is_to_xc:
            x_line = waypoints["xc"]
            rospy.loginfo("select next waypoint_xc")
        elif is_to_xb:
            x_line = waypoints["xb"]
            rospy.loginfo("select next waypoint_xb")
        elif is_to_xa:
            x_line = waypoints["xa"]
            rospy.loginfo("select next waypoint_xa")
        else:
            # a,b,cいずれにも移動できない場合
            x_line = waypoints["xb"]
            rospy.loginfo("select default waypoint")

        return x_line[current_stp]
    
    def correct_drawer_position(self, target_x, target_y):
        """62400746 阿部真子
        TFを使ってx・y座標を低速で補正する
        """

        for i in range(50):
            trans = self.get_relative_coordinate(
                "map", "base_footprint"
            )

            if trans is None:
                omni_base.cmd_vel(0.0, 0.0, 0.0)
                return False

            x = trans.translation.x
            y = trans.translation.y

            error_x = target_x - x
            error_y = target_y - y

            rospy.loginfo(
                "位置補正%d: x=%.3f, y=%.3f, 誤差=(%.3f, %.3f)",
                i + 1, x, y, error_x, error_y
            )

            # x・yともに1.0cm以内なら成功
            if abs(error_x) < 0.010 and abs(error_y) < 0.010:
                omni_base.cmd_vel(0.0, 0.0, 0.0)
                rospy.loginfo("位置補正成功")
                return True

            # 現在のロボットの向きを取得
            q = trans.rotation
            yaw = math.atan2(
                2.0 * (q.w * q.z + q.x * q.y),
                1.0 - 2.0 * (q.y**2 + q.z**2)
            )

            # map座標の誤差をロボット座標へ変換
            error_forward = (
                math.cos(yaw) * error_x
                + math.sin(yaw) * error_y
            )
            error_left = (
                -math.sin(yaw) * error_x
                + math.cos(yaw) * error_y
            )

            # 低速で移動（最大4cm/s）
            vel_x = max(-0.06, min(0.06, error_forward * 0.9))
            vel_y = max(-0.06, min(0.06, error_left * 0.9))

            omni_base.cmd_vel(vel_x, vel_y, 0.0)
            rospy.sleep(0.2)
            omni_base.cmd_vel(0.0, 0.0, 0.0)
            rospy.sleep(0.2)

        omni_base.cmd_vel(0.0, 0.0, 0.0)
        rospy.logwarn("位置補正が完了しませんでした")
        return False

    def correct_drawer_pose(self, target_x, target_y, target_deg):
        """62400746 阿部真子
        角度のみを低速で補正する
        """

        for i in range(100):
            trans = self.get_relative_coordinate(
                "map", "base_footprint"
            )

            if trans is None:
                omni_base.cmd_vel(0.0, 0.0, 0.0)
                return False

            q = trans.rotation
            yaw = math.atan2(
                2.0 * (q.w * q.z + q.x * q.y),
                1.0 - 2.0 * (q.y**2 + q.z**2)
            )
            current_deg = math.degrees(yaw)

            error_deg = (
                (target_deg - current_deg + 180.0) % 360.0
            ) - 180.0

            rospy.loginfo(
                "角度補正%d: 現在=%.2f, 目標=%.2f, 誤差=%.2f",
                i + 1, current_deg, target_deg, error_deg
            )

            if abs(error_deg) < 0.8:
                omni_base.cmd_vel(0.0, 0.0, 0.0)
                rospy.loginfo("角度補正成功")
                return True

            # 角度誤差に応じた回転速度（deg/s）
            angular_speed = max(
                -5.0,
                min(5.0, error_deg * 1.5)
            )

            rospy.loginfo(
                "回転指令: %.2f deg/s", angular_speed
            )

            omni_base.cmd_vel(0.0, 0.0, angular_speed)
            rospy.sleep(0.2)

            omni_base.cmd_vel(0.0, 0.0, 0.0)
            rospy.sleep(0.2)

        omni_base.cmd_vel(0.0, 0.0, 0.0)
        rospy.logwarn("角度補正が完了しませんでした")
        return False

    def move_forward_backward(self, duration=2.0, speed=0.05):
        """62400746 阿部真子
        指定した秒数だけ前進・後進する
        """

        rate = rospy.Rate(10)
        start = rospy.Time.now()

        try:
            while not rospy.is_shutdown():
                elapsed = (rospy.Time.now() - start).to_sec()

                if elapsed >= duration:
                    break

                # 正の速度なら前進、負の速度なら後進
                omni_base.cmd_vel(speed, 0.0, 0.0)
                rate.sleep()

        finally:
            # 移動終了後は必ず停止
            omni_base.cmd_vel(0.0, 0.0, 0.0)

    def get_base_xy(self):
        """62400746 阿部真子
        base_footprintのmap座標(x, y)を返す。取れなければNone
        """
        try:
            t = self.tf_buffer.lookup_transform(
                "map", "base_footprint", rospy.Time(0), rospy.Duration(1.0))
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException):
            return None
        return t.transform.translation.x, t.transform.translation.y

    def move_to_y(self, target_y, max_speed=0.05, timeout=30.0):
        """62400746 阿部真子
        yaw=-90度(-y向き)のとき、base_footprintのmap y座標がtarget_yになるまで動く。
        y > target_y なら前進、y < target_y なら後退する。
        """
        rate = rospy.Rate(10)
        start = rospy.Time.now()
        try:
            while not rospy.is_shutdown():
                if (rospy.Time.now() - start).to_sec() > timeout:
                    rospy.logwarn("move_to_y: タイムアウト")
                    return False

                pos = self.get_base_xy()
                if pos is None:
                    return False

                error = pos[1] - target_y
                if abs(error) < 0.005:
                    rospy.loginfo("move_to_y: 到達 y=%.3f", pos[1])
                    return True

                # 目標に近づくほど減速。遅すぎて止まらないよう下限を設ける
                speed = max(-max_speed, min(max_speed, error * 2.0))
                if abs(speed) < 0.015:
                    speed = math.copysign(0.015, speed)

                omni_base.cmd_vel(speed, 0.0, 0.0)
                rate.sleep()
        finally:
            omni_base.cmd_vel(0.0, 0.0, 0.0)
        return False

    def align_to_drawer(self, target_x, target_y, target_deg, retry=3):
        """62400746 阿部真子
        角度と位置を補正する。両方が許容内に入るまで最大retry回繰り返す
        """
        for _ in range(retry):
            ok_pos = self.correct_drawer_position(target_x, target_y)
            ok_pose = self.correct_drawer_pose(target_x, target_y, target_deg)
            if ok_pose and ok_pos:
                return True
        rospy.logwarn("align_to_drawer: 補正が完了しませんでした")
        return False

    def open_drawer(self, name, waypoint,
                target_x, target_y, target_deg,
                lift, arm_flex, grasp_y, pull_dist):
        """62400746 阿部真子
        引き出しを1つ開ける。
        grasp_y : 掴む位置のbase_footprintのy座標
        pull_dist: 引く距離[m]
        """
        rospy.loginfo("==== %s ====", name)
        if not self.goto_name(waypoint):
            return False
        omni_base.cmd_vel(0.0, 0.0, 0.0)
        rospy.sleep(0.5)
        gripper.command(1)

        # 腕を動かす前に補正
        self.align_to_drawer(target_x, target_y, target_deg)

        # アームの高さ・姿勢を設定
        grasp_pose = dict(self.poses["grasp_on_table"])
        grasp_pose["arm_lift_joint"] = lift
        grasp_pose["arm_flex_joint"] = arm_flex
        grasp_pose["wrist_flex_joint"] = 0.0
        grasp_pose["wrist_roll_joint"] = 1.5
        whole_body.move_to_joint_positions(grasp_pose)
        rospy.sleep(1.0)

        # 腕を動かした後にもう一度補正（角度＋位置）
        self.align_to_drawer(target_x, target_y, target_deg)
        rospy.sleep(0.5)

        # 取っ手に近づく（yで止める）
        self.move_to_y(grasp_y)
        rospy.sleep(0.5)
        pos = self.get_base_xy()
        if pos is not None:
            rospy.loginfo("%s 掴む直前: x=%.3f y=%.3f", name, pos[0], pos[1])

        # 取っ手をつかむ
        gripper.command(0)
        rospy.sleep(1.0)

        # 引き出す（yで止める）
        self.move_to_y(grasp_y + pull_dist)

        # 離して、さらに後退
        gripper.command(1)
        rospy.sleep(1.0)
        gripper.command(1)
        rospy.sleep(1.0)
        self.move_to_y(grasp_y + pull_dist + 0.1)

        self.change_pose("all_neutral")
        rospy.sleep(0.5)
        return True

    def execute_task1(self):
        """62400746 阿部真子
        3つの引き出し（左下・右下・右上）を順番に開ける
        """
        rospy.loginfo("#### Drawer Test ####")

        drawers = [
            dict(name="左下", waypoint="stair_like_drawer",
                 target_x=0.36, target_y=0.35, target_deg=-90.0,
                 lift=0.0, arm_flex = -1.7, grasp_y=0.32, pull_dist=0.3),
            dict(name="右下", waypoint="drawer",
                 target_x=0.07, target_y=0.34, target_deg=-90.0,
                 lift=0.0, arm_flex = -1.6, grasp_y=0.32, pull_dist=0.25),
            dict(name="右上", waypoint="drawer",
                 target_x=0.07, target_y=0.34, target_deg=-90.0,
                 lift=0.23, arm_flex = -1.5, grasp_y=0.25, pull_dist=0.3),
        ]

        for drawer in drawers:
            self.open_drawer(**drawer)

        
        # 物体探索、把持処理
        hsr_position = [
            ("tall_table", "look_at_tall_table"),
            ("near_long_table_l", "look_at_near_floor"),
            ("long_table_r", "look_at_tall_table"),
        ]

        for plc, pose in hsr_position:
            for _ in range(self.DETECT_CNT):
                # 移動と視線指示
                self.goto_name(plc)
                self.change_pose(pose)
                gripper.command(0)

                # 把持対象の有無チェック
                detected_objs = self.get_latest_detection()
                graspable_obj = self.get_most_graspable_obj(detected_objs.bboxes)

                if graspable_obj is None:
                    rospy.logwarn("Cannot determine object to grasp. Grasping is aborted.")
                    continue
                label = graspable_obj["label"]
                grasp_bbox = graspable_obj["bbox"]
                deposit = self.where_to_put(label)
                place, into_pose = deposit["waypoint"], deposit["pose"]
                if (place not in self.coordinates["positions"] or
                        into_pose not in self.poses):
                    rospy.logwarn("Cannot sort %s into %s: configure its waypoint and pose.",
                                  label, deposit["destination"])
                    continue
                rospy.loginfo("sort %s (%s) into %s", label, deposit["category"],
                              deposit["destination"])
                # TODO ラベル名を確認するためにコメントアウトを外す
                rospy.loginfo("grasp the " + label)

                # 把持対象がある場合は把持関数実施
                grasp_pos = self.get_grasp_coordinate(grasp_bbox)
                self.change_pose("grasp_on_table")
                if not self.exec_graspable_method(grasp_pos, label):
                    self.change_pose("all_neutral")
                    continue
                self.change_pose("all_neutral")

                # binに入れる
                self.put_in_place(place, into_pose)

    def execute_task2a(self):
        """
        task2aを実行する
        """
        rospy.loginfo("#### start Task 2a ####")
        self.change_pose("look_at_near_floor")
        gripper.command(0)
        self.change_pose("look_at_near_floor")
        self.goto_name("standby_2a")

        # 落ちているブロックを避けて移動
        self.execute_avoid_blocks()

        self.goto_name("go_throw_2a")
        whole_body.move_to_go()

    def execute_task2b(self):
        """有効な指示を待ち、指定された物を届ける"""
        rospy.loginfo("#### start Task 2b ####")

        self.change_pose("all_neutral")
        if self.goto_name("shelf") is False:
            rospy.logwarn("Cannot move to shelf")
            return
        self.change_pose("look_at_shelf")

        target_obj = None
        target_person = None

        for _ in range(30):
            if rospy.is_shutdown():
                return

            # 新しいメッセージから順に指示を探す
            for instruction in reversed(self.instruction_list[:]):
                obj, person = self.extract_target_obj_and_person(
                    instruction
                )

                if obj and person:
                    target_obj = obj
                    target_person = person
                    rospy.loginfo(
                        "received instruction: %s", instruction
                    )
                    break

            if target_obj and target_person:
                break

            rospy.sleep(1.0)

        if not target_obj or not target_person:
            rospy.logwarn("No valid instruction received")
            return

        self.deliver_to_target(target_obj, target_person)

    def run(self):
        """
        全てのタスクを実行する
        """
        self.change_pose("all_neutral")
        self.execute_task1()
        # self.execute_task2a()
        self.execute_task2b()


def main():
    """
    WRS環境内でタスクを実行するためのメインノードを起動する
    """
    rospy.init_node('main_controller')
    try:
        ctrl = WrsMainController()
        rospy.loginfo("node initialized [%s]", rospy.get_name())

        # タスクの実行モードを確認する
        if rospy.get_param("~test_mode", default=False) is True:
            rospy.loginfo("#### start with TEST mode. ####")
            ctrl.check_positions()
        else:
            rospy.loginfo("#### start with NORMAL mode. ####")
            ctrl.run()

    except rospy.ROSInterruptException:
        pass


if __name__ == '__main__':
    main()
