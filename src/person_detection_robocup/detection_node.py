#!/usr/bin/env python3
import traceback

import rospy
import message_filters
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Pose, PoseArray
from cv_bridge import CvBridge
from clf_person_recognition_msgs.srv import (
    SetPersonTemplate,
    SetPersonTemplateRequest,
    SetPersonTemplateResponse,
)
from std_srvs.srv import SetBool, SetBoolRequest
import numpy as np
import cv2
import os, time
import torch
from person_detection_robocup.submodules.SOD import SOD
import rospkg
import threading
from people_msgs.msg import People, Person


class CameraProcessingNode:
    def __init__(self):

        self.enabled = False
        self.enable_debug = True
        self.enable_people = True
        self.enable_poses = True

        self.lock = threading.Lock()

        # Single Person Detection model
        # Setting up Available CUDA device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        rospy.loginfo(f"running on {device}")
        rospack = rospkg.RosPack()
        package_path = rospack.get_path("person_detection_robocup")

        # Setting up model paths (YOLO for object detection and segmentation, and orientation estimation model)
        # yolo_path = os.path.join(pkg_shared_dir, 'models', 'yolov8n-segpose.engine')
        yolo_path = os.path.join(package_path, "models", "yolov8n-pose.engine")
        feature_extracture_model_path = os.path.join(
            package_path, "models", "kpr_reid_shape_inferred.onnx"
        )
        feature_extracture_cfg_path = os.path.join(
            package_path, "models", "kpr_market_test.yaml"
        )

        # Setting up Detection Pipeline
        self.model = SOD(
            yolo_path,
            feature_extracture_model_path,
            feature_extracture_cfg_path,
            logger_level=20,
        )
        self.model.to(device)
        rospy.loginfo("Deep Learning Model Armed")

        # Initialize the template
        # template_img_path = os.path.join(package_path, 'templates', 'temp_template.png')
        # self.template_img = cv2.imread(template_img_path)
        # self.model.template_update(self.template_img)

        # Warmup inference (GPU can be slow in the first inference)
        self.model.detect(
            img_rgb=np.ones((480, 640, 3), dtype=np.uint8),
            img_depth=np.ones((480, 640), dtype=np.uint16),
        )
        rospy.loginfo("Warmup Inference Executed")

        # Subscribing to topics
        image_sub = message_filters.Subscriber("~image", Image)
        depth_sub = message_filters.Subscriber("~depth", Image)
        info_sub = message_filters.Subscriber("~camera_info", CameraInfo)

        self.cv_bridge = CvBridge()

        # Synchronize topics
        ts = message_filters.TimeSynchronizer([image_sub, depth_sub, info_sub], 10)
        ts.registerCallback(self.callback)

        # Publisher for PoseArray
        self.pose_pub = rospy.Publisher("~detected_poses", PoseArray, queue_size=10)
        self.people_pub = rospy.Publisher("~people", People, queue_size=10)

        # Publisher for debug img
        self.debug_image_pub = rospy.Publisher("~debug_img", Image, queue_size=10)

        # Service to process images
        self.service = rospy.Service(
            "~set_template", SetPersonTemplate, self.handle_image_service
        )
        self.srv_enable = rospy.Service("~enable", SetBool, self.enable)

        rospy.loginfo("Camera Processing Node Ready")

    def callback(self, image, depth, camera_info):
        """Callback when all topics are received"""

        rospy.logdebug("Received synchronized image and depth.")

        if not self.enabled:
            return

        with self.lock:
            try:
                cv_rgb = self.cv_bridge.imgmsg_to_cv2(image, "bgr8")
                cv_depth = self.cv_bridge.imgmsg_to_cv2(
                    depth, "passthrough"
                )  # Assuming depth is float32

                fx = camera_info.K[0]
                fy = camera_info.K[4]
                cx = camera_info.K[2]
                cy = camera_info.K[5]

                start_time = time.time()
                ############################
                rospy.logdebug(f"Running Model...")
                results = self.model.detect(cv_rgb, cv_depth, [fx, fy, cx, cy])
                ############################
                end_time = time.time()
                execution_time = (end_time - start_time) * 1000

                rospy.logdebug(f"Model Inference Time: {execution_time}ms")
                # rospy.loginfo(f"{results}")

                person_poses = []
                bbox = []
                kpts = []
                conf = []
                valid_idxs = []
                tracked_ids = []

                if results is not None:

                    person_poses, bbox, kpts, tracked_ids, conf, valid_idxs = results

                    person_poses = person_poses[valid_idxs]

                    self.publish_human_pose(person_poses, depth.header)
                    self.publish_people(person_poses, tracked_ids, depth.header)

                self.publish_debug_img(
                    cv_rgb,
                    bbox,
                    kpts=kpts,
                    valid_idxs=valid_idxs,
                    confidences=conf,
                    tracked_ids=tracked_ids,
                    conf=conf,
                )

            except Exception as e:
                rospy.logerr(f"Error processing images: {e}")
                rospy.logwarn(f"{traceback.format_exc()}")

    def publish_debug_img(
        self, rgb_img, boxes, kpts, valid_idxs, confidences, tracked_ids, conf=0.5
    ):
        if not self.enable_debug:
            return

        color_kpts = (255, 0, 0)
        radius_kpts = 10
        thickness = 2

        # print("confidences", confidences)

        if len(boxes) > 0 and len(kpts) > 0:

            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box

                # cv2.putText(rgb_img, f"{conf * 100:.2f}%" , (x1, y1 + int((y2-y1)/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                cv2.rectangle(rgb_img, (x1, y1), (x2, y2), (0, 255, 0), thickness)

                # if i == target_idx and conf < 600.:
                # if target_idx[i]: #and conf < 0.8:
                if i in valid_idxs:
                    alpha = 0.2
                    overlay = rgb_img.copy()
                    cv2.rectangle(rgb_img, (x1, y1), (x2, y2), (0, 255, 0), -1)
                    cv2.addWeighted(overlay, alpha, rgb_img, 1 - alpha, 0, rgb_img)

                # Just for debugging
                cv2.putText(
                    rgb_img,
                    f"{confidences[i]:.2f}",
                    (x2 - 10, y2 - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2,
                )
                cv2.putText(
                    rgb_img,
                    f"ID: {tracked_ids[i]}",
                    (x1, y1 + int((y2 - y1) / 2)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 0, 0),
                    2,
                )

                kpt = kpts[i]
                for j in range(kpt.shape[0]):
                    u = kpt[j, 0]
                    v = kpt[j, 1]
                    cv2.circle(rgb_img, (u, v), radius_kpts, color_kpts, thickness)

        self.debug_image_pub.publish(
            self.cv_bridge.cv2_to_imgmsg(rgb_img, encoding="bgr8")
        )

    def publish_people(self, poses, tracked_ids, header):
        if not self.enable_people:
            return

        people_msg = People()
        people_msg.header = header

        for i, pose in enumerate(poses):
            if not tracked_ids[i] == 2222:
                continue
            person = Person()
            person.name = "tracked"
            person.position.x = pose[0]
            person.position.y = pose[1]
            person.position.z = pose[2]
            people_msg.people.append(person)

        self.people_pub.publish(people_msg)

    def publish_human_pose(self, poses, header):
        if not self.enable_poses:
            return

        # Publish the pose with covariance
        pose_array_msg = PoseArray()
        pose_array_msg.header = header

        for pose in poses:
            pose_msg = Pose()
            # Set the rotation using the composed quaternion
            pose_msg.position.x = pose[0]
            pose_msg.position.y = pose[1]
            pose_msg.position.z = pose[2]
            # Set the rotation using the composed quaternion
            pose_msg.orientation.x = 0.0
            pose_msg.orientation.y = 0.0
            pose_msg.orientation.z = 0.0
            pose_msg.orientation.w = 1.0
            # Create the pose Array
            pose_array_msg.poses.append(pose_msg)

        self.pose_pub.publish(pose_array_msg)

    def enable(self, req: SetBoolRequest):
        self.enabled = req.data
        rospy.loginfo(f"tracking enabled: {self.enabled}")

    def handle_image_service(self, req):
        """Service callback to Load the Template"""
        with self.lock:
            try:
                cv_image = self.cv_bridge.imgmsg_to_cv2(req.image, "bgr8")

                self.model.template_update(cv_image)

                # Dummy processing: Check if image is non-empty
                success = cv_image is not None and cv_image.size > 0
                rospy.loginfo("Service request processed, success: %s", success)
                self.enabled = True
                # Dummy processing: Check if image is non-empty
                success = cv_image is not None and cv_image.size > 0
                rospy.loginfo("Service request processed, success: %s", success)

                return SetPersonTemplateResponse(success)
            except Exception as e:
                rospy.logerr("Service processing failed: %s", str(e))
                return SetPersonTemplateResponse(False)


if __name__ == "__main__":
    rospy.init_node("robocup_tracker")

    node = CameraProcessingNode()
    rospy.spin()
