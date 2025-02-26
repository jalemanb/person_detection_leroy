#!/usr/bin/env python3

import rospy
import message_filters
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Pose, PoseArray
from cv_bridge import CvBridge
from person_detection_robocup.srv import TemplateImage, TemplateImageResponse
import numpy as np
import cv2
import os, time
import torch
from submodules.SOD import SOD
import rospkg

class CameraProcessingNode:
    def __init__(self):
        rospy.init_node("camera_processing_node")

        # Subscribing to topics
        image_sub = message_filters.Subscriber("/camera/camera/color/image_raw", Image)
        depth_sub = message_filters.Subscriber("/camera/camera/aligned_depth_to_color/image_raw", Image)
        info_sub = message_filters.Subscriber("/camera/camera/color/camera_info", CameraInfo)

        self.cv_bridge = CvBridge()

        # Synchronize topics
        ts = message_filters.TimeSynchronizer([image_sub, depth_sub, info_sub], 10)
        ts.registerCallback(self.callback)

        # Publisher for PoseArray
        self.pose_pub = rospy.Publisher("/detected_poses", PoseArray, queue_size=10)

        # Publisher for debug img
        self.debug_image_pub = rospy.Publisher("/debug_img", Image, queue_size=10)

        # Service to process images
        self.service = rospy.Service("template_image_upload_service", TemplateImage, self.handle_image_service)
        
        # Single Person Detection model
        # Setting up Available CUDA device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        rospack = rospkg.RosPack()
        package_path = rospack.get_path('person_detection_robocup')

        # Setting up model paths (YOLO for object detection and segmentation, and orientation estimation model)
        # yolo_path = os.path.join(pkg_shared_dir, 'models', 'yolov8n-segpose.engine')
        yolo_path = os.path.join(package_path, 'models', 'yolov8n-pose.engine')
        feature_extracture_model_path = os.path.join(package_path, 'models', 'kpr_reid_shape_inferred.onnx')
        feature_extracture_cfg_path = os.path.join(package_path, 'models', 'kpr_market_test.yaml')

        # Setting up Detection Pipeline
        self.model = SOD(yolo_path, feature_extracture_model_path, feature_extracture_cfg_path)
        self.model.to(device)
        rospy.loginfo('Deep Learning Model Armed')

        # Initialize the template
        # template_img_path = os.path.join(package_path, 'templates', 'temp_template.png')
        # self.template_img = cv2.imread(template_img_path)
        # self.model.template_update(self.template_img)

        # Warmup inference (GPU can be slow in the first inference)
        self.model.detect(img_rgb = np.ones((480, 640, 3), dtype=np.uint8), img_depth = np.ones((480, 640), dtype=np.uint16))
        rospy.loginfo('Warmup Inference Executed')

        rospy.loginfo("Camera Processing Node Ready")
        rospy.spin()

    def callback(self, image, depth, camera_info):
        """ Callback when all topics are received """
        try:
            cv_rgb = self.cv_bridge.imgmsg_to_cv2(image, "bgr8")
            cv_depth = self.cv_bridge.imgmsg_to_cv2(depth, "passthrough")  # Assuming depth is float32

            rospy.loginfo("Received synchronized image and depth.")


            fx = camera_info.K[0]
            fy = camera_info.K[4]
            cx = camera_info.K[2]
            cy = camera_info.K[5]

            start_time = time.time()
            ############################
            results = self.model.detect(cv_rgb, cv_depth, [fx, fy, cx, cy])
            ############################
            end_time = time.time()
            execution_time = (end_time - start_time) * 1000
                
            rospy.loginfo(f"Model Inference Time: {execution_time} ms")

            person_poses = []
            bbox = []
            kpts = []
            conf = []
            valid_idxs = []
            tracked_ids = []

            if results is not None:
                self.draw_box = True

                person_poses, bbox, kpts, tracked_ids, conf, valid_idxs = results

                person_poses = person_poses[valid_idxs]

                self.publish_human_pose(person_poses, camera_info.header.frame_id)

            self.publish_debug_img(cv_rgb, bbox, kpts = kpts, valid_idxs = valid_idxs, confidences = conf,  tracked_ids = tracked_ids, conf = conf)

        except Exception as e:
            rospy.logerr("Error processing images: %s", str(e))


    def publish_debug_img(self, rgb_img, boxes, kpts, valid_idxs, confidences, tracked_ids, conf = 0.5):
        color_kpts = (255, 0, 0) 
        radius_kpts = 10
        thickness = 2

        print("confidences", confidences)

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
                cv2.putText(rgb_img, f"{confidences[i]:.2f}" , (x2-10, y2-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.putText(rgb_img, f"ID: {tracked_ids[i]}" , (x1, y1 + int((y2-y1)/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            
                kpt = kpts[i]
                for j in range(kpt.shape[0]):
                    u = kpt[j, 0]
                    v = kpt[j, 1]
                    cv2.circle(rgb_img, (u, v), radius_kpts, color_kpts, thickness)

        self.debug_image_pub.publish(self.cv_bridge.cv2_to_imgmsg(rgb_img, encoding="bgr8"))
        
    def publish_human_pose(self, poses, frame_id):

        # Publish the pose with covariance
        pose_array_msg = PoseArray()
        pose_array_msg.header.stamp = rospy.Time.now()
        pose_array_msg.header.frame_id = frame_id

        for pose in poses:
            pose_msg = Pose()
            # Set the rotation using the composed quaternion
            pose_msg.position.x = pose[0]
            pose_msg.position.y = pose[1]
            pose_msg.position.z = 0.
            # Set the rotation using the composed quaternion
            pose_msg.orientation.x = 0.
            pose_msg.orientation.y = 0.
            pose_msg.orientation.z = 0.
            pose_msg.orientation.w = 1.
            # Create the pose Array
            pose_array_msg.poses.append(pose_msg)

        self.pose_pub.publish(pose_array_msg)

    def handle_image_service(self, req):
        """ Service callback to Load the Template"""
        try:
            cv_image = self.cv_bridge.imgmsg_to_cv2(req.image, "bgr8")

            self.model.template_update(cv_image)

            # Dummy processing: Check if image is non-empty
            success = cv_image is not None and cv_image.size > 0
            rospy.loginfo("Service request processed, success: %s", success)

            return TemplateImageResponse(success)
        except Exception as e:
            rospy.logerr("Service processing failed: %s", str(e))
            return TemplateImageResponse(False)

if __name__ == "__main__":
    CameraProcessingNode()
