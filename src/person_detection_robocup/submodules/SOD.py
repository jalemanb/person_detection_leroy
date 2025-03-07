import logging

from ultralytics import settings

settings.ONLINE = False

from ultralytics import YOLO
import torch.nn.functional as F
import torch, cv2
import time
import numpy as np

from .kpr_onnx import KPR
from .bbox_kalman_filter import BboxKalmanFilter, chi2inv95
from .utils import (
    kp_img_to_kp_bbox,
    rescale_keypoints,
    iou_vectorized,
    bbox_to_xyah,
    xyah_to_bbox,
)


class SOD:

    def __init__(
        self,
        yolo_model_path,
        feature_extracture_model_path,
        feature_extracture_cfg_path,
        tracker_system_path="",
        logger_level=logging.DEBUG,
    ) -> None:

        self.logger = logging.getLogger("SOD")
        formatter = logging.Formatter("{levelname} - {message}", style="{")
        self.logger.setLevel(logger_level)
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tracker_file = tracker_system_path

        # Detection Model
        # self.yolo = YOLO(yolo_model_path, task="segment_pose")  # load a pretrained model (recommended for training)
        self.yolo = YOLO(
            yolo_model_path
        )  # load a pretrained model (recommended for training)

        # ReID System
        self.kpr_reid = KPR(
            feature_extracture_cfg_path,
            feature_extracture_model_path,
            kpt_conf=0.8,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

        self.template = None
        self.template_features = None

        # Intrinsic Camera Parameters
        self.fx, self.fy, self.cx, self.cy = None, None, None, None

        self.erosion_kernel = np.ones(
            (9, 9), np.uint8
        )  # A 3x3 kernel, you can change the size

        self.tracker = BboxKalmanFilter()
        self.man_kf = None
        self.cov_kf = None
        self.border_thr = 10

        self.reid_thr = 0.8

        # Incremental KNN Utils #########################
        self.max_samples = 100
        self.gallery_feats = torch.zeros((self.max_samples, 6, 512)).cuda()
        self.gallery_vis = torch.zeros((self.max_samples, 6)).to(torch.bool).cuda()
        self.gallery_labels = torch.zeros((self.max_samples)).to(torch.bool).cuda()
        self.samples_num = 0
        #################################################

        self.logger.info("Tracker Armed")

        self.reid_mode = True
        self.is_tracking = False

    def to(self, device):
        self.device = device

    def store_feats(self, feats, vis, label):
        """
        Stores feature vectors, visibility masks, and labels into a fixed-size buffer.
        Uses `torch.roll` to implement a circular buffer. If the batch size is larger than `max_samples`,
        it discards excess samples.

        Args:
            feats (torch.Tensor): Feature tensor of shape [batch, 6, 512]
            vis (torch.Tensor): Visibility tensor of shape [batch, 6] (bool)
            label (torch.Tensor): Labels tensor of shape [batch] (bool)
        """
        new_feats_num = feats.shape[0]

        # If batch is larger than max_samples, keep only the most recent samples
        if new_feats_num > self.max_samples:
            feats = feats[-self.max_samples :]  # Keep only last `max_samples` samples
            vis = vis[-self.max_samples :]
            label = label[-self.max_samples :]
            new_feats_num = self.max_samples  # Adjust count

        if self.samples_num < self.max_samples:
            # Append new samples normally
            available_space = self.max_samples - self.samples_num
            num_to_store = min(new_feats_num, available_space)

            self.gallery_feats[self.samples_num : self.samples_num + num_to_store] = (
                feats[:num_to_store]
            )
            self.gallery_vis[self.samples_num : self.samples_num + num_to_store] = vis[
                :num_to_store
            ]
            self.gallery_labels[self.samples_num : self.samples_num + num_to_store] = (
                label[:num_to_store]
            )

            self.samples_num += num_to_store

        else:
            # Use torch.roll to shift old data and insert new samples at the beginning
            self.gallery_feats = torch.roll(
                self.gallery_feats, shifts=-new_feats_num, dims=0
            )
            self.gallery_vis = torch.roll(
                self.gallery_vis, shifts=-new_feats_num, dims=0
            )
            self.gallery_labels = torch.roll(
                self.gallery_labels, shifts=-new_feats_num, dims=0
            )

            # Overwrite the first `new_feats_num` positions with new data
            self.gallery_feats[-new_feats_num:] = feats
            self.gallery_vis[-new_feats_num:] = vis
            self.gallery_labels[-new_feats_num:] = label

    def iknn(self, feats, feats_vis, metric="euclidean", threshold=0.8):
        """
        Compare tensors A[N, 6, 512] and B[batch, 6, 512] part-by-part with visibility filtering.
        Retrieve k smallest distances, classify based on nearest neighbors' labels, and apply a threshold.

        Args:
            A (torch.Tensor): Feature tensor of shape [N, 6, 512]
            B (torch.Tensor): Feature tensor of shape [batch, 6, 512]
            visibility_A (torch.Tensor): Visibility mask for A [N, 6] (bool)
            visibility_B (torch.Tensor): Visibility mask for B [batch, 6] (bool)
            labels_A (torch.Tensor): Boolean labels of shape [N] corresponding to A
            metric (str): Distance metric, "euclidean" or "cosine"
            k (int): Number of smallest distances to retrieve per part.
            threshold (float): Values greater than this threshold will be masked.

        Returns:
            classification (torch.Tensor): Classification tensor of shape [batch, 6] (boolean values)
            binary_mask (torch.Tensor): Binary mask of shape [k, batch, 6] indicating which top-k distances are within threshold
        """

        A = self.gallery_feats[: self.samples_num]
        visibility_A = self.gallery_vis[: self.samples_num]
        labels_A = self.gallery_labels[: self.samples_num]
        B = feats
        visibility_B = feats_vis
        k = int(np.minimum(torch.sum(labels_A).item(), np.sqrt(self.max_samples)))

        N, parts, dim = A.shape
        batch = B.shape[0]

        # Expand A and B to match dimensions for pairwise comparison
        A_expanded = A.unsqueeze(1).expand(N, batch, parts, dim)  # [N, batch, 6, 512]
        B_expanded = B.unsqueeze(0).expand(N, batch, parts, dim)  # [N, batch, 6, 512]

        # Compute similarity/distance based on the selected metric
        if metric == "euclidean":
            distance = torch.norm(
                A_expanded - B_expanded, p=2, dim=-1
            )  # Euclidean distance
        elif metric == "cosine":
            distance = 1 - F.cosine_similarity(
                A_expanded, B_expanded, dim=-1
            )  # Cosine distance
        else:
            raise ValueError("Unsupported metric. Choose 'euclidean' or 'cosine'.")

        # Expand visibility masks for proper masking
        vis_A_expanded = visibility_A.unsqueeze(1).expand(
            N, batch, parts
        )  # [N, batch, 6]
        vis_B_expanded = visibility_B.unsqueeze(0).expand(
            N, batch, parts
        )  # [N, batch, 6]

        # Apply visibility mask: Only compare if both A and B parts are visible
        valid_mask = vis_A_expanded & vis_B_expanded  # Boolean mask
        distance[~valid_mask] = float("inf")  # Ignore invalid comparisons

        # Retrieve the k smallest distances along dim=0 (N dimension)
        top_k_values, top_k_indices = torch.topk(
            distance, k, dim=0, largest=False
        )  # [k, batch, 6]

        # Retrieve the corresponding labels for the k nearest neighbors This is the knn-prediction
        top_k_labels = labels_A[
            top_k_indices
        ]  # Shape [k, batch, 6], labels for nearest N indices

        # Create binary mask based on threshold
        binary_mask = top_k_values <= threshold  # [k, batch, 6]
        binary_mask = (top_k_values <= threshold) | (top_k_values > 10)

        # Apply threshold influence: Set labels to zero where distances exceed the threshold
        valid_labels = (
            top_k_labels * binary_mask
        )  # Zero out labels where threshold is exceeded

        # Perform classification by majority vote (sum up valid labels and classify based on majority vote)
        classification = (valid_labels.sum(dim=0) > (k // 2)).to(
            torch.bool
        )  # Shape [batch, 6]

        return classification.T

    def masked_detections(
        self,
        img_rgb,
        img_depth=None,
        detection_class=0,
        size=(128, 384),
        track=False,
        detection_thr=0.5,
    ):

        results = self.detect_mot(
            img_rgb,
            detection_class=detection_class,
            track=track,
            detection_thr=detection_thr,
        )
        # self.logger.info(f"LEN yolox detections: {len(results[0].boxes)}, kps: {len(results[0].keypoints)}")

        if not (len(results[0].boxes) > 0):
            return []

        subimages = []
        person_kpts = []
        total_keypoints = []
        poses = []
        bboxes = []
        track_ids = []
        for (
            result
        ) in (
            results
        ):  # Need to iterate because if batch is longer than one it should iterate more than once
            boxes = result.boxes  # Boxes object
            keypoints = result.keypoints

            for i, (box, kpts) in enumerate(zip(boxes, keypoints.data)):
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                track_id = -1 if box.id is None else box.id.int().cpu().item()
                # Crop the Image
                subimage = cv2.resize(img_rgb[y1:y2, x1:x2], size)
                # Getting Eyes+Torso+knees Keypoints for pose estimation
                torso_kpts = kpts[:, :2].cpu().numpy()[[1, 2, 5, 6, 11, 12, 13, 14], :]
                torso_kpts = (
                    torso_kpts[~np.all(torso_kpts == 0, axis=1)].astype(np.int32) - 1
                )  # Rest one to avoid incorrect pixel corrdinates
                # Getting the Person Central Pose (Based on Torso Keypoints)
                pose = self.get_person_pose(torso_kpts, img_depth)
                # Store all the bounding box detections and subimages in a tensor
                subimages.append(
                    torch.tensor(subimage, dtype=torch.float16)
                    .permute(2, 0, 1)
                    .unsqueeze(0)
                )
                bboxes.append((x1, y1, x2, y2))
                track_ids.append(track_id)
                person_kpts.append(torso_kpts)
                poses.append(pose)
                # Scaling points to be with respect to the bounding box
                kpts_box = kp_img_to_kp_bbox(kpts, (x1, y1, x2, y2))
                kpts_scaled = rescale_keypoints(
                    kpts_box, (x2 - x1, y2 - y1), (size[0], size[1])
                )
                total_keypoints.append(kpts_scaled)

        poses = np.array(poses)
        bboxes = np.array(bboxes)
        track_ids = np.array(track_ids)
        batched_tensor = torch.cat(subimages).to(device=self.device)
        batched_kpts = torch.stack(total_keypoints, dim=0).to(device=self.device)
        return [batched_tensor, batched_kpts, bboxes, person_kpts, poses, track_ids]

    def detect(
        self, img_rgb, img_depth, camera_params=[1.0, 1.0, 1.0, 1.0], detection_class=0
    ):

        # Get Image Dimensions (Assumes noisy message wth varying image size)
        img_h = img_rgb.shape[0]
        img_w = img_rgb.shape[1]

        self.fx, self.fy, self.cx, self.cy = camera_params

        with torch.no_grad():

            # If there is not template initialization then dont return anything
            if self.template is None:
                self.logger.warning("No template provided")
                return None

            total_execution_time = 0  # To accumulate total time

            # Measure time for `masked_detections`
            start_time = time.time()
            detections = self.masked_detections(
                img_rgb,
                img_depth,
                detection_class=detection_class,
                track=False,
                detection_thr=0.5,
            )
            end_time = time.time()
            masked_detections_time = (
                end_time - start_time
            ) * 1000  # Convert to milliseconds
            # print(f"masked_detections execution time: {masked_detections_time:.2f} ms")
            total_execution_time += masked_detections_time

            self.logger.debug(f"DETECTIONS: {len(detections)}")
            # If no detection (No human) then stay on reid mode and return Nothing
            if not (len(detections) > 0):
                self.reid_mode = True
                # self.is_tracking = False
                return None

            # YOLO Detection Results
            detections_imgs, detection_kpts, bboxes, person_kpts, poses, track_ids = (
                detections
            )

            batch_size = detections_imgs.shape[0]

            self.logger.debug(f"DETECTIONS bboxes: {len(bboxes)}")

            # Up to This Point There are Only Yolo Detections #####################################

            if self.reid_mode:  # ReId mode

                self.logger.debug("REID MODE")

                # Measure time for `feature_extraction` - Extract features to all subimages
                start_time = time.time()
                detections_features = self.feature_extraction(
                    detections_imgs=detections_imgs, detection_kpts=detection_kpts
                )
                end_time = time.time()
                feature_extraction_time = (
                    end_time - start_time
                ) * 1000  # Convert to milliseconds
                # self.logger.debug(
                #    f"feature_extraction execution time: {feature_extraction_time:.2f} ms"
                # )
                total_execution_time += feature_extraction_time

                # Measure time for `similarity_check`
                start_time = time.time()
                appearance_dist, part_dist = self.similarity_check(
                    self.template_features, detections_features, self.reid_thr
                )
                end_time = time.time()
                similarity_check_time = (
                    end_time - start_time
                ) * 1000  # Convert to milliseconds
                # self.logger.debug(
                #    f"similarity_check execution time: {similarity_check_time:.2f} ms"
                # )
                total_execution_time += similarity_check_time

                appearance_dist = appearance_dist[0]

                similarity = appearance_dist.tolist()

                classification = self.iknn(
                    detections_features[0], detections_features[1], threshold=0.9
                )
                self.logger.debug(f"self.is_tracking:  {self.is_tracking}")
                if self.is_tracking:

                    self.mean_kf, self.cov_kf = self.tracker.predict(
                        self.mean_kf, self.cov_kf
                    )
                    mb_dist = self.tracker.gating_distance(
                        self.mean_kf, self.cov_kf, bbox_to_xyah(bboxes)
                    )

                    # knn_gate = (torch.sum(classification, dim=0) > 4).cpu().numpy()

                    knn_gate = (
                        (
                            torch.sum(classification & detections_features[1].T, dim=0)
                            >= torch.sum(detections_features[1].T, dim=0) - 1
                        )
                        .cpu()
                        .numpy()
                    )

                    mb_dist = np.array(mb_dist)
                    mb_gate = mb_dist < chi2inv95[4]

                    appearance_dist = np.array(appearance_dist)
                    appearance_gate = appearance_dist < self.reid_thr

                    gate = knn_gate * mb_gate

                    # Get All indices belonging to valid Detections
                    best_idx = np.argwhere(gate == 1).flatten().tolist()

                    if np.sum(gate) == 1:
                        self.mean_kf, self.cov_kf = self.tracker.update(
                            self.mean_kf, self.cov_kf, bbox_to_xyah(bboxes)[best_idx[0]]
                        )

                        bool_tensor = torch.zeros(
                            batch_size, dtype=torch.bool, device="cuda"
                        )
                        bool_tensor[best_idx] = True
                        self.store_feats(
                            detections_features[0], detections_features[1], bool_tensor
                        )

                else:
                    # knn_gate = (torch.sum(classification, dim=0) >= 5).cpu().numpy()
                    knn_gate = (
                        (
                            torch.sum(classification & detections_features[1].T, dim=0)
                            >= torch.sum(detections_features[1].T, dim=0) - 1
                        )
                        .cpu()
                        .numpy()
                    )

                    appearance_dist = np.array(appearance_dist)
                    appearance_gate = appearance_dist < self.reid_thr
                    gate = knn_gate

                    # Get All indices belonging to valid Detections
                    best_idx = np.argwhere(gate == 1).flatten().tolist()

                valid_idxs = best_idx

                # If there is not a valid detection but a box is being tracked (keep prediction until box is out of fov)
                if not np.sum(gate) and self.is_tracking:
                    self.reid_mode = True
                    self.mean_kf, self.cov_kf = self.tracker.predict(
                        self.mean_kf, self.cov_kf
                    )
                    tracked_bbox = xyah_to_bbox(self.mean_kf[:4])[0]
                    # if tracked box is out of FOV then stop tracking and rely purely on visual appearance
                    if tracked_bbox[2] < 0 or tracked_bbox[0] > img_w:
                        self.is_tracking = False
                    self.logger.debug(
                        f"RETURN none: [not np.sum(gate) and self.is_tracking] BBOX: {tracked_bbox}"
                    )
                    return None

                # If there are no valid detection and no box is being tracked
                elif not np.sum(gate) and not self.is_tracking:
                    self.reid_mode = True
                    self.logger.debug(
                        f"RETURN none: [not np.sum(gate) and not self.is_tracking]"
                    )
                    return None

                # If there is only valid detection
                if np.sum(gate) == 1 and not self.is_tracking:
                    # Extra conditions
                    best_match_idx = best_idx[0]
                    target_bbox = bboxes[best_match_idx]

                    self.mean_kf, self.cov_kf = self.tracker.initiate(
                        bbox_to_xyah(target_bbox)[0]
                    )
                    self.is_tracking = True
                    self.reid_mode = False

                # If there is just one valid and there is a track
                elif np.sum(gate) == 1 and self.is_tracking:
                    # Extra conditions
                    best_match_idx = best_idx[0]
                    target_bbox = bboxes[best_match_idx]

                    # Check the bounding boxes are well separated amoung each other (distractor boxes from the target box)
                    distractor_bbox = np.delete(bboxes, best_match_idx, axis=0)
                    ious_to_target = iou_vectorized(target_bbox, distractor_bbox)

                    # Check the target Box is  far from the edge of the image (left or right)
                    if not np.any(ious_to_target > 0):
                        self.reid_mode = False

            else:  # Tracking mode
                self.logger.debug("TRACKING MODE")

                # Track using iou constant acceleration model or ay opencv tracker (KCF)
                self.mean_kf, self.cov_kf = self.tracker.predict(
                    self.mean_kf, self.cov_kf
                )
                # Data association Based on Only Spatial Information

                # Association Based on Mahalanobies Distance
                mb_dist = self.tracker.gating_distance(
                    self.mean_kf, self.cov_kf, bbox_to_xyah(bboxes)
                )
                best_match_idx = np.argmin(mb_dist)

                # Association Based on IOU
                # ious = iou_vectorized(xyah_to_bbox(self.mean_kf[:4])[0], bboxes)
                # best_match_idx = np.argmax(ious)

                target_bbox = bboxes[best_match_idx]

                # If the Association Metric (Mahalanobies Distance) is greater than the gate then return
                if mb_dist[best_match_idx] > chi2inv95[4]:
                    self.reid_mode = True
                    # self.is_tracking = False
                    self.logger.debug(f"mb_dist[best_match_idx] > chi2inv95[4]")
                    return None

                self.mean_kf, self.cov_kf = self.tracker.update(
                    self.mean_kf, self.cov_kf, bbox_to_xyah(bboxes)[best_match_idx]
                )

                # This is to visualize the Mahalanobies Distances
                ################### VISUALIZATION #####################
                # similarity = mb_dist
                similarity = mb_dist

                valid_idxs = [best_match_idx]

                for i in range(len(track_ids)):
                    track_ids[i] = 2222
                ##################################################

                #### IF SPATIAL AMBIGUITY IS PRESENT GO BACK TO ADD APPEARANCE INFORMATION FOR ASSOCIATION ###############

                # Check if bounding boxes are too close to the target
                # if so return nothing and change to reid_mode
                if len(bboxes) > 1:
                    # See one time steps into the futre if there will be an intersection
                    fut_mean, fut_cov = self.tracker.predict(self.mean_kf, self.cov_kf)
                    # fut_mean, fut_cov = self.tracker.predict(fut_mean, fut_cov)
                    fut_target_bbox = xyah_to_bbox(fut_mean[:4])[0]

                    distractor_bbox = np.delete(bboxes, best_match_idx, axis=0)
                    ious_to_target = iou_vectorized(fut_target_bbox, distractor_bbox)
                    if np.any(ious_to_target > 0):
                        self.reid_mode = True

                        for i in range(len(similarity)):
                            similarity[i] = 100.0

                tx1, ty1, tx2, ty2 = target_bbox

                # check if the target bbox is close to the image edges
                # if so return nothing and change to reid_mode
                if tx1 < self.border_thr or tx2 > img_rgb.shape[1] - self.border_thr:
                    self.reid_mode = True
                ###############################################################################################################

                if not self.reid_mode:

                    # Incremental Learning
                    # Add some sort of feature learning/ prototype augmentation, etc
                    # latest_features = self.feature_extraction(detections_imgs=detections_imgs[[best_match_idx]], detection_kpts=detection_kpts[[best_match_idx]])

                    latest_features = self.feature_extraction(
                        detections_imgs=detections_imgs, detection_kpts=detection_kpts
                    )

                    batch_size = latest_features[0].shape[0]

                    ##################################################################################################
                    ##################################################################################################
                    ##################################################################################################
                    ## In this part the Features will be Added to the Gallery for the Reid Mode to do Associations####
                    # When Adding Negative Samples vectorize the implmentation for labelling

                    # Create a tensor of all False values on CUDA
                    bool_tensor = torch.zeros(
                        batch_size, dtype=torch.bool, device="cuda"
                    )
                    bool_tensor[best_match_idx] = True
                    self.store_feats(
                        latest_features[0], latest_features[1], bool_tensor
                    )

                    ##################################################################################################
                    ##################################################################################################
                    ##################################################################################################

            # Return results
            return (poses, bboxes, person_kpts, track_ids, similarity, valid_idxs)

    def similarity_check(self, template_features, detections_features, similarity_thr):

        fq_, vq_ = template_features
        fg_, vg_ = detections_features

        dist, part_dist = self.kpr_reid.compare(fq_, fg_, vq_, vg_)

        return dist, part_dist

    def detect_mot(
        self, img, detection_class, track=False, detection_thr=0.5, verbose=False
    ):
        # Run multiple object detection with a given desired class
        if track:
            return self.yolo.track(
                img,
                persist=True,
                classes=detection_class,
                tracker=self.tracker_file,
                iou=0.2,
                conf=detection_thr,
                verbose=verbose,
            )
        else:
            return self.yolo(
                img,
                classes=detection_class,
                conf=detection_thr,
                verbose=verbose,
            )

    def template_update(self, template):

        detections = self.masked_detections(template, track=False, detection_thr=0.8)

        # Restart the Tracking States (Gallery & Tracking Stage)
        self.gallery_feats = torch.zeros((self.max_samples, 6, 512)).cuda()
        self.gallery_vis = torch.zeros((self.max_samples, 6)).to(torch.bool).cuda()
        self.gallery_labels = torch.zeros((self.max_samples)).to(torch.bool).cuda()
        self.samples_num = 0

        # self.logger.debug("len(detections)", len(detections))

        if len(detections):
            self.template = detections[0]
            self.template_kpts = detections[1]
        else:
            return False

        # self.logger.debug("self.template", self.template.shape)
        # self.logger.debug("self.template_kpts", self.template_kpts.shape)

        self.template_features = self.extract_features(
            self.template, self.template_kpts
        )

        # self.logger.debug("ALOHAWAY")

        # Store First Initial Features on Galery
        self.store_feats(
            self.template_features[0],
            self.template_features[1],
            torch.ones(1).to(torch.bool).cuda(),
        )

        self.reid_mode = True
        self.is_tracking = False
        return True

    def feature_extraction(self, detections_imgs, detection_kpts):
        # Extract features for similarity check
        return self.extract_features(detections_imgs, detection_kpts)

    def get_target_rgb_and_depth(self, rgb_img, depth_img, bbox, seg_mask):
        # Get the desired person subimage both in rgb and depth
        x1, y1, x2, y2 = map(int, bbox.xyxy[0])

        # Ensure the mask is binary (0 or 255)
        binary_mask = (seg_mask > 0).astype(np.uint8) * 255

        # Create an output image that shows only the highlighted pixels
        masked_rgb_img = cv2.bitwise_and(rgb_img, rgb_img, mask=binary_mask)
        masked_depth_img = cv2.bitwise_and(depth_img, depth_img, mask=binary_mask)

        # Return Target Images With no background of the target person for orientation estimation
        return masked_depth_img, masked_rgb_img

    def feature_distance(self, template_features, detections_features, mode="cosine"):

        # Compute Similarity Check
        if mode == "cosine":
            L = F.cosine_similarity(template_features, detections_features, dim=1)
        elif mode == "eucledian":
            L = torch.cdist(
                template_features.to(torch.float32),
                detections_features.to(torch.float32),
                p=2,
            )

        # Return most similar index
        return L

    def get_template_results(self, detections, most_similar_idx, img_size):
        # Get the segmentation mask
        segmentation_mask = detections[0].masks.data[most_similar_idx].to("cpu").numpy()
        # Resize the mask to match the image size
        segmentation_mask = cv2.resize(
            segmentation_mask, img_size, interpolation=cv2.INTER_NEAREST
        )
        # Get the corresponding bounding box
        bbox = detections[0].boxes[most_similar_idx].to("cpu")
        return bbox, segmentation_mask

    def extract_subimages(self, image, results, size=(224, 224)):
        subimages = []
        for result in results:
            boxes = result.boxes  # Boxes object
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                subimage = image[y1:y2, x1:x2]
                subimage = cv2.resize(subimage, size)
                subimages.append(subimage)
        batched_tensor = torch.stack(
            [
                torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)
                for img in subimages
            ]
        )
        return batched_tensor

    def extract_features(self, image, kpt):

        fg_, vg_ = self.kpr_reid.extract(image, kpt, return_heatmaps=False)

        return (fg_, vg_)

    def get_person_pose(
        self, kpts, depth_img
    ):  # 3d person pose estimation wrt the camera reference frame
        # Wrap the person around a 3D bounding box

        if depth_img is None:
            return None

        if (
            kpts.shape[0] < 2
        ):  # Not Enough Detected Keypoints proceed to compute the human pose
            return [0, 0, 0]

        scale = 0.001 if depth_img.dtype == np.uint16 else 1
        cy = scale / self.fy
        cx = scale / self.fx

        u = kpts[:, 0]
        v = kpts[:, 1]

        z = depth_img[v, u]
        x = z * (u - self.cx) * cx
        y = z * (v - self.cy) * cy

        return [x.mean(), y.mean(), z.mean() * scale]

    def yaw_to_quaternion(self, yaw):
        """
        Convert a yaw angle (in radians) to a quaternion.
        Parameters:
        yaw (float): The yaw angle in radians.
        Returns:
        np.ndarray: The quaternion [w, x, y, z].
        """
        half_yaw = yaw / 2.0
        qw = np.cos(half_yaw)
        qx = 0.0
        qy = 0.0
        qz = np.sin(half_yaw)

        return (qx, qy, qz, qw)
