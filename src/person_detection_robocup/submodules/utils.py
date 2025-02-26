import numpy as np
import torch

def kp_img_to_kp_bbox(kp_xyc_img, bbox_xyxy):
    """
    Convert keypoints in image coordinates to bounding box coordinates and filter out keypoints 
    that are outside the bounding box.
    
    Args:
        kp_xyc_img (np.ndarray): Keypoints in image coordinates, shape (K, 3) where columns are (x, y, c).
        bbox_xyxy (np.ndarray): Bounding box, shape (4,) as [x1, y1, x2, y2].
    
    Returns:
        np.ndarray: Keypoints in bounding box coordinates, shape (K, 3), with invalid keypoints set to 0.
    """
    # Unpack the bounding box coordinates
    x1, y1, x2, y2 = bbox_xyxy
    
    # Calculate the width and height of the bounding box
    w = x2 - x1
    h = y2 - y1
    
    # Create a copy of the keypoints to work on
    kp_xyc_bbox = kp_xyc_img.clone()
    
    # Transform keypoints to bounding box-relative coordinates
    kp_xyc_bbox[..., 0] = kp_xyc_img[..., 0] - x1
    kp_xyc_bbox[..., 1] = kp_xyc_img[..., 1] - y1
    
    # Filter out keypoints outside the bounding box or with confidence == 0
    invalid_mask = (
        (kp_xyc_img[..., 2] == 0) |    # Keypoint has 0 confidence
        (kp_xyc_bbox[..., 0] < 0) |   # X-coordinate is out of bounds (left)
        (kp_xyc_bbox[..., 0] >= w) |  # X-coordinate is out of bounds (right)
        (kp_xyc_bbox[..., 1] < 0) |   # Y-coordinate is out of bounds (top)
        (kp_xyc_bbox[..., 1] >= h)    # Y-coordinate is out of bounds (bottom)
    )
    
    # Set invalid keypoints to zero
    kp_xyc_bbox[invalid_mask] = 0
    
    return kp_xyc_bbox

def rescale_keypoints(rf_keypoints, size, new_size):
    """
    Rescale keypoints to new size.
    Args:
        rf_keypoints (np.ndarray): keypoints in relative coordinates, shape (K, 2)
        size (tuple): original size, (w, h)
        new_size (tuple): new size, (w, h)
    Returns:
        rf_keypoints (np.ndarray): rescaled keypoints in relative coordinates, shape (K, 2)
    """
    w, h = size
    new_w, new_h = new_size
    rf_keypoints = rf_keypoints.clone()
    rf_keypoints[..., 0] = rf_keypoints[..., 0] * new_w / w
    rf_keypoints[..., 1] = rf_keypoints[..., 1] * new_h / h

    assert ((rf_keypoints[..., 0] >= 0) & (rf_keypoints[..., 0] <= new_w)).all()
    assert ((rf_keypoints[..., 1] >= 0) & (rf_keypoints[..., 1] <= new_h)).all()

    return rf_keypoints

def get_indices_and_values_as_lists_torch(tensor, threshold, less_than = True):
    """
    Get the flattened indices and values of elements in a tensor smaller than a given threshold as Python lists.
    
    Args:
        tensor (torch.Tensor): The input tensor of any shape.
        threshold (float): The threshold value.
    
    Returns:
        list, list: Flattened indices and values as Python lists.
    """
    if less_than:
        # Create a boolean mask for elements smaller than the threshold
        mask = tensor < threshold
    else:
        mask = tensor > threshold
    
    # Get the flattened indices of the elements that match the condition
    valid_indices = torch.nonzero(mask.flatten(), as_tuple=False).squeeze(1)
    
    # Extract the corresponding values
    values = tensor[mask]
    
    # Convert indices and values to Python lists
    indices_list = valid_indices.tolist()
    values_list = values.tolist()
    
    return indices_list, values_list

def get_indices_and_values_as_lists_np(array, threshold, less_than=True):
    """
    Get the flattened indices and values of elements in a NumPy array smaller (or greater) than a given threshold as Python lists.
    
    Args:
        array (np.ndarray): The input array of any shape.
        threshold (float): The threshold value.
        less_than (bool): If True, look for values less than the threshold. Otherwise, greater than the threshold.
    
    Returns:
        list, list: Flattened indices and values as Python lists.
    """
    if less_than:
        # Create a boolean mask for elements smaller than the threshold
        mask = array < threshold
    else:
        mask = array > threshold
    
    # Get the flattened indices of the elements that match the condition
    valid_indices = np.flatnonzero(mask)
    
    # Extract the corresponding values
    values = array[mask]
    
    # Convert indices and values to Python lists
    indices_list = valid_indices.tolist()
    values_list = values.tolist()
    
    return indices_list, values_list

def iou_vectorized(box, boxes):
    """
    Compute the Intersection over Union (IoU) between one box and multiple boxes.

    Parameters:
    box (numpy.ndarray or list or tuple): A single box in the format [x1, y1, x2, y2].
    boxes (numpy.ndarray or list or tuple): Multiple boxes in the format [[x1, y1, x2, y2], ...].

    Returns:
    numpy.ndarray: Array of IoU values between the input box and each of the boxes.
    """
    # Convert inputs to numpy arrays
    box = np.array(box, dtype=np.float32).reshape(1, 4)
    boxes = np.array(boxes, dtype=np.float32)

    # Validate shapes
    if box.shape != (1, 4):
        raise ValueError("The 'box' parameter must have shape (4,).")
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError("The 'boxes' parameter must have shape (N, 4).")

    # Calculate intersection coordinates
    inter_x1 = np.maximum(box[:, 0], boxes[:, 0])
    inter_y1 = np.maximum(box[:, 1], boxes[:, 1])
    inter_x2 = np.minimum(box[:, 2], boxes[:, 2])
    inter_y2 = np.minimum(box[:, 3], boxes[:, 3])

    # Calculate intersection area
    inter_width = np.maximum(0, inter_x2 - inter_x1)
    inter_height = np.maximum(0, inter_y2 - inter_y1)
    inter_area = inter_width * inter_height

    # Calculate areas of the boxes
    box_area = (box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    # Calculate union area
    union_area = box_area + boxes_area - inter_area

    # Compute IoU
    iou = inter_area / union_area

    return iou


def bbox_to_xyah(bbox):
    """
    Convert bounding box format from [x1, y1, x2, y2] to [x, y, a, h].

    Parameters:
    bbox (list, tuple, or np.ndarray): Bounding box in [x1, y1, x2, y2] format.

    Returns:
    np.ndarray: Bounding box in [x, y, a, h] format.
    """
    # Convert input to numpy array
    bbox = np.array(bbox, dtype=np.float32).reshape(-1, 4)

    # Calculate width and height
    widths = bbox[:, 2] - bbox[:, 0]
    heights = bbox[:, 3] - bbox[:, 1]

    # Calculate aspect ratio
    aspect_ratios = np.where(heights != 0, widths / heights, 0)

    # Calculate center x and y
    center_x = bbox[:, 0] + widths / 2
    center_y = bbox[:, 1] + heights / 2

    # Create the output array
    xyah = np.stack((center_x, center_y, aspect_ratios, heights), axis=1)

    return xyah

def xyah_to_bbox(xyah):
    """
    Convert bounding box format from [x, y, a, h] to [x1, y1, x2, y2].

    Parameters:
    xyah (list, tuple, or np.ndarray): Bounding box in [x, y, a, h] format.

    Returns:
    np.ndarray: Bounding box in [x1, y1, x2, y2] format.
    """
    # Convert input to numpy array
    xyah = np.array(xyah, dtype=np.float32).reshape(-1, 4)

    # Extract components
    center_x = xyah[:, 0]
    center_y = xyah[:, 1]
    a = xyah[:, 2]
    h = xyah[:, 3]

    # Calculate width
    w = a * h

    # Calculate x1, y1, x2, y2
    x1 = center_x - w / 2
    y1 = center_y - h / 2
    x2 = center_x + w / 2
    y2 = center_y + h / 2

    # Create the output array
    bbox = np.stack((x1, y1, x2, y2), axis=1)

    return bbox