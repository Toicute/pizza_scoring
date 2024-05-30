import math
from typing import Optional, Tuple, List

import cv2
import numpy as np

from .engine import Yolov7Segmentation


def resize_image_to_match_mask(image: np.ndarray, mask_shape: Tuple[int, int, int]) -> np.ndarray:
    """
    Resize the image to match the mask shape.

    Args:
        image (np.ndarray): Input image.
        mask_shape (Tuple[int, int, int]): Shape of the mask.

    Returns:
        np.ndarray: Resized image.
    """
    if image.shape != mask_shape:
        return cv2.resize(image, (mask_shape[1], mask_shape[2]))
    
    return image


def apply_binary_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Apply a binary mask to the image.

    Args:
        image (np.ndarray): Input image.
        mask (np.ndarray): Binary mask.

    Returns:
        np.ndarray: Masked image.
    """
    for m in mask:
        image[m] = (255, 255, 255)
        image[~m] = (0, 0, 0)

    return image


def find_largest_contour(contours: List[np.ndarray]) -> Optional[np.ndarray]:
    """
    Find the largest contour.

    Args:
        contours (List[np.ndarray]): List of contours.

    Returns:
        Optional[np.ndarray]: Largest contour or None if no contours are found.
    """
    if contours:
        return max(contours, key=cv2.contourArea)
    return None


def calculate_contour_circularity(contour: np.ndarray) -> float:
    """
    Calculate the circularity of a contour.

    Args:
        contour (np.ndarray): Contour.

    Returns:
        float: Circularity of the contour.
    """
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    circularity = 4 * np.pi * (area / (perimeter ** 2))

    return min(1.0, circularity)


def get_preprocessed_and_masked_image(image: np.ndarray, engine: Yolov7Segmentation) -> np.ndarray:
    """
    Get the preprocessed and masked image.

    Args:
        image (np.ndarray): Input image.
        engine (Yolov7Segmentation): Yolov7Segmentation engine.

    Returns:
        np.ndarray: Preprocessed and masked image.
    """
    pred_mask, _, _ = engine.inference(image)
    preprocessed_image = resize_image_to_match_mask(image, pred_mask.shape)
    masked_image = apply_binary_mask(preprocessed_image, pred_mask)

    return cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)


def calculate_shape_score(image: np.ndarray, engine: Yolov7Segmentation) -> float:
    """
    Calculate the shape score of the image.

    Args:
        image (np.ndarray): Input image.
        engine (Yolov7Segmentation): Yolov7Segmentation engine.

    Returns:
        float: Shape score.
    """
    grayscale_image = get_preprocessed_and_masked_image(image, engine)
    edges = cv2.Canny(grayscale_image, 100, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = find_largest_contour(contours)
    if largest_contour is not None:
        return calculate_contour_circularity(largest_contour)
    
    return 0.0


def calculate_real_pizza_area(pixel_area: float, pixel_per_unit: float) -> float:
    """
    Calculate the real area of the pizza based on the pixel area and pixel per unit.

    Args:
        pixel_area (float): Area in pixels.
        pixel_per_unit (float): Pixels per unit.

    Returns:
        float: Real area of the pizza.
    """
    radius_in_units = math.sqrt(pixel_area / math.pi) / pixel_per_unit

    return math.pi * (radius_in_units ** 2)


def calculate_size_score(image: np.ndarray, engine: Yolov7Segmentation, expected_diameter: float, pixel_per_unit: float) -> float:
    """
    Calculate the size score of the image.

    Args:
        image (np.ndarray): Input image.
        engine (Yolov7Segmentation): Yolov7Segmentation engine.
        expected_diameter (float): Expected diameter of the pizza.
        pixel_per_unit (float): Pixels per unit.

    Returns:
        float: Size score.
    """
    grayscale_image = get_preprocessed_and_masked_image(image, engine)
    edges = cv2.Canny(grayscale_image, 100, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = find_largest_contour(contours)
    if largest_contour is not None:
        pixel_area = cv2.contourArea(largest_contour)
        real_area = calculate_real_pizza_area(pixel_area, pixel_per_unit)
        expected_area = np.pi * (expected_diameter / 2) ** 2
        return min(1.0, real_area / expected_area)
    
    return 0.0