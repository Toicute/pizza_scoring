from typing import Annotated

from fastapi import APIRouter, HTTPException, UploadFile, File, Path
import cv2
import numpy as np

from core.scoring import calculate_shape_score, calculate_size_score
from api.dependencies import engine


PIXEL_PER_UNIT = 24
IMAGINATIVE_MAX_VIEW_SIZE_CM = 50
router = APIRouter()

def read_imagefile(file: bytes) -> np.ndarray:
    """
    Reads an image file and converts it to a numpy array.

    Args:
        file (bytes): The image file in bytes.

    Returns:
        np.ndarray: The image in numpy array format.
    """
    image = np.frombuffer(file, np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    return image


@router.post("/shape")
async def shape_score(file: UploadFile = File(title="Image of pizza")) -> dict:
    """
    Calculates the shape score of the uploaded image using the YOLOv7 segmentation engine.

    Args:
        file (UploadFile): The uploaded image file.

    Returns:
        dict: A dictionary containing the shape score.
    """
    try:
        image = read_imagefile(await file.read())
        score = calculate_shape_score(image, engine['yolov7-seg'])
        return {"shape_score": score}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/size")
async def size_score(file: UploadFile = File(title="Image of pizza"),
                     expected_diameter: Annotated[int, Path(title="Expected diameter of pizza", ge=1, le=IMAGINATIVE_MAX_VIEW_SIZE_CM)] = 30) -> dict:
    """
    Calculates the size score of the uploaded image using the YOLOv7 segmentation engine.

    Args:
        file (UploadFile): The uploaded image file.
        expected_diameter (int): The expected diameter of the object in the image in centimeters.

    Returns:
        dict: A dictionary containing the size score.
    """
    try:
        image = read_imagefile(await file.read())
        score = calculate_size_score(image, engine['yolov7-seg'], expected_diameter=expected_diameter, pixel_per_unit=PIXEL_PER_UNIT)
        return {"size_score": score}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))