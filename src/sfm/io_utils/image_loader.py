import numpy as np
import cv2

from dataclasses import dataclass
from pathlib import Path


""" ImageData contains all relevant information about each image """
@dataclass
class ImageData:
    image_id: int
    name: str
    path: Path
    image: np.ndarray
    gray: np.ndarray


"""
Load in images into ImageData objects to get the pipeline started
Input: 
    image_dir: str, path to directory containing images
Output: 
    images: list[ImageData]
"""
def load_images(path: str) -> list[ImageData]:

    # Valid exts: jpg, jpeg, png
    exts = {".jpg", ".jpeg", ".png"} 
    paths = sorted([p for p in Path(path).iterdir() if p.suffix.lower() in exts])

    images = []

    for i, path in enumerate(paths):
        color = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if color is None:
            raise ValueError(f"Failed to load image: {path}")
        gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)

        temp = ImageData(i, path.name, path, color, gray)
        images.append(temp)
    
    return images
        
