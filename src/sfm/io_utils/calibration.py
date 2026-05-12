from pathlib import Path
import numpy as np
import yaml

"""
Load in the camera calibration matrix K
Input:
    config_path: str | Path, path to the yaml file containing the calibration matrix
Output:
    K: np.ndarray, camera calibration matrix
"""
def load_camera_matrix(config_path: str | Path) -> np.ndarray:

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    K = np.array(config["camera"]["K"], dtype=np.float64)

    if K.shape != (3,3):
        raise ValueError("Camera calibration matrix K must be 3x3.")
    
    if abs(K[2,2]) < 1e-12:
        raise ValueError("Invalid camera matrix: K[2,2] cannot be zero.")
    
    K = K / K[2,2]

    return K