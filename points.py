import glob
import numpy as np
import cv2

def get_points():
    """
        Extracts object points and image points for camera calibration.
    """
    # Prepare object points
    obgp = np.zeros((6*8, 3), dtype=np.float32)
    obgp[:,:2] = np.mgrid[0:8, 0:6].T.reshape(-1,2)
    
    # Object points and image points
    objpoints = []
    imgpoints = []

    # List of calibration images
    images = glob.glob('images/GO*.jpg')

    for i, filename in enumerate(images):
        image = cv2.imread(filename)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Find the corners
        ret, corners = cv2.findChessboardCorners(gray, (8,6), None)
        if ret == True:
            objpoints.append(obgp)
            imgpoints.append(corners)

    return objpoints, imgpoints


