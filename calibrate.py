import pickle
import cv2
from points import get_points

def undistort():
    # Test undistortion on an image
    img = cv2.imread('images/test_image.jpg')
    img_size = (img.shape[1], img.shape[0])

    objpoints, imgpoints = get_points()

    # Do camera calibration given object points and image points
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)


    dst = cv2.undistort(img, mtx, dist, None, mtx)
    cv2.imwrite('./test_undist.jpg',dst)

if __name__ == '__main__':
    undistort()