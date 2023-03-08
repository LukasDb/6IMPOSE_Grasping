import os
import cv2
import numpy as np
from lib.geometry import get_affine_matrix_from_6d_vector, invert_homogeneous
from scipy import optimize



def load_images(directory_path):
    """ from fieldview-client"""
    imgs = np.array(
        [f for f in os.listdir(directory_path) if f.endswith(".png")])
    order = np.argsort([int(p.split('.')[-2].split('/')[-1]) for p in imgs])
    return [os.path.join(directory_path, img) for img in imgs[order]]

def read_chessboards(images, aruco_dict, board):
    """
    Charuco base pose estimation.
    from fieldview-client
    """
    print("POSE ESTIMATION STARTS:")
    allCorners = []
    allIds = []
    decimator = 0
    markers = []
    # SUB PIXEL CORNER DETECTION CRITERION
    criteria = (cv2.TERM_CRITERIA_EPS +
                cv2.TERM_CRITERIA_MAX_ITER, 100, 0.00001)
    gray = None
    for im in images:
        print("=> Processing image {0}".format(im))
        frame = cv2.imread(im)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(
            gray, aruco_dict)

        if len(corners) > 0:
            # SUB PIXEL DETECTION
            for corner in corners:
                cv2.cornerSubPix(gray, corner,
                                winSize=(3, 3),
                                zeroZone=(-1, -1),
                                criteria=criteria)
            markers.append(corners)
            res2 = cv2.aruco.interpolateCornersCharuco(
                corners, ids, gray, board)
            if res2[1] is not None and res2[2] is not None and len(res2[1]) > 3 and decimator % 1 == 0:
                allCorners.append(res2[1])
                allIds.append(res2[2])

        decimator += 1

    imsize = gray.shape
    return allCorners, allIds, imsize, markers

def cv2_calibration(allCorners, allIds, imsize, board):
    """
    Calibrates the camera using the detected corners.
    from fieldview-client
    """
    cameraMatrixInit = np.array([[2500., 0., imsize[1] / 2.],
                                [0., 2500., imsize[0] / 2.],
                                [0., 0., 1.]])

    distCoeffsInit = np.zeros((5, 1))
    flags = (cv2.CALIB_USE_INTRINSIC_GUESS +
            cv2.CALIB_RATIONAL_MODEL + cv2.CALIB_FIX_ASPECT_RATIO)
    # flags = (cv2.CALIB_RATIONAL_MODEL)
    (ret, camera_matrix, distortion_coefficients0,
    rotation_vectors, translation_vectors,
    stdDeviationsIntrinsics, stdDeviationsExtrinsics,
    perViewErrors) = cv2.aruco.calibrateCameraCharucoExtended(
        charucoCorners=allCorners,
        charucoIds=allIds,
        board=board,
        imageSize=imsize,
        cameraMatrix=cameraMatrixInit,
        distCoeffs=distCoeffsInit,
        flags=flags,
        criteria=(cv2.TERM_CRITERIA_EPS & cv2.TERM_CRITERIA_COUNT, 10000, 1e-9))

    return ret, camera_matrix, distortion_coefficients0, rotation_vectors, translation_vectors

def load_arrays(directory_path, pattern):
    pattern = pattern.split('< >')
    files = np.array([f for f in os.listdir(directory_path) if f.endswith(pattern[-1])])
    order_func = lambda name: name.split(pattern[0])[1].split(pattern[1])[0].split('/')[-1] if len(pattern[0]) > 0 else name.split(pattern[1])[0].split('/')[-1]
    order = np.argsort([int(order_func(p)) for p in files])
    return [os.path.join(directory_path, f) for f in files[order]]

def optimize_camera_offeset(camera_poses, robot_poses):
    camera2tool_t = np.zeros((6,))
    camera2tool_t[5] = np.pi # initialize with 180° around z
    marker2wc_t = np.zeros((6,))
    marker2camera_t = [
        invert_homogeneous(get_affine_matrix_from_6d_vector('Rodriguez', x))
        for x in camera_poses
    ]
    tool2wc_t = [
        invert_homogeneous(x) # already homogeneous matrix
        for x in robot_poses
    ]
  

    x0 = np.array([camera2tool_t, marker2wc_t]).reshape(12)
    #ret = optimize.least_squares(res, x0, camera_poses=camera_poses, robot_poses=robot_poses)
    ret = optimize.least_squares(res, x0, kwargs={"marker2camera": marker2camera_t, "tool2wc": tool2wc_t})
    print(ret)
    return ret


def res(x, tool2wc=None, marker2camera=None):
    camera2tool = get_affine_matrix_from_6d_vector('xyz', x[:6])
    marker2wc = get_affine_matrix_from_6d_vector('xyz', x[6:])
    return res_func(marker2camera, tool2wc, camera2tool, marker2wc)

def res_func(marker2camera, tool2wc, camera2tool, marker2wc):
    res = []
    for i in range(len(marker2camera)):
        res += single_res_func(marker2camera[i], tool2wc[i], camera2tool, marker2wc)
    return np.array(res).reshape(16 * len(marker2camera))

def single_res_func(marker2camera, tool2wc, camera2tool, marker2wc):
    res_array = marker2camera @ camera2tool @ tool2wc - marker2wc
    return [res_array.reshape((16,))]