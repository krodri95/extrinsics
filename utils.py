import math
import numpy as np
import cv2

template = "# calibration, positions in millimeters, rotations in degrees\n# Position x\n{Tx}\n# Position y\n{Ty}\n# Position z\n{Tz}\n# Rotation x (roll)\n{roll}\n# Rotation y (pitch)\n{pitch}\n# Rotation z (yaw)\n{yaw}"

def isClose(x, y, rtol=1.e-5, atol=1.e-8):
    return abs(x-y) <= atol + rtol * abs(y)


def RM2EA(R, units):
    """Convert rotation matrix to Euler angles

    Args:   
        R (np.ndarray): Rotation matrix
        units (str): 'degrees' or 'radians'

    Returns:
        Euler angles

    Example:
        roll, pitch, yaw = RM2EA(R, 'degrees')
        print("roll:  {}\npitch: {}\nyaw:   {}".format(roll, pitch, yaw))
    
    """

    phi = 0.0
    if isClose(R[2,0],-1.0):
        theta = math.pi/2.0
        psi = math.atan2(R[0,1],R[0,2])
    elif isClose(R[2,0],1.0):
        theta = -math.pi/2.0
        psi = math.atan2(-R[0,1],-R[0,2])
    else:
        theta = -math.asin(R[2,0])
        cos_theta = math.cos(theta)
        psi = math.atan2(R[2,1]/cos_theta, R[2,2]/cos_theta)
        phi = math.atan2(R[1,0]/cos_theta, R[0,0]/cos_theta)

    if units == 'radians':
        return psi, theta, phi
    elif units == 'degrees':
        psi = psi * 180 / np.pi
        theta = theta * 180 / np.pi
        phi = phi * 180 / np.pi
        return psi, theta, phi
    else:
        assert False,'Euler angles should be in degrees or radians.'


def vis_tag_detections(img, tags, camera_params, tag_size):
    """Annotate image with aprilTag detections.

    Args:
        img (numpy.ndarray): undistored BGR image
        tags (List): tags info
        camera params (tuple): camera matric parameters
        tag size (float): size of individual AprilTags

    """

    fx, fy, cx, cy = camera_params
    K = np.array([fx, 0, cx, 0, fy, cy, 0, 0, 1]).reshape(3, 3)
    dcoeffs = np.zeros(5)

    for tag in tags:

        rvec = tag.pose_R
        tvec = tag.pose_t

        opoints = np.float32([[1,0,0],
                                 [0,1,0]]).reshape(-1,3) * tag_size/2

        ipoints, _ = cv2.projectPoints(opoints, rvec, tvec, K, dcoeffs)
        ipoints = np.round(ipoints).astype(int)

        center = np.round(tag.center).astype(int)
        center = tuple(center.ravel())

        cv2.line(img, center, tuple(ipoints[0].ravel()), (0,0,255), 2)
        cv2.line(img, center, tuple(ipoints[1].ravel()), (0,255,0), 2)

        for idx in range(len(tag.corners)):
            cv2.line(
                img,
                tuple(tag.corners[idx - 1, :].astype(int)),
                tuple(tag.corners[idx, :].astype(int)),
                (255, 0, 0),
            )

        cv2.putText(
            img,
            str(tag.tag_id),
            org=(
                tag.center[0].astype(int)-10,
                tag.center[1].astype(int)+10,
            ),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.5,
            thickness=2,
            color=(0, 255, 255),
        )


def vis_bundle_detections(img, bundles, camera_params, tag_size):
    """Annotate image with aprilTag bundle detections.

    Args:
        img (numpy.ndarray): undistored BGR image
        bundles (List): tags info
        camera params (tuple): camera matric parameters
        tag size (float): size of individual AprilTags
    """

    fx, fy, cx, cy = camera_params
    K = np.array([fx, 0, cx, 0, fy, cy, 0, 0, 1]).reshape(3, 3)
    dcoeffs = np.zeros(5)

    for bundle in bundles:

        rvec = bundle.pose_R
        tvec = bundle.pose_t

        opoints = np.float32([[1,0,0], [0,1,0], [0,0,1]]).reshape(-1,3) * tag_size / 8 * 14

        ipoints, _ = cv2.projectPoints(opoints, rvec, tvec, K, dcoeffs)
        ipoints = np.round(ipoints).astype(int)

        center = np.round(bundle.center).astype(int)
        center = tuple(center.ravel())

        cv2.line(img, center, tuple(ipoints[0].ravel()), (0,0,255), 2)
        cv2.line(img, center, tuple(ipoints[1].ravel()), (0,255,0), 2)
        cv2.line(img, center, tuple(ipoints[2].ravel()), (255,0,0), 2)


def save_extrinsic(R, T, fname):
    """Fill camera extrinsic template and save text file

    Args:
        R (numpy.ndarray): Rotation matrix
        T (numpy.ndarray): Translation matrix
        fname (str): filename
    """

    #Euler angles ZYX rotation
    roll, pitch, yaw = RM2EA(R, 'degrees') #angles in 'degrees' or 'radians'

    object = {
    "Tx" : T[0][0]*1000,
    "Ty" : T[1][0]*1000,
    "Tz" : T[2][0]*1000,
    "roll" : roll,
    "pitch" : pitch,
    "yaw" : yaw
    }

    file = open(fname, 'w')
    file.write(template.format(**object))
    file.close()