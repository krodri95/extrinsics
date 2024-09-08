import cv2
assert cv2.__version__[0] >= '3', 'The fisheye module requires opencv version >= 3.0.0'
import numpy as np
from tqdm import tqdm


#Fisheye camera model opencv
class FisheyeModel:
    def __init__(self, hcorners=9, vcorners=6):
        super().__init__()

        self.CHECKERBOARD = (hcorners, vcorners)

        # stop the iteration when specified accuracy, epsilon, is reached or specified number of iterations are completed.
        self.subpix_criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)

        self.calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC+cv2.fisheye.CALIB_CHECK_COND+cv2.fisheye.CALIB_FIX_SKEW

        self.objp = np.zeros((1, self.CHECKERBOARD[0]*self.CHECKERBOARD[1], 3), np.float32)
        self.objp[0,:,:2] = np.mgrid[0:self.CHECKERBOARD[0], 0:self.CHECKERBOARD[1]].T.reshape(-1, 2)

        self.DIM = None
        self.K = None
        self.D = None


    def calibrate(self, images=[], process=''):

        objpoints = [] # 3d point in real world space
        imgpoints = [] # 2d points in image plane.

        _img_shape = None
        N_OK = None
        invalid = []

        for fname in tqdm(images, total=len(images), desc=process, colour='white'):
            img = cv2.imread(fname)
            if _img_shape == None:
                _img_shape = img.shape[:2]
            else:
                assert _img_shape == img.shape[:2], "All images must share the same size."    
                
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray, self.CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE)
            
            # If found, add object points, image points (after refining them)
            if ret == True:
                objpoints.append(self.objp)
                cv2.cornerSubPix(gray,corners,(3,3),(-1,-1),self.subpix_criteria)
                cv2.drawChessboardCorners(img, self.CHECKERBOARD, corners, ret)
                imgpoints.append(corners)
                N_OK = len(objpoints)
            else:
                invalid.append(fname.split('/')[-1])

        print('Found {} valid images for calibration'.format(N_OK))
        if len(invalid)!=0:
            print('All grid corners not detected in {} . Skipped these images.'.format(invalid))
            
        DIM=_img_shape[::-1]
        K = np.zeros((3, 3))
        D = np.zeros((4, 1))
        rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
        tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]

        rms, _, _, _, _ = cv2.fisheye.calibrate(objpoints,
                                                imgpoints,
                                                gray.shape[::-1],
                                                K,
                                                D,
                                                rvecs,
                                                tvecs,
                                                self.calibration_flags,
                                                (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6))

        
        #print("DIM=" + str(_img_shape[::-1]))
        #print("K=np.array(" + str(K.tolist()) + ")")
        #print("D=np.array(" + str(D.tolist()) + ")")
        print('\n')

        self.DIM = DIM
        self.K = K
        self.D = D


    def undistort(self, img_path=None, balance=0.0, dim2=None, dim3=None):

        img = cv2.imread(img_path)
        dim1 = img.shape[:2][::-1]  #dim1 is the dimension of input image to un-distort
        assert dim1[0]/dim1[1] == self.DIM[0]/self.DIM[1], "Image to undistort needs to have same aspect ratio as the ones used in calibration"

        if not dim2:
            dim2 = dim1

        if not dim3:
            dim3 = dim1

        scaled_K = self.K * dim1[0] / self.DIM[0]  # The values of K is to scale with image dimension.
        scaled_K[2][2] = 1.0  # Except that K[2][2] is always 1.0

        # This is how scaled_K, dim2 and balance are used to determine the final K used to un-distort image.
        new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(scaled_K, self.D, dim2, np.eye(3), balance=balance)
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(scaled_K, self.D, np.eye(3), new_K, dim3, cv2.CV_16SC2)
        undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

        return undistorted_img, new_K


#Pinhole camera model opencv
class PinholeModel:
    def __init__(self, hcorners=9, vcorners=6):
        super().__init__()

        self.CHECKERBOARD = (hcorners, vcorners)

        # stop the iteration when specified accuracy, epsilon, is reached or specified number of iterations are completed.
        self.subpix_criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)

        self.calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC+cv2.fisheye.CALIB_CHECK_COND+cv2.fisheye.CALIB_FIX_SKEW

        self.objp = np.zeros((1, self.CHECKERBOARD[0]*self.CHECKERBOARD[1], 3), np.float32)
        self.objp[0,:,:2] = np.mgrid[0:self.CHECKERBOARD[0], 0:self.CHECKERBOARD[1]].T.reshape(-1, 2)

        self.DIM = None
        self.K = None
        self.D = None


    def calibrate(self, images=[], process=''):

        objpoints = [] # 3d point in real world space
        imgpoints = [] # 2d points in image plane.

        _img_shape = None
        N_OK = None
        invalid = []

        for fname in tqdm(images, total=len(images), desc=process, colour="white"):
            img = cv2.imread(fname)
            if _img_shape == None:
                _img_shape = img.shape[:2]
            else:
                assert _img_shape == img.shape[:2], "All images must share the same size."    
                
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray, self.CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE)
            
            # If found, add object points, image points (after refining them)
            if ret == True:
                objpoints.append(self.objp)
                cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),self.subpix_criteria)
                cv2.drawChessboardCorners(img, self.CHECKERBOARD, corners, ret)
                imgpoints.append(corners)
                N_OK = len(objpoints)
            else:
                invalid.append(fname.split('/')[-1])

        print('Found {} valid images for calibration'.format(N_OK))
        if len(invalid)!=0:
            print('All grid corners not detected in {} . Skipped these images.'.format(invalid))

        DIM = img.shape[:2][::-1]
        ret, K, D, r_vecs, t_vecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
  
        #print("DIM=" + str(_img_shape[::-1]))
        #print("K=np.array(" + str(K.tolist()) + ")")
        #print("D=np.array(" + str(D.tolist()) + ")")
        print('\n')

        self.DIM = DIM
        self.K = K
        self.D = D


    def undistort(self, img_path=None, balance=1.0):

        img = cv2.imread(img_path)
        dim1 = img.shape[:2][::-1]  #dim1 is the dimension of input image to un-distort
        assert dim1[0]/dim1[1] == self.DIM[0]/self.DIM[1], "Image to undistort needs to have same aspect ratio as the ones used in calibration"

        scaled_K = self.K * dim1[0] / self.DIM[0]  # The values of K is to scale with image dimension.
        scaled_K[2][2] = 1.0  # Except that K[2][2] is always 1.0

        new_K, roi = cv2.getOptimalNewCameraMatrix(scaled_K, self.D, self.DIM, balance, self.DIM)
        mapx, mapy = cv2.initUndistortRectifyMap(scaled_K, self.D, None, new_K, self.DIM, 5)
        undistorted_img = cv2.remap(img, mapx, mapy, interpolation=cv2.INTER_LINEAR)

        return undistorted_img, new_K