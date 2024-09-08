import numpy as np
import cv2
from collections import namedtuple 

class AprilTagBundles3x3:
    def __init__(self, tag_size=None):
        super().__init__()
        '''
        3x3 bundle : Total 9 aprilTags arranged as follows

        012
        345
        678

        Supports 7 bundles
        '''
        #Grid cell coordinates
        #bottom left, bottom right, top right, top left and center
        tagCoords0 = np.array([(-14.0, 6.0, 0.0),
                               (-6.0, 6.0, 0.0),
                               (-6.0, 14.0, 0.0),
                               (-14.0, 14.0, 0.0),
                               (-10.0, 10.0, 0.0)
                              ])

        tagCoords1 = np.array([(-4.0, 6.0, 0.0),
                               (4.0, 6.0, 0.0),
                               (4.0, 14.0, 0.0),
                               (-4.0, 14.0, 0.0),
                               (0.0, 10.0, 0.0)
                              ])

        tagCoords2 = np.array([(6.0, 6.0, 0.0),
                               (14.0, 6.0, 0.0),
                               (14.0, 14.0, 0.0),
                               (6.0, 14.0, 0.0),
                               (10.0, 10.0, 0.0)
                              ])

        tagCoords3 = np.array([(-14.0, -4.0, 0.0),
                               (-6.0, -4.0, 0.0),
                               (-6.0, 4.0, 0.0),
                               (-14.0, 4.0, 0.0),
                               (-10.0, 0.0, 0.0)
                              ])

        tagCoords4 = np.array([(-4.0, -4.0, 0.0),
                               (4.0, -4.0, 0.0),
                               (4.0, 4.0, 0.0),
                               (-4.0, 4.0, 0.0),
                               (0.0, 0.0, 0.0)
                              ])

        tagCoords5 = np.array([(6.0, -4.0, 0.0),
                               (14.0, -4.0, 0.0),
                               (14.0, 4.0, 0.0),
                               (6.0, 4.0, 0.0),
                               (10.0, 0.0, 0.0)
                              ])

        tagCoords6 = np.array([(-14.0, -14.0, 0.0),
                               (-6.0, -14.0, 0.0),
                               (-6.0, -6.0, 0.0),
                               (-14.0, -6.0, 0.0),
                               (-10.0, -10.0, 0.0)
                              ])

        tagCoords7 = np.array([(-4.0, -14.0, 0.0),
                               (4.0, -14.0, 0.0),
                               (4.0, -6.0, 0.0),
                               (-4.0, -6.0, 0.0),
                               (0.0, -10.0, 0.0)
                              ])

        tagCoords8 = np.array([(6.0, -14.0, 0.0),
                               (14.0, -14.0, 0.0),
                               (14.0, -6.0, 0.0),
                               (6.0, -6.0, 0.0),
                               (10.0, -10.0, 0.0)
                              ])

        self.bundleTagCoords = np.concatenate((tagCoords0,
                                               tagCoords1,
                                               tagCoords2,
                                               tagCoords3,
                                               tagCoords4,
                                               tagCoords5,
                                               tagCoords6,
                                               tagCoords7,
                                               tagCoords8),
                                               axis=0)

        #scale to true size
        self.bundleTagCoords *= (tag_size/8)  #tag size is equal to the size of 8 grid cells

        self.bundle0_tag_ids = {0,1,2,3,4,5,6,7,8}
        self.bundle1_tag_ids = {9,10,11,12,13,14,15,16,17}
        self.bundle2_tag_ids = {18,19,20,21,22,23,24,25,26}
        self.bundle3_tag_ids = {27,28,29,30,31,32,33,34,35}
        self.bundle4_tag_ids = {36,37,38,39,40,41,42,43,44}
        self.bundle5_tag_ids = {45,46,47,48,49,50,51,52,53}
        self.bundle6_tag_ids = {54,55,56,57,58,59,60,61,62}


    def detect_bundles(self, tags_info=[], K=None, D=None):
        """Detect AprilTag bundles from static images.

        Args: 
            tags_info (dictionary): Detected aprilTags info
            K (matrix): Camera intrinsic matrix
            D (list): Image distortion coefficients

        Returns: 
            Info of the detected bundles.

        """
        tag_dict = {}
        for tag in tags_info:
            tag_id = tag.tag_id

            #image coordinates
            #bottom left, bottom right, top right, top left and center
            tagImgPts = np.concatenate((tag.corners, np.vstack(tag.center).reshape((1,2))), axis=0)
            tag_dict[tag_id] = tagImgPts

        tag_ids = set(tag_dict.keys())

        bundles_info = []
        detectedObject = namedtuple('DetectedObject', ['bundle_id', 'center', 'corners', 'pose_R', 'pose_t'])

        #create bundle 0
        if self.bundle0_tag_ids.issubset(tag_ids):
            bundleImgCoords = np.concatenate((tag_dict[0],
                                              tag_dict[1],
                                              tag_dict[2],
                                              tag_dict[3],
                                              tag_dict[4],
                                              tag_dict[5],
                                              tag_dict[6],
                                              tag_dict[7],
                                              tag_dict[8]),
                                              axis=0)

            ret, rotation_vec, T = cv2.solvePnP(self.bundleTagCoords, bundleImgCoords, K, D, flags=cv2.SOLVEPNP_IPPE)
            R, _ = cv2.Rodrigues(rotation_vec)
            d_ob = detectedObject(0, tag_dict[4][4], bundleImgCoords, R, T)
            bundles_info.append(d_ob)

        #create bundle 1
        if self.bundle1_tag_ids.issubset(tag_ids):
            bundleImgCoords = np.concatenate((tag_dict[9],
                                              tag_dict[10],
                                              tag_dict[11],
                                              tag_dict[12],
                                              tag_dict[13],
                                              tag_dict[14],
                                              tag_dict[15],
                                              tag_dict[16],
                                              tag_dict[17]),
                                              axis=0)

            ret, rotation_vec, T = cv2.solvePnP(self.bundleTagCoords, bundleImgCoords, K, D, flags=cv2.SOLVEPNP_IPPE)
            R, _ = cv2.Rodrigues(rotation_vec)
            d_ob = detectedObject(1, tag_dict[13][4], bundleImgCoords, R, T)
            bundles_info.append(d_ob)

        #create bundle 2
        if self.bundle2_tag_ids.issubset(tag_ids):
            bundleImgCoords = np.concatenate((tag_dict[18],
                                              tag_dict[19],
                                              tag_dict[20],
                                              tag_dict[21],
                                              tag_dict[22],
                                              tag_dict[23],
                                              tag_dict[24],
                                              tag_dict[25],
                                              tag_dict[26]),
                                              axis=0)

            ret, rotation_vec, T = cv2.solvePnP(self.bundleTagCoords, bundleImgCoords, K, D, flags=cv2.SOLVEPNP_IPPE)
            R, _ = cv2.Rodrigues(rotation_vec)
            d_ob = detectedObject(2, tag_dict[22][4], bundleImgCoords, R, T)
            bundles_info.append(d_ob)

        #create bundle 3
        if self.bundle3_tag_ids.issubset(tag_ids):
            bundleImgCoords = np.concatenate((tag_dict[27],
                                              tag_dict[28],
                                              tag_dict[29],
                                              tag_dict[30],
                                              tag_dict[31],
                                              tag_dict[32],
                                              tag_dict[33],
                                              tag_dict[34],
                                              tag_dict[35]),
                                              axis=0)

            ret, rotation_vec, T = cv2.solvePnP(self.bundleTagCoords, bundleImgCoords, K, D, flags=cv2.SOLVEPNP_IPPE)
            R, _ = cv2.Rodrigues(rotation_vec)
            d_ob = detectedObject(3, tag_dict[31][4], bundleImgCoords, R, T)
            bundles_info.append(d_ob)

        #create bundle 4
        if self.bundle4_tag_ids.issubset(tag_ids):
            bundleImgCoords = np.concatenate((tag_dict[36],
                                              tag_dict[37],
                                              tag_dict[38],
                                              tag_dict[39],
                                              tag_dict[40],
                                              tag_dict[41],
                                              tag_dict[42],
                                              tag_dict[43],
                                              tag_dict[44]),
                                              axis=0)

            ret, rotation_vec, T = cv2.solvePnP(self.bundleTagCoords, bundleImgCoords, K, D, flags=cv2.SOLVEPNP_IPPE)
            R, _ = cv2.Rodrigues(rotation_vec)
            d_ob = detectedObject(4, tag_dict[40][4], bundleImgCoords, R, T)
            bundles_info.append(d_ob)

        #create bundle 5
        if self.bundle5_tag_ids.issubset(tag_ids):
            bundleImgCoords = np.concatenate((tag_dict[45],
                                              tag_dict[46],
                                              tag_dict[47],
                                              tag_dict[48],
                                              tag_dict[49],
                                              tag_dict[50],
                                              tag_dict[51],
                                              tag_dict[52],
                                              tag_dict[53]),
                                              axis=0)

            ret, rotation_vec, T = cv2.solvePnP(self.bundleTagCoords, bundleImgCoords, K, D, flags=cv2.SOLVEPNP_IPPE)
            R, _ = cv2.Rodrigues(rotation_vec)
            d_ob = detectedObject(5, tag_dict[49][4], bundleImgCoords, R, T)
            bundles_info.append(d_ob)

        #create bundle 6
        if self.bundle6_tag_ids.issubset(tag_ids):
            bundleImgCoords = np.concatenate((tag_dict[54],
                                              tag_dict[55],
                                              tag_dict[56],
                                              tag_dict[57],
                                              tag_dict[58],
                                              tag_dict[59],
                                              tag_dict[60],
                                              tag_dict[61],
                                              tag_dict[62]),
                                              axis=0)

            ret, rotation_vec, T = cv2.solvePnP(self.bundleTagCoords, bundleImgCoords, K, D, flags=cv2.SOLVEPNP_IPPE)
            R, _ = cv2.Rodrigues(rotation_vec)
            d_ob = detectedObject(6, tag_dict[58][4], bundleImgCoords, R, T)
            bundles_info.append(d_ob)

        return bundles_info