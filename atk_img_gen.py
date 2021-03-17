import numpy as np
import dccp
import cv2
import cvxpy as cvx
import matplotlib.pyplot as plt
from PIL import Image
import sys

class AtkImgGenerator:

    # def __init__(self):

    def GetCoefficients(self, Scale, s_m, s_n, t_m, t_n, IN_max):
        '''
          Output:
            CL (m',m)
            CR (n, n')
        '''

        # Identity matrices size m,m and n,n
        # where m and n are dimensions of the attack image (same as source)
        identity_m = np.identity(s_m)
        identity_n = np.identity(s_n)

        CL = Scale(identity_m * IN_max, t_m, s_m)
        for i in range(0, t_m):
            CL[i, :] = CL[i, :] / CL[i, :].sum()

        CR = Scale(identity_n * IN_max, s_n, t_n)
        for i in range(0, t_n):
            CR[:, i] = CR[:, i] / CR[:, i].sum()

        return CL, CR

    def resizeCV2(self, img, height, width):
        # height, width is not the normal way picture dimensions are described
        # but it's what you get back from .shape with cv2's imread
        # i.e cv2.imread(picture).shape --> (height, width)
        dim = (width, height)
        img = cv2.resize(img, dim)
        return img

    def GetPerturbationsCL(self, src, tgt, CL, e, IN_max, minmax):
        '''
        Input
          src     source image column
          tgt     target image column
          CL      convert matrix left
          e       constraint
          IN_max  maximum pixel value for image format
          minmax  minimise or maximise (for strong or weak attack forms respectively)

        Output
          delta_1 delta_1 column


        This function takes in two columns, src and tgt, and finds an optimal delta_1 column
        subject to the contraints given in the constraints list below.
        '''

        # Dimensions of src and target
        # No need for a second dimension as they are columns
        in_dim = src.shape[0]
        out_dim = tgt.shape[0]

        # CVX Variables
        delta_1 = cvx.Variable(src.shape)
        atk = cvx.Variable(src.shape)

        objective = None
        constraints = [
            atk >= 0,
            atk <= IN_max,
            atk == delta_1 + src,
            cvx.norm(CL @ (atk) - tgt, 'inf') <= e * IN_max
        ]

        if (minmax == 'min'):
            objective = cvx.Minimize(cvx.norm(delta_1, 2))
        elif (minmax == 'max'):
            objective = cvx.Maximize(cvx.norm(delta_1, 2))

        # Convex problem in the form
        # Min/Max (objective)
        # Such that (constraints) are met

        prob = cvx.Problem(objective, constraints)

        # Ensure problem is suitable for disciplined concave convex programming (dccp)
        assert dccp.is_dccp(prob)

        result = prob.solve(method='dccp', verbose=False)

        # Remove singleton dimensions
        delta_1 = np.squeeze(delta_1.value)

        return delta_1

    def GetPerturbationsCR(self, src, tgt, CR, e, IN_max, minmax):
        in_dim = src.shape[0]
        out_dim = tgt.shape[0]

        src = src.reshape(1, -1)
        tgt = tgt.reshape(1, -1)

        delta_1 = cvx.Variable(src.shape)
        atk = cvx.Variable(src.shape)

        objective = None
        constraints = [
            atk >= 0,
            atk <= IN_max,
            atk == delta_1 + src,
            cvx.norm(atk @ CR - tgt, 'inf') <= e * IN_max
        ]

        if (minmax == 'min'):
            objective = cvx.Minimize(cvx.norm(delta_1, 2))
        elif (minmax == 'max'):
            objective = cvx.Maximize(cvx.norm(delta_1, 2))

        prob = cvx.Problem(objective, constraints)
        assert dccp.is_dccp(prob)

        result = prob.solve(method='dccp', verbose=False)
        delta_1 = np.squeeze(delta_1.value)

        return delta_1

    def StrongAttackFormGray(self, Scale, src_img, tgt_img):

        # TODO: Try with another solver (solver=cxv.{solver})

        # Specify image factor
        IN_max = 255

        # Get dimensions of source and target images
        s_m, s_n = src_img.shape  # m,n
        t_m, t_n = tgt_img.shape  # m',n'
        
        # Returns approximations of scaling matrices
        CL, CR = self.GetCoefficients(Scale, s_m, s_n, t_m, t_n, IN_max)

        # Column-wise scaled to target image
        # Intermediate source image
        int_src_img = Scale(src_img, s_m, t_n)
        # Perturbation matrix of vertical attack

        # delta_v = np.zeros(src_img.shape)
        delta_v = np.zeros((s_m, t_n))  # , dtype=np.uintc)
        
        for col in range(0, t_n):
            delta_v[:, col] = self.GetPerturbationsCL(int_src_img[:, col], tgt_img[:, col], CL, 0.01, IN_max, 'min')
            sys.stdout.write("\r Processing column: {0}/{1}".format(col, t_n))
            sys.flush()
        int_atk_img = int_src_img + delta_v

        delta_h = np.zeros((src_img.shape))  # , dtype=np.uintc)

        for row in range(0, s_m):
            delta_h[row, :] = self.GetPerturbationsCR(src_img[row, :], int_atk_img[row, :], CR, 0.01, IN_max, 'min')
            sys.stdout.write("\r Processing row: {0}/{1}".format(row, t_n))
            sys.flush()
            
        atk_img = src_img + delta_h
        return atk_img

    def WeakAttackFormGray(self, Scale, src_height, src_width, tgt_img):

        # TODO: Try with another solver (solver=cxv.{solver})

        # Specify image factor
        IN_max = 255

        # Get dimensions of source and target images
        s_m = src_height
        s_n = src_width
        t_m, t_n = tgt_img.shape  # m',n'

        # Returns approximations of scaling matrices
        CL, CR = self.GetCoefficients(Scale, s_m, s_n, t_m, t_n, IN_max)

        # Column-wise scaled to target image
        # Intermediate source image
        int_src_img = Scale(tgt_img, s_m, t_n)

        # Perturbation matrix of vertical attack
        delta_v = np.zeros((s_m, t_n))  # , dtype=np.uintc)

        for col in range(0, t_n):
            delta_v[:, col] = self.GetPerturbationsCL(int_src_img[:, col], tgt_img[:, col], CL, 0.01, IN_max, 'max')
            sys.stdout.write("\r Processing column: {0}/{1}".format(col, t_n))
            sys.flush()
        src_img = Scale(tgt_img, s_m, s_n)

        int_atk_img = int_src_img + delta_v

        delta_h = np.zeros((s_m, s_n))  # , dtype=np.uintc)

        for row in range(0, s_m):
            delta_h[row, :] = self.GetPerturbationsCR(src_img[row, :], int_atk_img[row, :], CR, 0.01, IN_max, 'max')
            sys.stdout.write("\r Processing row: {0}/{1}".format(row, t_n))
            sys.flush()
            
        atk_img = src_img + delta_h
        return atk_img

    def plot_img(self, img_var, title):
        plt.figure(figsize=(20, 10))
        plt.subplot(2, 2, 1)
        plt.title(title)
        plt.imshow(img_var, cmap='gray', vmin=0, vmax=255)
        plt.show()


class CLI:

    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.atk_gen = AtkImgGenerator()

    def basicCLIWeak(self, tgt_img):
        img_path = self.folder_path + tgt_img
        img = cv2.imread(img_path, 0)
        print("This is the target image")
        self.atk_gen.plot_img(img, 'Target Image')
        print("This is the image in its Weak Attack Form")
        print(img.shape)
        wk = self.atk_gen.WeakAttackFormGray(self.atk_gen.resizeCV2, 1231, 1000, img)
        self.atk_gen.plot_img(wk, 'The Attack Image')


cmd = CLI('/mnt/WD_Blue_SATAData/Documents/Schoolwork/comp0055/comp0055_seeing_is_not_believing/img/')
cmd.basicCLIWeak('jeff.jpg')
