import cv2
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
import tensorflow.keras.preprocessing as pre
import tensorflow as tf
from PIL import Image
from termcolor import cprint
import sys


class Resizers:

    def __init__(self, folder_path, rsz):
        self.folder_path = folder_path
        self.rsz = rsz

    def plot_img(self, img_var, title):
        plt.figure(figsize=(20, 10))
        plt.subplot(2, 2, 1)
        plt.title(title)
        plt.imshow(img_var / 255, cmap='gray', vmin=0, vmax=255)
        plt.show()

    def pt_resize(self, atk_img_path):
        atk_img_path = self.folder_path + atk_img_path
        image = Image.open(atk_img_path)
        scaled_img = F.resize(image, [self.rsz, self.rsz])
        self.plot_atk_and_scale(image, scaled_img)

    def tf_resize(self, atk_img_path):
        atk_img_path = self.folder_path + atk_img_path
        image = Image.open(atk_img_path)
        image = pre.image.img_to_array(image)
        scaled_img = tf.image.resize(image, [self.rsz, self.rsz])
        self.plot_atk_and_scale(image, scaled_img)

    def plot_atk_and_scale(self, atk_img, scale_img):
        self.plot_img(atk_img, 'Attack Image')
        self.plot_img(scale_img, 'Scaled Image')

    def cv2_resize(self, atk_img_path):
        # height, width is not the normal way picture dimensions are described
        # but it's what you get back from .shape with cv2's imread
        # i.e cv2.imread(picture).shape --> (height, width)
        atk_img_path = self.folder_path + atk_img_path
        image = cv2.imread(atk_img_path)
        dim = (self.rsz, self.rsz)
        scaled_img = cv2.resize(image, dim)
        self.plot_atk_and_scale(image, scaled_img)


class CLI:

    def __init__(self, folder_path, resize):
        self.atk_img = None
        self.Rsz = Resizers(
            folder_path, resize)

    def selectResizer(self):
        cprint("Select a resizing method", 'green')
        cprint("Enter 1 for PyTorch", 'blue')
        cprint("Enter 2 for TensorFlow", 'blue')
        cprint("Enter 3 for OpenCV", 'blue')
        arg = input("Select Option Number for Resizer: ")

        if arg == '1':
            self.Rsz.pt_resize(self.atk_img)
        elif arg == '2':
            self.Rsz.tf_resize(self.atk_img)
        elif arg == '3':
            self.Rsz.cv2_resize(self.atk_img)
        else:
            cprint("Invalid input, please try again", 'red')
            self.selectResizer()

    def selectAtkImage(self):
        cprint("Select an attack image", 'green')
        cprint("Enter 1 for Cat", 'blue')
        cprint("Enter 2 for Sheep", 'blue')

        arg = input("Select Option Number for Attack Image: ")

        if arg == '1':
            self.atk_img = 'atk_cat.jpg'
        elif arg == '2':
            self.atk_img = 'atk_sheep.jpg'
        else:
            cprint("Invalid input, please try again", 'red')
            self.atk_img = self.selectAtkImage()

    def startInterface(self):
        self.selectAtkImage()
        self.selectResizer()


CLI("/mnt/WD_Blue_SATAData/Documents/Schoolwork/comp0055/comp0055_seeing_is_not_believing/attack_images/",
    229).startInterface()
