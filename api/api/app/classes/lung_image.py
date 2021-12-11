import cv2
import numpy as np
import pandas as pd
from PIL import Image
from io import BytesIO


class Lung_image():


    def __init__(self, uploaded_image):
        # this class will take the input image, turn it to a npumpy array and resize it zo it is the right shape for the auto encoder model.
        self.image_to_np_array(uploaded_image)
        self.image_resize()


    def image_resize(self):
        # reshape the image to a 400x400 image
        self.image = cv2.resize(self.image, (400, 400))
        # resize the image to have 3 dimensions instead of 2 - otherwise the model won't work
        self.image = self.image.reshape(-1, 400, 400)


    def image_to_np_array(self, uploaded_image):
        self.image = np.array(Image.open(BytesIO(uploaded_image))) / 255


    def get_image(self):
        # return the image
        return self.image

