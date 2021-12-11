import uuid
import numpy as np
import matplotlib.pyplot as plt

class Segmentation:

    def __init__(self, image_to_save):
        self.create_unique_name()
        self.save_image(image_to_save)
    

    def save_image(self, image_to_save):
        # This will save the image with the unique name to the correct directory, so it can be sent back to the user
        plt.imshow(np.squeeze(image_to_save), cmap="gray")
        plt.savefig(f".//images//{self.get_image_name()}")

    
    def create_unique_name(self):
        # assignes an unique name to the image
        self.name = f"{uuid.uuid1()}.jpg"

    
    def get_image_name(self):
        return self.name