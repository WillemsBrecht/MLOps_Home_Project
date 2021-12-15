from classes.lung_image import *
from classes.autoencoder import *
from classes.segmentation_image import *

from fastapi import APIRouter, File
from fastapi.responses import FileResponse


router = APIRouter(
    prefix = "/lungs",
    tags = ["Lungs"],
    responses = {404: {"Lungs": "Not found"}}
)


@router.post("")
async def upload_image_and_predict(input_image: bytes = File(...)):
    """
        input: a x-ray image of a chest - shows the lungs
        output: returns an image where the lungs are segmentated from the image
    """
    # Create an object of the lung_image class which will prepare the input image for the autencoder model - for more informatie see the lung image class
    lung_image = Lung_image(input_image)
    # Create an object of the auto encoder class that will contain the autoencoder
    autoencoder = Autoencoder()
    # Create an object of the Segementation image class which will save the image so it can be sent back through the PI
    segmentation_image = Segmentation(autoencoder.predict(lung_image.get_image()))
    # Return the saved segmentation file to the user
    return FileResponse(f".//images//{segmentation_image.get_image_name()}")