import os
from fastapi import FastAPI
from routers import lung_router as lung
from fastapi_utils.tasks import repeat_every

app = FastAPI()
app.include_router(lung.router)

name_queue = list()


@app.get("/")
async def root():
    return {"Message": "Welcome to our API, please visit /docs for the different routes/endpoints you can use."}

# Checks on startup and every 10 minutes
@app.on_event("startup")
@repeat_every(seconds=60*10)  # repeat every 10 minutes
def delete_images_from_directory():
    # Checks if the image directory is empty or not
    # if it's empty then delete all the files ( in this case images from the directory)
    if len(os.listdir(".//images")) > 0:
        [os.remove(f".//images//{image}") for image in os.listdir(".//images")]