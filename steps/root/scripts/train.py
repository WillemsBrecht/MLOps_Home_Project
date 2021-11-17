
import argparse
import os
import numpy as np
import glob
import cv2 
import random
import joblib
import pandas as pd
os.environ["KERAS_BACKEND"] = "tensorflow"
import tensorflow as tf
import onnxmltools
import tf2onnx
from sklearn.model_selection import train_test_split

from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K

from azureml.core import Workspace, Dataset, Run

# arguments - given during training step 02
parser = argparse.ArgumentParser()
parser.add_argument('--modelname', type=str, dest='modelname', default="model1")
parser.add_argument('--modelversion', type=float, dest='modelversion', default=0.1)
parser.add_argument('--epochs', type=int, dest='epochs', default=20)
parser.add_argument('--batchsize', type=int, dest='batchsize', default=64)
parser.add_argument('--dataset_name', type=str, dest='dataset_name', default='lungs')
args = parser.parse_args()

# data inlezen
dataset_folder = os.path.join(os.getcwd(), 'dataset')
os.makedirs(dataset_folder, exist_ok=True)

run = Run.get_context()
workspace = run.experiment.workspace

dataset = Dataset.get_by_name(workspace, name=args.dataset_name)
dataset.download(target_path=dataset_folder, overwrite=True)

# reading images
Lung_images_path = glob.glob('./dataset/Lung_images/*.png') 
Lung_masks_path = glob.glob('./dataset/Lung_masks/*.png*') 

def read_images(path):
    images = []
    for f in path: 
        img = cv2.imread(f) 
        im_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        resize = cv2.resize(im_rgb, (400,400))
        images.append(resize) 
    return images

X = np.array(read_images(Lung_images_path)) # images
y = np.array(read_images(Lung_masks_path)) # masks

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=20, random_state=42)

print(X_train.shape, y_train.shape, X_test.shape, y_test.shape, sep = '\n')

# normalisatie van de pixel waarden
X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255
y_train = y_train.astype('float32') / 255
y_test = y_test.astype('float32') / 255

# Creating autoencoder model
latent_dim = 64 

class Autoencoder(Model):
  def __init__(self, latent_dim):
    super(Autoencoder, self).__init__()
    self.latent_dim = latent_dim   
    self.encoder = tf.keras.Sequential([
      layers.Flatten(),
      layers.Dense(latent_dim, activation='relu'),
    ])
    self.decoder = tf.keras.Sequential([
      layers.Dense(400*400, activation='sigmoid'),
      layers.Reshape((400, 400))
    ])

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded

  def dice_coef(self, y_true, y_pred, smooth=1):
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    return (2.*intersection + smooth)/(K.sum(K.square(y_true),-1)+ K.sum(K.square(y_pred),-1) + smooth)

  def dice_coef_loss(self, y_true, y_pred):
    return 1-self.dice_coef(y_true, y_pred)

autoencoder = Autoencoder(latent_dim)

autoencoder.compile(optimizer='adam', loss=autoencoder.dice_coef_loss)
history = autoencoder.fit(X_train, y_train, epochs=args.epochs, batch_size=args.batchsize, shuffle=True)

print("Train an autoencoder with batchsize {}; epochs: {}".format(args.batchsize, args.epochs))

# Showing loss
print("loss training: {}".format(history.history['loss']))

# adding logs
run.log('batch size', np.int(args.batchsize))
run.log('epochs', np.int(args.epochs))

os.makedirs('outputs', exist_ok=True)
onnx_model = onnxmltools.convert_keras(autoencoder) 
onnxmltools.utils.save_model(onnx_model, 'outputs/keras_example.onnx')
