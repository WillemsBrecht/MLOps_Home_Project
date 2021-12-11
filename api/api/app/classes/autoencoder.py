import numpy as np
import onnxruntime as runtime


class Autoencoder:

    def __init__(self):
        self.load_onnx_model()

    
    def load_onnx_model(self):
        # This will load the model from the model folder
        self.model = runtime.InferenceSession(".//model//lung-model.onnx")


    def predict(self, image):
        # Get the input layer name
        input_name = self.model.get_inputs()[0].name
        # Get the output layer name
        label_name = self.model.get_outputs()[0].name
        # returns the prediction made with the lung image as input
        return self.model.run([label_name], {input_name: image.astype(np.float32)})[0]