from cnvrg import Endpoint
from PIL import Image
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
#perform the prediction
from keras.models import load_model
#include custom charts in logging
e = Endpoint()
model = load_model('mnist_model.h5')

def predict(file_path):
    #load the image as grayscale and make it the right size
    x = np.asarray(Image.open(file_path).convert('L').resize(size=(28,28)))
    #compute a bit-wise inversion so black becomes white and vice versa
    x = np.invert(x)
    #convert to a 4D tensor to feed into our model
    x = x.reshape(1,28,28,1)
    x = x.astype('float32')
    x /= 255
    # predict the class
    out = model.predict(x)
    #log the predicted digit
    e.log_metric("digit", np.argmax(out))
    return str(np.argmax(out))