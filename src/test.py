import config
import numpy as np 
from keras.models import load_model
import sys
import cv2 as cv 



model = load_model(config.model)

def test_prediction(img):
    """
    function takes the image as argument and return
    prediction as output.

    Args:
        sys (string): image_to_pred
    """
    image = cv.imread(img)
    image = image/255           ## normalizing the data
    image = np.expand_dims(image,axis=0)
    prediction = model.predict(image)
    pred = np.argmax(prediction)
    print("Image is {}".format(config.result[pred]))


if __name__ == "__main__":

    img = sys.argv[1]
    print(img)
    try:
        test_prediction(img) 
    except:
        print("Image not valid")
  
      
    


    


    