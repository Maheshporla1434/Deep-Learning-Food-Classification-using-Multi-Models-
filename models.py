import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import keras

model = keras.models.load_model('./vgg16_group1_model.h5')
labels=['Baked Potato','Crispy Chicken','Donut']
def fun(path):
  image = cv2.imread(path,1)
  print(f'Original Image shape  : {image.shape}')
  resized_image = cv2.resize(image,(256,256))
  print(f'Resized Image shape : {resized_image.shape}')
  input_image = np.expand_dims(resized_image,axis=0)
  print(f'Perfect Model Input : {input_image.shape}')
  result = model.predict(input_image)
  r = labels[np.argmax(result)]
  print(f'Predicted Class was : {r}')
  cv2.putText(image,r,(70,70),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
  cv2.imshow('image', image)
  cv2.waitKey()
  cv2.destroyAllWindows()
fun('C:\\Users\\Mahesh Porla\\Downloads\\foodproject\\testing_dataset\\Group_1\\Baked Potato\\Baked Potato-Train (1).jpeg')