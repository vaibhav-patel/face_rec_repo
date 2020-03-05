#!/usr/bin/env python
# coding: utf-8

# ## __Objective:__ Create a multiclass image classifier
# 
# ## __Purpose:__ Can be used to classify  species of animal
# 
# ### Use transfer learning and vgg16 model

# ### importing necessary libraries

# In[ ]:





# In[1]:



import numpy as np
import itertools
import keras
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img 
from keras.models import Sequential 
from keras import optimizers
from keras.preprocessing import image
from keras.layers import Dropout, Flatten, Dense  
from keras import applications  
from keras.utils.np_utils import to_categorical  
import math  
import datetime
import time


# Loading up our image datasets

# In[2]:


#Default dimensions we found online
img_width, img_height = 224, 224  
   
#Create a bottleneck file
top_model_weights_path = 'bottleneck_fc_model.h5' 

num_classes = 7


# In[3]:



import simpleaudio as sa
cat = sa.WaveObject.from_wave_file('cnnTest/cat.wav')
dog = sa.WaveObject.from_wave_file('cnnTest/dog.wav')
elephant = sa.WaveObject.from_wave_file('cnnTest/elephant.wav')
horse = sa.WaveObject.from_wave_file('cnnTest/horse.wav')
sheep = sa.WaveObject.from_wave_file('cnnTest/sheep.wav')
tree = sa.WaveObject.from_wave_file('cnnTest/tree.wav')
cow = sa.WaveObject.from_wave_file('cnnTest/cow.wav')

labelsAudio = {'cat': cat, 'dog': dog, 'elephant': elephant, 'horse': horse, 'sheep': sheep, 'tree': tree, 'cow': cow }
def playLabel(label):
    play_obj = labelsAudio[label].play()
    play_obj.wait_done()  # Wait until sound has


# In[4]:


#Loading vgc16 model
vgg16 = applications.VGG16(include_top=False, weights='imagenet') 


# In[5]:


from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import imutils
import time
import cv2
import os
import numpy as np
import csv
from keras.models import load_model
import argparse
import pickle
import cv2


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# # Training of model

# In[6]:


#This is the best model we found. For additional models, check out I_notebook.ipynb
start = datetime.datetime.now()
model = Sequential()  
model.add(Flatten(input_shape=[7,7,512]))  
model.add(Dense(100, activation=keras.layers.LeakyReLU(alpha=0.3)))  
model.add(Dropout(0.5))  
model.add(Dense(50, activation=keras.layers.LeakyReLU(alpha=0.3)))  
model.add(Dropout(0.3)) 
model.add(Dense(num_classes, activation='softmax'))  

model.load_weights(top_model_weights_path)  




end= datetime.datetime.now()
elapsed= end-start
print ('Time: ', elapsed)


# In[7]:


#Model summary
model.summary()


# ## Testing images on model

# In[8]:


def read_image(image2):
    print("[INFO] loading and preprocessing image...")  
    image = img_to_array(image2)  
    image = np.expand_dims(image, axis=0)
    image /= 255.
    return image


# In[9]:


def test_single_image(image):
    animals = ['cat', 'cow', 'dog', 'elephant', 'horse', 'sheep', 'tree']
    images = read_image(image)
    bt_prediction = vgg16.predict(images)  
    preds = model.predict_proba(bt_prediction)
    for idx, animal, x in zip(range(0,7), animals , preds[0]):
        print("ID: {}, Label: {} {}%".format(idx, animal, round(x*100,2) ))
    print('Final Decision:')
    
    class_predicted = model.predict_classes(bt_prediction)
    classPred = animals[int(class_predicted)]
    playLabel(classPred)
    


# In[10]:



print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)
# start the FPS throughput estimator


# loop over frames from the video file stream
while True:
    # grab the frame from the threaded video stream
    frame = vs.read()
    time.sleep(1)
    v1 = cv2.resize(frame, (224,224))
    test_single_image(v1)
    cv2.imshow("Frame", v1)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break


# In[ ]:


v1


# In[ ]:




