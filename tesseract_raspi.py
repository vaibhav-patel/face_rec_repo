#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# USAGE
# python ocr.py --image images/example_01.png 
# python ocr.py --image images/example_02.png  --preprocess blur

# import the necessary packages
from PIL import Image
import pytesseract
import argparse
import cv2
import os
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import pickle
import time
import cv2
import os
import numpy as np
import csv

from gtts import gTTS
import os
# os.system("mpg321 good.mp3")
# initialize the video stream, then allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)
# words = set(nltk.corpus.words.words())

# def getOnlyWords(text2):
#     return " ".join(w for w in nltk.wordpunct_tokenize(text2) if w.lower() in words or not w.isalpha())
def playTTS(text2):
    tts = gTTS(text=text2, lang='en')
    tts.save("speech.mp3")
    os.system("mpg321 speech.mp3")

    

image = 'images/example_02.png'
preprocess = 'thresh'

def preprocessImage(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    cv2.imshow("Image", gray)

    # check to see if we should apply thresholding to preprocess the
    # image
    if preprocess == "thresh":
        gray = cv2.threshold(gray, 0, 255,
            cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    # make a check to see if median blurring should be done to remove
    # noise
    elif preprocess == "blur":
        gray = cv2.medianBlur(gray, 3)

    # write the grayscale image to disk as a temporary file so we can
    # apply OCR to it
    filename = "{}.png".format(os.getpid())
    cv2.imwrite(filename, gray)

    # load the image as a PIL/Pillow image, apply OCR, and then delete
    # the temporary file
    text = pytesseract.image_to_string(Image.open(filename))
    os.remove(filename)
    if(text not in ['']):
        playTTS(text)
        print(text)

    # show the output images
    # cv2.imshow("Image", image)
    #     cv2.imshow("Output", gray)
    #     cv2.waitKey(0)


# In[ ]:


# loop over frames from the video file stream
while True:
    # grab the frame from the threaded video stream
    frame = vs.read()
    preprocessImage(frame)
    # show the output frame
    #cv2.imshow("Frame", frame)
    #key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    #if key == ord("q"):
    #    break


# do a bit of cleanup

vs.stop()


# In[ ]:


# import nltk
# words = set(nltk.corpus.words.words())
# def getOnlyWords(text2):
#     return " ".join(w for w in nltk.wordpunct_tokenize(text2) if w.lower() in words or not w.isalpha())


# In[ ]:


# sent = "Io andiamo to the beach with my amico."
# getOnlyWords(sent)


# In[ ]:




