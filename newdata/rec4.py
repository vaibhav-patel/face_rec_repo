# USAGE
# python recognize_video.py --detector face_detection_model \
#	--embedding-model openface_nn4.small2.v1.t7 \
#	--recognizer output/recognizer.pickle \
#	--le output/le.pickle

# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import pickle
import time
import cv2
import os
import numpy as np
import csv
# 9g74vq
knownEmbeddings = np.load('embeddings.npy')
knownNames = None
with open('labels.csv', 'r', newline='') as file:
    reader = csv.reader(file)
    for i in reader:
        knownNames = i

import simpleaudio as sa
neerav = sa.WaveObject.from_wave_file('Neeraav.wav')
smit = sa.WaveObject.from_wave_file('Smeet.wav')
unknown = sa.WaveObject.from_wave_file('Un_known.wav')

sandip = sa.WaveObject.from_wave_file('sandip.wav')
labelsAudio = {'neerav': neerav, 'smit': smit, 'unknown': unknown,  'sandip': sandip}
def playLabel(label):
    play_obj = labelsAudio[label].play()
    play_obj.wait_done()  # Wait until sound has

detector = 'face_detection_model'    
emb = 'openface_nn4.small2.v1.t7'
dataset = 'dataset'
confidenceArg =  0.5


# load our serialized face detector from disk
print("[INFO] loading face detector...")
protoPath = os.path.sep.join([detector, "deploy.prototxt"])
modelPath = os.path.sep.join([detector,
	"res10_300x300_ssd_iter_140000.caffemodel"])
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# load our serialized face embedding model from disk
print("[INFO] loading face recognizer...")
embedder = cv2.dnn.readNetFromTorch(emb)

# load the actual face recognition model along with the label encoder
#recognizer = pickle.loads(open(args["recognizer"], "rb").read())
#le = pickle.loads(open(args["le"], "rb").read())

# initialize the video stream, then allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

# start the FPS throughput estimator
fps = FPS().start()
knownNamesDic = {}
for i in knownNames:
    if(i not in knownNamesDic):
        knownNamesDic[i] = 0
    knownNamesDic[i] += 1
def checkIt(e):
    losses = []
    lossesDic = {}
    for ind,i in enumerate(knownEmbeddings):
        loss = np.sum((e-i)**2)
        losses.append(loss)
        if(knownNames[ind] not in lossesDic):
            lossesDic[knownNames[ind]] = 0
        lossesDic[knownNames[ind]] += loss
    for i in lossesDic:
        lossesDic[i] /= knownNamesDic[i]
    #print(lossesDic, knownNamesDic)
    return np.argmin(losses)
        
frameans = 0
labelsFound = []
# loop over frames from the video file stream
def findAllNames(ar):
    d1 = {}
    for i in ar:
        if(i not in d1):
            d1[i] = 0
        d1[i] += 1
    foundArrNames = []
    for i in d1:
        if(d1[i] > 1):
            foundArrNames.append(i)
    return foundArrNames


while True:
    # grab the frame from the threaded video stream
    frame = vs.read()

    # resize the frame to have a width of 600 pixels (while
    # maintaining the aspect ratio), and then grab the image
    # dimensions
    frame = imutils.resize(frame, width=600)
    (h, w) = frame.shape[:2]

    # construct a blob from the image
    imageBlob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)), 1.0, (300, 300),
        (104.0, 177.0, 123.0), swapRB=False, crop=False)

    # apply OpenCV's deep learning-based face detector to localize
    # faces in the input image
    detector.setInput(imageBlob)
    detections = detector.forward()

    # loop over the detections

    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with
        # the prediction
        confidence = detections[0, 0, i, 2]

        # filter out weak detections
        if confidence > confidenceArg:
            # compute the (x, y)-coordinates of the bounding box for
            # the face
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # extract the face ROI
            face = frame[startY:endY, startX:endX]
            (fH, fW) = face.shape[:2]

            # ensure the face width and height are sufficiently large
            if fW < 20 or fH < 20:
                continue

            # construct a blob for the face ROI, then pass the blob
            # through our face embedding model to obtain the 128-d
            # quantification of the face
            faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
                (96, 96), (0, 0, 0), swapRB=True, crop=False)
            embedder.setInput(faceBlob)
            vec = embedder.forward()

            # perform classification to recognize the face
            j = checkIt(vec)
            name = knownNames[j]
            labelsFound.append(name)

            # draw the bounding box of the face along with the
            # associated probability
            text = "{}".format(name)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(frame, (startX, startY), (endX, endY),
                (0, 0, 255), 2)
            cv2.putText(frame, text, (startX, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

    # update the FPS counter
    fps.update()
    frameans += 1
    if(frameans in [3]):
        print(labelsFound)
        labelsToSpeak = findAllNames(labelsFound)
        for i in labelsToSpeak:
            playLabel(i)
        labelsFound = []
        frameans = 0
        # show the output frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

# stop the timer and display FPS information
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
