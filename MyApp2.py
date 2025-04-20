import tensorflow as tf
import tflearn
from tflearn.layers.conv import conv_2d,max_pool_2d
from tflearn.layers.core import input_data,dropout,fully_connected
from tflearn.layers.estimator import regression
import numpy as np
from PIL import Image
import cv2
import imutils
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
# global variables
bg = None

finalClassName = ""
currentWord = ""
sentence = ""
def resizeImage(imageName):
    basewidth = 100
    img = Image.open(imageName)
    wpercent = (basewidth/float(img.size[0]))
    hsize = int((float(img.size[1])*float(wpercent)))
    img = img.resize((basewidth,hsize), Image.ANTIALIAS)
    img.save(imageName)

def run_avg(image, aWeight):
    global bg
    # initialize the background
    if bg is None:
        bg = image.copy().astype("float")
        return

    # compute weighted average, accumulate it and update the background
    cv2.accumulateWeighted(image, bg, aWeight)

def segment(image, threshold=25):
    global bg
    # find the absolute difference between background and current frame
    diff = cv2.absdiff(bg.astype("uint8"), image)

    # threshold the diff image so that we get the foreground
    thresholded = cv2.threshold(diff,
                                threshold,
                                255,
                                cv2.THRESH_BINARY)[1]

    # get the contours in the thresholded image
    (cnts,_) = cv2.findContours(thresholded.copy(),
                                    cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)

    # return None, if no contours detected
    if len(cnts) == 0:
        return
    else:
        # based on contour area, get the maximum contour which is the hand
        segmented = max(cnts, key=cv2.contourArea)
        return (thresholded, segmented)
def getClassName(predictedClass):
    className = ""
    if predictedClass == 0:
        className = "A"
    elif predictedClass == 1:
        className = "B"
    elif predictedClass == 2:
        className = "C"
    elif predictedClass == 3:
        className = "D"
    elif predictedClass == 4:
        className = "E"
    elif predictedClass == 5:
        className = "F"
    elif predictedClass == 6:
        className = "G"
    elif predictedClass == 7:
        className = "H"
    elif predictedClass == 8:
        className = "I"
    elif predictedClass == 9:
        className = "J"
    elif predictedClass == 10:
        className = "K"
    elif predictedClass == 11:
        className = "L"
    elif predictedClass == 12:
        className = "M"
    elif predictedClass == 13:
        className = "N"
    elif predictedClass == 14:
        className = "O"
    elif predictedClass == 15:
        className = "P"
    elif predictedClass == 16:
        className = "Q"
    elif predictedClass == 17:
        className = "R"
    elif predictedClass == 18:
        className = "S"
    elif predictedClass == 19:
        className = "T"
    elif predictedClass == 20:
        className = "U"
    elif predictedClass == 21:
        className = "V"
    elif predictedClass == 22:
        className = "W"
    elif predictedClass == 23:
        className = "X"
    elif predictedClass == 24:
        className = "Y"
    elif predictedClass == 25:
        className = "Z"
    elif predictedClass == 26:
        className = "Space"
    elif predictedClass == 27:
    	className = "Backspace"
    elif predictedClass == 28:
        className = "Blank"
    return className
def main():
    global finalClassName
    # initialize weight for running average
    aWeight = 0.5

    # get the reference to the webcam
    camera = cv2.VideoCapture(0)

    # region of interest (ROI) coordinates
    top, right, bottom, left = 70, 350, 285, 590

    # initialize num of frames
    num_frames = 0
    start_recording = False
    cnt=0
    flag = 0
    # keep looping, until interrupted
    dict={}
    for c in range(0,28):
        dict[c]=0
    frame_count=0
    blank_count=0
    while(True):
        #print(num_frames)
        # get the current frame
        (grabbed, frame) = camera.read()
        #print(cnt)
        cnt = cnt+1
        # resize the frame
        frame = imutils.resize(frame, width = 700)

        # flip the frame so that it is not the mirror view
        frame = cv2.flip(frame, 1)
    
        # clone the frame
        clone = frame.copy()
        
        # get the height and width of the frame
        (height, width) = frame.shape[:2]

        # get the ROI
        roi = frame[top:bottom, right:left]

        # convert the roi to grayscale and blur it
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)

        # to get the background, keep looking till a threshold is reached
        # so that our running average model gets calibrated
        if num_frames < 30:
            run_avg(gray, aWeight)
        else:
            # segment the hand region
            hand = segment(gray)
            
            if(flag==0):
                cv2.rectangle(clone, (left, top), (right, bottom), (0,255,0), 2) #green , gesture is expected
                #(450,630)
                cv2.putText(clone,"Gesture Expected",(30, 100),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0, 255, 0),2)
            else:
                cv2.rectangle(clone, (left, top), (right, bottom), (255,0,0), 2) #blue, blank is expected
                cv2.putText(clone,"Blank Expected",(30, 100),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255, 0, 0),2)
                    
            # check whether hand region is segmented 
            if hand is not None:
                # if yes, unpack the thresholded image and
                # segmented region
                
                (thresholded, segmented) = hand

                # draw the segmented region and display the frame
                cv2.drawContours(clone, [segmented + (right, top)], -1, (0, 0, 255))
                if start_recording and flag!=1:
                    cv2.imwrite('Temp.png', thresholded)
                    resizeImage('Temp.png')
                    predictedClass, confidence = getPredictedClass()
                    if(flag==0):
                        dict[predictedClass] += 1
                        frame_count += 1
                        if(frame_count == 180):
                            mx=0
                            mx_val = 0
                            for category in dict:
                                val = dict[category]
                                if(val>mx):
                                    mx = val
                                    mx_val=category
                            finalClassName = getClassName(mx_val)
                            showStatistics(predictedClass, confidence,1)
                            #print(chr(mx_val+65),sep='')
                            for z in range(0,28):
                                dict[z]=0
                            flag=1
                            frame_count = 0 
                    showStatistics(predictedClass, confidence)
                cv2.imshow("Thesholded", thresholded)
            elif (start_recording and flag==0):
                showStatistics(28,1)                    
            elif (start_recording and flag==1):
                showStatistics(28,1)
                blank_count += 1
                if(blank_count==90):
                    flag=0
                    blank_count=0
 
        # draw the segmented hand
        #cv2.rectangle(clone, (left, top), (right, bottom), (0,255,255), 2)

        # increment the number of frames
        num_frames += 1

        # display the frame with segmented hand
        
        cv2.imshow("Video Feed", clone)

        # observe the keypress by the user
        keypress = cv2.waitKey(1) & 0xFF

        # if the user pressed "q", then stop looping
        if keypress == ord("q"):
            break
        
        if keypress == ord("s"):
            print("Recording started")
            start_recording = True
  

def getPredictedClass():
    # Predict
    image = cv2.imread('Temp.png')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    prediction = model.predict([gray_image.reshape(89, 100, 1)])
    return np.argmax(prediction),np.amax(prediction)/np.sum(prediction)
def showStatistics(predictedClass, confidence,flag=0):
    global finalClassName
    global currentWord
    global sentence
    textImage = np.zeros((300,620,3), np.uint8)
    className = getClassName(predictedClass)
    cv2.putText(textImage,"Predicted Character : " + className, 
    (30, 30), 
    cv2.FONT_HERSHEY_SIMPLEX, 
    0.8,
    (77, 77, 255),
    2)

    cv2.putText(textImage,"Confidence : " + str(confidence * 100) + '%', 
    (30, 100), 
    cv2.FONT_HERSHEY_SIMPLEX, 
    0.8,
    (255,0, 0),
    2)
    if(flag==1):
        if(finalClassName == "Blank"):
            currentWord += ""
        elif(finalClassName == "Space"):
            currentWord += " "
            sentence += currentWord
            currentWord = ""
        elif(finalClassName == "Backspace"):
            currentWord = currentWord[:-1]
        else:
            currentWord = currentWord + finalClassName
        
    
    cv2.putText(textImage,"Word : " + currentWord, 
    (30, 170), 
    cv2.FONT_HERSHEY_SIMPLEX, 
    0.8,
    (26, 255, 26),
    2)
    
    cv2.putText(textImage,"Sentence : " + sentence, 
    (30, 240), 
    cv2.FONT_HERSHEY_SIMPLEX, 
    0.8,
    (255, 77, 255),
    2)
    cv2.imshow("Statistics", textImage)
        
# Model defined
tf.reset_default_graph()
convnet=input_data(shape=[None,89,100,1],name='input')
convnet=conv_2d(convnet,32,2,activation='relu')
convnet=max_pool_2d(convnet,2)
convnet=conv_2d(convnet,64,2,activation='relu')
convnet=max_pool_2d(convnet,2)

convnet=conv_2d(convnet,128,2,activation='relu')
convnet=max_pool_2d(convnet,2)

convnet=conv_2d(convnet,256,2,activation='relu')
convnet=max_pool_2d(convnet,2)

convnet=conv_2d(convnet,256,2,activation='relu')
convnet=max_pool_2d(convnet,2)

convnet=conv_2d(convnet,128,2,activation='relu')
convnet=max_pool_2d(convnet,2)

convnet=conv_2d(convnet,64,2,activation='relu')
convnet=max_pool_2d(convnet,2)

convnet=fully_connected(convnet,1000,activation='relu')
convnet=dropout(convnet,0.75)

convnet=fully_connected(convnet,28,activation='softmax')

convnet=regression(convnet,optimizer='adam',learning_rate=0.001,loss='categorical_crossentropy',name='regression')

model=tflearn.DNN(convnet,tensorboard_verbose=0)
print('Loding model...')
# Load Saved Model
model.load("TrainedModel/GestureRecogModelivp14.tfl")
print('Model loaded, App started.')
input('Press Enter key to key to start background determination.....')
print('Started Background Determination')
main()
