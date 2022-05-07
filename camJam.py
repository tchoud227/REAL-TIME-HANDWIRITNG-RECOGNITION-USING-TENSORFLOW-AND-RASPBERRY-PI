#imports
import numpy as np
from skimage import img_as_ubyte #convert float to uint8
from skimage.color import rgb2gray
import cv2
import imutils
import time
from time import sleep
from imutils.video import VideoStream
import tensorflow as tf
from tensorflow import keras
from sense_hat import SenseHat

#settings for sense hat 
sense = SenseHat()
sense.set_rotation(180)
sense.low_light = True

#loaded Model from jupyter script mentioned earlier
model= tf.keras.models.load_model('my_model.h5')
#debug msg
print('Model loaded succesfully')

# initialize the video stream and allow the cammera sensor to warmup
vs = VideoStream(0).start()

#process image to be fit for model and distinguishable with the minist dataset
def PreProcess(orig):
    gray = rgb2gray(orig) #convert original to gray image
    gray_u8 = img_as_ubyte(gray)    # convert gray image to uint8
    (thresh, im_bw) = cv2.threshold(gray_u8, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    resized = cv2.resize(bw,(28,28))
    gray_invert = 255 - resized

    #filters image to get rid of noise
    filtered_image = gray_invert
    for row in range(28):
            for col in range(28):
                if gray_invert[row,col] <180:
                    filtered_image[row,col] = 0
    im_final = filtered_image.reshape(1,28,28,1)

    #prediction
    ans = model.predict(im_final)
    print(ans)
    acc = max(ans[0].tolist())
    # choose the digit with greatest possibility as predicted dight
    ans = ans[0].tolist().index(max(ans[0].tolist()))

    #outputs 
    con = acc*100
    print('CNN predicted digit is: ',ans)
    sense.show_message(str(ans), text_colour=[255, 0, 0])
    sense.show_message(str(con), scroll_speed = 0.05, text_colour=[0, 0, 255])
    sense.clear()  




# infinite loop over the frames from the video stream
while True:
    
    # grab the frame from the  video stream and resize it
    # to have a maximum width of 400 pixels
    frame = vs.read()
    frame = imutils.resize(frame, width=400)
 

     
    # display the frame window
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
 
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break
        # do a bit of cleanup
        cv2.destroyAllWindows()
        vs.stop()
     # if the 't' key was pressed it will save image 
    elif key == ord("t"):
        cv2.imwrite("store.jpg", frame)  
        orig = cv2.imread("store.jpg")
        PreProcess(orig)

#getting rid of windows and clean up      
vs.stop()
cv2.destroyAllWindows()               
            
            


