from frames import dual_frame, approx, init_blocks, stack_blocks
from multiprocessing import Pool
from itertools import starmap
import cv2
import numpy as np
import time

# For webcam input:
cap = cv2.VideoCapture(0)
#cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
#cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
#cap.set(cv2.CAP_PROP_FRAME_WIDTH, 300)
#cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 300)
#IMG_WIDTH = cv2.CAP_PROP_FRAME_WIDTH
#IMG_HEIGHT = cv2.CAP_PROP_FRAME_HEIGHT

success, image = cap.read()
# Vertical and horizontal indices
L=10
N=9
step=10
#s=0.01
s=1
M, Mtilde, window = dual_frame(L=L,N=N,s=s)
block_idxs, vstack_idxs = init_blocks(image.shape, step, L)



#color = False
color = True
while cap.isOpened():
    old_image = image
    success, image = cap.read()
    #image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    #image.flags.writeable = False
    #cv2.imshow('Before', image)
    if color:
        with Pool() as p:
            blocks = np.array( p.starmap( approx, 
                ([image[bx:bx+L, by:by+L,0], M, Mtilde, window] for bx, by in block_idxs)))
            stack_blocks(blocks, image[:,:,0], vstack_idxs, L)

            blocks = np.array( p.starmap( approx, 
                ([image[bx:bx+L, by:by+L,1], M, Mtilde, window] for bx, by in block_idxs)))
            stack_blocks(blocks, image[:,:,1], vstack_idxs, L)

            blocks = np.array( p.starmap( approx, 
                ([image[bx:bx+L, by:by+L,2], M, Mtilde, window] for bx, by in block_idxs)))
            stack_blocks(blocks, image[:,:,2], vstack_idxs, L)
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        with Pool() as p:                                                                          
            blocks = np.array( p.starmap( approx, 
                ([image[bx:bx+L, by:by+L], M, Mtilde, window] for bx, by in block_idxs)))
            stack_blocks(blocks, image, vstack_idxs, L)

    cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()

