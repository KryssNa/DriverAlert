# import cv2
# import numpy as np

# cap = cv2.VideoCapture('http://192.168.43.191:8081/video')

# while(True):
#     ret, frame = cap.read()
#     cv2.imshow('frame',frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break 

import streamlit as st
import cv2

vid = cv2.VideoCapture( 'http://192.168.43.191:8081/video' )

st.title( 'Using Mobile Camera with Streamlit' )
frame_window = st.image( [] )
take_picture_button = st.button( 'Take Picture' )

while True:
    got_frame , frame = vid.read()
    frame = cv2.cvtColor( frame , cv2.COLOR_BGR2RGB )
    if got_frame:
        frame_window.image(frame)

    if take_picture_button:
        # Pass the frame to a model
        # And show the output here...
        break

vid.release()
