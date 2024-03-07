# camera github : https://gist.github.com/cbednarski/8450931

import cv2
import mediapipe as mp
import time
import utils, math
import numpy as np
import os
#import pygame 
#from pygame import mixer 

# variables 
frame_counter = 0
CEF_COUNTER = 0
TOTAL_BLINKS = 0
start_voice = False
counter_right = 0
counter_left = 0
counter_center = 0

TIME_INTERVAL = 5

last_action_time = 0
# constants
CLOSED_EYES_FRAME = 2
FONTS = cv2.FONT_HERSHEY_COMPLEX

"""
# initialize mixer 
mixer.init()
# loading in the voices/sounds 
voice_left = mixer.Sound('Voice/left.wav')
voice_right = mixer.Sound('Voice/Right.wav')
voice_center = mixer.Sound('Voice/center.wav')
"""

# face bounder indices 
FACE_OVAL=[ 10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103,67, 109]

# lips indices for Landmarks
LIPS=[ 61, 146, 91, 181, 84, 17, 314, 405, 321, 375,291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95,185, 40, 39, 37,0 ,267 ,269 ,270 ,409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78 ]
LOWER_LIPS =[61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
UPPER_LIPS=[ 185, 40, 39, 37,0 ,267 ,269 ,270 ,409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78] 
# Left eyes indices 
LEFT_EYE =[ 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398 ]
LEFT_EYEBROW =[ 336, 296, 334, 293, 300, 276, 283, 282, 295, 285 ]

# right eyes indices
RIGHT_EYE=[ 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246 ]  
RIGHT_EYEBROW=[ 70, 63, 105, 66, 107, 55, 65, 52, 53, 46 ]

map_face_mesh = mp.solutions.face_mesh

# camera object 
camera = cv2.VideoCapture(0)
_, frame = camera.read()
img = cv2.resize(frame, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
img_hieght, img_width = img.shape[:2]
print(img_hieght, img_width)



def on_mouse_click(event, x, y, flags, param):
    global TIME_INTERVAL
    if event == cv2.EVENT_LBUTTONDOWN:
        if 20 <= x <= 70 and 20 <= y <= 70 and TIME_INTERVAL < 10:
            TIME_INTERVAL += 1
        elif 20 <= x <= 70 and 120 <= y <= 170 and TIME_INTERVAL > 1:
            TIME_INTERVAL -= 1

# video Recording setup 
#fourcc = cv2.VideoWriter_fourcc(*'XVID')
#out = cv2.VideoWriter('output21.mp4', fourcc, 30.0, (img_width, img_hieght))

# landmark detection function 
def landmarksDetection(img, results, draw=False):
    img_height, img_width= img.shape[:2]
    # list[(x,y), (x,y)....]
    mesh_coord = [(int(point.x * img_width), int(point.y * img_height)) for point in results.multi_face_landmarks[0].landmark]
    if draw :
        [cv2.circle(img, p, 2, (0,255,0), -1) for p in mesh_coord]

    # returning the list of tuples for each landmarks 
    return mesh_coord

# Euclaidean distance 
def euclaideanDistance(point, point1):
    x, y = point
    x1, y1 = point1
    distance = math.sqrt((x1 - x)**2 + (y1 - y)**2)
    return distance

# Blinking Ratio
def blinkRatio(img, landmarks, right_indices, left_indices):
    # Right eyes 
    # horizontal line 
    rh_right = landmarks[right_indices[0]]
    rh_left = landmarks[right_indices[8]]
    # vertical line 
    rv_top = landmarks[right_indices[12]]
    rv_bottom = landmarks[right_indices[4]]
    # draw lines on right eyes 
    # cv2.line(img, rh_right, rh_left, utils.GREEN, 2)
    # cv2.line(img, rv_top, rv_bottom, utils.WHITE, 2)

    # LEFT_EYE 
    # horizontal line 
    lh_right = landmarks[left_indices[0]]
    lh_left = landmarks[left_indices[8]]

    # vertical line 
    lv_top = landmarks[left_indices[12]]
    lv_bottom = landmarks[left_indices[4]]

    rhDistance = euclaideanDistance(rh_right, rh_left)
    rvDistance = euclaideanDistance(rv_top, rv_bottom)

    lvDistance = euclaideanDistance(lv_top, lv_bottom)
    lhDistance = euclaideanDistance(lh_right, lh_left)

    reRatio = rhDistance/rvDistance
    leRatio = lhDistance/lvDistance

    ratio = (reRatio+leRatio) / 2
    return ratio 

# Eyes Extrctor function,
def eyesExtractor(img, right_eye_coords, left_eye_coords):
    # converting color image to  scale image 
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # getting the dimension of image 
    dim = gray.shape

    # creating mask from gray scale dim
    mask = np.zeros(dim, dtype=np.uint8)

    # drawing Eyes Shape on mask with white color 
    cv2.fillPoly(mask, [np.array(right_eye_coords, dtype=np.int32)], 255)
    cv2.fillPoly(mask, [np.array(left_eye_coords, dtype=np.int32)], 255)

    # showing the mask 
    # cv2.imshow('mask', mask)
    
    # draw eyes image on mask, where white shape is 
    eyes = cv2.bitwise_and(gray, gray, mask=mask)
    # change black color to gray other than eys 
    # cv2.imshow('eyes draw', eyes)
    eyes[mask==0]=155
    
    # getting minium and maximum x and y  for right and left eyes 
    # For Right Eye 
    r_max_x = (max(right_eye_coords, key=lambda item: item[0]))[0]
    r_min_x = (min(right_eye_coords, key=lambda item: item[0]))[0]
    r_max_y = (max(right_eye_coords, key=lambda item : item[1]))[1]
    r_min_y = (min(right_eye_coords, key=lambda item: item[1]))[1]

    # For LEFT Eye
    l_max_x = (max(left_eye_coords, key=lambda item: item[0]))[0]
    l_min_x = (min(left_eye_coords, key=lambda item: item[0]))[0]
    l_max_y = (max(left_eye_coords, key=lambda item : item[1]))[1]
    l_min_y = (min(left_eye_coords, key=lambda item: item[1]))[1]

    # croping the eyes from mask 
    cropped_right = eyes[r_min_y: r_max_y, r_min_x: r_max_x]
    cropped_left = eyes[l_min_y: l_max_y, l_min_x: l_max_x]

    # returning the cropped eyes 
    return cropped_right, cropped_left

# Eyes Postion Estimator 
def positionEstimator(cropped_eye):
    # getting height and width of eye 
    h, w =cropped_eye.shape
    
    # remove the noise from images
    gaussain_blur = cv2.GaussianBlur(cropped_eye, (9,9),0)
    median_blur = cv2.medianBlur(gaussain_blur, 3)

    # applying thrsholding to convert binary_image
    ret, threshed_eye = cv2.threshold(median_blur, 130, 255, cv2.THRESH_BINARY)

    # create fixd part for eye with 
    piece = int(w/3) 

    # slicing the eyes into three parts 
    right_piece = threshed_eye[0:h, 0:piece]
    center_piece = threshed_eye[0:h, piece: piece+piece]
    left_piece = threshed_eye[0:h, piece +piece:w]
    
    # calling pixel counter function
    eye_position, color = pixelCounter(right_piece, center_piece, left_piece)

    return eye_position, color 

# creating pixel counter function 
def pixelCounter(first_piece, second_piece, third_piece):
    # counting black pixel in each part 
    right_part = np.sum(first_piece==0)
    center_part = np.sum(second_piece==0)
    left_part = np.sum(third_piece==0)
    # creating list of these values
    eye_parts = [right_part, center_part, left_part]

    # getting the index of max values in the list 
    max_index = eye_parts.index(max(eye_parts))
    pos_eye ='' 
    if max_index==0:
        pos_eye="RIGHT"
        color=[utils.BLACK, utils.GREEN]
    elif max_index==1:
        pos_eye = 'CENTER'
        color = [utils.YELLOW, utils.PINK]
    elif max_index ==2:
        pos_eye = 'LEFT'
        color = [utils.GRAY, utils.YELLOW]
    else:
        pos_eye="Closed"
        color = [utils.GRAY, utils.YELLOW]
    return pos_eye, color


with map_face_mesh.FaceMesh(min_detection_confidence =0.5, min_tracking_confidence=0.5) as face_mesh:

    # starting time here 
    start_time = time.time()

    while True:
        ret, frame = camera.read()
        if not ret:
            break

        frame = cv2.resize(frame, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
        frame_height, frame_width = frame.shape[:2]
        cv2.putText(frame, 'Press "'"S"'" to start detection and take photos', (int(frame_width / 2) - 600, frame_height - 50), FONTS, 1.7, utils.PINK, 2)
        cv2.putText(frame, 'Press "'"Q"'" to quit the app', (int(frame_width / 2) - 300, frame_height - 115), FONTS, 1.7, utils.PINK, 2)
        cv2.imshow('frame', frame)
        key = cv2.waitKey(1)
        if key == ord('q') or key == ord('Q'):
            quit()
        elif key == ord('s') or key == ord('S'):
            cv2.destroyAllWindows()
            break

    # starting Video loop here.
    while True:
        frame_counter += 1 # frame counter
        ret, frame = camera.read() # getting frame from camera 
        if not ret: 
            break # no more frames break
        #  resizing frame
        
        frame = cv2.resize(frame, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
        frame_height, frame_width= frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        shots_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
        results  = face_mesh.process(rgb_frame)
        if results.multi_face_landmarks:
            mesh_coords = landmarksDetection(frame, results, False)
            ratio = blinkRatio(frame, mesh_coords, RIGHT_EYE, LEFT_EYE)
            #cv2.putText(frame, f'ratio {round(ratio, 2)}', (100, 100), FONTS, 1.0, utils.YELLOW, 2)
            #utils.colorBackgroundText(frame,  f'Ratio : {round(ratio,2)}', FONTS, 0.7, (30,100),2, utils.PINK, utils.YELLOW)

            if ratio > 3.3:
                CEF_COUNTER += 1
                #cv2.putText(frame, 'Blink', (200, 50), FONTS, 1.7, utils.PINK, 2)
                
                #utils.colorBackgroundText(frame,  f'Blink', FONTS, 1.7, (int(frame_height/2), 100), 2, utils.YELLOW, pad_x=6, pad_y=6, )

            else:
                current_time = time.time()
                if CEF_COUNTER > CLOSED_EYES_FRAME:
                    TOTAL_BLINKS += 1
                    CEF_COUNTER = 0
                else:
                    if current_time - last_action_time >= TIME_INTERVAL:
                       output_filename = f'capture_{time.time()}.jpg'
                       output_path = os.path.join('img', output_filename)
                       #out = cv2.imwrite(output_path, shots_frame)
                       cv2.putText(frame, 'Photo taken', (200, 50), FONTS, 1.7, utils.PINK, 2)
                       last_action_time = current_time
            #cv2.putText(frame, f'Total Blinks: {TOTAL_BLINKS}', (200, 150), FONTS, 0.7, utils.GREEN, 2)
            #utils.colorBackgroundText(frame,  f'Total Blinks: {TOTAL_BLINKS}', FONTS, 0.7, (30,150),2)
            
            cv2.polylines(frame,  [np.array([mesh_coords[p] for p in LEFT_EYE ], dtype=np.int32)], True, utils.GREEN, 1, cv2.LINE_AA)
            cv2.polylines(frame,  [np.array([mesh_coords[p] for p in RIGHT_EYE ], dtype=np.int32)], True, utils.GREEN, 1, cv2.LINE_AA)

            # Blink Detector Counter Completed
            right_coords = [mesh_coords[p] for p in RIGHT_EYE]
            left_coords = [mesh_coords[p] for p in LEFT_EYE]
            crop_right, crop_left = eyesExtractor(frame, right_coords, left_coords)
            # cv2.imshow('right', crop_right)
            # cv2.imshow('left', crop_left)
            """
            eye_position_right, color = positionEstimator(crop_right)
            utils.colorBackgroundText(frame, f'R: {eye_position_right}', FONTS, 1.0, (40, 220), 2, color[0], color[1], 8, 8)
            eye_position_left, color = positionEstimator(crop_left)
            utils.colorBackgroundText(frame, f'L: {eye_position_left}', FONTS, 1.0, (40, 320), 2, color[0], color[1], 8, 8)
            """
            
            """
            # Starting Voice Indicator 
            if eye_position_right=="RIGHT" and pygame.mixer.get_busy()==0 and counter_right<2:
                # starting counter 
                counter_right+=1
                # resetting counters 
                counter_center=0
                counter_left=0
                # playing voice 
                voice_right.play()


            if eye_position_right=="CENTER" and pygame.mixer.get_busy()==0 and counter_center <2:
                # starting Counter 
                counter_center +=1
                # resetting counters 
                counter_right=0
                counter_left=0
                # playing voice 
                voice_center.play()
            
            if eye_position_right=="LEFT" and pygame.mixer.get_busy()==0 and counter_left<2: 
                counter_left +=1
                # resetting counters 
                counter_center=0
                counter_right=0
                # playing Voice 
                voice_left.play()
"""


        # calculating  frame per seconds FPS
        end_time = time.time()-start_time
        fps = frame_counter/end_time

        #frame =utils.textWithBackground(frame,f'FPS: {round(fps,1)}',FONTS, 1.0, (30, 50), bgOpacity=0.9, textThickness=2)
        # writing image for thumbnail drawing shape
        # cv2.imwrite(f'img/frame_{frame_counter}.png', frame)
        # wirting the video for demo purpose 
        cv2.putText(frame, 'Press "'"Q"'" to quit the app', (int(frame_width / 2) - 300, frame_height - 50), FONTS, 1.7, utils.PINK, 2)
        cv2.setMouseCallback('frame', on_mouse_click)
        cv2.rectangle(frame, (25, 20), (70, 70), (0, 255, 0), 2)
        cv2.rectangle(frame, (25, 120), (70, 170), (0, 255, 0), 2)
        cv2.putText(frame, '+', (27, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, '-', (27, 160), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, f'Time Interval: {TIME_INTERVAL} seconds', (25, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('frame', frame)
        key = cv2.waitKey(2)
        if key == ord('q') or key == ord('Q'):
            break
    cv2.destroyAllWindows()
    camera.release()