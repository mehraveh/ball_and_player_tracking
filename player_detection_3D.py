import cv2
import os
import numpy as np
import sys
import time
from params import *
from utils import *

vidcap = cv2.VideoCapture(sys.argv[1])
tracker = cv2.TrackerGOTURN_create()
success,image = vidcap.read()
success = True
count = 0
idx = 0
pl1 = 1
pl2 = 0
ball_coordinate = (0,0)
distance = 0
cor = (0,0)
owner = 0
while success:
    blue_player_coordinates = []
    red_player_coordinates = []
    hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_green, upper_green)
    res = cv2.bitwise_and(image, image, mask=mask)
    res_gray = cv2.cvtColor(res,cv2.COLOR_BGR2GRAY)

    kernel = np.ones((3,3),np.uint8)
    thresh = cv2.threshold(res_gray,127,255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    contours,hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        
    for c in contours:
        x,y,w,h = cv2.boundingRect(c)
        
        if(h>=(1.5)*w):
            if(w>15 and h>= 15):
                idx = idx+1
                player_img = image[y:y+h,x:x+w]
                player_hsv = cv2.cvtColor(player_img,cv2.COLOR_BGR2HSV)
                mask1 = cv2.inRange(player_hsv, lower_blue, upper_blue)
                res1 = cv2.bitwise_and(player_img, player_img, mask=mask1)
                res1 = cv2.cvtColor(res1,cv2.COLOR_HSV2BGR)
                res1 = cv2.cvtColor(res1,cv2.COLOR_BGR2GRAY)
                blue_count = cv2.countNonZero(res1)
                mask2 = cv2.inRange(player_hsv, lower_red, upper_red)
                res2 = cv2.bitwise_and(player_img, player_img, mask=mask2)
                res2 = cv2.cvtColor(res2,cv2.COLOR_HSV2BGR)
                res2 = cv2.cvtColor(res2,cv2.COLOR_BGR2GRAY)
                red_count = cv2.countNonZero(res2)

                if(blue_count >= 20):
                    cv2.putText(image, '1', (x-2, y-2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2, cv2.LINE_AA)
                    cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),3)
                if(red_count>=20):
                    cv2.putText(image, '2', (x-2, y-2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2, cv2.LINE_AA)
                    cv2.rectangle(image,(x,y),(x+w,y+h),(0,0,255),3)
        if((h>=1 and w>=1) and (h<=30 and w<=30)):
            player_img = image[y:y+h,x:x+w]
        
            player_hsv = cv2.cvtColor(player_img,cv2.COLOR_BGR2HSV)
            mask1 = cv2.inRange(player_hsv, lower_white, upper_white)
            res1 = cv2.bitwise_and(player_img, player_img, mask=mask1)
            res1 = cv2.cvtColor(res1,cv2.COLOR_HSV2BGR)
            res1 = cv2.cvtColor(res1,cv2.COLOR_BGR2GRAY)
            ball_count = cv2.countNonZero(res1)

            if(ball_count >= 2):
                cv2.putText(image, 'football', (x-2, y-2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2, cv2.LINE_AA)
                cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),3)

    new = False
    if(len(blue_player_coordinates)>0 or len(red_player_coordinates)>0):
        distance, cor, owner, new = detect_owner(blue_player_coordinates, red_player_coordinates, ball_coordinate, distance, cor, owner)
    # if new:
    #     bbox = (cor[0]-10, cor[1]-10, cor[0]+10, cor[1]+10)
    #     ok = tracker.init(image, bbox)
    #     image = cv2.circle(image, cor, 20, (255, 0, 255) , 3) 
    # else:
    #     ok, bbox = tracker.update(image)
    #     if ok:
    #         image = cv2.circle(image, (int(bbox[0]), int(bbox[1])), 20, (255, 0, 255) , 3) 

    owner_txt = ''
    if owner is 1:
        owner_txt = "player1"
        pl1 += 1
    if owner is 2:
        owner_txt = "player2"
        pl2 += 1
    count += 1
    cv2.imshow('Match Detection',image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    success,image = vidcap.read()
print(str(pl1/(pl1+pl2)*100), str(pl2/(pl1+pl2)*100))
vidcap.release()
cv2.destroyAllWindows()

