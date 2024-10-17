# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 13:02:35 2023

@author: sadham
"""

import cv2

full_body=cv2.CascadeClassifier(r"C:\Users\sadham\Downloads\haarcascade_fullbody.xml")

cap=cv2.VideoCapture(r"C:\Users\sadham\Downloads\Top View Pedestrian Dataset Sample 1.mp4")

while cap.isOpened():
    ret,input1=cap.read()
    gray=cv2.cvtColor(input1,cv2.COLOR_BGR2GRAY)
    body=full_body.detectMultiScale(gray,1.2,3)
    for (x,y,w,h) in body:
        cv2.rectangle(input1,(x,y),(x+w,y+h),(0,0,255),2)
        cv2.imshow('pat',input1)
        cv2.waitKey(1)
cv2.destroyAllWindows()
