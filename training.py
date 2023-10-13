import cv2
import numpy as np
haar_classifier=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
capture=cv2.VideoCapture(0)
data=[]
while True:
    flag,img=capture.read()
    if flag:
        faces=haar_classifier.detectMultiScale(img)
        for x,y,w,h in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,255),4)
            face=img[y:y+h,x:x+w,:]
            face=cv2.resize(face,(50,50))
            data.append(face)
            print(len(data))
        cv2.imshow('result', img)
        if cv2.waitKey(2)== 27 or len(data)>=200:
            break
np.save("without_mask.npy",data)
capture.release()
cv2.destroyAllWindows()