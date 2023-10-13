import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
with_mask=np.load('with_mask.npy',allow_pickle=True)
without_mask=np.load('without_mask.npy',allow_pickle=True)
with_mask=with_mask.reshape(200,50*50*3)
without_mask=without_mask.reshape(200,50*50*3)
X=np.r_[with_mask,without_mask]
# print(X.shape)
labels=np.zeros(X.shape[0])
labels[200:]=1.0
x_train,x_test,y_train,y_test=train_test_split(X,labels,test_size=0.80)
print(x_train.shape)
pca=PCA(n_components=3)
x_train=pca.fit_transform(x_train)
print(x_train.shape)
svm=SVC()
svm.fit(x_train,y_train)
x_test=pca.transform(x_test)
y_pred=svm.predict(x_test)
print(accuracy_score(y_test, y_pred))
haar_classifier=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
# img=cv2.imread("WIN_20230502_00_45_14_Pro.jpg")
# flag=1
capture=cv2.VideoCapture(0)
while True:
    flag,img=capture.read()
    if flag:
        faces=haar_classifier.detectMultiScale(img)
        for x,y,w,h in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,255),4)
            face=img[y:y+h,x:x+w,:]
            face=cv2.resize(face,(50,50))
            face=face.reshape(1,-1)
            face=pca.transform(face)
            pred=svm.predict(face)[0]
            if pred==0:
                print("mask")
            elif pred==1:
                print("no mask")
                # temp=True
        cv2.imshow('result', img)
        if cv2.waitKey(2)== 27 :
            break
capture.release()
cv2.destroyAllWindows()




