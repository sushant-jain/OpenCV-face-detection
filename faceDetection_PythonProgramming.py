import numpy as np
import cv2

# multiple cascades: https://github.com/Itseez/opencv/tree/master/data/haarcascades

#https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
#https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_eye.xml
head_shoulder_cascade = cv2.CascadeClassifier('HS.xml')

test_cascade=cv2.CascadeClassifier('abc.xml')


# cap = cv2.VideoCapture(0)

imNo=1
imgC=cv2.imread("{}.jpg".format(imNo))
g_imgC=cv2.cvtColor(imgC, cv2.COLOR_BGR2GRAY)

#for class img1
#facesx = face_cascade.detectMultiScale(g_imgC, 1.04, 5)
facesx = face_cascade.detectMultiScale(g_imgC, 1.04, 5)

for (x,y,w,h) in facesx:
    cv2.rectangle(imgC,(x,y),(x+w,y+h),(255,0,0),2)

#for class img2
headsx=head_shoulder_cascade.detectMultiScale(g_imgC,1.24,3)
#headsx=head_shoulder_cascade.detectMultiScale(g_imgC,1.015,8)

for (x,y,w,h) in headsx:
    cv2.rectangle(imgC,(x,y),(x+w,y+h),(0,255,0),2)

testx=test_cascade.detectMultiScale(g_imgC,1.01,6)

for (x,y,w,h) in testx:
    cv2.rectangle(imgC,(x,y),(x+w,y+h),(0,0,255),2)

# bodyx=full_body_cascade.detectMultiScale(g_imgC,1.05,5)

# for (x,y,w,h) in bodyx:
#     cv2.rectangle(imgC,(x,y),(x+w,y+h),(0,255,0),2)

cv2.imwrite("out{}.jpg".format(imNo),imgC)
        
# while 1:
#     ret, img = cap.read()
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     faces = face_cascade.detectMultiScale(gray, 1.3, 5)

#     for (x,y,w,h) in faces:
#         cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
#         roi_gray = gray[y:y+h, x:x+w]
#         roi_color = img[y:y+h, x:x+w]
        
#         eyes = eye_cascade.detectMultiScale(roi_gray)
#         for (ex,ey,ew,eh) in eyes:
#             cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

#     cv2.imshow('img',img)
#     k = cv2.waitKey(30) & 0xff
#     if k == 27:
#         break

# cap.release()
cv2.destroyAllWindows()