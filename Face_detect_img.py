import cv2 as cv

img = cv.imread('Face_detect_test.jpg')

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # as face detection take place from the edges hence image need to be grayscale
cv.imshow('Grayscale', gray)

haar_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=2)

print(f'Number of faces found = {len(faces_rect)}')

for (x,y,w,h) in faces_rect:
    cv.rectangle(img, (x,y), (x+w,y+h), (0,255,0), thickness = 2)

cv.imshow('Final', img)

cv.waitKey(0)
