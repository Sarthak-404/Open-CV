import cv2 as cv 
import mediapipe as mp 
import time

capture = cv.VideoCapture(0)

pTime = 0
cTime = 0

mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
FaceDetection = mpFaceDetection.FaceDetection() #press ctrl + left click on the library syntax to know what variables it can have  #use 0.75 for incrs acc

while True:
    isTrue,frame = capture.read()

    rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    results = FaceDetection.process(rgb)
    if results.detections:
        for id,detection in enumerate(results.detections):
            # mpDraw.draw_detection(frame, detection)  #use this no need for below syntax in this loop it detects eyes nose and ears with face
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, ic = frame.shape
            bbox = int(bboxC.xmin * iw),int(bboxC.ymin * ih),\
                int(bboxC.width * iw),int(bboxC.height * ih)
            cv.rectangle(frame, bbox, (255,0,255), 2)
            cv.putText(frame, f'{int(detection.score[0]*100)}%', (bbox[0],bbox[1]-20), cv.FONT_HERSHEY_COMPLEX,3,(0,255,0),2)
            #it is used only to calculate the accuracy of face

    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime
    cv.putText(frame, f'FPS: {int(fps)}', (10,70), cv.FONT_HERSHEY_COMPLEX,3,(255,0,255),3)

    cv.imshow('Video', frame)
    if cv.waitKey(20) & 0xFF == ord('d'):
        break

capture.release()
cv.destroyAllWindows()