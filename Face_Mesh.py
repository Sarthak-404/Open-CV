import cv2 as cv 
import mediapipe as mp 
import time

capture = cv.VideoCapture(0)

cTime = 0
pTime = 0

mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces = 2)
drawSpec = mpDraw.DrawingSpec(thickness=1 , circle_radius=1)

while True:
    isTrue,frame = capture.read()

    rgb = cv.cvtColor(frame,cv.COLOR_BGR2RGB)
    results = faceMesh.process(rgb)
    if results.multi_face_landmarks:
        for faceLms in results.multi_face_landmarks:
            mpDraw.draw_landmarks(frame, faceLms, mpFaceMesh.FACEMESH_TESSELATION,drawSpec,drawSpec)


    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime
    cv.putText(frame, f'FPS: {int(fps)}', (10,70), cv.FONT_HERSHEY_COMPLEX,3,(255,0,255),3)

    cv.imshow('Video', frame)
    if cv.waitKey(20) & 0xFF == ord('d'):
        break

capture.release()
cv.destroyAllWindows()