import cv2 as cv
import mediapipe as mp # library by google
import time

capture = cv.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0

while True:
    isTrue,frame = capture.read()
    rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id,lm in enumerate(handLms.landmark):   #not necessary just for spot marking big one
                h,w,c = frame.shape
                cx,cy = int(lm.x*w), int(lm.y*h)
                print(id,cx,cy)
                
                #to spot a specific point id no. can be found on google
                if id==9:    
                    cv.circle(frame, (cx,cy), 25, (255,0,255), cv.FILLED)

            mpDraw.draw_landmarks(frame, handLms, mpHands.HAND_CONNECTIONS)
    #to show fps
    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime
    cv.putText(frame, str(int(fps)), (10,70), cv.FONT_HERSHEY_COMPLEX,3,(255,0,255),3)

    cv.imshow('Video',frame)
    if cv.waitKey(20) & 0xFF == ord('d'):
        break

capture.release()
cv.destroyAllWindows()

#refer to video to learn how to convert it into module