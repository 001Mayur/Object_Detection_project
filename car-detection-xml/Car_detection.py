import cv2

import random
import sys



def generator():

    N = 10
    n=[random.randint(0,sys.maxsize) for _ in range(N)]
    b=str(n[0])
    a=b[:8]
    
    return a

cap = cv2.VideoCapture("G:/Git-projects/Object_Detection_project/car-detection-xml/input_video2.mp4")
car_cascade = cv2.CascadeClassifier('G:/Git-projects/Object_Detection_project/car-detection-xml/cars.xml')

while True:
    ret, frames = cap.read()
    gray = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
    cars = car_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=9)

    # if str(np.array(cars).shape[0]) == '1':
    #     i += 1
    #     continue
    for (x,y,w,h) in cars:
        plate = frames[y:y + h, x:x + w]
        cv2.rectangle(frames,(x,y),(x +w, y +h) ,(51 ,51,255),thickness=5)
        # cv2.rectangle(frames, (x, y - 40), (x + w, y), (51,51,255), -2)
        cv2.putText(frames, 'Car', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 5, (255, 255, 255), 5)
        cv2.imshow('car',plate)
        cv2.imwrite(F"G:\\Git-projects\\Object_Detection_project\\car-detection-xml\\save\\{generator()}.jpg",frames)

    # lab1 = "Car Count: " + str(i)
    # cv2.putText(frames, lab1, (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (147, 20, 255), 3)
    frames = cv2.resize(frames,(600,400))
    cv2.imshow('Car Detection System', frames)
    # cv2.resizeWindow('Car Detection System', 800, 800)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
cv2.destroyAllWindows()