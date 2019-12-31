import numpy as np
import cv2

right_fist = cv2.CascadeClassifier('fist.xml')
right_palm = cv2.CascadeClassifier('rpalm.xml')

cam = cv2.VideoCapture(0)
detections = ['random']
while True:
    _ , img = cam.read()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    right_fists = right_fist.detectMultiScale(gray, 1.3, 5)
    for (x, y,w, h) in right_fists:
        # Inserting the value if fist is detected only once 
        # to avoid multiple inserts in multiple frames
        if detections[-1] != 'f':
            detections.append('f')
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

    right_palms = right_palm.detectMultiScale(gray, 1.3, 5)

    for (x, y,w, h) in right_palms:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        if detections[-1] != 'p':
            detections.append('p')

    if detections[-3:] == ['p', 'f', 'p']:
        # The video start stop action should be here
        print('Toggle start stop')
        detections = ['random']
    print(detections)
    cv2.imshow('img',img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()
