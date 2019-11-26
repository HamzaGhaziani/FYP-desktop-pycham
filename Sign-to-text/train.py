import numpy as np
import cv2
import time


def find_object_backsub(imb, imf, filtert):
    filterthresh = filtert
    imf = np.int64(imf)
    imb = np.int64(imb)
    img = imf - imb

    img[img < -filterthresh] = 255
    img[img < filterthresh] = 0
    img[img > filterthresh] = 255

    img = np.uint8(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY)
    img = cv2.erode(img, None, iterations=1)
    imgmask = cv2.dilate(img, None, iterations=2)

    return imgmask



filterthresh = 25
thresh = 3

cap = cv2.VideoCapture(0)
kernel = None
imb = None
ret, frame = cap.read()
frame = cv2.flip(frame, 1)
cv2.rectangle(frame, (430, 0), (635, 210), (0, 200, 10), 2)
boxf = frame[5:205, 435:630]
realpart = np.zeros(boxf.shape, dtype=np.uint8)

while (True):
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    if not ret:
        break
    cv2.rectangle(frame, (430, 0), (635, 210), (0, 200, 10), 2)
    boxf = frame[5:205, 435:630]
    realpart = np.zeros(boxf.shape, dtype=np.uint8)
    print(boxf.shape)
    if imb is None:
        time.sleep(2)
        imb = boxf
        continue
    canvas = np.zeros([200, 195], dtype=np.uint8)
    canvas = np.zeros(boxf.shape[:2], dtype=np.uint8)
    closing = find_object_backsub(imb, boxf, filterthresh)
    contours, hierarchy = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours and cv2.contourArea(max(contours, key=cv2.contourArea)) > 1000:
        c = max(contours, key=cv2.contourArea)
        epsilon = 0.005 * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, epsilon, True)
        cv2.drawContours(canvas, [approx], -1, 255, -1)
        realpart = cv2.bitwise_and(boxf, boxf, mask=canvas)

    cv2.line(frame, (430, 105), (635, 105), (0, 0, 255), 3)
    cv2.line(frame, (532, 0), (532, 210), (0, 0, 255), 3)
    cv2.imshow('closing', closing)
    cv2.imshow('canvas', canvas)
    cv2.imshow('realpart', realpart)

    cv2.imshow('frame', frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

    if cv2.waitKey(10) & 0xFF == ord('s'):
        cv2.imwrite('E.jpg', realpart)
        break

cap.release()
cv2.destroyAllWindows()