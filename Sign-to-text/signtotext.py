import numpy as np
import cv2
import time

cap = cv2.VideoCapture(0)
kernel = None
imb = None
text = np.zeros((120,400,3),np.uint8)

a = cv2.imread("A.jpg", 0)
b = cv2.imread("B.jpg", 0)
c = cv2.imread("C.jpg", 0)
d = cv2.imread("D.jpg", 0)
e = cv2.imread("E.jpg", 0)
f = cv2.imread("F.jpg", 0)
g = cv2.imread("G.jpg", 0)

alphabets = [a, b, c, d, e, f, g]
names = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
size = 0
high_score = 0
label = 'None'
filterthresh = 25
thresh = 3
def find_object_backsub(imb, imf, filtert):
    filterthresh = filtert
    imf = np.int16(imf)
    imb = np.int16(imb)
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

while (True):
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    if not ret:
        break

    boxf = frame[120:350, 350:600]
    realpart = np.zeros(boxf.shape, dtype=np.uint8)

    if imb is None:
        time.sleep(2)
        imb = boxf
        continue

    graybox = cv2.cvtColor(boxf, cv2.COLOR_BGR2GRAY)
    canvas = np.zeros(boxf.shape[:2], dtype=np.uint8)
    closing = find_object_backsub(imb, boxf, filterthresh)
    contours, hierarchy = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours and cv2.contourArea(max(contours, key=cv2.contourArea)) > 1200:
        c = max(contours, key=cv2.contourArea)
        epsilon = 0.005 * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, epsilon, True)
        cv2.drawContours(canvas, [approx], -1, 255, -1)
        realpart = cv2.bitwise_and(boxf, boxf, mask=canvas)

        scores=[]
        for sign in alphabets:
            ret, mask = cv2.threshold(sign, 3, 255, cv2.THRESH_BINARY)
            res = cv2.matchTemplate(graybox, sign, cv2.TM_CCOEFF ,mask)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            scores.append(max_val)
        high_score = max_val
        label = names[np.argmax(scores)]

    cv2.putText(frame, label, (0, 130), cv2.TM_CCOEFF, 2, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imshow('Sentence', text)
    cv2.imshow('frame', frame)

    if cv2.waitKey(10) & 0xFF == ord('s'):
        cv2.putText(text, label, (size,22), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0))
        size += 20

    k = cv2.waitKey(10)
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()