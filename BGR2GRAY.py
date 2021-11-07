import cv2
import imutils

cap = cv2.VideoCapture(0)
rotate = 0
while True:
    ret, frame = cap.read()

    if ret:
        frame = cv2.resize(frame, dsize=None, fx=1, fy=1)
        frame = imutils.rotate(frame, rotate)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 11)
        canny = cv2.Canny(gray, threshold1=30, threshold2=50)
        cv2.imshow('Video', frame)
        cv2.imshow('Gray', gray)
        cv2.imshow('Hsv', hsv)
        cv2.imshow('Thresh', thresh)
        cv2.imshow('Canny', canny)
    k = cv2.waitKey(1)
    if k == ord('q'):
        break
    if k == ord('a'):
        rotate = 90
    if k == ord('s'):
        rotate = 180
    if k == ord('d'):
        rotate = -90
    if k == ord('e'):
        rotate = 0
cap.release()
cv2.destroyAllWindows()
