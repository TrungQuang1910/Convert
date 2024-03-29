import cv2

img = cv2.imread('D:\MaDoc\sample1.png', 1)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
thresh = cv2.adaptiveThreshold(
    gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 41, 11)

contours, _ = cv2.findContours(
    thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
cv2.drawContours(img, contours, -1, (0, 255, 0), 2)
number = 0
for c in contours:
    x, y, w, h = cv2.boundingRect(c)
    print(x, y, w, h)
    # cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
    if 140 < h < 160:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
        number += 1
        crop = img[y:y+h, x:x+w]
        cv2.imshow('{}'.format(number), crop)
print('Number is: ' + str(number))
cv2.imshow('Picture', img)
cv2.imshow('Gray', gray)
cv2.imshow('Thres', thresh)
cv2.waitKey()
cv2.destroyAllWindows()
