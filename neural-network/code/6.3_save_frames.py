import cv2

cap = cv2.VideoCapture('../data/real.mp4')

i = 0
while (cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:

        filename = '../data/real_data/1/' + str(i) + '.jpg'
        cv2.imwrite(filename, frame)
        i+=1
    else:
        break

cap.release()
cv2.destroyAllWindows()
