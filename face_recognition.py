import cv2 as cv
import matplotlib.pyplot as plt 

for i in range(1,6):
    img =cv.imread('Sample/'+str(i)+'.jpg')
    cv.imshow(str(i),img)

    gray =cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    #cv.imshow('Gray',gray)

    haar_cascade =cv.CascadeClassifier('haar_face.xml') #Reads the haarcascade file and store in the variable

    faces_rect =haar_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5)

    print(f'Number of faces detected = {len(faces_rect)}')

    for (x,y,w,h) in faces_rect:
        cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),1)

    cv.imshow('Detected Image',img)

    if(len(faces_rect)>0):
        print("Human")

    else:
        print("Non Human")

    cv.waitKey(0)
    cv.destroyAllWindows()