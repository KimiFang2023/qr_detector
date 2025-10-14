import cv2
from pyzbar.pyzbar import decode
#import webbrowser

#cap=cv2.VideoCapture(0)

data=['link']

while True:
    img=cv2.imread('qr_code.jpg')
    QR_code=decode(img)
    #print(QR_code)

    for QR in QR_code:
        QR_data=QR.data.decode('utf-8')
        #print(QR_data)

        if(QR_data!=data[-1]):
            data.append(QR_data)
            #webbrowser.open(QR_data)
            print(QR_data)

        point=QR.rect
        #print(point)
        cv2.rectangle(img,(point[0],point[1]),(point[0]+point[2],point[1]+point[3]),(255,255,0),5)
        cv2.putText(img,QR_data,(point[0],point[1]-10),cv2.FONT_HERSHEY_COMPLEX,0.6,(255,0,255),1)

    cv2.imshow('output',img)

    if cv2.waitKey(1)&0xFF==27:
        break