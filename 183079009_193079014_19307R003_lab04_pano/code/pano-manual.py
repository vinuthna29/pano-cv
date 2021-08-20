import cv2 as cv
import numpy as np
import math as m
import argparse
import os

import sys
if __name__=='__main__':
    my_parser = argparse.ArgumentParser()
    my_parser.add_argument('Path',
                       metavar='path',
                       type=str,
                       default='../data/manual/campus/',
                       help='../data/manual/campus/')
    args = my_parser.parse_args()
    input_path = args.Path
    

    dirs=sorted(os.listdir(str(input_path)))
    path=["" for i in range(len(dirs))]
    for i  in range(len(dirs)):
        path[i]=input_path+dirs[i]





#path1="/home/arjun/Desktop/130010009_140076001_150050001_lab03_midsem/data/manual/scene/scene1.png"
#path2="/home/arjun/Desktop/130010009_140076001_150050001_lab03_midsem/data/manual/scene/scene2.png"
# mouse callback function
    image_1_correspondance=[]
    image_2_correspondance=[]
    def do_i_need_to_resize(image):
        height=image.shape[0]
        width=image.shape[1]
        if height>=768 or width>=1366/2:
            return True
        else:
            return False
    def resize(image,scale_percent=50):
#calculate the scale_percent of original dimensions
        width = int(image.shape[1] * scale_percent / 100)
        height = int(image.shape[0] * scale_percent / 100)
        dsize = (width, height)
        output = cv.resize(image, dsize)
        return output



    def draw_circle1(event,x,y,flags,param):
        (H1,W1,_)=I1.shape
        if event == cv.EVENT_LBUTTONDBLCLK:
            cv.circle(I1,(x,y),2,(0,0,255),-1)
            image_1_correspondance.append([x+2*W1,y+m.floor(H1/2)])


    def draw_circle2(event,x,y,flags,param):
        (H2,W2,_)=I2.shape
        if event == cv.EVENT_LBUTTONDBLCLK:
            cv.circle(I2,(x,y),2,(0,0,255),-1)
            image_2_correspondance.append([x+2*W2,y+m.floor(H2/2)])

# Create a black image, a window and bind the function to window

    I1 =cv.imread(path[0])
    if (do_i_need_to_resize(I1)):
        I1=resize(I1,50)
    I2 =cv.imread(path[1])
    if (do_i_need_to_resize(I2)):
        I2=resize(I2,50)
    (H1,W1,_)=I1.shape
    (H2,W2,_)=I2.shape
    H1_by_2=m.floor(H1/2)
    H2_by_2=m.floor(H2/2)

    I1_=np.zeros((2*H1_by_2+H1,5*W1,3),dtype="uint8")
    I2_=np.zeros((2*H2_by_2+H2,5*W2,3),dtype="uint8")
    I1_[H1_by_2:H1+H1_by_2,2*W1:3*W1,:]=I1
    I2_[H2_by_2:H2+H2_by_2,2*W2:3*W2,:]=I2

    cv.namedWindow('I2')
    cv.imshow('I2',I2)

    cv.setMouseCallback('I2',draw_circle2)

    cv.namedWindow('I1')
    cv.imshow('I1',I1)
    cv.moveWindow('I1',650,0)
    cv.setMouseCallback('I1',draw_circle1)
    while(1):
        cv.imshow('I1',I1)
        cv.imshow('I2',I2)
        if cv.waitKey(20) & 0xFF == 27:
            break
    cv.destroyAllWindows()
#print(image_2_correspondance,image_1_correspondance)
    h, status = cv.findHomography(np.array(image_1_correspondance),np.array(image_2_correspondance))
    Iout= cv.warpPerspective(I1_, h, (I1_.shape[1],I1_.shape[0]))


    cv.namedWindow('Iout')
    cv.imshow('Iout',Iout)
    cv.moveWindow('Iout',610,480)
    new=np.zeros(Iout.shape)
    new=Iout
#print(new[H2_by_2:H2+H2_by_2,2*W2:3*W2,:].shape)
#print(I2_[H2_by_2:H2+H2_by_2,2*W2:3*W2,:].shape)
    new[H2_by_2:H2+H2_by_2,2*W2:3*W2,:]=I2_[H2_by_2:H2+H2_by_2,2*W2:3*W2,:]
    while(1):

        cv.imshow("new",new)
        if cv.waitKey(20) & 0xFF == 27:
            break
    cv.destroyAllWindows()
    gray = cv.cvtColor(new, cv.COLOR_BGR2GRAY)
    sum_along_row=np.sum(gray,axis=1)
    sum_along_col=np.sum(gray,axis=0)

    sum_along_row=np.append(sum_along_row,0)
    sum_along_col=np.append(sum_along_col,0)
    for i in range(len(sum_along_col)):
        if sum_along_col[i]!=0:
            c1=i
            break
    for i in range(len(sum_along_col)-1,-1,-1):
        if sum_along_col[i]!=0:
            c2=i
            break
    for i in range(len(sum_along_row)):
        if sum_along_row[i]!=0:
            r1=i
            break
    for i in range(len(sum_along_row)-1,-1,-1):
        if sum_along_row[i]!=0:
            r2=i
            break
    final_out=new[r1:r2+1,c1:c2+1,:]
    while(1):

        cv.imshow("output",final_out)
        if cv.waitKey(20) & 0xFF == 27:
            break
    cv.destroyAllWindows()
    cv.imwrite('../results/pano-auto-results/{}'.format(dirs[0]),final_out)

