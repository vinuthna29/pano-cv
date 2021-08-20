#working
import cv2 as cv
import numpy as np
import math as m
import argparse
import os
import sys

num_of_correspondance=25


my_parser = argparse.ArgumentParser()
my_parser.add_argument('Path',
                       metavar='path',
                       type=str,
                       default='../data/auto/campus/',
                       help='../data/auto/campus/')
args = my_parser.parse_args()
input_path = args.Path
    

dirs=sorted(os.listdir(str(input_path)))
path=["" for i in range(len(dirs))]
for i  in range(len(dirs)):
   path[i]=input_path+dirs[i]

#path1="/home/arjun/Desktop/130010009_140076001_150050001_lab03_midsem/data/manual/scene/scene2.jpg"#to be altered wrt imageat path 2
#path2="/home/arjun/Desktop/130010009_140076001_150050001_lab03_midsem/data/manual/scene/scene1.jpg"
def do_i_need_to_resize(image):
    height=image.shape[0]
    width=image.shape[0]
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

I1 =cv.imread(path[0])
while 1:
    if (do_i_need_to_resize(I1)):
        I1=resize(I1,50)
    else:
        break
I2 =cv.imread(path[1])
while 1:

    if (do_i_need_to_resize(I2)):
        I2=resize(I2,50)
    else:
        break
(H1,W1,_)=I1.shape
(H2,W2,_)=I2.shape
#print("shape of image 1: "+str(I1.shape))
#print("shape of image 2: "+str(I2.shape))

orb = cv.ORB_create()##create instance of orb
kp1,des1=orb.detectAndCompute(I1,None)#query
kp2,des2=orb.detectAndCompute(I2,None)#
#print(kp1,kp2,des1,des2)
bf=cv.BFMatcher(cv.NORM_HAMMING,True)
matches=bf.match(des1,des2)
matches_=sorted(matches,key=lambda x:x.distance)
I3=cv.drawMatches(I1,kp1,I2,kp2,matches_[:num_of_correspondance],None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
while(1):

    cv.imshow("matching_diagram(Press Esc to go to next window)",I3)
    if cv.waitKey(20) & 0xFF == 27:
        break
cv.destroyAllWindows()

image_1_correspondance=[]
image_2_correspondance=[]

for m in (matches_[:num_of_correspondance]):
   image_1_correspondance.append([int(kp1[m.queryIdx].pt[0]),int(kp1[m.queryIdx].pt[1])])
   image_2_correspondance.append([int(kp2[m.trainIdx].pt[0]),int(kp2[m.trainIdx].pt[1])])


#while(1):

 #   cv.imshow("points marked on img 1",I1)

  #  cv.imshow("points marked on img 2",I2)
   # cv.moveWindow("points marked on img 2",650,0)
    #if cv.waitKey(20) & 0xFF == 27:
     #   break
#cv.destroyAllWindows()


image_1_correspondance=np.array(image_1_correspondance)
image_2_correspondance=np.array(image_2_correspondance)
#print(image_1_correspondance_)
#print(image_2_correspondance_)

h, status = cv.findHomography(image_1_correspondance,image_2_correspondance)
#print(h,status)

Iout= cv.warpPerspective(I1, h, (W1+W2,H1+H2))
Iout[0:H2,0:W2,:]=I2
#cv.namedWindow('Iout')
#cv.imshow('Iout',Iout)
#cv.moveWindow('Iout',610,480)

#while(1):

 #   cv.imshow("Iout",Iout)
  #  if cv.waitKey(20) & 0xFF == 27:
   #     break
#cv.destroyAllWindows()
gray = cv.cvtColor(Iout, cv.COLOR_BGR2GRAY)
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
final_out=Iout[r1:r2+1,c1:c2+1,:]
while(1):

    cv.imshow("output",final_out)
    if cv.waitKey(20) & 0xFF == 27:
        break
cv.destroyAllWindows()
cv.imwrite('../results/pano-auto-results/{}'.format(dirs[0]),final_out)

