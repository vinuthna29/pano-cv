import argparse
import numpy as np
from os import listdir
from os.path import isfile, join
import cv2 as cv
from matplotlib import pyplot as plt
number_of_matches=25
rev=0

def resize(image,scale_percent=50):
#calculate the scale_percent of original dimensions
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dsize = (width, height)
    output = cv.resize(image, dsize)
    return output

def pad_with(vector, pad_width, iaxis, kwargs):
    pad_value = kwargs.get('padder', 10)
    vector[:pad_width[0]] = pad_value
    vector[-pad_width[1]:] = pad_value

def remove_unwanted_black(output):
    '''Removes unwanted blank space from an image and crops it to the smallest rectangle which fits all the nonzero pixel'''
    shape=output.shape
    output=output.astype("uint8")
    if  len(shape)>=3:
        gray = cv.cvtColor(output, cv.COLOR_BGR2GRAY)
    else:
        gray=output
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
    final_out=output[r1:r2+1,c1:c2+1,:]
    final_out=np.array(final_out,dtype="uint8")
    return  final_out

def stitch_to_left(img1,img2):
    '''Takes the left image (img1) as reference and stiches the right image (img2) to the reference image'''
    descriptor = cv.ORB_create()
    keypoints=[]
    features_list=[]
    (kps, features) = descriptor.detectAndCompute(img1, None)
    keypoints.append(kps)
    features_list.append(features)
    (kps, features) = descriptor.detectAndCompute(img2, None)
    keypoints.append(kps)
    features_list.append(features)
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    best_matches = bf.match(features_list[0],features_list[1])
    rawMatches = sorted(best_matches, key = lambda x:x.distance)
    matches=rawMatches[:number_of_matches]
    img3 = cv.drawMatches(img1,keypoints[0],img2,keypoints[1],matches,
                           None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    if len(matches) >= 4:
        src = np.float32([ keypoints[0][m.queryIdx].pt for m in matches]).reshape(-1,1,2)
        dst = np.float32([ keypoints[1][m.trainIdx].pt for m in matches]).reshape(-1,1,2)
    H, masked = cv.findHomography(dst, src, cv.RANSAC, 5.0)
    width = img1.shape[1] + img2.shape[1]
    height = img1.shape[0] + img2.shape[0]
    result = cv.warpPerspective(img2,H, (width, height))
    mask=(img1<1)*1
    result[0:img1.shape[0], 0:img1.shape[1]] =result[0:img1.shape[0], 0:img1.shape[1]]*mask+img1
    return result

def stitch_to_right(img1,img2):
    '''Takes the right image (img2) as reference and stiches the left image (img1) to the reference image'''
    return cv.flip(stitch_to_left(cv.flip(img1,1),cv.flip(img2,1)),1)

if __name__=='__main__':
    my_parser = argparse.ArgumentParser()
    my_parser.add_argument('Path',metavar='path',type=str,default='..\data\general\mountain',help='..\data\general\mountain')
    my_parser.add_argument('index',metavar='index',type=int,default=1,help='give index')
    args = my_parser.parse_args()
    mypath=args.Path
    mypath=mypath.rstrip()
    temp=mypath.split("\\")
    if temp[-2]=="campus" or temp[-2]=="yard":
        rev=1
    if temp[-2]=="ledge" or temp[-2]=="campus":
        number_of_matches=20
    if temp[-2]=="yosemite":
        number_of_matches=9
    n=args.index
    n=n-1
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath,f))]
    images = np.empty(len(onlyfiles), dtype=object)
    for i in range(0, len(onlyfiles)):
        images[i] = cv.imread( join(mypath,onlyfiles[i]))
    if rev==1:
        images=images[::-1]
    for i in range(n,n+1):
        r=images[n]     #reference image
        # cv.imshow("ref img",r)
        r=cv.copyMakeBorder(r, r.shape[0]//8, r.shape[0]//8, r.shape[1]//8, r.shape[1]//8, cv.BORDER_CONSTANT, None, 0)
        # cv.imshow("padded ref img",r)
        # stitching images that are to the right of the reference image (homography to the left)
        for i in range(n+1,len(images)):
            r=stitch_to_left(r,images[i])
            r=remove_unwanted_black(r)

        #stitching images that are to the left of the reference image (homograpy to the right)
        for i in range(n-1,-1,-1):
            r=stitch_to_right(r,images[i])
            r=remove_unwanted_black(r)


        cv.imshow("final stitched",resize(remove_unwanted_black(r),40))
        # cv.imshow("final stitched w black",resize(r,50))
        cv.waitKey(0)
        cv.destroyAllWindows()
        mypath=mypath.rstrip()
        temp=mypath.split("\\")
        # print(temp[-2])
        path_save="..\\results\pano-general-results\\"
        # if rev==1:
        #     print('{}_ref={}_matches={}_rev.jpg'.format(temp[-2],n+1,number_of_matches))
        #     cv.imwrite(join(path_save , '{}_ref={}_matches={}_rev.jpg'.format(temp[-2],n+1,number_of_matches)), r)
        # else:
        #     print('{}_ref={}_matches={}.jpg'.format(temp[-2],n+1,number_of_matches))
        #     cv.imwrite(join(path_save , '{}_ref={}_matches={}.jpg'.format(temp[-2],n+1,number_of_matches)), r)
