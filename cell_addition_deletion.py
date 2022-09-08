import cv2
from skimage.measure import label, regionprops
from sklearn.mixture import GaussianMixture as GMM
from sklearn.cluster import KMeans


import copy
import os



import parameters_cell_mod as params_m
from utility import *
from components import *


import numpy as np
import math
import statistics
import functools

def read_input(path):
    img = cv2.imread(path)
    return img

def resizing(img_l, img_s):
    dim = (img_l.shape[1], img_l.shape[0])
    img = cv2.resize(img_s, dim, interpolation = cv2.INTER_AREA)
    return img

def bboxed(bin_img):
    blobs = connected_components(bin_img)
    bboxes, _ , _,min_area, omitted_bbox = regions(blobs, bin_img, partial_check=True)
    print('min area: ',min_area)
    return bboxes, omitted_bbox

def get_centroid(r_bboxes):
    centroids = []
    for bboxes in r_bboxes:
        c = [[(bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2] for bbox in bboxes]
        centroids.append(c)
    return centroids

def check_with_centroid(img_1, img_2, row_centroid,case=''):
    subtracted_image = img_1 - img_2
    name = 'subtracted'+case
    cv2.imwrite(os.path.join(output_path, name+'.jpg'),subtracted_image)   

    flag = False
    for i,centroids in enumerate(row_centroid):
        for j,centroid in enumerate(centroids):
            if subtracted_image[round(centroid[1]),round(centroid[0])] > 0:
                flag=True
                print('subrow: ',i,' comp: ',j,' compromised')
    return flag

if __name__=="__main__":
    layout = read_input(params_m.layout_row_path)
    sem = read_input(params_m.sem_row_path)
    output_path = os.path.join(params_m.output,'addition_deletion','row_34')

    print(layout.shape,' ',sem.shape)

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    #exit()
    sem = resizing(layout,sem)

    cell_sem = sem[:,:,0]
    cell_lout = layout[:,:,0]

    denoised_sem = cv2.fastNlMeansDenoising(cell_sem,None,20,5,30)
    cv2.imwrite(os.path.join(output_path, 'denoised_sem.jpg'),denoised_sem)
    
    bin_sem, _ = binarized(denoised_sem, output_path)
    cv2.imwrite(os.path.join(output_path,'binarized_sem.jpg'),bin_sem)

    bin_lout, _ = binarized(cell_lout, output_path)
    cv2.imwrite(os.path.join(output_path,'binarized_lout.jpg'),bin_lout)

    bboxes_sem, omitted_bbox_sem = bboxed(bin_sem)
    bboxes_lout, omitted_bbox_lout = bboxed(bin_lout)

    print('sem components: ',len(bboxes_sem))
    print('layout components: ',len(bboxes_lout))

    for bbox in omitted_bbox_sem:
        x1,y1,x2,y2 = bbox[0],bbox[1],bbox[2],bbox[3]
        bin_sem[y1:y2+1,x1:x2+1] = 0

    for bbox in omitted_bbox_lout:
        x1,y1,x2,y2 = bbox[0],bbox[1],bbox[2],bbox[3]
        bin_lout[y1:y2+1,x1:x2+1] = 0

    layout_bboxed = draw_bboxes(layout, bboxes_lout, color=(0,255,0))
    cv2.imwrite(os.path.join(output_path,'lout_boxed.jpg'),layout_bboxed)
    sem_bboxed = draw_bboxes(sem, bboxes_sem, color=(0,0,255))
    cv2.imwrite(os.path.join(output_path,'sem_boxed.jpg'),sem_bboxed)

    lines_sem, terminal_bboxes_img_sem, bboxes_sorted_sem = bbox_sorting(bin_sem, bboxes_sem, output_path)
    lines_lout, terminal_bboxes_img_lout, bboxes_sorted_lout = bbox_sorting(bin_lout, bboxes_lout, output_path)

    print('###########Checking on Initial Image#################')

    row_centroid_sorted_lout = get_centroid(bboxes_sorted_lout)
    row_centroid_sorted_sem = get_centroid(bboxes_sorted_sem)
    #print(row_centroid_sorted_lout)
    bin_lout_centroid = draw_centroid(row_centroid_sorted_lout,bin_lout,output_path, row_type='cell', img_type='layout')
    cv2.imwrite(os.path.join(output_path,'lout_centroid.jpg'),bin_lout_centroid)

    bin_sem_centroid = draw_centroid(row_centroid_sorted_sem,bin_sem,output_path, row_type='cell', img_type='sem')
    cv2.imwrite(os.path.join(output_path,'sem_centroid.jpg'),bin_sem_centroid)

    deletion_flag = check_with_centroid(bin_lout_centroid, bin_sem_centroid, row_centroid_sorted_lout)
    addition_flag = check_with_centroid(bin_sem_centroid, bin_lout_centroid,row_centroid_sorted_sem)
    

    if not addition_flag:
        print('No Addition')
    else:
        print('Addition Happened')
    if not deletion_flag:
        print('No deleteion')

    '''  
    print("##############deleting from SEM################")
    modified_bbox = []
    bbox_1 = bboxes_sorted_sem[0][7]
    bbox_2 = bboxes_sorted_sem[1][7]
    modified_bbox.append(bbox_1)
    modified_bbox.append(bbox_2)

    for bbox in modified_bbox:
        x1,y1,x2,y2 = bbox[0],bbox[1],bbox[2],bbox[3]
        bin_sem[y1:y2+1,x1:x2+1] = 0
    cv2.imwrite(os.path.join(output_path,'new_bin_sem.jpg'),bin_sem)
    deletion_flag = check_with_centroid(bin_lout, bin_sem, row_centroid_sorted_lout,case='Deletion')
    if not deletion_flag:
        print('No deleteion')
    else:
        print('Deletion')
    '''
    '''
    print("##############Adding in SEM = Deleting from Layout################")
    modified_bbox = []
    bbox_1 = bboxes_sorted_lout[0][11]
    bbox_2 = bboxes_sorted_lout[1][11]
    modified_bbox.append(bbox_1)
    modified_bbox.append(bbox_2)

    for bbox in modified_bbox:
        x1,y1,x2,y2 = bbox[0],bbox[1],bbox[2],bbox[3]
        bin_lout[y1:y2+1,x1:x2+1] = 0
    cv2.imwrite(os.path.join(output_path,'new_bin_lout.jpg'),bin_lout)
    addition_flag = check_with_centroid(bin_sem, bin_lout, row_centroid_sorted_sem,case='Addition')
    if not addition_flag:
        print('No Addition')
    else:
        print('Addition')
    '''
    
    '''
    lout_centroid = copy.deepcopy(bin_lout)
    lout_centroid = cv2.merge((lout_centroid,lout_centroid,lout_centroid))
    for centroids in row_centroid_sorted_lout:
        lout_centroid = draw_centroid(centroids, lout_centroid, output_path, radius=2, img_type='sem')
    cv2.imwrite(os.path.join(output_path,'lout_centroid.jpg'),lout_centroid)

    for i,bboxes in enumerate(bboxes_sorted_lout):
        print('row: ',i,': ',len(bboxes))
        layout_bboxed = draw_bboxes(layout, bboxes, color=(0,0,255))
        name = 'lout_sorted_boxed_'+str(i)
        cv2.imwrite(os.path.join(output_path,name+'.jpg'),layout_bboxed)


    print('##############for sem##################')

    row_centroid_sorted_sem = get_centroid(bboxes_sorted_sem)
    sem_centroid = copy.deepcopy(bin_sem)
    sem_centroid = cv2.merge((sem_centroid, sem_centroid, sem_centroid))
    for centroids in row_centroid_sorted_sem:
        sem_centroid = draw_centroid(centroids, sem_centroid, output_path, radius=2, img_type='sem')
    cv2.imwrite(os.path.join(output_path,'sem_centroid.jpg'),sem_centroid)


    for i,bboxes in enumerate(bboxes_sorted_sem):
        print('row: ',i,': ',len(bboxes))
        sem_bboxed = draw_bboxes(sem, bboxes, color=(0,0,255))
        name = 'sem_sorted_boxed_'+str(i)
        cv2.imwrite(os.path.join(output_path,name+'.jpg'),sem_bboxed)
    '''
