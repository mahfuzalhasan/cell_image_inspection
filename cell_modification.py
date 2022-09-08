import cv2
from skimage.measure import label, regionprops
from sklearn.mixture import GaussianMixture as GMM
from sklearn.cluster import KMeans


import copy
import os



import parameters_cell_mod as params_m
from utility import binarized, draw_bboxes, draw_rectangle, visualization
from components import connected_components, regions
from scipy import signal
from skimage.metrics import structural_similarity as ssim
from nms import NMS


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
    bboxes, _ , _, _, _ = regions(blobs, bin_img, partial_check=False)
    return bboxes, _


def calculate_iou(box_lout, box_sem):
    box_lout = list(box_lout)
    box_sem = list(box_sem)

    x_left = max(box_lout[0], box_sem[0])
    y_top = max(box_lout[1], box_sem[1])
    x_right = min(box_lout[2], box_sem[2])
    y_bottom = min(box_lout[3], box_sem[3])
    
    if (x_right < x_left) or (y_bottom < y_top):
        return 0.0
    
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb_lout_area = (box_lout[2] - box_lout[0]) * (box_lout[3] - box_lout[1])
    bb_sem_area = (box_sem[2] - box_sem[0]) * (box_sem[3] - box_sem[1])

    iou = intersection_area / float(bb_lout_area + bb_sem_area - intersection_area)
    return iou


def finding_neighbor(bboxes_sem, bboxes_lout):
    neighbors = np.full(len(bboxes_sem),-1)
    centroid = []
    
    centroid_sem = [[(bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2] for bbox in bboxes_sem]
    centroid.extend(centroid_sem)
    
    centroid_lout = [[(bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2] for bbox in bboxes_lout]
    centroid.extend(centroid_lout)

    centroid = np.asarray(centroid)

    n_cluster = int(len(centroid)/2)
    print('n_cluster: ',n_cluster)
    kmeans = KMeans(n_clusters=n_cluster, random_state=0).fit(centroid)
    labels = np.array(kmeans.labels_)
    print('labels: ',labels)

    for i in range(n_cluster):
        indx = np.where(labels==i)
        print(indx[0])
        index_1, index_2 = int(indx[0][0]), int(indx[0][1])

        neighbors[index_1] = index_2-len(neighbors)    
    
    return neighbors, centroid_sem, centroid_lout




if __name__=="__main__":
    layout = read_input(params_m.layout_cell_path)
    sem = read_input(params_m.sem_cell_path)
    output_path = os.path.join(params_m.output,'cell_modification','row_12')

    print(layout.shape,' ',sem.shape)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    sem = resizing(layout,sem)

    cell_sem = sem[:,:,0]
    cell_lout = layout[:,:,0]

    denoised_sem = cv2.fastNlMeansDenoising(cell_sem,None,20,5,20)
    cv2.imwrite(os.path.join(output_path, 'denoised_sem.jpg'),denoised_sem)
    
    bin_sem, _ = binarized(denoised_sem, output_path)
    cv2.imwrite(os.path.join(output_path,'binarized_sem.jpg'),bin_sem)

    bin_lout, _ = binarized(cell_lout, output_path)
    cv2.imwrite(os.path.join(output_path,'binarized_lout.jpg'),bin_lout)

    cor = signal.correlate2d (bin_lout, bin_sem)

    s = ssim(bin_lout, bin_sem)
    print('structural similarity: ',s)

    bboxes_sem, _ = bboxed(bin_sem)
    bboxes_lout, _ = bboxed(bin_lout)
    #print('layout components before: ',bboxes_lout)
    #bboxes_lout = NMS(bboxes_lout, 0.9)

    print('sem components: ',bboxes_sem)
    print('layout components: ',bboxes_lout)

    layout_bboxed = draw_bboxes(layout, bboxes_lout, color=(0,255,0))
    cv2.imwrite(os.path.join(output_path,'lout_boxed.jpg'),layout_bboxed)

    sem_bboxed = draw_bboxes(sem, bboxes_sem, color=(0,0,255))
    cv2.imwrite(os.path.join(output_path,'sem_boxed.jpg'),sem_bboxed)


    #visualization(layout, sem, bin_sem, bin_lout, bboxes_sem, bboxes_lout, centroid_sem, centroid_lout, output_path)

    neighbors, centroid_sem, centroid_lout = finding_neighbor(bboxes_sem, bboxes_lout)

    

    print('neighbors: ',neighbors)
    for i in range(len(neighbors)):
        iou = calculate_iou(bboxes_lout[neighbors[i]], bboxes_sem[i])
        print('iou between layout comp ',neighbors[i], ' and sem comp ',i, ': ',iou)


    ##drawing
    #layout[:,:,1:3] = 0
    #sem[:,:,0:2] = 0
    visualization(layout, sem, bin_sem, bin_lout, bboxes_sem, bboxes_lout, centroid_sem, centroid_lout, output_path)
    


    






