import cv2
import os
import copy

import numpy as np
import math

from sklearn.mixture import GaussianMixture as GMM


def read_input(path):
    img = cv2.imread(path)
    img = img[:,:,0]
    print('img shape: ',img.shape)
    return img


def denoised(img, output_path):
    denoised_img = cv2.fastNlMeansDenoising(img,None,30,7,30)
    cv2.imwrite(os.path.join(output_path, 'denoised.jpg'),denoised_img)
    return denoised_img
    
    
def binarized(img, output_path, threshold = -1):
    th, img_th = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    print(' OT threshold: ',th)
    #print('threshold using: ',th+10)
    th, img_th = cv2.threshold(img, th+10, 255 , cv2.THRESH_BINARY)
    #cv2.imwrite(os.path.join(output_path,'binarized.jpg'),img_th)
    #print('threshold using: ',th)
    return img_th, th

def draw_centroid(centroids, image, output_path, radius = 3, row_type = 'comp', img_type = 'sem'):

    img = copy.deepcopy(image)
    if 'sem' in img_type:
        color = (0,0,0)
    else:
        color = (0,255,0)
    if row_type == 'cell':
        for row_centroids in centroids:
            for centroid in row_centroids:
                cv2.circle(img, (round(centroid[0]), round(centroid[1])), radius, color, -1)
    else:
        for centroid in centroids:
            cv2.circle(img, (round(centroid[0]), round(centroid[1])), radius, color, -1)

    
    return img
    

def visualization(layout, sem, bin_sem, bin_lout, bboxes_sem, bboxes_lout, centroid_sem, centroid_lout, output_path):

    layout_bboxed = draw_bboxes(layout, bboxes_lout, color=(0,255,0))
    cv2.imwrite(os.path.join(output_path,'lout_boxed.jpg'),layout_bboxed)

    img = draw_centroid(centroid_lout,layout_bboxed,output_path,img_type='layout')
    cv2.imwrite(os.path.join(output_path,'lout_boxed_centroid.jpg'),img)

    layout_bboxed = draw_bboxes(layout_bboxed, bboxes_sem, color=(0,0,255))
    cv2.imwrite(os.path.join(output_path,'lout_double_boxed.jpg'),layout_bboxed)

    sem_bboxed = draw_bboxes(sem, bboxes_sem, color=(0,0,255))
    cv2.imwrite(os.path.join(output_path,'sem_boxed.jpg'),sem_bboxed)
    img = draw_centroid(centroid_sem,sem_bboxed,output_path,img_type='sem')
    cv2.imwrite(os.path.join(output_path,'sem_boxed_centroid.jpg'),img)

    cv2.imwrite(os.path.join(output_path,'cell_layout.jpg'),layout)
    cv2.imwrite(os.path.join(output_path,'cell_sem.jpg'),sem)

    ###for drawing overlay with different color
    layout[:,:,1:3] = 0
    #layout[:,:,2] = 0
    cv2.imwrite(os.path.join(output_path,'layout_rgb.jpg'),layout)


    bin_sem_2 = cv2.merge((bin_sem,bin_sem,bin_sem))
    bin_sem_2[:,:,0:2] = 0
    cv2.imwrite(os.path.join(output_path,'sem_rgb.jpg'),bin_sem_2)

    overlay = cv2.addWeighted(layout,0.8,bin_sem_2,0.5,0)
    cv2.imwrite(os.path.join(output_path,'overlay.jpg'),overlay)

    overlay_centroid = draw_centroid(centroid_sem, overlay, output_path, radius=2, img_type='sem')
    overlay_centroid = draw_centroid(centroid_lout, overlay_centroid, output_path, radius=2, img_type='layout')
    cv2.imwrite(os.path.join(output_path,'overlay_centroid.jpg'),overlay_centroid)

    overlay_bboxed = draw_bboxes(overlay, bboxes_sem, color=(255,255,255))
    overlay_bboxed = draw_bboxes(overlay_bboxed, bboxes_lout, color=(0,255,0))
    cv2.imwrite(os.path.join(output_path,'overlay_bboxed.jpg'),overlay_bboxed)

    
    overlay_bboxed_centroid = draw_centroid(centroid_sem, overlay_bboxed, output_path, radius=2, img_type='sem')
    overlay_bboxed_centroid = draw_centroid(centroid_lout, overlay_bboxed_centroid, output_path, radius=2, img_type='layout')
    cv2.imwrite(os.path.join(output_path,'overlay_bboxed_centroid.jpg'),overlay_bboxed_centroid)

def draw_rectangle(bbox, image, color=(0,0,0), wh=False, thickness=1):
    start_point = (bbox[0],bbox[1])
    end_point = (bbox[2],bbox[3])
    #print(start_point)
    if wh:
        end_point = (bbox[0]+bbox[2], bbox[1]+bbox[3])
    #print(end_point)
    image = cv2.rectangle(image, start_point, end_point, color, thickness)
    return image

def draw_bboxes(image, bboxes,color=(0,0,0),wh=False):
    image = copy.deepcopy(image)
    for _,bbox in enumerate(bboxes):
        if wh:          #bbox in x,y,w,h format
            bbox = [bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]] #x1,y1,x2,y2
        image = draw_rectangle(bbox, image, color)
    return image


def draw_sorted_boxes(image, row_bboxes):
    for i,row_box_list in enumerate(row_bboxes):
        img = copy.deepcopy(image)
        for _,bbox in enumerate(row_box_list):
            img = draw_rectangle(bbox, img,thickness=2)
        cv2.imwrite(os.path.join('dummy','boxed_'+str(i)+'.jpg'),img)


def generate_row_images(image, lines, bboxes_row_sorted, output_path):
    #lines = lines[1:len(lines)-1]
    omit = 0
    output = []
    cell_rows = []
    r_bboxes_local = []

    output_row_image_path = os.path.join(output_path,"rows")
    if not os.path.exists(output_row_image_path):
        os.makedirs(output_row_image_path)

    if len(lines)%2==0:
        omit = 2
    else:
        omit = 1

    gate_row_count = 0

    for i in range(0, len(lines)-omit):   #some portion from bottom are omitted    
        bboxes = bboxes_row_sorted[i]
        curr_line = lines[i]
        next_line = lines[i+1]
        x1, y1 = curr_line[0], curr_line[1]
        x2, y2 = next_line[2], next_line[3] 
        row_image = image[y1:y2+1, x1:x2+1]

        bboxes_local = [[bbox[0], bbox[1]-y1, bbox[2], bbox[3]-y1] for bbox in bboxes]

        if i%2 == 0:
            next_next_line = lines[i+2]
            x2, y2 = next_next_line[2], next_next_line[3]
            cell_row = image[y1:y2+1, x1:x2+1]
            #clean_rows(gate_row, gate_row_count, components, bboxes_row_sorted[i+1], next_line[1]-curr_line[1],output_path)
            gate_row_count += 1
            cell_rows.append(cell_row)
            cv2.imwrite(os.path.join(output_row_image_path,'cell_row_'+str(i)+'.png'),cell_row)

        output.append(row_image)
        r_bboxes_local.append(bboxes_local)
        # to check out correctedness
        row_image = draw_bboxes(row_image, bboxes_local)
        cv2.imwrite(os.path.join(output_row_image_path,'row_'+str(i)+'.png'),row_image)

    return output, r_bboxes_local, cell_rows



def calculate_distance(comps):
    distance_x = []
    for i in range(1,len(comps)):
        pre_bbox = comps[i-1]
        bbox = comps[i]
        distance_x.append(bbox[0]-pre_bbox[2]) #next_box_x1 - prev_box_x2
    #print('distance x: ',distance_x)
    return distance_x

def merging(merge):
    return [min(x[0] for x in merge),min(x[1] for x in merge),max(x[2] for x in merge),max(x[3] for x in merge)]

def splating_bbox(components, img):
    height, width = img.shape
    c = []
    for comp in components:
        comp = [max(0,comp[0]-3), max(0,comp[1]-3), min(width-1,comp[2]+3), min(height-1,comp[3]+3)]
        c.append(comp)
    return c

def cropped_coordinates(height,width,component):
    return component[0], 0, component[2], height-1

def save_cell_image(image, row_no, comp_no, path):
    output_image_path = os.path.join(path,"cell_images",str(row_no))
    if not os.path.exists(output_image_path):
        os.makedirs(output_image_path)
    cv2.imwrite(os.path.join(output_image_path,str(comp_no)+'.jpg'),image)

def th_selection(distances):
    sorted_distances = sorted(distances)
    #print('sorted: ',sorted_distances)
    average = sum(sorted_distances)/len(sorted_distances)
    std = np.std(sorted_distances)
    #print('avg: ',average,' std: ',std)
    s_distances = np.asarray(sorted_distances).reshape(-1,1)
    #print('sorted: ',s_distances)
    gmm_model = GMM(n_components=3, covariance_type='tied').fit(s_distances)
    gmm_labels = np.asarray(gmm_model.predict(s_distances))
    means = gmm_model.means_
    lm_cluster_index = np.argmin(means)
    mean = means[lm_cluster_index][0]
    
    terminal_distances = np.count_nonzero(np.where(sorted_distances==mean))
    #print('terminal distances: ',terminal_distances)
    threshold = int(math.ceil(mean))


    '''
    lm_cluster_samples_index = np.where(gmm_labels == lm_cluster_index)
    sorted_distances = np.asarray(sorted_distances)
    lm_cluster = sorted_distances[lm_cluster_samples_index]
    std = np.std(lm_cluster)
    
    


    print('--------------------------------------')
    print('gmm means: ',means)
    print('gmm_labels: ',gmm_labels)
    print("lowest mean cluster: ",lm_cluster_index)
    #print('lm_cluster index: ',lm_cluster_samples_index)
    #print('lm_cluster: ',lm_cluster)
    print('std: ', std)
    print('lowest mean: ', means[lm_cluster_index][0] )
    print('threshold: ',threshold)
    print('---------------------------------------')
    '''

    return threshold