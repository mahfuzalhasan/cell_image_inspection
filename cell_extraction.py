import cv2
import numpy as np
import copy
from skimage.measure import label, regionprops
from sklearn.mixture import GaussianMixture as GMM
import parameters as params
import os
import functools
from utility import *
from components import *
#import input_processing as input_reading
#import component_generation as comp_gen

import math
import statistics


def pre_processing(output_path, path):
    img = read_input(path)
    img_denoised = denoised(img, output_path)
    img_bin, th = binarized(img_denoised, output_path)
    return img, img_denoised, img_bin, th




if __name__=="__main__":
    for setId in os.listdir(params.paths):
        #print(setId)
        paths = os.path.join(params.paths, setId)
        for img in os.listdir(paths):
            print('----------------------------------------------------------------')
            print("Set: ",setId, " Image: ",img)
            img_name = img[:img.rindex('.')]
            output_path = os.path.join(params.output, setId, img_name)
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            img, img_denoised, img_bin, th = pre_processing(output_path, os.path.join(paths,img))

            blobs = connected_components(img_bin)
            
            bboxes, top_corners, max_height, _, _ = regions(blobs, img)

            
            bboxes_wh = [(bbox[0],bbox[1],bbox[2]-bbox[0],bbox[3]-bbox[1]) for bbox in bboxes]

            print('#bbox: ',len(bboxes_wh))
            img = np.array(img)
            img_boxed = draw_bboxes(img, bboxes)
            cv2.imwrite(os.path.join(output_path,'img_bboxed.jpg'),img_boxed)
            row_separator_lines, terminal_boxed_img, row_bboxes = bbox_sorting(img, bboxes, output_path)
            #print(sorted_bboxes_wh)
            #row_bboxes_wh = find_rows(img, sorted_bboxes_wh)
            print('row of bboxes: ',len(row_bboxes))
            #draw_sorted_boxes(img, row_bboxes)
            
            
            row_images, row_bboxes_local, cell_rows = generate_row_images(img, row_separator_lines, row_bboxes, output_path)
            print("row images: ",len(row_images), " #row_bboxes:", len(row_bboxes_local), " #cell_rows: ",len(cell_rows))

            merged_cells = components_merging(row_images, row_bboxes_local, output_path)
            #exit()
            cell_generation(cell_rows, merged_cells, output_path)
        print(" _________________   __________________  ___________________")