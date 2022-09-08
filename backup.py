def bbox_sorting(img, bboxes, top_corner):

    #print('th: ',th)
    keypoints_to_search = copy.deepcopy(top_corner)
    print(len(keypoints_to_search))
    points = []
    while len(keypoints_to_search.keys()) > 0:
        #a = sorted(keypoints_to_search, key=lambda p: (p.pt[0]) + (p.pt[1]))[0]  
        a = dict(sorted(keypoints_to_search.items(), key=lambda item: (item[1][0]+item[1][1])))   # find upper left point
        #b = sorted(keypoints_to_search, key=lambda p: (p.pt[0]) - (p.pt[1]))[-1]  
        b = dict(sorted(keypoints_to_search.items(), key=lambda item: (item[1][0]-item[1][1])))   # find upper right point
        #print(a)
        top_leftmost = a[list(a.keys())[0]]
        top_rightmost = b[list(b.keys())[-1]]
        #print(list(a.keys())[0], ' ',top_leftmost)
        #print(list(b.keys())[-1],' ',top_rightmost)
        #exit()
        #cv2.line(img, (int(top_leftmost[0]), int(top_leftmost[1]), (int(top_rightmost[0]), int(top_rightmost[1])), 255, 1)

        
        # convert opencv keypoint to numpy 3d point
        top_leftmost = np.asarray(top_leftmost)
        top_rightmost = np.asarray(top_rightmost)
        #top_rightmost = np.asarray([top_rightmost[0], top_rightmost[1], 0])

        box_tl = bboxes[list(a.keys())[0]]
        box_tr = bboxes[list(b.keys())[-1]]
        
        th = (abs(box_tl[3] - box_tl[1]) + abs(box_tr[3] - box_tr[1]))/2  #avg height of two box is set as thrteshold
        row_points = {}
        remaining_points = {}
        for i,(key, pt) in enumerate(keypoints_to_search.items()):
            p = np.asarray(pt)
            #d = k.size  # diameter of the keypoint (might be a theshold)
            dist = np.linalg.norm(np.cross(np.subtract(top_rightmost, top_leftmost),np.subtract(top_leftmost,p))) / np.linalg.norm(np.subtract(top_rightmost,top_leftmost))   # distance between keypoint and line a->b
            if dist <= th:
                row_points[key] = pt
            else:
                remaining_points[key] = pt

        points.append(sorted(row_points.items(), key=lambda item: item[1][0]))
        #print(points)
        #print(len(row_points))
        keypoints_to_search = copy.deepcopy(remaining_points)
        #print(len(keypoints_to_search.keys()))
    return points


def draw_sorted_boxes(image,bboxes,sorted_coords):
    for i,row_coords in enumerate(sorted_coords):
        img = copy.deepcopy(image)
        for _,tuple in enumerate(row_coords):
            index = tuple[0]
            bbox = bboxes[index]
            img = draw_rectangle(bbox, img,thickness=2)
        cv2.imwrite(os.path.join('dummy','boxed_'+str(i)+'.jpg'),img)
    return image

def components_merging(row_images, row_bboxes_local, output_path):
    updated_list = []
    distances = []
    for index, (row_image, bboxes) in enumerate(zip(row_images,row_bboxes_local)):
        distance = calculate_distance(bboxes)
        distances.extend(distance)
        threshold = th_selection(distance)
        #exit()
        updated_bboxes = composite_component_formation(bboxes, distance, threshold=threshold)
        row_image = draw_bboxes(row_image,updated_bboxes)
        splated_bboxes = splating_bbox(updated_bboxes,row_image)

        if index%2==1:  #every 2nd one
            len_last_comp = len(updated_list[len(updated_list)-1])  #length of last component list
            if len(splated_bboxes) < len_last_comp:
                updated_list[len(updated_list)-1] = splated_bboxes
        else:
            updated_list.append(splated_bboxes)
    #print('on whole image')
    #th_selection(distances)
    return updated_list

def bbox_sorting_2(image, bboxes_wh, max_height):
    #sorted_bboxes_wh = sorted(bboxes_wh, key=lambda bbox: bbox[0] + bbox[1] * image.shape[1] )
    nearest = max_height * 1.2
    print('max h, nearest: ', max_height,'  ',nearest)

    bboxes_wh.sort(key=lambda r: [int(nearest * round(float(r[1]) / nearest)), r[0]])

    '''
    for x, y, w, h in bboxes_wh:
        print(f"{x:4} {y:4} {w:4} {h:4}") 
    '''
    return bboxes_wh

def find_rows(image, bboxes_wh):
    row_bboxes_wh = []
    row = 0
    holder = []
    img = copy.deepcopy(image)
    for i in range(1,len(bboxes_wh)):
        holder.append(bboxes_wh[i-1])
        if bboxes_wh[i-1][0] > bboxes_wh[i][0]: #x-coordinate of previous is greater than x-coordinate of current
            img = draw_rectangle(bboxes_wh[i-1], img, wh=True, thickness=2)
            row+=1
            row_bboxes_wh.append(copy.deepcopy(holder))
            holder = []
    print('#row: ',row)
    print('#row_bboxes: ', len(row_bboxes_wh))
   
    cv2.imwrite(os.path.join('dummy','terminal_bbox.jpg'),img)
    return row_bboxes_wh