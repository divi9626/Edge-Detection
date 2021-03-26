
import numpy as np
import cv2 as cv
import copy
import imutils

def tag_orient(warp):
    map = {'TR': (2, 3, 0, 1), 'BR': (1, 2, 3, 0), 'TL': (3, 0, 1, 2), 'BL': (0, 1, 2, 3)}
    # check for orientation nd determine if tr/br/tl/bl
    # take the key from the map
    if warp[20:30,20:30].mean() < 175:
        return map['TL']
    elif warp[20:30,50:60].mean() < 175:
        return map['TR']
    elif warp[50:60,20:30].mean() < 175:
        return map['BL']
    elif warp[50:60,50:60].mean() < 175:
        return map['BR']
    else:
        return map['BL']
    
def tag_id(warp):
    # a list to store the returned value of orient
    check = tag_orient(warp)
    id_list = []
    sum = 0
    for i in check:
        id = cell_check(i)
        if warp[id[0]:id[1],id[2]:id[3]].mean() > 125:
            id_list.append(0)
        else:
            id_list.append(1)
    for j in range(len(id_list)):
        if id_list[j] == 0:
            sum = sum + 0
        else:
            sum = sum + 2**j
    return sum, check

def cell_check(i):
    if i == 0:
        a,b,c,d = 30,40,30,40
    if i == 1:
        a,b,c,d = 30,40,40,50
    if i == 2:
        a,b,c,d = 40,50,40,50
    if i == 3:
        a,b,c,d = 40,50,30,40
    return [a,b,c,d]

def get_corners(contour):
    epsilon = 0.05
    count = 0
    while True:
        perimeter = cv.arcLength(contour,True)
        perimeter = epsilon*perimeter
        if perimeter > 100 or perimeter < 1:
            return None
        approx = cv.approxPolyDP(contour,perimeter,True)
        print(perimeter)
        hull = cv.convexHull(approx)
        if len(hull) == 4:
            return hull
        else:
            if len(hull) > 4:
                epsilon += 0.01
            else:
                epsilon -= 0.01
        if count > 10:
            return []
        
def ar_tag_contours(contours, contour_hierarchy):
    paper_contours_ind = []
    ar_tag_contours = []
    for ind, contour in enumerate(contour_hierarchy[0]):
        if contour[3] == 0:
            paper_contours_ind.append(ind)
            
    if (len(paper_contours_ind) > 3):
        return None
    for ind in paper_contours_ind:
        ar_tag_contour_ind = contour_hierarchy[0][ind][2]
        ar_tag_contours.append(contours[ar_tag_contour_ind])
        
    return ar_tag_contours
        
        
def arrange(corners):
    corners = corners.reshape((4, 2))
    new = np.zeros((4, 1, 2), dtype=np.int32)
    add = corners.sum(1)
    
    new[0] = corners[np.argmin(add)]
    new[2] =corners[np.argmax(add)]
    diff = np.diff(corners, axis=1)
    new[1] =corners[np.argmin(diff)]
    new[3] = corners[np.argmax(diff)]
    return new

def homograph(src_plane, dest_plane):
    A = []

    for i in range(0, len(src_plane)):
        x, y = src_plane[i][0], src_plane[i][1]
        xp, yp = dest_plane[i][0], dest_plane[i][1]
        A.append([-x, -y, -1, 0, 0, 0, x * xp, y * xp, xp])
        A.append([0, 0, 0, -x, -y, -1, x * yp, y * yp, yp])
        
    A = np.asarray(A) 
    
    U, S, VT = np.linalg.svd(A)
    
    # normalizing
    l = VT[-1, :] / VT[-1, -1]
    
    H_matrix = l.reshape(3,3)

    return H_matrix
    
def get_warp(H,dest,src_img,dest_img):
    
    ans = copy.deepcopy(dest_img)
    
    H_inv = np.linalg.inv(H)
    H_inv = H_inv/H_inv[2][2]
    col_min,row_min = np.min(dest, axis =0)
    col_max, row_max = np.max(dest, axis = 0)
    #height = int(row_max - row_min)
    #width = int(col_max - col_min)
    src_img_dim = src_img.shape
    
    
    for y in range(int(row_min)+1, int(row_max)):
        for x in range(int(col_min)+1,int(col_max)):
            dest_pt = np.float32([x, y, 1]).T
            src_pt = np.dot(H_inv, dest_pt)
            a = src_pt[1]/src_pt[2]
            b = src_pt[0]/src_pt[2]
            #src_pt = (src_pt/src_pt[2]).astype(int)
            if ((int(a) in range(0, src_img_dim[0])) and (int(b) in range(0, src_img_dim[1]))):
                    try:
                        ans[y][x] = src_img[int(np.round(src_pt[1]/src_pt[2]))][int(np.round(src_pt[0]/src_pt[2]))]
                    except:
                        continue
                    
    return ans

cap = cv.VideoCapture('Tag0.mp4')
if cap.isOpened() == False:
    print("Error opening the image")
    
img = None
count = 0

while cap.isOpened():
    count += 1
    ret, frame = cap.read()
    if ret == False:
        break
    try:
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    except:
        break
    gray = cv.resize(gray,(600,600))
    frame = cv.resize(frame,(600,600))
    _,thresh = cv.threshold(gray,200,255, cv.THRESH_BINARY_INV)

    if count == 50:
        #img_c = frame
        img_frame = frame
        img = thresh
        img_c = gray
        
    cv.imshow('imgg',gray)
    
    if cv.waitKey(1) == 27:
        break
#img = np.float32(img)
print(gray.shape)
cap.release()
cv.destroyAllWindows()

width, height = 80,80

contours, hierarchy = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)  # get all contours
#print("Number of contours = " + str(len(contours)))
contour = contours[2]  # get the tag contour
cv.drawContours(frame, contours, 2, (0,0,255), 1)



get_ar_tag_contours = ar_tag_contours(contours,hierarchy)
warp_test = img_c
if get_ar_tag_contours is not None:

    for contour in get_ar_tag_contours:
        
        cv.drawContours(img_frame, contours, 2, (0,0,255), 1)
    
        New = get_corners(contour)
        if New is not None:# get corners of the tag
            source  = arrange(New)  # arrange the corners in proper orientation (for warping)
            #show_corners(New)
            source = np.float32(source)
            cor = []
            for i in source:
                cor.append(i[0])
            cor = np.asarray(cor)
            
            cor = np.float32(cor)
        
            New = get_corners(contour)  # get corners of the tag
            source  = arrange(New)  # arrange the corners in proper orientation (for warping)
            #show_corners(New)
            source = np.float32(source)
            cor = []
            for i in source:
                cor.append(i[0])
            cor = np.asarray(cor)
            cor_handle = cor[3]
            cor_handle = (int(cor_handle[0]) + 20, int(cor_handle[1]) + 20)
            
            cor = np.float32(cor)
            
            dest  = np.float32([[0,0],[width,0], [width,height], [0,height]])
            dest_img = np.zeros((80,80))
            H = homograph(cor,dest)
            warp = get_warp(H,dest,img,dest_img)



tag_num = tag_id(warp)
print(tag_num[0],tag_num[1])


# font 
font = cv.FONT_HERSHEY_SIMPLEX 
  
  
# fontScale 
fontScale = 0.5
   
# Blue color in BGR 
color = (0, 0, 255) 
  
# Line thickness of 2 px 
thickness = 1
   
# Using cv2.putText() method 
image = cv.putText(img_frame, str(tag_num[0]), cor_handle, font,  
                   fontScale, color, thickness, cv.LINE_AA)

while (cv.waitKey(1) != 27):
    cv.imshow('img' , img_frame)
    
cv.destroyAllWindows()