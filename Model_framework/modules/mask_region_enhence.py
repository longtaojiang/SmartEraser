from PIL import Image, ImageDraw
from tqdm import tqdm
import numpy as np
import random
import os
import cv2
import bezier
import copy

def gray2whiteblack(gray_image):
    
    img_array = np.array(gray_image)
    binary_img = (img_array > 128).astype(np.uint8) * 255
    mask_image = Image.fromarray(binary_img)
    
    return mask_image

def overlay_mask_on_image(image, mask, color=(255, 0, 0), alpha=0.3):

    if mask.mode != 'L':
        mask = mask.convert('L')
    
    color_layer = Image.new("RGB", mask.size, color)
    mask_with_color = Image.composite(color_layer, Image.new("RGB", mask.size), mask)
    mask_with_color = mask_with_color.convert("RGBA")
    mask_with_color.putalpha(int(alpha * 255))
    image = image.convert("RGBA")
    result_image = Image.alpha_composite(image, mask_with_color)
    result_image = Image.composite(result_image, image, mask)

    return result_image

def bessel_curve_fit(mask_img):
    
    W,H = mask_img.size
    bbox = list(mask_img.getbbox())
    bbox_w = bbox[2] - bbox[0]
    bbox_y = bbox[3] - bbox[1]

    extended_bbox=copy.copy(bbox)
    left_freespace=min(0.5*(bbox[0]-0), 0.2*bbox_w)
    right_freespace=min(0.5*(W-bbox[2]), 0.2*bbox_w)
    up_freespace=min(0.5*(bbox[1]-0), 0.2*bbox_y)
    down_freespace=min(0.5*(H-bbox[3]), 0.2*bbox_y)
    extended_bbox[0]=bbox[0]-random.randint(0,int(left_freespace))
    extended_bbox[1]=bbox[1]-random.randint(0,int(up_freespace))
    extended_bbox[2]=bbox[2]+random.randint(0,int(right_freespace))
    extended_bbox[3]=bbox[3]+random.randint(0,int(down_freespace))

    mask_img = Image.new('RGB', (W, H), (0, 0, 0)) 
    bbox_mask=copy.copy(bbox)
    extended_bbox_mask=copy.copy(extended_bbox)
    top_nodes = np.asfortranarray([
                    [bbox_mask[0],(bbox_mask[0]+bbox_mask[2])/2 , bbox_mask[2]],
                    [bbox_mask[1], extended_bbox_mask[1], bbox_mask[1]],
                ])
    down_nodes = np.asfortranarray([
            [bbox_mask[2],(bbox_mask[0]+bbox_mask[2])/2 , bbox_mask[0]],
            [bbox_mask[3], extended_bbox_mask[3], bbox_mask[3]],
        ])
    left_nodes = np.asfortranarray([
            [bbox_mask[0],extended_bbox_mask[0] , bbox_mask[0]],
            [bbox_mask[3], (bbox_mask[1]+bbox_mask[3])/2, bbox_mask[1]],
        ])
    right_nodes = np.asfortranarray([
            [bbox_mask[2],extended_bbox_mask[2] , bbox_mask[2]],
            [bbox_mask[1], (bbox_mask[1]+bbox_mask[3])/2, bbox_mask[3]],
        ])
    top_curve = bezier.Curve(top_nodes, degree=2)
    right_curve = bezier.Curve(right_nodes, degree=2)
    down_curve = bezier.Curve(down_nodes, degree=2)
    left_curve = bezier.Curve(left_nodes, degree=2)
    curve_list=[top_curve,right_curve,down_curve,left_curve]
    pt_list=[]
    random_width=5
    for curve in curve_list:
        x_list=[]
        y_list=[]
        for i in range(1,19):
            if (curve.evaluate(i*0.05)[0][0]) not in x_list and (curve.evaluate(i*0.05)[1][0] not in y_list):
                pt_list.append((curve.evaluate(i*0.05)[0][0]+random.randint(-random_width,random_width),curve.evaluate(i*0.05)[1][0]+random.randint(-random_width,random_width)))
                x_list.append(curve.evaluate(i*0.05)[0][0])
                y_list.append(curve.evaluate(i*0.05)[1][0])
    mask_img_draw=ImageDraw.Draw(mask_img)
    mask_img_draw.polygon(pt_list,fill=(255,255,255))
    
    return mask_img.convert('L')

def images_composite(img_list, resolution=512):

    width, height = resolution, resolution
    total_width = width * len(img_list)
    max_height = height
    
    new_image = Image.new('RGB', (total_width, max_height))

    for i, img in enumerate(img_list):
        new_image.paste(img, (i*width, 0))
    
    return new_image

def get_area_ratio(img):
    
    img_array = np.array(img)
    binary_img = (img_array > 128).astype(np.uint8) * 255
    mask_area = np.sum(binary_img > 128)
    img_area = img_array.size
    mask_area_ratio = mask_area*1.0 / img_area

    return mask_area_ratio

def get_bbox_and_area_ratio(img):
    
    img_array = np.array(img)
    binary_img = (img_array > 128).astype(np.uint8) * 255
    ypos, xpos = np.where(binary_img > 128)
    min_x, max_x = xpos.min(), xpos.max()
    min_y, max_y = ypos.min(), ypos.max()
    bbox = (min_x, min_y, max_x - min_x, max_y - min_y)
    mask_area = np.sum(binary_img > 128)
    img_area = img_array.size
    mask_area_ratio = mask_area*1.0 / img_area
    bbox_area_ratio = bbox[-2]*bbox[-1]*1.0 / img_area

    return mask_area_ratio, bbox_area_ratio

def get_convexhull_ratio(mask):
    
    mask_np = np.array(mask)
    contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hull = [cv2.convexHull(c) for c in contours]
    hull_mask = np.zeros_like(mask_np)
    cv2.drawContours(hull_mask, hull, -1, 255, -1)
    img_area = mask_np.size
    convexhull_area = np.sum(hull_mask > 128)
    convexhull_ratio = convexhull_area*1.0 / img_area
    
    return convexhull_ratio

class MaskAugmentor:
    def __init__(self, dilate_kernel_range=(5 ,15), erode_kernel_range=(2, 8), max_translation=10):
        self.dilate_kernel_range = dilate_kernel_range
        self.erode_kernel_range = erode_kernel_range
        self.max_translation = max_translation
    
    def dilate_by_area(self, mask, target_ratio, max_iter=100):
        
        mask = np.array(mask)
        initial_area = np.sum(mask > 128)
        target_area = int(initial_area * target_ratio)
        
        if target_area <= initial_area:
            print(target_area, initial_area)
            return Image.fromarray(mask)
        
        current_mask = mask.copy()
        current_area = initial_area
        iteration = 0
        
        while iteration < max_iter and abs(current_area - target_area) > 0:

            area_diff = abs(current_area - target_area)
            kernel_size = max(3, int(np.sqrt(area_diff / 10)))
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            current_mask = cv2.dilate(current_mask, kernel, iterations=1)
            current_area = np.sum(current_mask > 128)
            if current_area >= target_area:
                break
            iteration += 1
            
        return Image.fromarray(current_mask)
    
    def erode_by_area(self, mask, target_ratio, max_iter=100):
        
        mask = np.array(mask)
        initial_area = np.sum(mask > 128)
        target_area = int(initial_area * target_ratio)
        
        if target_area >= initial_area:
            print(target_area, initial_area)
            return Image.fromarray(mask)
            
        
        current_mask = mask.copy()
        current_area = initial_area
        iteration = 0
        
        while iteration < max_iter and abs(current_area - target_area) > 0:

            area_diff = abs(current_area - target_area)
            kernel_size = max(3, int(np.sqrt(area_diff / 10)))
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            current_mask = cv2.erode(current_mask, kernel, iterations=1)
            current_area = np.sum(current_mask > 128)
            if current_area <= target_area:
                break
            iteration += 1
            
        return Image.fromarray(current_mask)
        
    def dilate_mask(self, mask: Image.Image, kernel_size=None):
        mask_np = np.array(mask)
        dilate_kernel_size = random.randint(self.dilate_kernel_range[0], self.dilate_kernel_range[1])
        if kernel_size:
            kernel = np.ones(kernel_size, np.uint8)
        else:
            kernel = np.ones(dilate_kernel_size, np.uint8)
        dilated_mask = cv2.dilate(mask_np, kernel, iterations=3)
        return Image.fromarray(dilated_mask)

    def erode_mask(self, mask: Image.Image, kernel_size=None):
        mask_np = np.array(mask)
        erode_kernel_size = random.randint(self.erode_kernel_range[0], self.erode_kernel_range[1])
        if kernel_size:
            kernel = np.ones(kernel_size, np.uint8)
        else:
            kernel = np.ones(erode_kernel_size, np.uint8)
        erode_mask = cv2.erode(mask_np, kernel, iterations=3)
        return Image.fromarray(erode_mask)
    
    def convex_hull_mask(self, mask: Image.Image):
        mask_np = np.array(mask)
        contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        hull = [cv2.convexHull(c) for c in contours]
        hull_mask = np.zeros_like(mask_np)
        cv2.drawContours(hull_mask, hull, -1, 255, -1)
        return Image.fromarray(hull_mask)
    
    def bounding_box_mask(self, mask: Image.Image):
        mask_np = np.array(mask)
        x, y, w, h = cv2.boundingRect(mask_np)
        bbox_mask = np.zeros_like(mask_np)
        bbox_mask[y:y+h, x:x+w] = 255
        return Image.fromarray(bbox_mask)
    
    def spherical_enclose_mask(self, mask: Image.Image):

        mask_np = np.array(mask)
        mask_np = mask_np.T
        coords = np.column_stack(np.where(mask_np > 128))
        ellipse = cv2.fitEllipse(coords)
        elliptical_mask = np.zeros_like(mask)
        cv2.ellipse(elliptical_mask, ellipse, 255, -1)
        
        return Image.fromarray(elliptical_mask)
    
    def augment_mask(self, mask: Image.Image, mode=None):
        if mode:
            choice = mode
        else:
            mask_area_ratio, bbox_area_ratio = get_bbox_and_area_ratio(mask)
            convexhull_area_ratio = get_convexhull_ratio(mask)
            if (mask_area_ratio/bbox_area_ratio < 0.5 or bbox_area_ratio > 0.4):
                if (mask_area_ratio/convexhull_area_ratio < 0.6 or convexhull_area_ratio > 0.5):
                    if mask_area_ratio > 0.5:
                        choice = 1
                    else:
                        choice = random.randint(1, 3)
                else:
                    choice = random.randint(1, 5)
            else:
                choice = random.randint(1, 6)

        if choice == 1:
            augmented_mask = mask
            
        elif choice == 2:
            if mode:
                ratio = 0.85
            else:
                ratio = 1 - random.randint(10, 20) * 0.01
            augmented_mask = self.erode_by_area(mask, ratio)
        elif choice == 3:
            if mode:
                ratio = 1.10
            else:
                ratio = 1 + random.randint(25, 50) * 0.01
            augmented_mask = self.dilate_by_area(mask, ratio)
        elif choice == 4:
            if mode:
                ratio = 1.35
            else:
                ratio = 1 + random.randint(25, 50) * 0.01
            augmented_mask = self.convex_hull_mask(mask)
            augmented_mask = self.dilate_by_area(augmented_mask, ratio)
        elif choice == 5:
            if mode:
                ratio = 2.25
            else:
                ratio = 1 + random.randint(100, 150) * 0.01
            augmented_mask = self.dilate_by_area(mask, ratio)
            augmented_mask = self.spherical_enclose_mask(augmented_mask)
        elif choice == 6:
            if mode:
                ratio = 1.15
            else:
                ratio = 1 + random.randint(10, 20) * 0.01
            augmented_mask = self.bounding_box_mask(mask)
            augmented_mask = self.dilate_by_area(augmented_mask, ratio)
            augmented_mask = bessel_curve_fit(augmented_mask)
        else:
            ValueError('no such mode')
        
        if get_area_ratio(augmented_mask) > 0.7:
            augmented_mask = mask
        
        return augmented_mask
