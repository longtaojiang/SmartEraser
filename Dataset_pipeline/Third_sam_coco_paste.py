import json
import os
import argparse
import numpy as np
import multiprocessing as mp
from PIL import Image, ImageFile, ImageDraw, ImageChops
from torchvision import transforms
from tqdm import tqdm
import random as rd
from numpy import random
import cv2
from pymatting import estimate_alpha

image_transforms_resize_and_crop = transforms.Compose(
            [
                transforms.Resize(512, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(512),
            ]
        )

def erode_to_gray(image):

    img = image.convert("L")
    img = np.array(img)
    img[img > 128] = 255
    img[img < 128] = 0
    kernel = np.ones((3, 3), np.uint8)
    eroded_img = cv2.erode(img, kernel, iterations=3)
    eroded_img[eroded_img > 128] = 255
    eroded_img[eroded_img < 128] = 0
    diff = cv2.absdiff(img, eroded_img)
    eroded_img[diff > 128] = 128

    return eroded_img

#Scale the instance based on the mean of the instance and the total proportion of bbox, and take the minimum value of the remaining 1/2 of the mean and bbox proportion
def ins_resize(overlay, background_size, area_samples, bbox_list, img_id):
    ins_width, ins_height = overlay.size
    img_width, img_height = background_size

    area_all = bboxes_area_all_compute(bbox_list, background_size)
    img_area = img_width * img_height
    ins_area = ins_width * ins_height
    area_blank = 1.0 - (area_all*1.0 / img_area)
    area_blank = area_blank / 2.0

    area_need = min(area_blank, area_samples)
    area_need = area_need if area_need > 0.05 else 0.05
    area_need = area_need * img_area
    scale_factor = (area_need / ins_area) ** 0.5

    new_width = int(ins_width * scale_factor)
    new_height = int(ins_height * scale_factor)
    
    #sepical case
    if new_width <= 0 or new_width <= 0:
        print(img_id)
        print(overlay.size)
        print(scale_factor)
        print(area_need, area_blank, area_samples)
        return overlay

    overlay = overlay.resize((new_width, new_height))

    return overlay

def bboxes_area_all_compute(bbox_list, background_size):
    
    img_width, img_height = background_size
    mask = np.zeros((int(img_width) + 1, int(img_height) + 1), dtype=np.int8)
    
    for x, y, w, h in bbox_list:
        x = int(x)
        y = int(y)
        w = int(w)
        h = int(h)
        mask[x:x+w, y:y+h] = 1
    
    total_area = np.sum(mask)
    
    return total_area

#Calibrate all bbox and edges, and randomly select a point within the optional area
def random_pas_location(ins_size, image_size, bbox_list, args, img_id):

    img_width, img_height = image_size
    ins_width, ins_height = ins_size
    mask = Image.new('L', (img_width, img_height), 255)
    draw = ImageDraw.Draw(mask)

    edge_ratio = args.edge_mask_ratio
    width_edge = int(img_width * edge_ratio)
    width_edge_start = int(img_width * (1 - edge_ratio))
    hight_edge = int(img_height * edge_ratio)
    hight_edge_start = int(img_height * (1 - edge_ratio))

    draw.rectangle([0, 0, width_edge, img_height], fill=0)
    draw.rectangle([width_edge_start, 0, img_width, img_height], fill=0)
    draw.rectangle([0, 0, img_width, hight_edge], fill=0)
    draw.rectangle([0, hight_edge_start, img_width, img_height], fill=0)

    for x, y, w, h in bbox_list:
        draw.rectangle([x, y, x + w, y + h], fill=0)

    mask_array = np.array(mask)
    mask_array = (mask_array==255).astype(bool)

    indices = np.argwhere(mask_array)
    y_coords, x_coords = indices[:, 0], indices[:, 1]
    max_instance_iou = args.max_instance_iou
    min_self_iou = args.instance_self_ration
    max_ins_resize = args.max_ins_resize

    paste_x = -999
    paste_y = -999
    fail_time = -999
    paste_resize = -1
    min_ins_ratio = 0.05

    if len(x_coords) == 0 or len(y_coords) == 0:
        paste_x = np.random.randint(width_edge, width_edge_start)
        paste_y = np.random.randint(hight_edge, hight_edge_start)
        fail_time = -1
        paste_resize = max_ins_resize
        
    else:
        
        for k in range(max_ins_resize):
            ins_width_new = ins_width // (k + 1)
            ins_height_new = ins_height // (k + 1)
        
            for i in range(10):
                random_index = np.random.randint(0, len(x_coords))
                random_x = x_coords[random_index]
                random_y = y_coords[random_index]
                bbox_a = [random_x - ins_width_new // 2, random_y - ins_height_new // 2, ins_width_new, ins_height_new]
                iou_ok = compute_iou(bbox_a, bbox_list, max_instance_iou)
                self_iou_ok = compute_self_iou(bbox_a, image_size, min_self_iou)

                if iou_ok == True and self_iou_ok == True:
                    paste_x = random_x
                    paste_y = random_y
                    fail_time = i
                    paste_resize = (k + 1)
                    break
        
        if paste_resize == -1:
            paste_x = np.random.randint(width_edge, width_edge_start)
            paste_y = np.random.randint(hight_edge, hight_edge_start)
            fail_time = -1
            paste_resize = max_ins_resize
            print(f"Random fail more than 10 times, ins size {ins_size}, img size{image_size} and have to 1/{max_ins_resize*max_ins_resize} {ins_width_new, ins_height_new}, img_id:{img_id}.")
            ins_width_new = ins_width // max_ins_resize
            ins_height_new = ins_height // max_ins_resize
            for i in range(10):
                random_index = np.random.randint(0, len(x_coords))
                random_x = x_coords[random_index]
                random_y = y_coords[random_index]
                bbox_a = [random_x - ins_width_new // 2, random_y - ins_height_new // 2, ins_width_new, ins_height_new]
                self_iou_ok = compute_self_iou(bbox_a, image_size, min_self_iou)

                if self_iou_ok == True:
                    paste_x = random_x
                    paste_y = random_y
                    fail_time = i
                    paste_resize = max_ins_resize
                    break
            
    paste_resize_max = paste_resize
    paste_resize = 1
    for i in range(paste_resize_max, 0, -1):
        ins_width_new = ins_width // i
        ins_height_new = ins_height // i
        
        if ins_width_new*ins_height_new >= min_ins_ratio*img_width*img_height:
            paste_resize = i
            break
        
    return paste_x, paste_y, fail_time, paste_resize

def compute_iou(bbox_a, bbox_list, max_instance_iou):

    for bbox_b in (bbox_list):
        xA = max(bbox_a[0], bbox_b[0])
        yA = max(bbox_a[1], bbox_b[1])
        xB = min(bbox_a[0] + bbox_a[2], bbox_b[0] + bbox_b[2])
        yB = min(bbox_a[1] + bbox_a[3], bbox_b[1] + bbox_b[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        area_b = bbox_b[2]*bbox_b[3]
        iou = interArea / area_b if area_b != 0 else 0
        if iou > max_instance_iou:
            return False

    return True

def compute_self_iou(bbox_a, image_size, min_self_iou):

    img_width, img_height = image_size
    x, y, w, h = bbox_a[0], bbox_a[1], bbox_a[2], bbox_a[3]
    ori_area = w * h

    if x < 0:
        w = w + x
    elif x + w > img_width:
        w = w - (x + w - img_width)

    if y < 0:
        h = h + y
    elif y + h > img_height:
        h = h - (h + y - img_height)

    area_iou = w * h * 1.0 / ori_area

    if area_iou >= min_self_iou:
        return True
    else:
        return False

#Paste ins to the specified location, including random rotation and mask generation
def paste_ins2img(position, overlay, background):

    center_x, center_y = position
    rotation_degree = rd.uniform(-45, 45)  # Random rotation angle range
    rotated_overlay = overlay.rotate(rotation_degree, expand=True)

    overlay_width, overlay_height = rotated_overlay.size
    start_x = center_x - overlay_width // 2 - 1
    start_y = center_y - overlay_height // 2 - 1
    end_x = center_x + overlay_width // 2 + 1
    end_y = center_y + overlay_height // 2 + 1

    if start_x < 0:
        start_x = 0
        rotated_overlay = rotated_overlay.crop((-start_x, 0, overlay_width, overlay_height))
    elif end_x > background.width:
        rotated_overlay = rotated_overlay.crop((0, 0, overlay_width - (end_x - background.width), overlay_height))
    
    if start_y < 0:
        start_y = 0
        rotated_overlay = rotated_overlay.crop((0, -start_y, overlay_width, overlay_height))
    elif end_y > background.height:
        rotated_overlay = rotated_overlay.crop((0, 0, overlay_width, overlay_height - (end_y - background.height)))

    mask = rotated_overlay.split()[3]  # Obtain the mask of the transparency channel and reverse it
    ori_image = background.copy()
    background.paste(rotated_overlay, (start_x, start_y), mask)
    background_mask = Image.new("L", background.size, 0)
    background_mask.paste(mask, (start_x, start_y), mask)
    overlay_width_crop, overlay_height_crop = rotated_overlay.size
    paste_bbox = [int(start_x), int(start_y), overlay_width_crop, overlay_height_crop]

    return ori_image, background_mask, background, paste_bbox

def resize_and_crop_bboxes(image_size, bboxes):

    original_width, original_height = image_size

    short_side = min(original_width, original_height)
    scale_factor = 512 / short_side
    new_width = int(original_width * scale_factor)
    new_height = int(original_height * scale_factor)

    # Calculate the coordinates of the center crop
    start_x = (new_width - 512) // 2
    start_y = (new_height - 512) // 2

    resized_bboxes = []
    for bbox in bboxes:
        x, y, w, h = bbox
        # Convert to resized coordinate system
        x = x * scale_factor
        y = y * scale_factor
        w = w * scale_factor
        h = h * scale_factor
        resized_bboxes.append([x, y, w, h])
    
    # Update bbox coordinates and consider crop offset
    cropped_bboxes = []
    for bbox in resized_bboxes:
        x, y, bbox_w, bbox_h = bbox
        cropped_x = x - start_x
        cropped_y = y - start_y
        
        # Only keep the bbox within the crop area
        if cropped_x + bbox_w > 0 and cropped_y + bbox_h > 0 and cropped_x < 512 and cropped_y < 512:
            cropped_x = max(0, cropped_x)
            cropped_y = max(0, cropped_y)
            cropped_bbox_w = min(bbox_w, 512 - cropped_x)
            cropped_bbox_h = min(bbox_h, 512 - cropped_y)
            cropped_bboxes.append([cropped_x, cropped_y, cropped_bbox_w, cropped_bbox_h])

    return cropped_bboxes

#Paste processing main function, input as a type of instance and corresponding quantity of images
#{"name", "number", "mean_scores", "var_scores", "mean_areas", "var_areas", "main_path", "paths"}
#{(img_id, item(Multiple objects bbox)),(),...}
def work_paste(ins_cls_list, paste_img_list, paste_img_dir, output_main_dir, args, part_number):

    output_gt_dir = os.path.join(output_main_dir, 'gt')
    output_mask_dir = os.path.join(output_main_dir, 'mask')
    output_paste_dir = os.path.join(output_main_dir, 'paste')

    print(f'INFO --- Start paste {part_number} part to backgrounds, total {len(paste_img_list)} number.')
    imgs_info = {}
    
    assert len(ins_cls_list) == len(paste_img_list)

    for pas_idx, pas_img in enumerate(tqdm(paste_img_list)):
        
        instance_item = ins_cls_list[pas_idx]
        
        instance_name = instance_item['name']
        ins_mean_areas = instance_item['mean_areas']
        ins_var_areas = instance_item['var_areas']
        instance_paths = instance_item['paths']
        
        img_name = pas_img[0]
        img_item = pas_img[1]
        img_item = img_item['annotations']
        ins_path = instance_paths
        img_path = os.path.join(paste_img_dir, img_name)

        bbox_list = []
        for item in img_item:
            bbox = item[0]
            area_norm = item[2]
            if area_norm < args.max_ins_area_norm and area_norm > args.min_ins_area_norm:
                bbox_list.append(bbox)
        
        background = Image.open(img_path)
        background = background.convert('RGB')
        bbox_list = resize_and_crop_bboxes(background.size, bbox_list)
        background = image_transforms_resize_and_crop(background)
        overlay = Image.open(ins_path)
        overlay = overlay.convert('RGBA')
        overlay_w, overlay_h = overlay.size

        area_samples = random.normal(loc=ins_mean_areas, scale=np.sqrt(ins_var_areas), size=(1, 1))[0][0]
        if area_samples < ins_mean_areas - ins_var_areas:
            area_samples = ins_mean_areas - ins_var_areas
        elif area_samples > ins_mean_areas + ins_var_areas:
            area_samples = ins_mean_areas + ins_var_areas
        else:
            area_samples = area_samples

        overlay = ins_resize(overlay, background.size, area_samples, bbox_list, img_name)
        overlay_w, overlay_h = overlay.size
        pas_x, pas_y, fail_times, resize_times = random_pas_location(overlay.size, background.size, bbox_list, args, img_name)
        overlay = overlay.resize((overlay_w // resize_times, overlay_h // resize_times))
        try:
            gt_img, mask_img, paste_img, paste_bbox = paste_ins2img((pas_x, pas_y), overlay, background)
        except:
            print('three image not ok!!!')
            print(resize_times, overlay.size)
            print((pas_x, pas_y), background.size)

        save_name = img_name
        imgs_info[save_name] = []
        imgs_info[save_name].append(instance_name)
        imgs_info[save_name].append(paste_bbox)
        imgs_info[save_name].append(paste_img.size)
        
        #Alpha Blending for Instance Pasting
        gt_tri = np.array(gt_img) / 255.0
        paste_tri = np.array(paste_img) / 255.0
        eroded_img = erode_to_gray(mask_img)
        trimap = np.array(eroded_img) / 255.0

        alpha_map = estimate_alpha(paste_tri, trimap)
        alpha_map = np.expand_dims(alpha_map, axis=-1)
        gt_tri = np.array(gt_tri)
        paste_tri = np.array(paste_tri)
        output_img = (alpha_map * paste_tri + (1 - alpha_map) * gt_tri) * 255.0
        output_img = output_img.astype(np.uint8)
        output_img = Image.fromarray(output_img)
        
        gt_img.save(os.path.join(output_gt_dir, save_name))
        mask_img.save(os.path.join(output_mask_dir, save_name))
        output_img.save(os.path.join(output_paste_dir, save_name))
        
    return imgs_info


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--paste_dir', type=str, default='/path/to/SAM_COCONut_images')
    parser.add_argument('--instance_dir', type=str, default='/path/to/open-images-v7-instance-filter-1000k')
    parser.add_argument('--split', type=str, default='train', choices=['train'])
    parser.add_argument('--output_dir', type=str, default='/path/to/SAM_COCONut_paste')
    parser.add_argument('--max_instance_iou', type=float, default=0.5)
    parser.add_argument('--edge_mask_ratio', type=float, default=0.2)
    parser.add_argument('--instance_self_ration', type=float, default=0.6)
    parser.add_argument('--max_ins_resize', type=float, default=4)
    parser.add_argument('--max_ins_area_norm', type=float, default=0.4)
    parser.add_argument('--min_ins_area_norm', type=float, default=0.0)
    parser.add_argument('--per_process_number', type=int, default=1000)
    
    args = parser.parse_args()

    #Read files and create folders
    instance_dir = os.path.join(args.instance_dir, args.split)
    instance_json_path = os.path.join(args.instance_dir, args.split, 'result.json')
    
    paste_img_dir_1 = os.path.join(args.paste_dir, f'{args.split}_paste', 'coconut-b')
    paste_img_dir_2 = os.path.join(args.paste_dir, f'{args.split}_paste', 'sam_1400k')
    paste_json_path_1 =  os.path.join(args.paste_dir, 'annotations', 'coconut_b_bbox.json')
    paste_json_path_2 =  os.path.join(args.paste_dir, 'annotations')

    output_main_dir = os.path.join(args.output_dir, args.split)
    output_path_file = os.path.join(output_main_dir, 'result.json')
    os.makedirs(output_main_dir, exist_ok=True)
    output_gt_dir = os.path.join(output_main_dir, 'gt')
    os.makedirs(output_gt_dir, exist_ok=True)
    output_mask_dir = os.path.join(output_main_dir, 'mask')
    os.makedirs(output_mask_dir, exist_ok=True)
    output_paste_dir = os.path.join(output_main_dir, 'paste')
    os.makedirs(output_paste_dir, exist_ok=True)

    print('Start loading instance_json.')
    with open(instance_json_path,'r') as f:
        instance_json = json.load(f)
    print('Start loading paste_json_path_1.')
    with open(paste_json_path_1,'r') as f:
        paste_json_1 = json.load(f)
    print(f'The length of paste_json_path_1 is {len(paste_json_1)}.')
    print('Start loading paste_json_path_2.')
    paste_json_2 = {}
    for i in tqdm(range(8)):
        paste_json_subset_2_path = os.path.join(paste_json_path_2, f'sam_bbox_1420k_{i}.json')
        with open(paste_json_subset_2_path,'r') as f:
                paste_json_subset_2 = json.load(f)
        paste_json_2.update(paste_json_subset_2)
    print(f'The length of paste_json_path_2 is {len(paste_json_2)}.')

    #Reorganize the instances in instance_json according to image units and perform cyclic sampling
    print('Start cyclic sampling of instances.')
    
    ins_number_count = 0
    for item in instance_json:
        instance_num = item['number']
        ins_number_count += instance_num
        
    ins_pbar = tqdm(ins_number_count)
    instance_total_list = []
    
    while len(instance_json) > 0:
        
        item_paths_no_zero = []
        for item in instance_json:
            ins_name = item['name']
            ins_mean_scores = item['mean_scores']
            ins_var_scores = item['var_scores']
            ins_mean_areas = item['mean_areas']
            ins_var_areas = item['var_areas']
            
            ins_main_path = item['main_path']
            ins_path = item['paths'].pop()
            ins_path = os.path.join(ins_main_path, ins_path)
            
            instance_total_list.append({'name':ins_name, 'mean_scores':ins_mean_scores, 'var_scores':ins_var_scores, 'mean_areas':ins_mean_areas, 'var_areas':ins_var_areas, 'paths':ins_path})
            ins_pbar.update(1)
            
            if len(item['paths']) > 0:
                item_paths_no_zero.append(item)
        instance_json = item_paths_no_zero
        
    print(f'total instance num is {len(instance_total_list)}!!!')
    #Reorganize the instances in paste_json according to the image units, and directly take the first total of __instance_num
    #dict_keys(['info', 'licenses', 'images', 'annotations', 'categories']
    #dict_keys(['segmentation', 'area', 'iscrowd', 'image_id', 'bbox', 'category_id', 'id'])
    #dict_keys(['supercategory', 'id', 'name'])
    print('Start organizing paste JSON.')
    annotations_paste_1 = []
    for key, value in paste_json_1.items():
        annotations_paste_1.append((key, value))
    print(f'total annotations_paste_1 num is {len(annotations_paste_1)}!!!')
    annotations_paste_2 = []
    for key, value in paste_json_2.items():
        annotations_paste_2.append((key, value))
    print(f'total annotations_paste_2 num is {len(annotations_paste_2)}!!!')

    #Paste according to the total number of totals, instances, and num, and save them separately to three folders
    parts = []
    results_json_filter = {}
    per_process_number = args.per_process_number
    
    instance_total_list_1 = instance_total_list[:len(annotations_paste_1)]
    instance_total_list_2 = instance_total_list[len(annotations_paste_1):(len(annotations_paste_1)+len(annotations_paste_2))]
    del instance_total_list
    
    total_parts_num_1 = (len(annotations_paste_1)) // per_process_number + 1
    total_parts_num_2 = (len(annotations_paste_2)) // per_process_number + 1
        
    for i in range(total_parts_num_2):
        
        if i == total_parts_num_2-1:
            instance_part = instance_total_list_2[i*per_process_number:]
            annotations_paste_part = annotations_paste_2[i*per_process_number:]
        else:
            instance_part = instance_total_list_2[i*per_process_number:(i+1)*per_process_number]
            annotations_paste_part = annotations_paste_2[i*per_process_number:(i+1)*per_process_number]
            
        parts.append((instance_part, annotations_paste_part, paste_img_dir_2, output_main_dir, args, i))
        
    for i in range(total_parts_num_1):
        
        if i == total_parts_num_1-1:
            instance_part = instance_total_list_1[i*per_process_number:]
            annotations_paste_part = annotations_paste_1[i*per_process_number:]
        else:
            instance_part = instance_total_list_1[i*per_process_number:(i+1)*per_process_number]
            annotations_paste_part = annotations_paste_1[i*per_process_number:(i+1)*per_process_number]
            
        parts.append((instance_part, annotations_paste_part, paste_img_dir_1, output_main_dir, args, i))

    print(f'total parts num is {len(parts)}!!!')
    print(f'start paste instances to backgrounds!!!')

    mp.set_start_method('spawn',force=True)
    num_threads = 32
    pool = mp.Pool(processes=num_threads)
    with tqdm(total=len(parts)) as pbar:
        for res in (pool.starmap(work_paste, parts, 1)):
            results_json_filter.update(res)
            pbar.update()
    
    with open(output_path_file,'w') as f:
        json.dump(results_json_filter,f)