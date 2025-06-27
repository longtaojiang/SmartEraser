import json
import os
import argparse
import numpy as np
from collections import defaultdict
import torch.multiprocessing as mp
from PIL import Image,ImageFile
from torchvision import transforms
import shutil
from tqdm import tqdm

def work_filter(classname, data, output_path):
    
    main_path = data['main_path']
    number = data['number']
    paths = data['paths_scores']
    mean_scores = data['mean_scores']
    mean_areas = data['mean_areas']
    var_scores = data['var_scores']
    var_areas = data['var_areas']
    paths_new = []

    print(f'Start copying {classname}, total have {number} samples.')
    output_path = os.path.join(output_path, classname)
    os.makedirs(output_path, exist_ok=True)
    
    #Create a folder based on the class ID. Since chunksize is 1, there is only one class in the part
    for i in tqdm(range(number)):
        src_path = os.path.join(main_path[2], paths[i][0][0])
        dest_path = os.path.join(output_path, paths[i][0][0])
        paths_new.append(paths[i][0][0])
        shutil.copy(src_path, dest_path)
    
    return {'name':classname, 'number':number, 'mean_scores':mean_scores, 'var_scores':var_scores, 'mean_areas':mean_areas, 'var_areas':var_areas, 'main_path':output_path, 'paths':paths_new}

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='/path/to/open-images-v7-instance')
    parser.add_argument('--split', type=str, default='val', choices=['train', 'val'])
    parser.add_argument('--output_dir', type=str, default='/path/to/open-images-v7-instance-filter-1000k/')
    parser.add_argument('--min_clip', type=float, default=0.2)
    parser.add_argument('--min_area', type=float, default=0.05)
    parser.add_argument('--max_area', type=float, default=0.95)
    parser.add_argument('--tolerance', type=float, default=0.02)
    parser.add_argument('--min_cls_number', type=float, default=6)
    parser.add_argument('--filter_number', type=float, default=1100000)

    args = parser.parse_args()

    #Initialize file path
    main_path = os.path.join(args.input_dir, args.split)
    results_json_path = os.path.join(main_path, 'result.json')
    with open(results_json_path,'r') as f:
        results_json = json.load(f)

    #Initialize the list and retain the final segmentation result
    datadict = {}
    total_number = 0
    total_class = 0

    for data in tqdm(results_json):
        #Single class images, clip scores and masks for each method
        classname = data['name']
        classnum = data['number']
        classnum_clip_scores = data['statisic'][0]
        classnum_areas = data['statisic'][1]
        classnum_main_path = data['main_path']
        classnum_paths = data['paths']
        datadict[classname] = {}
        datadict[classname]['paths_scores'] = []
        datadict[classname]['number'] = 0
        datadict[classname]['mean_scores'] = 0.0
        datadict[classname]['mean_areas'] = 0.0
        datadict[classname]['main_path'] = classnum_main_path
        cls_scores = []
        cls_areas = []

        #Clip threshold, filter low threshold mask
        this_bar = min(args.min_clip, np.max(classnum_clip_scores)-args.tolerance)
        # print(this_bar)

        #Filter out masks with low scores and large or small areas based on their scores
        for k in range(len(classnum_areas)):

            if classnum_clip_scores[k] < this_bar or classnum_areas[k]<args.min_area or classnum_areas[k]>args.max_area:
                continue

            cls_scores.append(classnum_clip_scores[k])
            cls_areas.append(classnum_areas[k])
            datadict[classname]['paths_scores'].append((classnum_paths[k], classnum_clip_scores[k]))
            #Record quantity
            datadict[classname]['number']+=1

        #If the total number of classes is less than a certain value, discard the class
        if datadict[classname]['number'] < args.min_cls_number:
            del datadict[classname]
            continue
        
        #Sort the instances of each class in ascending order of scores size
        datadict[classname]['paths_scores'] = sorted(datadict[classname]['paths_scores'], key=lambda item: item[1], reverse=True)

        #Count some attributes of each category
        datadict[classname]['mean_scores'] = sum(cls_scores)/datadict[classname]['number']
        datadict[classname]['var_scores'] = sum([(x - datadict[classname]['mean_scores']) ** 2 for x in cls_scores]) / datadict[classname]['number']
        datadict[classname]['mean_areas'] = sum(cls_areas)/datadict[classname]['number']
        datadict[classname]['var_areas'] = sum([(x - datadict[classname]['mean_areas']) ** 2 for x in cls_areas]) / datadict[classname]['number']

    #Sort by quantity and size
    datadict = sorted(datadict.items(), key=lambda item: item[1]['number'], reverse=True)
    datadict = dict(datadict)
    results_json_filter = []
    
    #Establish an information dictionary based on the group ID
    output_path = os.path.join(args.output_dir, args.split)
    output_path_file = os.path.join(output_path, 'result.json')
    parts=[(i, datadict[i], output_path) for i in datadict]

    #Multi process data processing
    mp.set_start_method('spawn',force=True)
    num_threads = 64
    pool = mp.Pool(processes=num_threads)
    with tqdm(total=len(parts)) as pbar:
        for res in (pool.starmap(work_filter, parts, 1)):
            results_json_filter.append(res)
            pbar.update()
    
    with open(output_path_file,'w') as f:
        json.dump(results_json_filter,f)
                