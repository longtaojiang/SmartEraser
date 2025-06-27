from IPython.display import display
import json
import sys,os
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
from tqdm import tqdm
import clip
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
from matplotlib import pyplot as plt
import csv
from torchvision import transforms
from torchvision.transforms import ToPILImage
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
torch.set_grad_enabled(False)

def read_csv_id2name(filename):
    with open(filename, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        result_dict = {}
        for row in reader:
            if len(row) >= 2:
                key = row[0].strip()
                value = row[1].strip()
                result_dict[key] = value
    return result_dict


def read_csv_to_image2mask(filename):
    with open(filename, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        headers = next(reader)
        key_list = headers[:9]
        dict_list = []

        for row in reader:
            assert len(row) >= 9
            row_dict = dict(zip(key_list, row[:9]))
            dict_list.append(row_dict)

    return dict_list

def read_lines_to_list(filename):

    with open(filename, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    lines = [line.strip() for line in lines]

    return lines

def init(args,rank=None):
    global clip_model, preprocess, resize, to_pil_image
    if rank is None:
        rank=torch.multiprocessing.current_process()._identity[0]-1
    print("init process GPU:",rank)
    device = torch.device('cuda:%s'%rank)
    torch.cuda.set_device(device)
    torch.cuda.empty_cache()
    clip_model, preprocess = clip.load(args.clip_path, device=rank)
    n_px=224
    preprocess=transforms.Compose([
        transforms.Resize(n_px, interpolation=BICUBIC),
        transforms.CenterCrop(n_px),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])
    resize=transforms.Compose([
        transforms.Resize(n_px, interpolation=BICUBIC),
        transforms.CenterCrop(n_px),
    ])
    to_pil_image = ToPILImage()

def seg_image(classname, path, total, args, output_path):

    batch_size = args.batch_size
    split = args.split
    print(f'Now compute {split} {classname} class clip scores and area, total {total} samples.')

    text='a photo of ' + classname
    clips=[]
    areas=[]
    os.makedirs(os.path.join(output_path, classname), exist_ok=True)

    last_num = total%batch_size
    
    for i in tqdm(range(total//batch_size+1)):

        next_num = batch_size if i < (total//batch_size) else last_num
        if next_num == 0:
            continue

        #read files
        imgs = [transforms.ToTensor()(Image.open(fn[0]).convert('RGB')) for fn in path[i*batch_size:(i)*batch_size+next_num]] #[0-1]
        mask_ims = [transforms.ToTensor()(Image.open(fn[1]).convert('L')) for fn in path[i*batch_size:(i)*batch_size+next_num]] #[0,1]
        mask_region = [fn[2] for fn in path[i*batch_size:(i)*batch_size+next_num]] #[0,1]
        imgs_RGBA = [transforms.ToTensor()(Image.open(fn[0]).convert('RGBA')) for fn in path[i*batch_size:(i)*batch_size+next_num]] #[0-1]

        #resize images
        mask_ims_re = [(mask_im.shape[1], mask_im.shape[2]) for mask_im in mask_ims]
        imgs = [transforms.Resize(size, interpolation=BICUBIC)(img) for img, size in zip(imgs, mask_ims_re)]
        imgs_RGBA = [transforms.Resize(size, interpolation=BICUBIC)(img) for img, size in zip(imgs_RGBA, mask_ims_re)]

        #compute area
        mask_area = [torch.sum(mask_im,dim=[0,1,2])/mask_im.shape[0]/mask_im.shape[1]/mask_im.shape[2] for mask_im in mask_ims]

        #compute clip score
        wb_ims = [img * mask_im + torch.ones_like(img) * (1-mask_im) for img, mask_im in zip(imgs, mask_ims)]
        wb_ims = torch.stack([preprocess(wb_im).cuda() for wb_im in wb_ims], dim=0)
        text_feature = clip.tokenize(text).cuda()   
        _, logits_per_text = clip_model(wb_ims, text_feature)
        logits_per_text=logits_per_text.view(-1).cpu().tolist()
        mask_area = [mask_a.cpu().item() for mask_a in mask_area]
        
        #Crop and save files
        for img_RGBA, mask in zip(imgs_RGBA, mask_ims):
            img_RGBA[-1,:,:] = mask
        imgs_RGBA_new = []
        for idx, (img, region) in enumerate(zip(imgs_RGBA, mask_region)):
            if region[1]>region[0] and region[3]>region[2]:
                y_l, x_l = img.shape[1], img.shape[2]
                region[0], region[1], region[2], region[3] = int(region[0]*x_l), int(region[1]*x_l), int(region[2]*y_l), int(region[3]*y_l)
                img = img[:, region[2]:region[3]+1,region[0]:region[1]+1]
                imgs_RGBA_new.append(img)
            else:
                img = img[:, 0:1,0:1]
                logits_per_text[idx] = 0
                mask_area[idx] = 0
                imgs_RGBA_new.append(img)
        
        imgs_RGBA_new = [to_pil_image((wb_im_RGBA).cpu()) for wb_im_RGBA in imgs_RGBA_new]

        #Save the processed image
        for j, fn in enumerate(path[i*batch_size:(i)*batch_size+next_num]):
            wb_wr_path = os.path.join(output_path, classname, f'image_{(i*batch_size+j):06d}.png')
            wb_wr = imgs_RGBA_new[j]
            wb_wr.save(wb_wr_path)
            fn.pop()
            fn.append(wb_wr_path)
            
        #Save clip score and area for each image
        clips += logits_per_text
        areas += mask_area

    main_path = []
    main_path.append(os.path.dirname(path[0][0]))
    main_path.append(os.path.dirname(os.path.dirname(path[0][1])))
    main_path.append(os.path.dirname(path[0][2]))

    for fn in path:
        img_path = fn.pop()
        mask_path = fn.pop()
        clip_path = fn.pop()

        fn.append(os.path.basename(img_path))
        fn.append(os.path.basename(mask_path)[0].upper()+'/'+os.path.basename(mask_path))
        fn.append(os.path.basename(clip_path))

    return {'name':classname, 'number':total, 'statisic':[clips,areas], 'main_path':main_path, 'paths':path}


from collections import defaultdict
from copy import deepcopy
import numpy as np
import argparse
import os
d1=defaultdict(list)
d2=defaultdict(list)
if __name__=="__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='/path/to/open-images-v7/')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'test', 'val'])
    parser.add_argument('--output_dir', type=str, default='/path/to/open-images-v7-instance/')
    parser.add_argument('--clip_path', type=str, default='./ckpts/ViT-L-14.pt')
    parser.add_argument('--batch_size', type=int, default=5)
    args = parser.parse_args()

    print('Clip model start load...')
    _ = clip.load(args.clip_path)
    del _
    print('Clip model load done...')

    main_path = os.path.join(args.input_dir, args.split)
    image_path = os.path.join(main_path, 'data')
    mask_path = os.path.join(main_path, 'labels')
    label_path = os.path.join(main_path, 'metadata')

    seg_id = read_lines_to_list(os.path.join(label_path, 'oidv7-classes-segmentation.txt'))
    seg_id2name = read_csv_id2name(os.path.join(label_path, 'all_classes_id.csv'))
    results = read_csv_to_image2mask(os.path.join(label_path, f'{args.split}-annotations-object-segmentation.csv'))
    
    os.makedirs(os.path.join(args.output_dir, args.split), exist_ok=True)
    output_path_json=os.path.join(args.output_dir, args.split, 'result.json')
    output_path=os.path.join(args.output_dir, args.split)
    mp.set_start_method('spawn',force=True)
    datadict=defaultdict(list)
    
    # MaskPath ImageID LabelName BoxID BoxXMin BoxXMax BoxYMin BoxYMax PredictedIoU
    for data in tqdm(results):
        label_id = data['LabelName']
        assert label_id in seg_id
        data_label = seg_id2name[label_id]
        if data_label not in datadict:
            datadict[data_label] = {}
            datadict[data_label]['path'] = []
        
        data_mask = data['MaskPath']
        mini_dir = data_mask[0].upper()
        data_mask = os.path.join(mask_path, mini_dir, data_mask)
        data_image = data['ImageID'] + '.jpg'
        data_image = os.path.join(image_path, data_image)

        datadict[data_label]['path'].append([data_image, data_mask, [float(data['BoxXMin']), float(data['BoxXMax']), float(data['BoxYMin']), float(data['BoxYMax'])]])

    print('All class statistic done...')
    #all class
    classes=[key for key in datadict.keys()]
    num_gpus=torch.cuda.device_count()
    print(f'Total have {num_gpus} gpus.')

    #Split each class
    if num_gpus>1:
        pool = mp.Pool(processes=num_gpus, initializer=init, initargs=(args,))
    else:
        init(args, rank=0)

    results_json = []
    if num_gpus>1:
        with tqdm(total=len(classes)) as pbar:
            for i, res in enumerate(pool.starmap(seg_image, [(i, datadict[i]['path'], len(datadict[i]['path']), args, output_path) for i in classes], 1)):
                results_json.append(res)
                pbar.update()
    else:
        results_json = [seg_image(i, datadict[i]['path'], len(datadict[i]['path']), args, output_path) for i in classes]

    print(f'Start save {args.split} clip_scores and areas')

    with open(output_path_json,'w') as f:
        json.dump(results_json,f)
