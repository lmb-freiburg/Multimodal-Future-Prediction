import numpy as np
import os, sys
from os import listdir
from os.path import isfile, join, isdir
import cv2
import argparse
import imageio

parser = argparse.ArgumentParser()
parser.add_argument("scene_ind")
args = parser.parse_args()

source_path = 'stanford_campus_dataset/annotations/'
img_path = 'processed'
target_path = 'new_processed'

scene_index = 0
scenes = sorted([d for d in listdir(source_path) if isdir(join(source_path, d))])
categories = {'Biker': 0,
              'Pedestrian': 1,
              'Skater': 2,
              'Cart': 3,
              'Car': 4,
              'Bus': 5}
categories_colors = {0: [255, 0, 0],
                     1: [0, 255, 0],
                     2: [0, 0, 255],
                     3: [255, 255, 0],
                     4: [0, 255, 255],
                     5: [255, 0, 255]}


target_width = 320
target_height = 576

def writeFloat(name, data):
    f = open(name, 'wb')

    dim=len(data.shape)
    # if dim>3:
    #     raise Exception('bad float file dimension: %d' % dim)

    f.write(('float\n').encode('ascii'))
    f.write(('%d\n' % dim).encode('ascii'))

    if dim == 1:
        f.write(('%d\n' % data.shape[0]).encode('ascii'))
    else:
        f.write(('%d\n' % data.shape[1]).encode('ascii'))
        f.write(('%d\n' % data.shape[0]).encode('ascii'))
        for i in range(2, dim):
            f.write(('%d\n' % data.shape[i]).encode('ascii'))

    data = data.astype(np.float32)
    if dim==2:
        data.tofile(f)
    elif dim==3:
        np.transpose(data, (2, 0, 1)).tofile(f)
    elif dim==4:
        np.transpose(data, (3, 2, 0, 1)).tofile(f)
    else:
        raise Exception('bad float file dimension: %d' % dim)

def get_mask(width, height, bboxs, categs):
    result = np.zeros((height, width, 3))
    for ind, bbox in enumerate(bboxs):
        try:
            color = categories_colors[categs[ind]]
            for i in range(bbox[0], bbox[2]):
                result[bbox[1], i, :] = color
                result[bbox[3], i, :] = color
            for j in range(bbox[1], bbox[3]):
                result[j, bbox[0], :] = color
                result[j, bbox[2], :] = color
        except IndexError:
            print("Oops! ", bbox)
            raise

    return result

for i in range(len(scenes)):
    videos_scene = sorted([d for d in listdir(join(source_path, scenes[i])) if isdir(join(source_path, scenes[i], d))])
    for j in range(len(videos_scene)):
        if scene_index != int(args.scene_ind):
            scene_index += 1
            continue
        else:
            print('processing %s' % join(scenes[i], videos_scene[j]))
            if not os.path.exists(join(target_path, 'scene_%03d' % scene_index)):
                os.makedirs(join(target_path, 'scene_%03d' % scene_index))
            src_path = join(source_path, scenes[i], videos_scene[j])
            with open(join(src_path, 'annotations.txt')) as f:
                content = [x.strip() for x in f.readlines()]
                length = len(content)
                obj_id = np.zeros(length, int)
                tl_x = np.zeros(length, int)
                tl_y = np.zeros(length, int)
                br_x = np.zeros(length, int)
                br_y = np.zeros(length, int)
                frame_id = np.zeros(length, int)
                out_of_frame = np.zeros(length, int)
                occluded = np.zeros(length, int)
                generated = np.zeros(length, int)
                category = np.zeros(length, int)
                for k in range(length):
                    # fill the arrays by parsing each line of the annotation file
                    x = content[k]
                    res = x.split(' ')
                    obj_id[k] = int(res[0])
                    tl_x[k] = int(res[1])
                    tl_y[k] = int(res[2])
                    br_x[k] = int(res[3])
                    br_y[k] = int(res[4])
                    frame_id[k] = int(res[5])
                    out_of_frame[k] = int(res[6])
                    occluded[k] = int(res[7])
                    generated[k] = int(res[8])
                    category[k] = categories[str(res[9].replace('"',''))]

                for k in range(frame_id.max()):
                    if k % 5 != 0:
                        continue
                    if os.path.exists(join(target_path, 'scene_%03d' % scene_index, '%07d-mask.jpg' % k)):
                        continue
                    # read and transpose the image
                    img = np.transpose(imageio.imread(join(img_path, 'scene_%03d' % scene_index, '%07d-img.jpg' % k)), [1, 0, 2])
                    height, width = img.shape[0], img.shape[1]
                    scale_x = width / target_width
                    scale_y = height / target_height

                    # filter gts which are occluded or out of view
                    indexs = np.where(frame_id==k)[0]
                    filtered_indexes = [ind for ind in indexs if out_of_frame[ind]==0 and occluded[ind]==0]

                    # compute the labels of each frame (the set of obj_ids corresponding to all objects appearing in this frame)
                    # compute the features of each frame containing bbox and category
                    labels = []
                    features = np.full((1154, 5), None)
                    bboxes = []
                    categs = []
                    for fi in filtered_indexes:
                        labels.append(obj_id[fi])
                        bbox = [int((tl_y[fi]) / scale_x),
                                int((tl_x[fi]) / scale_y),
                                min(int((br_y[fi]) / scale_x), target_width-1),
                                min(int((br_x[fi]) / scale_y), target_height-1)]
                        categs.append(category[fi])
                        features[obj_id[fi],:] = [bbox[0], bbox[1], bbox[2], bbox[3], category[fi]]
                        bboxes.append(bbox)

                    # create a mask image with bbox colored according to categories
                    mask = get_mask(target_width, target_height, bboxes, categs)

                    # resize the image
                    resized_img = cv2.resize(img, dsize=(target_width, target_height), interpolation=cv2.INTER_CUBIC)

                    # save the outputs: mask, resized_img, labels and features
                    imageio.imwrite(join(target_path, 'scene_%03d' % scene_index, '%07d-mask.jpg' % k), mask)
                    imageio.imwrite(join(target_path, 'scene_%03d' % scene_index, '%07d-img-resized.jpg' % k), resized_img)
                    writeFloat(join(target_path, 'scene_%03d' % scene_index, '%07d-labels.float3' % k), np.array(labels).reshape((len(labels),1)))
                    writeFloat(join(target_path, 'scene_%03d' % scene_index, '%07d-features.float3' % k), features)

            scene_index += 1
