import numpy as np
import os
import math
import cv2
import argparse
import tensorflow as tf
from dataset_loader import Dataset
from net import EWTA_MDF
from utils_np import *
from utils_tf import *
from config import *
import argparse

parser = argparse.ArgumentParser(description='Test all scenes in a dataset')
parser.add_argument('--output', help='write output images', action='store_true')
args = parser.parse_args()
dataset_name = 'SDD'
write_output_flag = args.output

output_folder = OUTPUT_FOLDER
if write_output_flag:
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

width = DATASET_RESOLUTION[dataset_name][0]
height = DATASET_RESOLUTION[dataset_name][1]
model_path = MODEL_PATH
data_path = DATASET_PATH[dataset_name]

session = create_session()

x_objects = tf.placeholder(tf.float32, shape=(3, 1, 1, 5, 1))
x_imgs = tf.placeholder(tf.float32, shape=(3, 1, 3, height, width))

# Build the network graph
network = EWTA_MDF(x_imgs, x_objects)
output = network.make_graph()

# Load the model snapshot
optimistic_restore(session, model_path)

# Load the input dataset
dataset = Dataset(data_path)

nll_sum = 0
semd_sum = 0
counter = 0
# Run the test for each sequence for each scene
for scene_index in range(len(dataset.scenes)):
    scene = dataset.scenes[scene_index]
    scene_name = scene.scene_path.split('/')[-1]
    print('---------------- Scene %s ---------------------' % scene_name)
    if write_output_flag:
        result_scene_path = os.path.join(output_folder, dataset_name, scene_name)
        os.makedirs(result_scene_path, exist_ok=True)
    for i in range(len(scene.sequences)):
        testing_sequence = scene.sequences[i]
        objects_list = []
        imgs_list = []
        for k in range(3):
            objects_list.append(decode_obj(testing_sequence.objects[k], testing_sequence.id))
            imgs_list.append(decode_img(testing_sequence.imgs[k], width=width, height=height))
        objects = np.stack(objects_list, axis=0)
        imgs = np.stack(imgs_list, axis=0)

        means, sigmas, mixture_weights, hyps, hyps_sigmas, input_blob, output_blob, tmp = session.run(output,
                                                           feed_dict={x_objects: objects,
                                                                      x_imgs: imgs})

        gt_object = decode_obj(testing_sequence.objects[-1], testing_sequence.id)
        if write_output_flag:
            drawn_img_hyps = draw_hyps(testing_sequence.imgs[-1], hyps, gt_object, objects)
            cv2.imwrite(os.path.join(result_scene_path, '%d-hyps.jpg' % i), drawn_img_hyps)
            draw_heatmap(testing_sequence.imgs[-1], means, sigmas, mixture_weights, objects, width, height,
                         os.path.join(result_scene_path, '%d-heatmap.jpg' % i), gt=gt_object)
        nll = compute_nll(means, sigmas, mixture_weights, gt_object)
        semd = get_multimodality_score(means, sigmas, mixture_weights)
        print('NLL: %5.2f,\t SEMD: %5.2f' % (nll, semd))
        nll_sum += nll
        semd_sum += semd
        counter += 1
print('--------------- AVERAGE METRICS ---------------')
print('NLL: %.2f,\t SEMD: %.2f, Number of samples: %d' %
      (nll_sum/counter, semd_sum/counter, counter))