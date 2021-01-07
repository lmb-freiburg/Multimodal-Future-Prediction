import numpy as np
import os, sys
from os import listdir
from os.path import isfile, join, isdir

source_path = 'stanford_campus_dataset/videos/'
target_path = 'processed'

scene_index = 0
scenes = sorted([d for d in listdir(source_path) if isdir(join(source_path, d))])
for i in range(len(scenes)):
    print('processing %s' % scenes[i])
    print('--------------------------')
    videos_scene = sorted([d for d in listdir(join(source_path, scenes[i])) if isdir(join(source_path, scenes[i], d))])
    for j in range(len(videos_scene)):
        src_path = join(source_path, scenes[i], videos_scene[j])
        tgt_path = join(target_path, 'scene_%03d' % scene_index)
        os.makedirs(tgt_path)
        os.system("ffmpeg -i %s/video.mov %s/img%%07d.jpg \n" % (src_path, tgt_path))
        scene_index += 1
    print('*******************')
