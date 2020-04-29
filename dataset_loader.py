import numpy as np
import os
from config import *
from utils_np import *

class Dataset():
    def __init__(self, path):
        self.path = path
        self.scenes = []
        self.load_scenes()

    def load_scenes(self):
        scenes_names = sorted(os.listdir(self.path))
        for scene_name in scenes_names:
            if os.path.exists(os.path.join(self.path, scene_name, 'scene.txt')):
                    self.scenes.append(Scene(os.path.join(self.path, scene_name)))


class Scene():
    def __init__(self, scene_path):
        self.scene_path = scene_path
        self.img_ext = '-img-resized.jpg'
        self.sequences = []
        self.load_sequences()

    def parse_string(self, c):
        id, ss = c.split(' ')
        img_0, img_1, img_2, img_f = ss.split(',')
        return int(id), str(img_0).strip(), str(img_1).strip(), str(img_2).strip(), str(img_f).strip()

    def load_sequences(self):
        with open(os.path.join(self.scene_path, 'scene.txt')) as f:
            content = f.readlines()
            for c in content:
                id, img_0, img_1, img_2, img_f = self.parse_string(c)
                img_0_path = os.path.join(self.scene_path, 'imgs', '%s-img-resized.jpg' % img_0)
                img_1_path = os.path.join(self.scene_path, 'imgs', '%s-img-resized.jpg' % img_1)
                img_2_path = os.path.join(self.scene_path, 'imgs', '%s-img-resized.jpg' % img_2)
                img_f_path = os.path.join(self.scene_path, 'imgs', '%s-img-resized.jpg' % img_f)

                obj_0_path = os.path.join(self.scene_path, 'floats', '%s-features.float3' % img_0)
                obj_1_path = os.path.join(self.scene_path, 'floats', '%s-features.float3' % img_1)
                obj_2_path = os.path.join(self.scene_path, 'floats', '%s-features.float3' % img_2)
                obj_f_path = os.path.join(self.scene_path, 'floats', '%s-features.float3' % img_f)

                self.sequences.append(Sequence(id, [img_0_path, img_1_path, img_2_path, img_f_path],
                                               [obj_0_path, obj_1_path, obj_2_path, obj_f_path]))

class Sequence():
    def __init__(self, id, imgs, objects):
        self.id = id
        self.imgs = imgs
        self.objects = objects