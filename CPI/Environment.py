import numpy as np
from PIL import Image, ImageDraw
from Pedestrian import Pedestrian
from Vehicle import Vehicle
from utils import Rect
import copy

Colors = {
    'background' : (255, 255, 255, 0),
    'pedestrian_area' : (200, 200, 200, 255),
    'shared_area' : (200, 0, 0, 255),
    'vehicle_area' : (0, 150, 0, 255),
    'pedestrian' : (50, 50, 50, 255),
    'vehicle' : (150, 50, 150, 255)
}

Sizes = {
    'pedestrian': 20,
    'vehicle': 40
}
class Environment():
    
    def __init__(self, width, height, num_pedestrians=1, num_vehicles=1):
        self.width = width
        self.height = height
        self.num_pedestrians = num_pedestrians # 1
        self.num_vehicles = num_vehicles # 1
        self.pedestrians = []
        self.vehicles = []
        self.background = None
        self.shared_rectangles = []
        self.pedestrians_rectangles = []
        self.drawing = None
    
    def get_copy(self):
        result = Environment(self.width, self.height, self.num_pedestrians, self.num_vehicles)
        result.pedestrians = [p.get_copy(env=result) for p in self.pedestrians]
        result.vehicles = [v.get_copy(env=result) for v in self.vehicles]
        result.background = copy.deepcopy(self.background)
        result.shared_rectangles = [r.get_copy() for r in self.shared_rectangles]
        result.pedestrians_rectangles = [r.get_copy() for r in self.pedestrians_rectangles]
        result.drawing = None
        return result

    def draw_cross_road(self):
        self.background = Image.new('RGB', (self.width, self.height), color=Colors['background'])
        self.drawing = ImageDraw.Draw(self.background)
        p1 = (self.width * 0.4, self.height * 0.4)
        p2 = (self.width * 0.4, self.height * 0.6)
        p3 = (self.width * 0.6, self.height * 0.4)
        p4 = (self.width * 0.6, self.height * 0.6)

        crossing_area_width = 10
        self.pedestrians_rectangles.clear()
        self.pedestrians_rectangles.append(Rect(0, 0, p1[0], p1[1]))
        self.pedestrians_rectangles.append(Rect(0, p2[1], p2[0], self.height))
        self.pedestrians_rectangles.append(Rect(p3[0], 0, self.width, p3[1]))
        self.pedestrians_rectangles.append(Rect(p4[0], p4[1], self.width, self.height))

        self.shared_rectangles.clear()
        self.shared_rectangles.append(Rect(p1[0] - crossing_area_width, p1[1], p2[0], p2[1]))
        self.shared_rectangles.append(Rect(p2[0], p2[1], p4[0], p4[1] + crossing_area_width))
        self.shared_rectangles.append(Rect(p3[0], p3[1], p4[0] + crossing_area_width, p4[1]))
        self.shared_rectangles.append(Rect(p1[0], p1[1] - crossing_area_width, p3[0], p3[1]))

        for r in self.pedestrians_rectangles:
            r.draw(self.drawing, color=Colors['pedestrian_area'])
        for r in self.shared_rectangles:
            r.draw(self.drawing, color=Colors['shared_area'])


    def init_pedestrian(self):
        size = Sizes['pedestrian']
        i = 0
        while (i < self.num_pedestrians):
            x = np.random.randint(size, self.width-size)
            y = np.random.randint(size, self.height-size)
            r = Rect(x, y, x+size, y+size)
            if r.within_any(self.pedestrians_rectangles):
                p = Pedestrian(r, size=size, env=self)
                p.draw(self.drawing, color=Colors['pedestrian'])
                self.pedestrians.append(p)
                i += 1

    def init_vehicle(self):
        size = Sizes['vehicle']
        i = 0
        while (i < self.num_vehicles):
            x = np.random.randint(size, self.width-size)
            y = np.random.randint(size, self.height-size)
            r = Rect(x, y, x+size, y+size)
            if not r.intersects_any(self.pedestrians_rectangles):
                v = Vehicle(r, size=size, env=self)
                v.draw(self.drawing, color=Colors['vehicle'])
                self.vehicles.append(v)
                i += 1

    def get_image(self):
        return self.background

    def next_state_get_flow(self):
        flow = np.zeros((self.width, self.height, 2))
        for p in self.pedestrians:
            mask = p.get_mask()
            old_tl = p.rect.top_left
            p.next_state()
            new_tl = p.rect.top_left
            fx = new_tl[0] - old_tl[0]
            fy = new_tl[1] - old_tl[1]
            flow[mask > 0, 0] = fx
            flow[mask > 0, 1] = fy
        for v in self.vehicles:
            mask = v.get_mask()
            old_tl = v.rect.top_left
            v.next_state()
            new_tl = v.rect.top_left
            fx = new_tl[0] - old_tl[0]
            fy = new_tl[1] - old_tl[1]
            flow[mask > 0, 0] = fx
            flow[mask > 0, 1] = fy
        return flow

    def get_flow(self, new_tl_p, new_tl_v):
        flow = np.zeros((self.width, self.height, 2))
        p = self.pedestrians[0]
        mask = p.get_mask()
        old_tl = p.rect.top_left
        fx = new_tl_p[0] - old_tl[0]
        fy = new_tl_p[1] - old_tl[1]
        flow[mask > 0, 0] = fx
        flow[mask > 0, 1] = fy
        v = self.vehicles[0]
        mask = v.get_mask()
        old_tl = v.rect.top_left
        fx = new_tl_v[0] - old_tl[0]
        fy = new_tl_v[1] - old_tl[1]
        flow[mask > 0, 0] = fx
        flow[mask > 0, 1] = fy
        return flow

    def draw_objects(self):
        for p in self.pedestrians:
            p.draw(self.drawing, color=Colors['pedestrian'])
        for v in self.vehicles:
            v.draw(self.drawing, color=Colors['vehicle'])

    def get_objects_locations(self):
        results = np.zeros((1, 1, (self.num_pedestrians+self.num_vehicles)*3))
        i = 0
        for p in self.pedestrians:
            results[0, 0, 0] = p.rect.top_left[0]
            results[0, 0, 1] = p.rect.top_left[1]
            results[0, 0, 2] = p.size
            i += 1
        for v in self.vehicles:
            results[0, 0, 3] = v.rect.top_left[0]
            results[0, 0, 4] = v.rect.top_left[1]
            results[0, 0, 5] = v.size
            i += 1
        return results

    def next_state(self):
        for p in self.pedestrians:
            p.next_state()
            p.draw(self.drawing, color=Colors['pedestrian'])
        for v in self.vehicles:
            v.next_state()
            v.draw(self.drawing, color=Colors['vehicle'])
