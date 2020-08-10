import numpy as np
import math
from utils import *
from utils import Pedestrian_States as State
from utils import Vehicle_States as VehicleStates
from PIL import Image, ImageDraw

Poses = [22.5, 67.5, 112.5, 157.5, 202.5, 247.5, 292.5, 337.5, -1] # last one for stay



class Pedestrian():
    
    def __init__(self, r, size, env):
        self.rect = r
        self.size = size
        self.env = env
        self.pose = np.random.choice(Poses, 1, p=[1.0/8, 1.0/8, 1.0/8, 1.0/8, 1.0/8, 1.0/8, 1.0/8, 1.0/8, 0.0])
        self.speed = 10
        self.states = []
        self.num_moves_away_from_crossing = 0
        self.nearest_shared_rect = None
        self.pedestrian_area_rect = None
        self.should_stay = False
        self.deviation_speed = 0
        self.deviation_pose = 0

    def get_copy(self, env=None):
        import copy
        result = Pedestrian(self.rect.get_copy(), self.size, self.env if env is None else env)
        result.states = copy.deepcopy(self.states)
        result.pose = self.pose
        result.speed = self.speed
        result.num_moves_away_from_crossing = self.num_moves_away_from_crossing
        result.nearest_shared_rect = self.nearest_shared_rect.get_copy() if self.nearest_shared_rect is not None else None
        result.pedestrian_area_rect = self.pedestrian_area_rect.get_copy() if self.pedestrian_area_rect is not None else None
        result.should_stay = self.should_stay
        result.deviation_speed = self.deviation_speed
        result.deviation_pose = self.deviation_pose
        return result

    def draw(self, drawing, color):
        self.rect.draw(drawing, color)
        vec = self.get_pose_vector(self.pose)
        # if vec is None:
        #     center = self.rect.get_center()
        #     r = Rect(center[0], center[1], center[0] + 2, center[1] + 2)
        #     r.draw(drawing, (0, 150, 255, 255))
        # else:
        #     vec.draw(drawing, (0, 150, 255, 255))
        #if self.nearest_shared_rect is not None:
            #self.nearest_shared_rect.draw(drawing, (0, 150, 255, 255))

    def sample_deviation(self):
        self.deviation_speed = np.random.normal(scale=2.)
        self.deviation_pose = np.random.normal(scale=2.)

    def get_pose_vector(self, pose):
        if pose == -1:
            return None
        center = self.rect.get_center()
        point = self.rect.get_right_edge_center()
        v = Vector(center, point)
        pose_vec = v.rotate(pose)
        return pose_vec

    def get_new_location(self, pose):
        if pose == -1:
            return self.rect.top_left
        pose += self.deviation_pose
        tl_shifted = self.rect.top_left[0] + self.speed + self.deviation_speed, self.rect.top_left[1]
        v = Vector(self.rect.top_left, tl_shifted)
        return v.rotate(pose).p2

    def next_state(self):
        self.sample_deviation()
        self.pose = self.next_action()
        new_tl = self.get_new_location(self.pose)
        self.rect = Rect(new_tl[0], new_tl[1], new_tl[0]+self.rect.width, new_tl[1]+self.rect.height)

    def get_mask(self):
        mask = Image.new('L', (self.env.width, self.env.height), color=0)
        ImageDraw.Draw(mask).rectangle(self.rect.get_coordinates(), fill=255)
        return np.array(mask)

    def get_state(self):
        if self.rect.within_any(self.env.pedestrians_rectangles) \
                and not self.rect.intersects_any(self.env.shared_rectangles) \
                and State.Crossing not in self.states:
            return State.TowardCrossing
        elif self.rect.intersects_any(self.env.shared_rectangles) \
                and self.rect.intersects_any(self.env.pedestrians_rectangles) \
                and State.Crossing not in self.states:
            return State.StartCrossing
        elif self.rect.intersects_any(self.env.shared_rectangles) \
                and self.rect.intersects_any(self.env.pedestrians_rectangles) \
                and State.Crossing in self.states:
            return State.FinishCrossing
        elif self.rect.intersects_any(self.env.shared_rectangles):
            return State.Crossing
        elif self.rect.within_any(self.env.pedestrians_rectangles) \
                and not self.rect.intersects_any(self.env.shared_rectangles) \
                and State.Crossing in self.states \
                and self.num_moves_away_from_crossing < 15:
            self.num_moves_away_from_crossing += 1
            return State.AwayFromCrossing
        elif self.rect.within_any(self.env.pedestrians_rectangles) \
                and not self.rect.intersects_any(self.env.shared_rectangles) \
                and State.Crossing in self.states \
                and self.num_moves_away_from_crossing >=15:
            self.states.clear()
            self.num_moves_away_from_crossing = 0
            self.nearest_shared_rect = None
            self.pedestrian_area_rect = None
            return self.get_state()

    def get_weights(self, state):
        # sorted as the Poses (from 22.5 to 337.5)
        weights = np.zeros(len(Poses))

        if state == State.TowardCrossing:
            distances = np.zeros(len(Poses))
            #if self.nearest_shared_rect is None:
            self.nearest_shared_rect = self.get_nearest_shared_rect_random()

            if self.pedestrian_area_rect is None:
                self.pedestrian_area_rect = self.rect.get_within_rect(self.env.pedestrians_rectangles)

            intesection_corner = self.pedestrian_area_rect.get_corner_intersect_with_other(self.nearest_shared_rect)
            # compute the distances of the possible distances to the nearest shared area
            # avoid out of allowed area options
            for i in range(len(Poses)):
                new_tl = self.get_new_location(Poses[i])
                if not self.is_location_valid(new_tl) or Poses[i]==-1:
                    distances[i] = 1000
                else:
                    dis = distance_between_two_points(new_tl, intesection_corner)
                    distances[i] = dis

            # keep the smallest 2 distances
            #print_array(distances, 'distances')
            smallest_distance_indexes = np.argsort(distances)[:2]
            for i in range(len(distances)):
                if i in smallest_distance_indexes:
                    if i == smallest_distance_indexes[0]:
                        weights[i] = 0.7
                    else:
                        weights[i] = 0.3

        elif state == State.StartCrossing:
            self.should_stay = False
            for v in self.env.vehicles:
                if v.states != [] and v.states[-1] in [VehicleStates.AboutToCross, VehicleStates.AwayFromCrossing]:
                    weights[-1] = 1.0
                    self.should_stay = True
            if not self.should_stay:
                #debug = []
                #debug2 = []
                smallest_index = 0
                smallest_weight = 1000
                for i in range(len(Poses)):
                    if Poses[i] == -1:
                        continue
                    angle = self.get_pose_vector(Poses[i]).angle(self.nearest_shared_rect.get_center())
                    new_tl = self.get_new_location(Poses[i])
                    new_r = Rect(new_tl[0], new_tl[1], new_tl[0] + self.size, new_tl[1] + self.size)
                    overlap_area = self.nearest_shared_rect.get_overlap_area(new_r)
                    weight = angle - overlap_area
                    #debug.append(angle)
                    #debug2.append(overlap_area)
                    if weight < smallest_weight and self.nearest_shared_rect.intersects(new_r):
                        smallest_weight = weight
                        smallest_index = i
                for i in range(len(Poses)):
                    if i == smallest_index:
                        weights[i] = 1.0
                #print_two_arrays(debug2, debug, 'overlap', 'angles')

        elif state == State.Crossing:
            #debug = []
            #debug2 = []
            smallest_index = 0
            smallest_weight = 1000
            for i in range(len(Poses)):
                diff_angle = diff_between_two_angles(Poses[i], self.pose)
                new_tl = self.get_new_location(Poses[i])
                new_r = Rect(new_tl[0], new_tl[1], new_tl[0] + self.size, new_tl[1] + self.size)
                overlap_area = self.nearest_shared_rect.get_overlap_area(new_r)
                weight = 2*diff_angle - overlap_area
                #debug.append(diff_angle)
                #debug2.append(overlap_area)
                if weight < smallest_weight and self.is_location_valid(new_tl) and Poses[i]!=-1:
                    smallest_weight = weight
                    smallest_index = i
            for i in range(len(Poses)):
                if i == smallest_index:
                    weights[i] = 1.0
            #print_two_arrays(debug2, debug, 'overlap', 'diff-angle')

        elif state == State.FinishCrossing:
            #debug = []
            #debug2 = []
            smallest_index = 0
            smallest_weight = 1000
            for i in range(len(Poses)):
                diff_angle = diff_between_two_angles(Poses[i], self.pose)
                new_tl = self.get_new_location(Poses[i])
                new_r = Rect(new_tl[0], new_tl[1], new_tl[0] + self.size, new_tl[1] + self.size)
                intersected_pedestrian_rect = new_r.get_any_intersection(self.env.pedestrians_rectangles)
                overlap_area = 0
                if intersected_pedestrian_rect is not None:
                    overlap_area = intersected_pedestrian_rect.get_overlap_area(new_r)
                weight = diff_angle - overlap_area
                #debug.append(diff_angle)
                #debug2.append(overlap_area)
                if weight < smallest_weight and self.is_location_valid(new_tl) and Poses[i]!=-1:
                    smallest_weight = weight
                    smallest_index = i
            for i in range(len(Poses)):
                if i == smallest_index:
                    weights[i] = 1.0
            #print_two_arrays(debug2, debug, 'overlap', 'diff-angle')

        elif state == State.AwayFromCrossing:
            distances = np.zeros(len(Poses))
            # compute the distances of the possible distances to the nearest shared area
            # avoid out of allowed area options
            intesection_corner = self.pedestrian_area_rect.get_corner_intersect_with_other(self.nearest_shared_rect)
            for i in range(len(Poses)):
                new_tl = self.get_new_location(Poses[i])
                if not self.is_location_valid(new_tl) or Poses[i]==-1:
                    distances[i] = -1000
                else:
                    dis = distance_between_two_points(new_tl, intesection_corner)
                    distances[i] = dis

            # keep the largest 3 distances
            largest_distance_indexes = np.argsort(distances)[-3:]
            #print_array(distances, 'distances')
            for i in range(len(distances)):
                if i in largest_distance_indexes:
                    if i == largest_distance_indexes[-1]:
                        weights[i] = 0.4
                    else:
                        weights[i] = 0.3


        return weights

    def get_nearest_shared_rect(self):
        smallest_distance = 1000
        nearest_shared_rect = None
        for r in self.env.shared_rectangles:
            dis = r.distance_to_center(self.rect.top_left)
            if dis < smallest_distance:
                smallest_distance = dis
                nearest_shared_rect = r
        return nearest_shared_rect

    def get_nearest_shared_rect_random(self):
        current_pedestrian_rect = None
        for r in self.env.pedestrians_rectangles:
            if self.rect.within(r):
                current_pedestrian_rect = r
                break

        nearest_shared_rects = []
        for r in self.env.shared_rectangles:
            if r.intersects(current_pedestrian_rect):
                nearest_shared_rects.append(r)
        nearest_shared_rect_index = np.random.choice([0,1], 1)[0]
        return nearest_shared_rects[nearest_shared_rect_index]

    def is_location_valid(self, new_tl):
        new_r = Rect(new_tl[0], new_tl[1], new_tl[0] + self.size, new_tl[1] + self.size)
        return new_r.within_any(self.env.pedestrians_rectangles) \
               or new_r.intersects(self.nearest_shared_rect)
               #or new_r.intersects_any(self.env.shared_rectangles)\


    def next_action(self):
        current_state = self.get_state()
        #print(current_state)
        #print(self.rect)
        if current_state is None:
            print('----------------NONE------------')
            print(self.rect)
            print(self.states)
            for r in self.env.shared_rectangles:
                print(r)
            print('----')
            for r in self.env.pedestrians_rectangles:
                print(r)
            print('****')
            print(self.nearest_shared_rect)
        self.states.append(current_state)
        actions_weights = self.get_weights(current_state)
        selected_action = np.random.choice(Poses, 1, p=actions_weights)
        #print(selected_action)
        return selected_action
