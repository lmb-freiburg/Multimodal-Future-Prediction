import numpy as np
import math
from utils import *
from enum import Enum
from utils import Vehicle_States as State
from utils import Pedestrian_States as PedestrianStates
from PIL import Image, ImageDraw

# stay, right, bottom, left, top
Actions = [-1.0, 0.0, 90.0, 180.0, 270.0]

class Vehicle():
    
    def __init__(self, r, size, env):
        self.rect = r
        self.size = size
        self.env = env
        self.action = self.init_action()
        self.speed = 10
        self.states = []
        self.prev_action = self.action # only valid if the current action is stay
        self.deviation_speed = 0
        self.deviation_pose = 0


    def get_copy(self, env=None):
        import copy
        result = Vehicle(self.rect.get_copy(), self.size, self.env if env is None else env)
        result.states = copy.deepcopy(self.states)
        result.action = self.action
        result.speed = self.speed
        result.prev_action = self.prev_action
        result.deviation_speed = self.deviation_speed
        return result

    def draw(self, drawing, color):
        self.rect.draw(drawing, color)
        vec = self.get_pose_vector(self.action)
        # if vec is not None:
        #     vec.draw(drawing, (255, 0, 0, 0))
        # else:
        #     center = self.rect.get_center()
        #     r = Rect(center[0], center[1], center[0]+2, center[1]+2)
        #     r.draw(drawing, (255, 0, 0, 0))

    def get_state(self):
        if self.all_modes_valid() \
                and State.InCrossing not in self.states:
            return State.InCrossing
        elif (self.rect.intersects_any(self.env.shared_rectangles)) \
                and State.InCrossing not in self.states:
            return State.AboutToCross
        elif (self.all_modes_valid() or self.rect.intersects_any(self.env.shared_rectangles)) \
                and State.InCrossing in self.states:
            return State.AwayFromCrossing
        elif not self.all_modes_valid():
            self.states.clear()
            return State.NotInCrossing

    def init_action(self):
        # select the initial pose of the moving car to the longest path
        max_length = 0
        init_action = -1
        for action in Actions:
            if action == -1:
                continue
            if self.is_valid_move(action):
                v = self.get_longest_vector(action)
                if v.get_length() > max_length:
                    max_length = v.get_length()
                    init_action = action
        return init_action

    def all_modes_valid(self):
        for a in Actions:
            if not self.is_valid_move(a):
                return False
        return True
    def is_valid_move(self, action):
        if action == -1:
            return True
        v1, v2 = self.get_longest_two_vectors(action)
        for rect in self.env.pedestrians_rectangles:
            if rect.contains_point(v1.p2) or rect.contains_point(v2.p2):
                return False
        return True

    def get_longest_vector(self, action):
        center = self.rect.get_center()
        end_point = (self.env.width, center[1])
        if action == 90.0:
            end_point = (center[0], self.env.height)
        elif action == 180.0:
            end_point = (0, center[1])
        elif action == 270.0:
            end_point = (center[0], 0)
        elif action == -1:
            return None
        return Vector(center, end_point)

    def get_longest_two_vectors(self, action):
        center_1 = self.rect.get_top_edge_center()
        center_2 = self.rect.get_bottom_edge_center()
        end_point_1 = (self.env.width, center_1[1])
        end_point_2 = (self.env.width, center_2[1])
        if action == 90.0: # bottom
            center_1 = self.rect.get_left_edge_center()
            center_2 = self.rect.get_right_edge_center()
            end_point_1 = (center_1[0], self.env.height)
            end_point_2 = (center_2[0], self.env.height)
        elif action == 180.0: # left
            center_1 = self.rect.get_top_edge_center()
            center_2 = self.rect.get_bottom_edge_center()
            end_point_1 = (0, center_1[1])
            end_point_2 = (0, center_2[1])
        elif action == 270.0: # top
            center_1 = self.rect.get_left_edge_center()
            center_2 = self.rect.get_right_edge_center()
            end_point_1 = (center_1[0], 0)
            end_point_2 = (center_2[0], 0)
        elif action == -1:
            return None
        return Vector(center_1, end_point_1), Vector(center_2, end_point_2)


    def get_pose_vector(self, action):
        center = self.rect.get_center()
        edge_right = self.rect.get_right_edge_center()
        center_right_edge_vector = Vector(center, edge_right)
        if action != -1:
            return center_right_edge_vector.rotate(action)
        else:
            return None

    def get_weights(self, state):
        # sorted as the Actions (from stay to backward)
        weights = np.zeros(len(Actions))
        for p in self.env.pedestrians:
            if p.states != [] and p.states[-1] in [PedestrianStates.Crossing, PedestrianStates.StartCrossing, PedestrianStates.FinishCrossing] and not p.should_stay:
                weights[0] = 1.0
                return weights
        action = self.action if self.action != -1 else self.prev_action
        if state == State.InCrossing:
            diffs = [diff_between_two_angles(Actions[i], action)
                     if Actions[i] != -1 else -1 for i in range(len(Actions))]
            max_index = np.argmax(np.array(diffs))
            for i in range(len(Actions)):
                if i != max_index and Actions[i]!=-1:
                    weights[i] = 1.0
            non_zeros = np.count_nonzero(weights)
            weights /= non_zeros
        elif state == State.AwayFromCrossing or state == State.AboutToCross:
            diffs = [diff_between_two_angles(Actions[i], action)
                     if Actions[i]!=-1 and self.is_valid_move(Actions[i]) else 1000 for i in range(len(Actions))]
            index = np.argmin(np.array(diffs))
            weights[index] = 0.7
            weights[0] = 0.3
        elif state == State.NotInCrossing:
            diffs = [diff_between_two_angles(Actions[i], action)
                     if Actions[i] != -1 and self.is_valid_move(Actions[i]) else 1000 for i in
                     range(len(Actions))]
            index = np.argmin(np.array(diffs))
            if not self.in_window(Actions[index]):
                diffs = [diff_between_two_angles(Actions[i], action)
                         if Actions[i] != -1 and self.is_valid_move(Actions[i]) else -1 for i in
                         range(len(Actions))]
                index = np.argmax(np.array(diffs))
            weights[index] = 0.8
            weights[0] = 0.2

        #print_array(weights, 'weights')
        return weights

    def next_action(self):
        state = self.get_state()
        self.states.append(state)
        actions_weights = self.get_weights(state)
        selected_action = np.random.choice(Actions, 1, p=actions_weights)
        return selected_action

    def get_mask(self):
        mask = Image.new('L', (self.env.width, self.env.height), color=0)
        ImageDraw.Draw(mask).rectangle(self.rect.get_coordinates(), fill=255)
        return np.array(mask)

    def next_state(self):
        new_action = self.next_action()
        self.sample_deviation(new_action)
        if new_action == -1 and self.action != -1:
            self.prev_action = self.action
        self.action = new_action
        new_tl = self.get_new_location(self.action)
        self.rect = Rect(new_tl[0], new_tl[1], new_tl[0]+self.size, new_tl[1]+self.size)

    def sample_deviation(self, new_action):
        found = False
        num_tries = 0
        while not found and num_tries < 50:
            self.deviation_speed = np.random.normal(scale=3.)
            self.deviation_pose = np.random.normal(scale=4.)
            found = self.is_new_location_valid(new_action)
            num_tries += 1
        if num_tries == 50:
            self.deviation_speed = 0
            self.deviation_pose = 0

    def get_new_location(self, action):
        tl_shifted = self.rect.top_left[0] + self.speed + self.deviation_speed, self.rect.top_left[1]
        v = Vector(self.rect.top_left, tl_shifted)
        if action == -1:
            return self.rect.top_left
        else:
            return v.rotate(action+self.deviation_pose).p2

    def in_window(self, action):
        new_tl = self.get_new_location(action)
        new_rect = Rect(new_tl[0], new_tl[1], new_tl[0] + self.size, new_tl[1] + self.size)
        window_rect = Rect(0, 0, self.env.width, self.env.height)
        return new_rect.within(window_rect)

    def is_new_location_valid(self, action):
        new_tl = self.get_new_location(action)
        new_rect = Rect(new_tl[0], new_tl[1], new_tl[0] + self.size, new_tl[1] + self.size)
        return not new_rect.intersects_any(self.env.pedestrians_rectangles)
