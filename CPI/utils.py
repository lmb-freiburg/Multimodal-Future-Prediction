import numpy as np
import math

class Rect:
    def __init__(self, tl_x, tl_y, br_x, br_y):
        self.top_left = (tl_x, tl_y)
        self.top_right = (br_x, tl_y)
        self.bottom_left = (tl_x, br_y)
        self.bottom_right = (br_x, br_y)
        self.width = br_x - tl_x
        self.height = br_y - tl_y

    def get_copy(self):
        return Rect(self.top_left[0], self.top_left[1], self.bottom_right[0], self.bottom_right[1])
    def intersects(self, other):
        return not ( self.top_right[0] < other.bottom_left[0]
                  or self.bottom_left[0] > other.top_right[0]
                  or self.top_right[1] > other.bottom_left[1]
                  or self.bottom_left[1] < other.top_right[1])

    def get_overlap_area(self, other):
        x_overlap = max(0, min(self.top_right[0], other.top_right[0]) - max(self.top_left[0], other.top_left[0]))
        y_overlap = max(0, min(self.bottom_left[1], other.bottom_left[1]) - max(self.top_left[1], other.top_left[1]));
        overlapArea = x_overlap * y_overlap
        return overlapArea

    def contains_point(self, point):
        return  ( self.top_left[0] <= point[0] <= self.top_right[0]
                  and self.top_left[1] <= point[1] <= self.bottom_left[1])

    def intersects_any(self, others):
        for r in others:
            if self.intersects(r):
                return True
        return False

    def get_any_intersection(self, others):
        for r in others:
            if self.intersects(r):
                return r
        return None

    def within(self, other):
        return (other.top_left[0] <= self.top_left[0] <= other.bottom_right[0] and
                other.top_left[1] <= self.top_left[1] <= other.bottom_right[1] and
                other.top_left[0] <= self.bottom_right[0] <= other.bottom_right[0] and
                other.top_left[1] <= self.bottom_right[1] <= other.bottom_right[1] )

    def within_any(self, others):
        for r in others:
            if self.within(r):
                return True
        return False

    def get_within_rect(self, others):
        for r in others:
            if self.within(r):
                return r
        return None

    def get_corner_intersect_with_other(self, other):
        if other.contains_point(self.top_left):
            return self.top_left
        elif other.contains_point(self.top_right):
            return self.top_right
        elif other.contains_point(self.bottom_left):
            return self.bottom_left
        elif other.contains_point(self.bottom_right):
            return self.bottom_right
        else:
            return None

    def get_coordinates(self):
        return (self.top_left[0], self.top_left[1], self.bottom_right[0], self.bottom_right[1])

    def draw(self, drawing, color):
        drawing.rectangle(self.get_coordinates(), fill=color)

    def get_center(self):
        return ((self.top_left[0] + self.width/2), (self.top_left[1] + self.height/2))

    def get_right_edge_center(self):
        return (self.top_right[0], (self.top_right[1] + self.height/2))

    def get_left_edge_center(self):
        return (self.top_left[0], (self.top_left[1] + self.height/2))

    def get_top_edge_center(self):
        return (self.top_left[0] + self.width/2, self.top_left[1])

    def get_bottom_edge_center(self):
        return (self.bottom_left[0] + self.width/2, self.bottom_left[1])

    def distance_to_center(self, point):
        center = self.get_center()
        return math.sqrt((center[0]-point[0])*(center[0]-point[0])+(center[1]-point[1])*(center[1]-point[1]))

    def __str__(self):
        return '(%d,%d) > (%d,%d)' %(self.top_left[0], self.top_left[1], self.bottom_right[0], self.bottom_right[1])

class Vector:
    def __init__(self, p1, p2):
        self.p1 = p1
        self.p2 = p2

    def get_length(self):
        return math.fabs(self.p1[0]-self.p2[0]) + math.fabs(self.p1[1]-self.p2[1])

    def get_coordinates(self):
        return (self.p1[0], self.p1[1], self.p2[0], self.p2[1])

    def draw(self, drawing, color):
        drawing.line(self.get_coordinates(), fill=color)

    def rotate(self, angle):
        angle = math.radians(angle)
        ox, oy = self.p1
        px, py = self.p2
        qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
        qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
        return Vector(self.p1, (qx, qy))

    def angle(self, p3):
        dx21 = self.p2[0] - self.p1[0]
        dy21 = self.p2[1] - self.p1[1]
        dx31 = p3[0] - self.p1[0]
        dy31 = p3[1] - self.p1[1]
        m12 = math.sqrt(dx21 * dx21 + dy21 * dy21)
        m13 = math.sqrt(dx31 * dx31 + dy31 * dy31)
        theta = math.acos((dx21 * dx31 + dy21 * dy31) / (m12 * m13))
        return math.degrees(theta)

from enum import Enum

class Pedestrian_States(Enum):
    TowardCrossing = 1
    StartCrossing = 2
    Crossing = 3
    FinishCrossing = 4
    AwayFromCrossing = 5

class Vehicle_States(Enum):
    InCrossing = 1
    NotInCrossing = 2
    AwayFromCrossing = 3
    AboutToCross = 4

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def distance_between_two_points(x,y):
    return math.sqrt((x[0] - y[0]) * (x[0] - y[0]) + (x[1] - y[1]) * (x[1] - y[1]))

def diff_between_two_angles(x,y):
    a = x - y
    if a > 180:
        a -= 360
    elif a < -180:
        a += 360
    return math.fabs(a)

def print_array(arr, name):
    str = ''
    str += '%010s   :' % name
    for i in range(len(arr)):
        str += "%5s, "% round(arr[i], 1)
    print(str)

def print_two_arrays(pos_a, neg_a, pos_name, neg_name):
    str = ''
    str += '%010s ++:' % pos_name
    for i in range(len(pos_a)):
        str += "%5s, "% round(pos_a[i], 1)
    print(str)
    str = ''
    str += '%010s --:' % neg_name
    for i in range(len(neg_a)):
        str += "%5s, "% round(neg_a[i], 1)
    print(str)