import cv2
from skimage.measure import label, regionprops
from sklearn.mixture import GaussianMixture as GMM
from sklearn.cluster import KMeans
from scipy import signal

import copy
import os



import parameters_cell_mod as params_m
from utility import binarized, draw_bboxes, draw_rectangle, visualization
from components import connected_components, regions


import numpy as np
import math
import statistics
import functools


def cosine_similarity():



