#!/usr/bin/env python
"""
Author: Ge Li, ge.li@tuebingen.mpg.de

This script receives a json file including landing positions of the ball given certain time parameters of hitting ProMP. This script will generate a 2D coutour map, showing an intuitive distribution of the landing points.
"""

import json
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colors
import numpy as np
import argparse
import os
import math
from scipy.spatial import distance


class AnalyseTime():

    """ 
    A python script to analyse T and t0 of ProMP
    """

    def __init__(self, args):
        self.json_file_name = args.file
        self.save_dir_name = args.save
        self.display_range = args.max_range
        self.T_list = None
        self.len_T = None
        self.delta_t0_list = None
        self.len_t0 = None

        self.landing_list = list()

        self.target_coordinate = [0.35, -3.13, -0.99]
        self.distance_to_target = None
        self.distance_reward = None

        self.load_json()
        self.compute_distance_to_target()
        self.draw_distance_contour()

    def load_json(self):
        with open(self.json_file_name) as f_obj:
            data_set = json.load(f_obj)

        self.T_list = np.unique(np.array([data["T"] for data in data_set]))
        self.len_T = len(self.T_list)

        self.delta_t0_list = np.unique(
            np.array([data['delta_t0'] for data in data_set]))
        self.len_t0 = len(self.delta_t0_list)

        #print("number of T is {}.".format(self.len_T))
        #print("number of delta_t0 is {}.".format(self.len_t0))

        self.landing_list = [data['landing_info'] for data in data_set]

    def compute_distance_to_target(self):
        temp_distance_list = list()        
        for landing in self.landing_list:
            distance_2d = distance.euclidean(self.target_coordinate[0:2], landing[0:2])
            temp_distance_list.append(distance_2d)
        self.distance_to_target = np.reshape(np.array(temp_distance_list), (self.len_T, self.len_t0))

    def compute_distance_reward(self):
        pass

    def draw_distance_contour(self):

        levels = np.linspace(0.0, self.display_range, 100)
        #plt.contourf(self.delta_t0_list,self.T_list, self.distance_to_target)

        plt.contourf(self.delta_t0_list, self.T_list,
                     self.distance_to_target, levels, cmap="jet_r")
        plt.colorbar()
        plt.ylabel("T, [s]")
        plt.xlabel("delta_t0, [s]")

        plt.show()
        #plt.savefig(self.save_dir_name + ".png", bbox_inches='tight', dpi=200)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    def rel_path(fname): return os.path.join(os.path.dirname(__file__), fname)

    parser.add_argument('file', help="File name where the data is stored")
    parser.add_argument(
        'save', help="File name where the picture should be stored")
    #parser.add_argument('--reward_range', nargs='+', type=float, help="range of reward")
    #parser.add_argument('--reward', nargs='+', type=int, help="Type of reward, 1 for ")
    parser.add_argument('--max_range', default=4.0, type=float, help="Maximum range displayed in the picture")
    args = parser.parse_args()

    analyseTime = AnalyseTime(args)
