#!/usr/bin/env python
"""
Author: Ge Li, ge.li@tuebingen.mpg.de

This script receives a json file including landing positions of the ball given certain time parameters of hitting ProMP. This script will generate a 2D coutour map, showing an intuitive distribution of the landing points.
"""

import json
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import argparse
import os
import math


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
        self.t_obs_1 = list()
        self.x_list = list()
        self.y_list = list()
        self.target = [0.35, -3.13]
        self.distance_to_target = None
        self.distance_reward = None

    def load_json(self):
        with open(self.json_file_name) as f_obj:
            data_set = json.load(f_obj)
        
        self.T_list = np.unique(np.array(data_set['T']))
        self.len_T = len(self.T_list)

        self.delta_t0_list = np.unique(np.array(data_set['delta_t0']))
        self.len_t0 = len(self.delta_t0_list)        
        
        print("number of T is {}.".format(self.len_T))
        print("number of delta_t0 is {}.".format(self.len_t0))

        self.x_list = data_set['x']
        self.y_list = data_set['y']

    def compute_distance_to_target(self):
        temp_distance_list = list()
        for counter in range(len(self.x_list)):
            distance = math.sqrt(math.pow(
                (self.x_list[counter] - self.target[0]), 2) + math.pow(self.y_list[counter] - self.target[1], 2))
            temp_distance_list.append(distance)
        #self.distance_to_target = np.reshape(temp_distance_list, (31, 21))
        self.distance_to_target = np.reshape(temp_distance_list, (self.len_T, self.len_t0))

    def compute_distance_reward(self):
        pass

    def draw_landing_scatter(self):
        plt.scatter(self.x_list,self.y_list)
        plt.show()

    def draw_distance_contour(self):
        
        levels = np.linspace(0.0,self.display_range,31)
        #plt.contourf(self.delta_t0_list,self.T_list, self.distance_to_target)
        plt.contourf(self.delta_t0_list,self.T_list, self.distance_to_target, levels)
        plt.colorbar()
        plt.ylabel("T, [s]")
        plt.xlabel("delta_t0, [s]")

        
        #plt.show()
        plt.savefig(self.save_dir_name + ".png",bbox_inches='tight', dpi = 200)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    def rel_path(fname): return os.path.join(os.path.dirname(__file__), fname)

    parser.add_argument('file', help="File name where the data is stored")
    parser.add_argument('save', help="File name where the picture should be stored")
    #parser.add_argument('--reward_range', nargs='+', type=float, help="range of reward")
    #parser.add_argument('--reward', nargs='+', type=int, help="Type of reward, 1 for ")
    parser.add_argument('--max_range', nargs='+', type=float, help="Maximum range displayed in the picture")
    args = parser.parse_args()

    analyseTime = AnalyseTime(args)
    analyseTime.load_json()
    analyseTime.compute_distance_to_target()
    analyseTime.draw_distance_contour()
    #analyseTime.draw_landing_scatter()
