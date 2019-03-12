"""
Author Ge Li, ge.li@tuebingen.mpg.de
Script to process loss data
"""
import json
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import math
import pathlib
import scipy.signal as ss

class ProcessLoss():
    def __init__(self, args):
        self.file = args.file
        self.pic_dir_file = args.pic_dir_file
        self.raw_data = list()
        self.processed_data = list()
        self.load_json()
        self.savgol_filter()
        self.draw()

    def load_json(self):
        path = pathlib.Path(self.file)
        file_json = str(path.resolve().parent) + str(path.resolve().anchor) + str(path.stem) + ".json"
        with open(file_json) as json_obj:
            self.raw_data = json.load(json_obj)

    def savgol_filter(self):
        self.processed_data = ss.savgol_filter(self.raw_data, 151, 5, mode="nearest")


    def draw(self):
        #plt.plot(self.processed_data)
        plt.plot(self.raw_data)
        plt.xlabel("episodes")
        plt.ylabel("loss")
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    def rel_path(fname): return os.path.join(os.path.dirname(__file__), fname)
    parser.add_argument('file', help="File where the loss data is stored.")
    parser.add_argument(
        'pic_dir_file', help="Picture file where the output picture should be stored.")
    args = parser.parse_args()
    pl = ProcessLoss(args)
