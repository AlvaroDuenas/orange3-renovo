
import enum

import cv2


import pandas as pd
import typing
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import numpy as np
import sys
import math
from time import time
if typing.TYPE_CHECKING:
    # pylint: disable=invalid-name
    T = typing.TypeVar("T")
    K = typing.TypeVar("K")
    E = typing.TypeVar("E", bound=enum.Enum)
import numpy as np

from scipy.io import loadmat


class Graficos:
    def __init__(self, mz_array_list, i_array_list, im_list):
        self.lastxind = 0
        self.lastyind = 0
        self.lastid = 0
        self.mz_array_list = mz_array_list
        self.i_array_list = i_array_list
        self.im_list = im_list
        max_x, min_x = get_min_max(im_list, 0)
        max_y, min_y = get_min_max(im_list, 1)
        im_crude = np.zeros((max_x, max_y), dtype=np.float32)
        for i in im_list:
            im_crude[int(i[0])-1, int(i[1])-1] = i[3]
        out = np.zeros(im_crude.shape, np.float32)
        normalized = cv2.normalize(im_crude, out, 1.0, 0.0, cv2.NORM_MINMAX)
        print(normalized)
        print(normalized.shape)

        self.fig, (self.ax, self.ax2,self.ax3) = plt.subplots(3, 1)
        im = self.ax.imshow(normalized)
        self.ax.set_picker(True)
        self.ax.set_title("Image's Heatmap")
        self.text = self.ax.text(0.05, 0.95, 'selected: none',
                                 transform=self.ax.transAxes, va='top')
        self.selected, = self.ax.plot([0], [0], 'o', ms=12, alpha=0.4,
                                      color='yellow', visible=False)

        self.fig.colorbar(im, ax=self.ax, label='Colorbar')
        self.fig.canvas.mpl_connect('pick_event', self.on_pick)
        # fig.canvas.mpl_connect('key_press_event', browser.on_press)
        self.get_spectrasum()
        self.show_graphs()

    # def on_press(self, event):
    #     if self.lastind is None:
    #         return
    #     if event.key not in ('n', 'p'):
    #         return
    #     if event.key == 'n':
    #         inc = 1
    #     else:
    #         inc = -1

    #     self.lastind += inc
    #     self.lastind = np.clip(self.lastind, 0, len(xs) - 1)
    #     self.update()
    def show_graphs(self):
        plt.show()
    def on_pick(self, event):

        # if event.artist != normalized:
        #     return True

        # the click locations
        x = event.mouseevent.xdata
        y = event.mouseevent.ydata
        self.lastxind = int(x)
        self.lastyind = int(y)
        print(self.lastxind, self.lastyind)
        xy = loadmat("addon/xy")['data']
        for id, coord in enumerate(xy):
            if (int(coord[1]) == self.lastxind and int(coord[0]) == self.lastyind):
                self.lastid = id
                break
        else:
            return
        self.update()
        
    def get_spectrasum(self):
        mz_array, i_array = sum_spectrum(self.mz_array_list, self.i_array_list)
        self.new_mz_array = reduce_mz_array(mz_array, 0.0005)
        print(len(self.new_mz_array), len(np.unique(np.asarray(self.new_mz_array))))
        self.new_i_array = reassign_mz_array(self.new_mz_array, mz_array, i_array)
        self.spectrasum = dict(zip(self.new_mz_array, self.new_i_array))
        self.ax3.set_title("Spectrum Plot")
        self.ax3.set_xlabel("Mz")
        self.ax3.set_ylabel("Intensity")
        self.ax3.stem(self.new_mz_array, self.new_i_array, linefmt="grey", markerfmt ='.')
        

    def update(self):
        if self.lastxind is None or self.lastyind is None:
            return

        dataxind = self.lastxind
        datayind = self.lastyind

        self.ax2.clear()
        mz_array = loadmat("addon/mz_array_list.mat")['data']
        i_array = loadmat("addon/i_array_list.mat")['data']
        x = mz_array[self.lastid]
        y = i_array[self.lastid]
        # plt.rcParams["figure.figsize"] = [7.50, 3.50]
        self.ax2.plot(x, y)
        # ax2.plot(X[dataind])

        # ax2.text(0.05, 0.9, f'mu={xs[dataind]:1.3f}\nsigma={ys[dataind]:1.3f}',
        #          transform=ax2.transAxes, va='top')
        # ax2.set_ylim(-0.5, 1.5)
        self.selected.set_visible(True)
        self.selected.set_data(self.lastxind, self.lastyind)

        # self.text.set_text('selected: %d' % dataind)
        self.fig.canvas.draw()


def get_min_max(arr, col):
    return int(np.max(arr[:, [col]])), int(np.min(arr[:, [col]]))


def sum_spectrum(mz_array, i_array):
    ''' calculates the sum of all spectra accross the matrix. '''

    total_spectra = dict()
    for i in range(len(mz_array)):        
        for j in range(len(mz_array[i])):  
            if mz_array[i][j] == 0 and j != 0:
                break
            if mz_array[i][j] in total_spectra.keys():
                total_spectra[mz_array[i][j]] += i_array[i][j]
            else:
                total_spectra[mz_array[i][j]] = i_array[i][j]
    total_spectra = dict(sorted(total_spectra.items()))
    mz_values = list(total_spectra.keys())
    i_values = list(total_spectra.values())
    return mz_values[0:1000], i_values[0:1000]


def reduce_mz_array_list(mz_array_list=loadmat("addon/mz_array_list.mat")['data'], precision=0.0001):
    cleaned_mz_array_list = []
    for mz_array in mz_array_list:
        cleaned_mz_array_list.append(reduce_mz_array(mz_array, precision))
    full_mz_array = list(np.concatenate(cleaned_mz_array_list).flat)
    print(
        f"Precision:{precision} Uncleaned:{len(full_mz_array)} Cleaned: {len(np.unique(full_mz_array))}")
    return cleaned_mz_array_list

def timer_func(func):
    # This function shows the execution time of 
    # the function object passed
    def wrap_func(*args, **kwargs):
        t1 = time()
        result = func(*args, **kwargs)
        t2 = time()
        print(f'Function {func.__name__!r} executed in {(t2-t1):.4f}s')
        return result
    return wrap_func

def reduce_mz_array(mz_array, precision=0.0001):
    cleaned_mz_array = []
    previous_mz = None
    for mz in mz_array:
        if previous_mz is None or (previous_mz is not None and (abs(mz - previous_mz) > precision)):
            cleaned_mz_array.append(mz)
            previous_mz = mz
    return cleaned_mz_array


def reassign_mz_array(new_mz_array, old_mz_array, old_i_array):
    new_i_array = np.zeros(len(new_mz_array))
    index_new = 0
    print(len(new_mz_array), len(old_mz_array), len(old_i_array))
    print(type(new_mz_array), type(old_mz_array), type(old_i_array))
    for index_old in range(len(old_mz_array)):
        index_new += find_nearest(
            new_mz_array[index_new:], old_mz_array[index_old])
        new_i_array[index_new] += old_i_array[index_old]
        #print(old_mz_array[index_old], new_mz_array[index_new-2:index_new+2], new_mz_array[index_new])
        
    return new_i_array

def find_nearest(mz_array, value, prev_diff=None, idx=0):
    if len(mz_array)==0 or (prev_diff is not None and math.fabs(value - mz_array[0]) > prev_diff):
        return idx-1
    else:
        return find_nearest(mz_array[1:], value, math.fabs(value - mz_array[0]), idx+1)


def heatmap():
    im_list = loadmat("addon/imgi.mat")['data']
    max_x, min_x = get_min_max(im_list, 0)

    max_y, min_y = get_min_max(im_list, 1)
    im_crude = np.zeros((max_x, max_y), dtype=np.float32)
    for i in im_list:
        im_crude[int(i[0])-1, int(i[1])-1] = i[3]
    out = np.zeros(im_crude.shape, np.float32)
    normalized = cv2.normalize(im_crude, out, 1.0, 0.0, cv2.NORM_MINMAX)
    colormap = get_cmap('inferno')
    heatmap = (colormap(normalized) * 2**16).astype(np.uint16)[:, :, :3]
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR)
    cv2.imshow('normalized', normalized)
    cv2.imshow('heatmap', heatmap)





def stem_plot(mz_array_path, i_array_path):
    mz_array = loadmat(mz_array_path)['data']
    i_array = loadmat(i_array_path)['data']
    print(mz_array[0][0:5], i_array[0][0:5])


def get_min_max(arr, col):
    return int(np.max(arr[:, [col]])), int(np.min(arr[:, [col]]))



if __name__ == "__main__":
    graficos = Graficos(loadmat("addon/mz_array_list.mat")['data'],loadmat("addon/i_array_list.mat")['data'],loadmat("addon/imgi.mat")['data'])
    graficos.show_graphs()