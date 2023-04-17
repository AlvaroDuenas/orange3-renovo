import numpy as np
import cv2
from scipy.io import savemat, loadmat
def sum_spectrum(mz_array, i_array):
    ''' calculates the sum of all spectra accross the matrix. '''
    
    total_spectra = dict()
    for i in range(len(mz_array)):        # loop over pixels
        for j in range(len(mz_array[i])): # loop over mz values
            if mz_array[i][j] == 0 and j != 0:
                break
            if mz_array[i][j] in total_spectra.keys():
                total_spectra[mz_array[i][j]] += i_array[i][j]
            else:
                total_spectra[mz_array[i][j]] = i_array[i][j]
                
    mz_values = list(total_spectra.keys())
    i_values = list(total_spectra.values())
    
    return mz_values, i_values
  
def visualize():
  im_list = loadmat("addon/imgi.mat")['data']
  max_x, min_x = get_min_max(im_list,0)

  max_y, min_y = get_min_max(im_list,1)
  im_crude = np.zeros((max_x, max_y), dtype=np.float32)
  for i in im_list:
      im_crude[int(i[0])-1, int(i[1])-1] = i[3]
  out = np.zeros(im_crude.shape, np.float32)
  normalized = cv2.normalize(im_crude, out, 1.0, 0.0, cv2.NORM_MINMAX)
  cv2.imshow("image",normalized)
  cv2.waitKey(0) 

import matplotlib.pyplot as plt
import numpy as np
import cv2
def heatmap(image):
    
    colormap = plt.get_cmap('inferno')
    heatmap = (colormap(image) * 2**16).astype(np.uint16)[:,:,:3]
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR)

cv2.imshow('image', image)
cv2.imshow('heatmap', heatmap)
cv2.waitKey()

def get_min_max(arr, col):
  return int(np.max(arr[:,[col]])), int(np.min(arr[:,[col]]))

if __name__ == "__main__": 
    visualize()