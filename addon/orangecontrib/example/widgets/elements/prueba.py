from typing import Tuple
import xml.etree.ElementTree as ET
from lxml.etree import QName
from settings import ParameterGroup, Parameter, ParameterGroupList, Spectrum, SpectrumList, ConfigIBD
import sys
import numpy as np
from scipy.io import savemat, loadmat
from time import time
import cv2

def load_imzml(filepath: str) -> Tuple[ParameterGroupList, SpectrumList]:
    PARAMETER_GROUP_LIST_KEYS = [
        "referenceableParamGroupList", "scanSettingsList"]
    parameter_group_list = ParameterGroupList()
    spectrum_list = SpectrumList()
    im_list = []
    for event, element in ET.iterparse(filepath):
        tag = QName(element.tag).localname
        # print(tag)
        if tag in PARAMETER_GROUP_LIST_KEYS:
            for ref_param_group in element:
                parameter_group = ParameterGroup.from_element(ref_param_group)
                parameter_group_list.add(parameter_group)
        elif tag == "spectrum":
            spectrum, im = Spectrum.from_element(element)
            im_list.append(im)
            spectrum_list.add(spectrum)
    mz_format = parameter_group_list.get_parameter_group("mzArray").get_parameter("format")
    i_format = parameter_group_list.get_parameter_group("intensities").get_parameter("format")
    mz_array_len = int(max(map(lambda im:im[4],im_list)))
    i_array_len = int(max(map(lambda im:im[7],im_list)))
    configIBD = ConfigIBD(mz_format, i_format,mz_array_len,i_array_len)
    im_list = np.array(im_list)
    configIBD.export("pruebas/")

    print(parameter_group_list.to_string())
    quit()
    savemat('pruebas/data.mat', {'data': im_list})
    savemat('pruebas/map.mat', {'map':im_list[:,:2]})
    return parameter_group_list, spectrum_list

def get_format(format:str) -> type:
    format_converter = {
        "64-bit float":np.float64,
        "32-bit float":np.float32,
        "16-bit float":np.float16,
    }
    return format_converter[format]

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

def numpy_fillna(data):
    # Get lengths of each row of data
    lens = np.array([len(i) for i in data])

    # Mask of valid places in each row
    mask = np.arange(lens.max()) < lens[:,None]

    # Setup output array and put elements from data into masked positions
    out = np.zeros(mask.shape, dtype=data.dtype)
    out[mask] = np.concatenate(data)
    return out

def fix(arr, length, format):
    result = np.zeros(length,format)
    result[:len(arr)] = arr
    return result

@timer_func
def load_ibd(filepath:str, im_list_filepath:str, configIBD: ConfigIBD) -> None:
    im_list = loadmat(im_list_filepath)['data']
    mz_format = get_format(configIBD.mz_format)
    i_format = get_format(configIBD.i_format)
    mz_array_list = []
    i_array_list = []
    with open(filepath, 'rb') as ibd_file:
        for im in im_list:
            ibd_file.seek(int(im[2]))
            
            mz_array = np.fromfile(ibd_file,mz_format,int(im[4]))
            mz_array_list.append(fix(mz_array, configIBD.max_length_mz_array, mz_format))
            ibd_file.seek(int(im[6]))
            i_array = np.fromfile(ibd_file,i_format,int(im[7]))
            i_array_list.append(fix(i_array, configIBD.max_length_i_array, i_format))
    savemat('pruebas/mz_array_list.mat', {'data': np.array(mz_array_list)})
    savemat('pruebas/i_array_list.mat', {'data': np.array(i_array_list)})

                    # imgi(i,1) = xIndex;
                    # imgi(i,2) = yIndex;
                    # imgi(i,3) = externalOffset;
                    # imgi(i,4) = xy_ion_current;
                    # imgi(i,5) = externalArrayLength;
                    # imgi(i,6) = externalEncodedLength;
                    # imgi(i,7) = externalOffseti;
                    # imgi(i,8) = externalArrayLengthi;
                    # imgi(i,9) = externalEncodedLengthi;



def get_min_max(arr, col):
    return int(np.max(arr[:,[col]])), int(np.min(arr[:,[col]]))
def visualize():
    im_list = loadmat("pruebas\data.mat")['data']
    max_x, min_x = get_min_max(im_list,0)

    max_y, min_y = get_min_max(im_list,1)
    im_crude = np.zeros((max_x, max_y), dtype=np.float32)
    for i in im_list:
        im_crude[int(i[0])-1, int(i[1])-1] = i[3]
    out = np.zeros(im_crude.shape, np.float32)
    normalized = cv2.normalize(im_crude, out, 1.0, 0.0, cv2.NORM_MINMAX)
    cv2.imshow("image",normalized)
    cv2.waitKey(0) 
  
#closing all open windows 
    cv2.destroyAllWindows()
    quit()

def main(filepath, im_list):
    config = ConfigIBD("64-bit float", "32-bit float",2001,2001)
    load_ibd(filepath, im_list, config)


if __name__ == "__main__":
    load_imzml(sys.argv[1])
    # if len(sys.argv) == 3:
    #     sys.exit(main(sys.argv[1], sys.argv[2]))
    # else:
    #     print("faltan inputs")

#"C:\Users\User\Desktop\experimento equipo nuevo\220225cerratambtsub10um-no_normalization.ibd" "pruebas\data.mat"

# F:\experimentos_Kratos\ASMS 2019_imzml\IBD 013 1A\IBD 013 1A.imzML
# C:\Users\User\Desktop\experimento equipo nuevo\220225cerratambtsub10um-no_normalization.imzml"                                           <cvparam accession="IMS:1000401" cvreo nuevo\220225cerratambtsub10um-no_normalization.imzml
