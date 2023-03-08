from scipy.io import savemat, loadmat
import sys
import os
from PyQt5.QtWidgets import (QApplication,
                             QHBoxLayout, QLabel, QWidget, QPushButton, QFileDialog)
from settings import ParameterGroupList,SpectrumList, ParameterGroup, Spectrum, ConfigIBD
from typing import Tuple
import xml.etree.ElementTree as ET
from lxml.etree import QName
import numpy as np

class Loader(QWidget):
    def __init__(self):
        super().__init__()
        open_folder_layout = QHBoxLayout()
        self.label_dir_path = QLabel("Select the dir path")
        self.button_open_folder = QPushButton('Choose Folder')
        self.button_open_folder.clicked.connect(self.open_workspace)
        open_folder_layout.addWidget(self.label_dir_path)
        open_folder_layout.addWidget(self.button_open_folder)
        self.setLayout(open_folder_layout)
        self.imzml_loaded = False
        self.configIBD_loaded = False
    

    def open_workspace(self):
        self.workspace = str(QFileDialog.getExistingDirectory(
            self, "Select Directory"))
        self.label_dir_path.setText(self.workspace)
        print(self.get_file("data.mat"))

    def get_file(self, file):
        if(os.path.isdir(self.workspace)):
            file_path = os.path.join(self.workspace,file)
            if(not os.path.isfile(file_path)):
                file_path = str(QFileDialog.getOpenFileUrl(self, f"Select {file} file",filter=self.tr(f"File (*{os.path.splitext(file)[1]})")))
            return file_path
    def load_imzml(self,filepath: str) -> Tuple[ParameterGroupList, SpectrumList]:
        PARAMETER_GROUP_LIST_KEYS = [
            "referenceableParamGroupList", "scanSettingsList"]
        self.experiment_settings = ParameterGroupList()
        self.spectrum_list = SpectrumList()
        self.im_list = []
        for event, element in ET.iterparse(filepath):
            tag = QName(element.tag).localname
            # print(tag)
            if tag in PARAMETER_GROUP_LIST_KEYS:
                for ref_param_group in element:
                    parameter_group = ParameterGroup.from_element(ref_param_group)
                    self.experiment_settings.add(parameter_group)
            elif tag == "spectrum":
                spectrum, im = Spectrum.from_element(element)
                self.im_list.append(im)
                self.spectrum_list.add(spectrum)
        self.mz_format = self.experiment_settings.get_parameter_group("mzArray").get_parameter("format")
        self.i_format = self.experiment_settings.get_parameter_group("intensities").get_parameter("format")
        self.mz_array_len = int(max(map(lambda im:im[4],self.im_list)))
        self.i_array_len = int(max(map(lambda im:im[7],self.im_list)))
        self.configIBD = ConfigIBD(self.mz_format, self.i_format,self.mz_array_len,self.i_array_len)
        self.im_list = np.array(self.im_list)
        self.configIBD.export(self.workspace)
        savemat(os.path.join(self.workspace,"imgi.mat"), {'data': self.im_list})
        savemat(os.path.join(self.workspace,"xy.mat"), {'data':self.im_list[:,:2]})
        self.imzml_loaded = True
        return self.experiment_settings, self.spectrum_list
    @staticmethod
    def get_format(format:str) -> type:
        format_converter = {
            "64-bit float":np.float64,
            "32-bit float":np.float32,
            "16-bit float":np.float16,
        }
        return format_converter[format]
    def get_config_IBD(self) -> ConfigIBD:
        if not self.configIBD_loaded:
            self.configIBD = ConfigIBD("64-bit float", "32-bit float",2001,2001)
            self.configIBD_loaded = False
        return self.configIBD
    @staticmethod
    def fix(arr, length, format):
        result = np.zeros(length,format)
        result[:len(arr)] = arr
        return result
    def load_ibd(self, ibd_filepath:str) -> None:
        settings_IBD = self.get_config_IBD()
        mz_format = self.get_format(settings_IBD.mz_format)
        i_format = self.get_format(settings_IBD.i_format)
        mz_array_list = []
        i_array_list = []
        with open(ibd_filepath, 'rb') as ibd_file:
            for im in self.im_list:
                ibd_file.seek(int(im[2]))
                
                mz_array = np.fromfile(ibd_file,mz_format,int(im[4]))
                mz_array_list.append(self.fix(mz_array, settings_IBD.max_length_mz_array, mz_format))
                ibd_file.seek(int(im[6]))
                i_array = np.fromfile(ibd_file,i_format,int(im[7]))
                i_array_list.append(self.fix(i_array, settings_IBD.max_length_i_array, i_format))
        savemat(os.path.join(self.workspace,'mz_array_list.mat'), {'data': np.array(mz_array_list)})
        savemat(os.path.join(self.workspace,'i_array_list.mat'), {'data': np.array(i_array_list)})


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Loader()
    window.load_imzml(sys.argv[1])
    quit()
    window.show()
    sys.exit(app.exec_())