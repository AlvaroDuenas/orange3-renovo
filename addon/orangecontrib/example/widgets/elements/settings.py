from operator import contains
from tokenize import Double
from lxml.etree import QName
from typing import Tuple, List
import numpy as np
from io import StringIO
import os
from scipy.io import savemat, loadmat


class StringBuilder:
    _file_str = None

    def __init__(self):
        self._file_str = StringIO()

    def add_line(self, line: str) -> None:
        self._file_str.write(line + "\n")

    def __str__(self):
        return self._file_str.getvalue()


class Parameter:
    def __init__(self, parameter_dict: dict):
        self.name = parameter_dict["name"]
        if "unitName" in parameter_dict:
            self.unit_name = parameter_dict["unitName"]
        if "value" in parameter_dict:
            self.value = parameter_dict["value"]

    def __repr__(self) -> str:
        return self.name

    def __str__(self) -> str:
        return self.name

    def to_string(self) -> str:
        string_builder = StringBuilder()
        if hasattr(self, "unit_name"):
            name_str = f"Parameter Name: {self.name}({self.unit_name})"
        else:
            name_str = f"Parameter Name: {self.name}"
        string_builder.add_line(name_str)
        if hasattr(self, "value"):
            string_builder.add_line(f"\t - Value:\t\t{self.value}")

        return str(string_builder)


class ParameterGroup:
    def __init__(self, id):
        self.parameters = {}
        self.id = id

    def add(self, parameter: Parameter) -> None:
        self.parameters[parameter.name] = parameter
    staticmethod

    def from_element(elements) -> 'ParameterGroup':
        ACCEPTED_ID = ["id", "instrumentConfigurationRef"]
        for key in ACCEPTED_ID:
            if key in elements.attrib:
                id = elements.attrib[key]
                break
        else:
            for element in elements:
                tag = QName(element.tag).localname
                if tag == "referenceableParamGroupRef":
                    id = element.attrib["ref"]
                    break
            else:
                raise Exception
        parameter_group = ParameterGroup(id)
        parameter_group.process(elements)
        return parameter_group

    def process(self, params):
        for cv_param in params:
            if QName(cv_param.tag).localname == "cvParam":
                attribs = cv_param.attrib
                setting_dict = {}
                for key in ["name", "unitName", "value"]:
                    if key in attribs:
                        # TODO: format access
                        if "float" in attribs[key]:
                            setting_dict["name"] = "format"
                            setting_dict["value"] = attribs[key]
                        else:
                            if not key in setting_dict:
                                setting_dict[key] = attribs[key]
                self.add(Parameter(setting_dict))

    def get_parameter(self, key: str) -> Parameter:
        return self.parameters[key]

    def to_string(self) -> str:
        string_builder = StringBuilder()
        string_builder.add_line(f"PARAMETER GROUP {self.id}")
        for key in self.parameters:
            string_builder.add_line(self.get_parameter(key).to_string())
        return str(string_builder)


class ParameterGroupList:
    def __init__(self):
        self.parameter_groups = {}

    def add(self, parameter_group: ParameterGroup) -> None:
        self.parameter_groups[parameter_group.id] = parameter_group

    def get_parameter_group(self, key: str) -> ParameterGroup:
        return self.parameter_groups[key]

    def to_string(self) -> str:
        string_builder = StringBuilder()
        for key in self.parameter_groups:
            string_builder.add_line(self.get_parameter_group(key).to_string())
        return str(string_builder)


class SpectrumList:
    def __init__(self):
        self.spectrums = []

    def add(self, spectrum: 'Spectrum') -> None:
        self.spectrums.append(spectrum)

    def length(self) -> int:
        return len(self.spectrums)


class Spectrum:
    def __init__(self) -> None:
        self.parameter_group_list = ParameterGroupList()

    def set_ion_current(self, ion_current) -> None:
        self.ion_current = ion_current

    def set_id(self, id) -> None:
        self.id = id
    staticmethod

    def from_element(elements) -> Tuple['Spectrum', List[float]]:
        im = np.zeros(9, float)
        spectrum = Spectrum()
        spectrum.set_id(elements.attrib["id"])

        for element in elements:
            tag = QName(element.tag).localname
            if tag == "cvParam" and element.attrib["name"] == "total ion current":
                spectrum.set_ion_current(element.attrib["value"])
            if tag == "scanList":
                for data in element:
                    if QName(data.tag).localname == "scan":
                        spectrum.parameter_group_list.add(
                            ParameterGroup.from_element(data))
            elif tag == "binaryDataArrayList":
                for binaryDataArray in element:
                    spectrum.parameter_group_list.add(
                        ParameterGroup.from_element(binaryDataArray))
        if "intensities" in spectrum.parameter_group_list.parameter_groups.keys():
            intensity_key = "intensities"
        else:
            intensity_key = "intensityArray"
        im[3] = spectrum.ion_current
        im[0] = spectrum.parameter_group_list.get_parameter_group(
            'instrumentConfiguration0').get_parameter('position x').value
        im[1] = spectrum.parameter_group_list.get_parameter_group(
            'instrumentConfiguration0').get_parameter('position y').value
        im[2] = spectrum.parameter_group_list.get_parameter_group(
            'mzArray').get_parameter('external offset').value
        im[4] = spectrum.parameter_group_list.get_parameter_group(
            'mzArray').get_parameter('external array length').value
        im[5] = spectrum.parameter_group_list.get_parameter_group(
            'mzArray').get_parameter('external encoded length').value
        im[6] = spectrum.parameter_group_list.get_parameter_group(
            intensity_key).get_parameter('external offset').value
        im[7] = spectrum.parameter_group_list.get_parameter_group(
            intensity_key).get_parameter('external array length').value
        
        im[8] = spectrum.parameter_group_list.get_parameter_group(
            intensity_key).get_parameter('external encoded length').value
        return spectrum, im, intensity_key


class ConfigIBD:
    FORMAT_MZ_NAME = 'mz_format'
    FORMAT_I_NAME = 'i_format'
    ARRAY_LENGTH_MZ = 'array_len_mz'
    ARRAY_LENGTH_I = 'array_len_i'
    DEFAULT_FILENAME = "config.mat"
    def __init__(self, mz_format: str, i_format: str, max_length_mz_array: int, max_length_i_array: int):
        self.mz_format = mz_format
        self.i_format = i_format
        self.max_length_mz_array = max_length_mz_array
        self.max_length_i_array = max_length_i_array

    def _export(self, filename):
        savemat(filename, {ConfigIBD.FORMAT_MZ_NAME: self.mz_format,
                           ConfigIBD.FORMAT_I_NAME: self.i_format,
                           ConfigIBD.ARRAY_LENGTH_MZ: self.max_length_mz_array,
                           ConfigIBD.ARRAY_LENGTH_I: self.max_length_i_array
                           })

    def _from_file(filename) -> 'ConfigIBD':
        config = loadmat(filename)
        return ConfigIBD(config[ConfigIBD.FORMAT_MZ_NAME], config[ConfigIBD.FORMAT_I_NAME],
                         config[ConfigIBD.ARRAY_LENGTH_MZ], config[ConfigIBD.ARRAY_LENGTH_I])

    def from_file(workspace, filename=DEFAULT_FILENAME) -> 'ConfigIBD':
        return ConfigIBD._from_file(os.path.join(workspace, filename))

    def export(self, workspace, filename=DEFAULT_FILENAME):
        self._export(os.path.join(workspace, filename))
