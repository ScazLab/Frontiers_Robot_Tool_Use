#!/usr/bin/env python

# Software License Agreement (MIT License)
#
# Copyright (c) 2020, tri_star
# All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Author: Meiying Qin, Jake Brawer

import os
import re
import shutil
import glob
import json

import numpy as np

import rospy

from tri_star import transformation_util

DIRNAME_ARCHIVE = "archive_{}"

"""
manage files/dirs
"""
def get_filenames_in_dir(dir_path, ext=None):
    if ext:
        dir_path += "/*." + ext
    else:
        if not dir_path.endswith("/"):
            dir_path += "/*"
    return [os.path.basename(i) for i in glob.glob(dir_path)]

def get_dirnames_in_dir(dir_path):
    all_content = get_all_in_dir(dir_path)
    abs_dir = [i for i in all_content if os.path.isdir(i)]
    return [os.path.basename(i) for i in abs_dir]

def get_all_in_dir(dir_path):
    return [os.path.abspath(i) for i in glob.glob(dir_path + "/*")]

def archive(base_data_dir):
    # if nothing to archive, then do not do anything
    to_archive = False
    for name in get_all_in_dir(base_data_dir):
        if re.search(DIRNAME_ARCHIVE, name) is None: # find a file or folder that is not the archive
            to_archive = True
            break
    
    if not to_archive:
        return
    
    archive_index = get_index_in_dir(base_data_dir, get_re_from_file_name(DIRNAME_ARCHIVE))
    archive_dir_name = DIRNAME_ARCHIVE.format(archive_index)        
    archive_dir_path = os.path.join(base_data_dir, archive_dir_name)
    
    create_dir(archive_dir_path)
    
    contents = get_all_in_dir(base_data_dir)
    for content in contents:
        if re.search(get_re_from_file_name(DIRNAME_ARCHIVE), os.path.basename(content)) is None: # not one of the archives
            shutil.move(content, archive_dir_path)    

# the default is either a directory with a number as the name, like 1,2,3
# or a file with the name 1.txt, 2.subl, etc.
def get_index_in_dir(dir_path, file_name_template="^[0-9]+$|^[0-9]+\.\S+$"):
    list_file_name = get_filenames_in_dir(dir_path)
    used_index = []
    
    for file_name in list_file_name:
        if not re.search(file_name_template, file_name) is None: # find the files that matches with it
            # get the number in the string
            matched = re.findall("[0-9]+", file_name)[0]
            number = str_to_int(matched)
            if number is not None:
                used_index.append(number)
    
    if len(used_index) == 0:
        return 1
    
    return max(used_index) + 1

def get_re_from_file_name(filename):
    return "^" + filename.replace("{}", "[0-9]+") + "$" # add v and $ to match the entire string

# create all the directories on the path
def create_dir(dir_path):
    #https://stackoverflow.com/questions/273192/how-can-i-safely-create-a-nested-directory
    try: 
        os.makedirs(dir_path)
    except OSError:
        if not os.path.isdir(dir_path):
            raise Exception("{} is not a directory path".format(dir_path))

"""
i/o related
"""

# https://stackoverflow.com/questions/26646362/numpy-array-is-not-json-serializable
class ToolUseEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool) or isinstance(obj, np.bool_):
            return bool(obj)        
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, transformation_util.AngleGroup):
            return obj.to_json()
        return json.JSONEncoder.default(self, obj)

# variable type
#TYPE_STR = "string"
#TYPE_MATRIX = "matrix"
#TYPE_ARRAY = "array"
#TYPE_LIST = "list"
#TYPE_NESTED_LIST = "nested_list"
#TYPE_INT = "int"
#TYPE_FLOAT = "float"
TYPE_LIST = "list"
TYPE_NUMPY = "numpy"
TYPE_ANGLEGROUP = "anglegroup"

def str_to_int(string):
    string = string.strip()
    number = None
    try:
        if string == "None":
            number = None
        else:
            number = int(string)
    except ValueError as e:
        pass
    return number

def str_to_float(string):
    string = string.strip()
    number = None
    try:
        if string == "None":
            number = None
        else:        
            number = float(string)
    except ValueError as e:
        pass
    return number

def str_to_npmatrix(string): # e.g., "[[1 2] [3 5]]"
    string = string.strip()
    
    if string == "None":
        return None   
    
    value = None
    current_list = None
    lists = []
    number_str = ""
    
    for char in string:
        if char == "[":
            if value is None:
                value = []
                current_list = value
                lists.append(current_list)
            else:
                new_list = []
                current_list.append(new_list)
                current_list = new_list
                lists.append(current_list)
        elif char == " ":
            if number_str == "":
                pass
            else:
                number = str_to_float(number_str)
                current_list.append(number)
                number_str = ""
        elif char == "]":
            if number_str != "":
                number = str_to_float(number_str)
                current_list.append(number)
                number_str = ""
            lists.pop()
            if len(lists) != 0:
                current_list = lists[-1]
        else:
            number_str += char
    
    return np.array(value)
            
def str_to_nparray(string): # e.g., specifically numpy array with shape (n,), like [1 2 3]
    value = string.strip()
    value = value.replace("[", "")
    value = value.replace("]", "")
    value = value.split()
    value = [str_to_float(i) for i in value]
    
    return np.array(value)

def nparray_to_str(matrix):
    value = str(matrix.tolist()).replace(",", " ")
    value = value.replace("\n", "")
    
    return value

def variable_to_string_no_name(variable, variable_collection_type=None):
    content = ""
    
    if variable_collection_type is None:
        if isinstance(variable, np.ndarray):
            content += nparray_to_str(variable) + "\n"
        else:
            variable = str(variable).replace("\n", "")
            content += str(variable) + "\n"
    elif variable_collection_type == TYPE_LIST:
        content += str(len(variable)) + "\n"
        for element in variable:
            if isinstance(element, np.ndarray):
                content += nparray_to_str(element) + "\n"
            else:
                element = str(element).replace("\n", "")
                content += str(element) + "\n"
    elif variable_collection_type == TYPE_NESTED_LIST: # 2 layers
        content += str(len(variable)) + "\n"
        for element in variable:
            content += variable_to_string_no_name(element, TYPE_LIST)
    
    return content    

# for saving purposes
def variable_to_string(name, variable, variable_collection_type=None):
    content = name + "\n"
    content += variable_to_string_no_name(variable, variable_collection_type)
 
    return content

# variable is a str
def convert_variable(variable, variable_type):
    variable = variable.strip()
    
    if variable == "None":
        variable = None
    elif variable_type == TYPE_STR:
        variable = variable
    elif variable_type == TYPE_INT:
        variable = str_to_int(variable)
    elif variable_type == TYPE_FLOAT:
        variable = str_to_float(variable)
    elif variable_type == TYPE_MATRIX:
        variable = str_to_npmatrix(variable)
    elif variable_type == TYPE_ARRAY:
        variable = str_to_nparray(variable)
    
    return variable

def read_variable(file_path, name, variable_type=None, variable_collection_type=None):
    json_result = {}
    with open(file_path, "r") as read_file:
        json_result = json.load(read_file)
    
    variable = json_result[name]
    
    if variable_collection_type == TYPE_LIST:
        if variable_type == TYPE_NUMPY:
            for i in range(len(variable)):
                variable[i] = np.array(variable[i])
    else:
        if variable_type == TYPE_NUMPY:
            variable = np.array(variable)
        elif variable_type == TYPE_ANGLEGROUP:
            variable = transformation_util.AngleGroup.from_json(variable)
    
    return variable
