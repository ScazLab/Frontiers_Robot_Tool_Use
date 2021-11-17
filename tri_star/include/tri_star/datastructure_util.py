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

import numpy as np

class ArrayList(object):
    def __init__(self, content=[], unique=True):
        self.content = []
        self.shape = None
        self.unique = unique
        
        for array in content:
            self.append(T)
    
    def _convert_element(self, i):
        element = self.content[i]
        
        return np.array(element)
    
    def __getitem__(self, i):
        return self._convert_element(i)
    
    def __iter__(self):
        for i in range(len(self.content)):
            yield self._convert_element(i)
    
    def __contains__(self, value):
        return list(value) in self.content
    
    def __len__(self):
        return len(self.content)
    
    # T: numpy array
    def append(self, array):
        if self.shape is None:
            self.shape = array.shape
        
        assert self.shape == array.shape, "the shape of the array to be inserted {} is not the same as other arrays {}".format(array.shape, self.shape)
                
        arrayT = list(array)
        
        if self.unique:
            if arrayT not in self.content:
                self.content.append(arrayT)
        else:
            self.content.append(arrayT)
    
    def remove(self, value):
        value = list(value)
        self.content.remove(value)
        
class ArrayDict(object):
    def __init__(self, unique=True):
        self.content = {}
        self.unique = unique
    
    def __getitem__(self, key):
        return list(self.content[key])
    
    def __setitem__(self, key, value):
        key = set(key)
        self.content[key] = ArrayList(value, unique)
    
    def keys():
        return [np.array(i) for i in self.content.keys()]
    
    def has_key(self, key):
        key = set(key)
        return key in self.content.keys()