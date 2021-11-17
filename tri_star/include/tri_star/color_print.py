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

# https://misc.flogisoft.com/bash/tip_colors_and_formatting
# https://gist.github.com/vratiu/9780109

# tested under unbuntu 18 bash

RESET = '\033[0m'

BLACK      = '0' # light grey is dark grey
RED        = '1'
GREEN      = '2'
YELLOW     = '3'
BLUE       = '4'
MAGENTA    = '5'
CYAN       = '6'
LIGHT_GRAY = '7' # light grey is white

FOREGROUND_REGULAR = '3'
FOREGROUND_LIGHT   = '9'

BACKGROUND_REGULAR = '4'
BACKGROUND_LIGHT   = '10'

BOLD          = '1'
DIM           = '2'
ITALIC        = '3'
UNDERLINE     = '4'
BLINK         = '5'
REVERSE       = '7'
HIDDEN        = '8'
STRIKETHROUGH = '9'

def colored_text(text, text_color='', text_color_light=False, background_color='', background_color_light=False, bold=False, dim=False, italic=False, underline=False, blink=False, reverse=False, hidden=False, strikethrough=False):
    formatting = ''
    template = "\033[{}m{}\033[00m"
    
    if bold:
        formatting += BOLD + ';'
    
    if dim:
        formatting += DIM + ';'
    
    if italic:
        formatting += ITALIC + ';'
    
    if underline:
        formatting += UNDERLINE + ';'
    
    if blink:
        formatting += BLINK + ';'
    
    if reverse:
        formatting += REVERSE + ';'
    
    if hidden:
        formatting += HIDDEN + ';'
    
    if strikethrough:
        formatting += STRIKETHROUGH + ';'
    
    if text_color:
        if text_color_light:
            formatting += FOREGROUND_LIGHT + text_color + ';'
        else:
            formatting += FOREGROUND_REGULAR + text_color + ';'
    
    if background_color:
        if background_color_light:
            formatting += BACKGROUND_LIGHT + background_color + ';'
        else:
            formatting += BACKGROUND_REGULAR + background_color + ';'
    
    if formatting:
        formatting = formatting[:-1]
        return template.format(formatting, text)
    else:
        return text