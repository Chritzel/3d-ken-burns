#!/usr/bin/env python

import torch
import torchvision

import cv2
import math
import moviepy
import moviepy.editor
import numpy
import getopt
import sys

from common import *

from models.disparity_adjustment import *
from models.disparity_estimation import *
from models.disparity_refinement import *
from models.pointcloud_inpainting import *

##########################################################

assert(int(str('').join(torch.__version__.split('.')[0:2])) >= 12) # requires at least pytorch version 1.2.0

torch.set_grad_enabled(False) # make sure to not compute gradients for computational performance

torch.backends.cudnn.enabled = True # make sure to use cudnn for computational performance

##########################################################

objCommon = {}

##########################################################

arguments_strIn = './images/doublestrike.jpg'
arguments_strOut = './autozoom.mp4'

for strOption, strArgument in getopt.getopt(sys.argv[1:], '', [ strParameter[2:] + '=' for strParameter in sys.argv[1::2] ])[0]:
	if strOption == '--in' and strArgument != '': arguments_strIn = strArgument # path to the input image
	if strOption == '--out' and strArgument != '': arguments_strOut = strArgument # path to where the output should be stored
# end

##########################################################

if __name__ == '__main__':
	npyImage = cv2.imread(filename=arguments_strIn, flags=cv2.IMREAD_COLOR)

	intWidth = npyImage.shape[1]
	intHeight = npyImage.shape[0]

	fltRatio = float(intWidth) / float(intHeight)

	intWidth = min(int(1024 * fltRatio), 1024)
	intHeight = min(int(1024 / fltRatio), 1024)

	npyImage = cv2.resize(src=npyImage, dsize=(intWidth, intHeight), fx=0.0, fy=0.0, interpolation=cv2.INTER_AREA)

	objCommon = process_load(npyImage, {}, objCommon)

	objFrom = {
		'fltCenterU': intWidth / 2.0,
		'fltCenterV': intHeight / 2.0,
		'intCropWidth': int(math.floor(0.97 * intWidth)),
		'intCropHeight': int(math.floor(0.97 * intHeight))
	}

	objTo = process_autozoom({
		'fltShift': 100.0,
		'fltZoom': 1.25,
		'objFrom': objFrom
	}, objCommon)

	npyResult, objCommon = process_kenburns({
		'fltSteps': numpy.linspace(0.0, 1.0, 75).tolist(),
		'objFrom': objFrom,
		'objTo': objTo,
		'boolInpaint': True
	}, objCommon)

	moviepy.editor.ImageSequenceClip(sequence=[ npyFrame[:, :, ::-1] for npyFrame in npyResult + list(reversed(npyResult))[1:] ], fps=25).write_videofile(arguments_strOut)
# end