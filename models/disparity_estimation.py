import torch
import torchvision

from models.models_common import *

class Semantics(torch.nn.Module):
	def __init__(self):
		super(Semantics, self).__init__()

		moduleVgg = torchvision.models.vgg19_bn(pretrained=True).features.eval()

		self.moduleVgg = torch.nn.Sequential(
			moduleVgg[0:3],
			moduleVgg[3:6],
			torch.nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
			moduleVgg[7:10],
			moduleVgg[10:13],
			torch.nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
			moduleVgg[14:17],
			moduleVgg[17:20],
			moduleVgg[20:23],
			moduleVgg[23:26],
			torch.nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
			moduleVgg[27:30],
			moduleVgg[30:33],
			moduleVgg[33:36],
			moduleVgg[36:39],
			torch.nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
		)
	# end

	def forward(self, tenInput):
		tenPreprocessed = tenInput[:, [ 2, 1, 0 ], :, :]

		tenPreprocessed[:, 0, :, :] = (tenPreprocessed[:, 0, :, :] - 0.485) / 0.229
		tenPreprocessed[:, 1, :, :] = (tenPreprocessed[:, 1, :, :] - 0.456) / 0.224
		tenPreprocessed[:, 2, :, :] = (tenPreprocessed[:, 2, :, :] - 0.406) / 0.225

		return self.moduleVgg(tenPreprocessed)
	# end
# end

class Disparity(torch.nn.Module):
	def __init__(self):
		super(Disparity, self).__init__()

		self.moduleImage = torch.nn.Conv2d(in_channels=3, out_channels=32, kernel_size=7, stride=2, padding=3)
		self.moduleSemantics = torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)

		for intRow, intFeatures in [ (0, 32), (1, 48), (2, 64), (3, 512), (4, 512), (5, 512) ]:
			self.add_module(str(intRow) + 'x0' + ' - ' + str(intRow) + 'x1', Basic('relu-conv-relu-conv', [ intFeatures, intFeatures, intFeatures ]))
			self.add_module(str(intRow) + 'x1' + ' - ' + str(intRow) + 'x2', Basic('relu-conv-relu-conv', [ intFeatures, intFeatures, intFeatures ]))
			self.add_module(str(intRow) + 'x2' + ' - ' + str(intRow) + 'x3', Basic('relu-conv-relu-conv', [ intFeatures, intFeatures, intFeatures ]))
		# end

		for intCol in [ 0, 1 ]:
			self.add_module('0x' + str(intCol) + ' - ' + '1x' + str(intCol), Downsample([ 32, 48, 48 ]))
			self.add_module('1x' + str(intCol) + ' - ' + '2x' + str(intCol), Downsample([ 48, 64, 64 ]))
			self.add_module('2x' + str(intCol) + ' - ' + '3x' + str(intCol), Downsample([ 64, 512, 512 ]))
			self.add_module('3x' + str(intCol) + ' - ' + '4x' + str(intCol), Downsample([ 512, 512, 512 ]))
			self.add_module('4x' + str(intCol) + ' - ' + '5x' + str(intCol), Downsample([ 512, 512, 512 ]))
		# end

		for intCol in [ 2, 3 ]:
			self.add_module('5x' + str(intCol) + ' - ' + '4x' + str(intCol), Upsample([ 512, 512, 512 ]))
			self.add_module('4x' + str(intCol) + ' - ' + '3x' + str(intCol), Upsample([ 512, 512, 512 ]))
			self.add_module('3x' + str(intCol) + ' - ' + '2x' + str(intCol), Upsample([ 512, 64, 64 ]))
			self.add_module('2x' + str(intCol) + ' - ' + '1x' + str(intCol), Upsample([ 64, 48, 48 ]))
			self.add_module('1x' + str(intCol) + ' - ' + '0x' + str(intCol), Upsample([ 48, 32, 32 ]))
		# end

		self.moduleDisparity = Basic('conv-relu-conv', [ 32, 32, 1 ])
	# end

	def forward(self, tenImage, tenSemantics):
		tenColumn = [ None, None, None, None, None, None ]

		tenColumn[0] = self.moduleImage(tenImage)
		tenColumn[1] = self._modules['0x0 - 1x0'](tenColumn[0])
		tenColumn[2] = self._modules['1x0 - 2x0'](tenColumn[1])
		tenColumn[3] = self._modules['2x0 - 3x0'](tenColumn[2]) + self.moduleSemantics(tenSemantics)
		tenColumn[4] = self._modules['3x0 - 4x0'](tenColumn[3])
		tenColumn[5] = self._modules['4x0 - 5x0'](tenColumn[4])

		intColumn = 1
		for intRow in range(len(tenColumn)):
			tenColumn[intRow] = self._modules[str(intRow) + 'x' + str(intColumn - 1) + ' - ' + str(intRow) + 'x' + str(intColumn)](tenColumn[intRow])
			if intRow != 0:
				tenColumn[intRow] += self._modules[str(intRow - 1) + 'x' + str(intColumn) + ' - ' + str(intRow) + 'x' + str(intColumn)](tenColumn[intRow - 1])
			# end
		# end

		intColumn = 2
		for intRow in range(len(tenColumn) -1, -1, -1):
			tenColumn[intRow] = self._modules[str(intRow) + 'x' + str(intColumn - 1) + ' - ' + str(intRow) + 'x' + str(intColumn)](tenColumn[intRow])
			if intRow != len(tenColumn) - 1:
				tenUp = self._modules[str(intRow + 1) + 'x' + str(intColumn) + ' - ' + str(intRow) + 'x' + str(intColumn)](tenColumn[intRow + 1])

				if tenUp.shape[2] != tenColumn[intRow].shape[2]: tenUp = torch.nn.functional.pad(input=tenUp, pad=[ 0, 0, 0, -1 ], mode='constant', value=0.0)
				if tenUp.shape[3] != tenColumn[intRow].shape[3]: tenUp = torch.nn.functional.pad(input=tenUp, pad=[ 0, -1, 0, 0 ], mode='constant', value=0.0)

				tenColumn[intRow] += tenUp
			# end
		# end

		intColumn = 3
		for intRow in range(len(tenColumn) -1, -1, -1):
			tenColumn[intRow] = self._modules[str(intRow) + 'x' + str(intColumn - 1) + ' - ' + str(intRow) + 'x' + str(intColumn)](tenColumn[intRow])
			if intRow != len(tenColumn) - 1:
				tenUp = self._modules[str(intRow + 1) + 'x' + str(intColumn) + ' - ' + str(intRow) + 'x' + str(intColumn)](tenColumn[intRow + 1])

				if tenUp.shape[2] != tenColumn[intRow].shape[2]: tenUp = torch.nn.functional.pad(input=tenUp, pad=[ 0, 0, 0, -1 ], mode='constant', value=0.0)
				if tenUp.shape[3] != tenColumn[intRow].shape[3]: tenUp = torch.nn.functional.pad(input=tenUp, pad=[ 0, -1, 0, 0 ], mode='constant', value=0.0)

				tenColumn[intRow] += tenUp
			# end
		# end

		return torch.nn.functional.threshold(input=self.moduleDisparity(tenColumn[0]), threshold=0.0, value=0.0)
	# end
# end

moduleSemantics = Semantics().cuda().eval()
moduleDisparity = Disparity().cuda().eval(); moduleDisparity.load_state_dict(torch.load('./models/disparity-estimation.pytorch'))

def disparity_estimation(tenImage):
	intWidth = tenImage.shape[3]
	intHeight = tenImage.shape[2]

	fltRatio = float(intWidth) / float(intHeight)

	intWidth = min(int(512 * fltRatio), 512)
	intHeight = min(int(512 / fltRatio), 512)

	tenImage = torch.nn.functional.interpolate(input=tenImage, size=(intHeight, intWidth), mode='bilinear', align_corners=False)

	return moduleDisparity(tenImage, moduleSemantics(tenImage))
# end