import torch
import numpy as np
from PIL import Image
import os
import pdb


epoch = 81
model_date = '2017-10-25_13-08-32'
model = torch.load('/data/vision/torralba/deepscene/lib/captionGen/results/%s/checkpoint_epoch_%d.pth.tar'%(model_date, epoch))['model']


cnn_model = model.cnn
torch.save(cnn_model, 'whole_imagecaptioning.pth.tar')
