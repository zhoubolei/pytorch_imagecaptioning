import torch
import numpy as np
from PIL import Image
import os
import pdb

__COCO_IMG_PATH = "/data/vision/oliva/scenedataset/vqa_cache/coco_allimage2014"

epoch = 91
#model_date = '2017-10-25_10-47-15' # this is the finetuned model
model_date = '2017-10-25_13-08-32'
model = torch.load('/data/vision/torralba/deepscene/lib/captionGen/results/%s/checkpoint_epoch_%d.pth.tar'%(model_date, epoch))['model']


def show_and_tell(filename, beam_size=3):
    img = Image.open(filename, 'r')
#    imshow(np.asarray(img))
    captions = model.generate(img, beam_size=beam_size)
    print(captions)

show_and_tell(os.path.join(__COCO_IMG_PATH, 'COCO_val2014_000000000073.jpg'))

