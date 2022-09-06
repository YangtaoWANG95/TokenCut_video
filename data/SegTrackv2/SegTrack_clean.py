#import cv2
from PIL import Image
import glob as gb
import numpy as np


def combine(dir1, dir2):
	for i in range(len(dir1)):
		im1 = np.array(Image.open(dir1[i]))
		im2 = np.array(Image.open(dir2[i]))
		Image.fromarray(im1[:,:,1]).save('./test.png')
		ims = np.clip(im1[:,:,1]+im2[:,:,1], 0, 255)
		Image.fromarray(ims).save(dir1[i].replace('/1/', '/').replace('.bmp','.png'))

cats = ['hummingbird', 'drift', 'bmx', 'monkeydog', 'cheetah']
for cat in cats:
	dir1 = sorted(gb.glob('./GroundTruth/{}/1/*.png'.format(cat)) + gb.glob('./GroundTruth/{}/1/*.bmp'.format(cat)))
	dir2 = sorted(gb.glob('./GroundTruth/{}/2/*.png'.format(cat)) + gb.glob('./GroundTruth/{}/2/*.bmp'.format(cat)))
	combine(dir1, dir2)

cat = 'penguin'
dir1 = sorted(gb.glob('./GroundTruth/{}/1/*.png'.format(cat)))
dir2 = sorted(gb.glob('./GroundTruth/{}/2/*.png'.format(cat)))
dir3 = sorted(gb.glob('./GroundTruth/{}/3/*.png'.format(cat)))
dir4 = sorted(gb.glob('./GroundTruth/{}/4/*.png'.format(cat)))
dir5 = sorted(gb.glob('./GroundTruth/{}/5/*.png'.format(cat)))
dir6 = sorted(gb.glob('./GroundTruth/{}/6/*.png'.format(cat)))
for i in range(len(dir1)):
	im1 = np.array(Image.open(dir1[i]))
	im2 = np.array(Image.open(dir2[i]))
	im3 = np.array(Image.open(dir3[i]))
	im4 = np.array(Image.open(dir4[i]))
	im5 = np.array(Image.open(dir5[i]))
	im6 = np.array(Image.open(dir6[i]))

	ims = np.clip(im1+im2+im3+im4+im5+im6, 0, 255)
	Image.fromarray(ims[:,:,1]).save(dir1[i].replace('/1/', '/'))

cat = 'girl'
dir1 = sorted(gb.glob('./GroundTruth/{}/*.bmp'.format(cat)))
for i in range(len(dir1)):
	im = np.array(Image.open(dir1[i]))
	Image.fromarray(im[:,:,0]).save(dir1[i].replace('.bmp', '.png'))