import imageio
from PIL import Image
import PIL
import os
import utilities

range = 202599
x = 1
while x <= range :
	raw = imageio.imread('img_align_celeba\%06d.jpg'%x)
	cropped = raw[20:198,:,:]
	imageio.imwrite('storage.jpg',cropped)
	cropped = Image.open('storage.jpg')
	resized = cropped.resize((64,64),PIL.Image.ANTIALIAS)
	resized.save('64x64_celeba\%06d.jpg'%x)
	utilities.show_loop_progress(x,range)
	x += 1

#there's probably a smoother way of cropping then resizing an image
os.remove('storage.jpg')