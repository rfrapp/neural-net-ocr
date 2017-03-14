import os
import scipy.misc
import scipy.io
import numpy
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.mlab as mlab

def display_image(imgdata):
	fig = plt.figure()

	ax0 = fig.add_subplot(221)
	ax0.axis('off')
	ax0.imshow(imgdata, cmap='gray')

	plt.tight_layout()
	plt.show(block=False)
	s = input("Press Enter to close...")
	plt.close()

	return s

numpy.set_printoptions(threshold=numpy.inf)

def read_image_grayscale(filename):
	return scipy.misc.imread(name=filename, flatten=True)

DATA_PATH = "/Users/ryan/Downloads/by_class"
LETTERS = [i for i in range(ord('0'), ord('9') + 1)] + \
          [i for i in range(ord('a'), ord('z') + 1)] + \
          [i for i in range(ord('A'), ord('Z') + 1)]

COL_SIZE = 36
IMG_LIMIT = 100
REAL_IMG_SIZE = 128
c = 0

if __name__ == '__main__':
	for i, letter_ascii in enumerate(LETTERS):
		d = DATA_PATH + "/{0:x}/train_{0:x}".format(letter_ascii)
		imgs = [name for name in os.listdir(d)]
		print("Processing images for '%s'..." % chr(letter_ascii))
		inputs = []
		num_imgs = 0

		for img in imgs:

			if num_imgs == IMG_LIMIT:
				break

			img_path = d + "/{0}".format(img)
			img_arr = read_image_grayscale(img_path)
			# print("img_arr:", img_arr)
			img_arr = scipy.misc.imresize(img_arr, (COL_SIZE, COL_SIZE))
			img_arr = img_arr / 255.
			img_arr = 1 - img_arr

			# if c == 0:
			# 	display_image(img_arr)
			# 	c += 1
			inputs.append([j for j in img_arr.flat])
			num_imgs += 1


		print("num images:", num_imgs)
		input_mat = numpy.matrix(inputs).reshape((num_imgs, COL_SIZE ** 2))

		data_dict = {'X': input_mat}
		print("Saving training data...")
		scipy.io.savemat("data/trainingdata_%d.mat" % letter_ascii, data_dict,
			             do_compression=True)
		print("Finished saving data.")
