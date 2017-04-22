import os
import scipy.misc
import scipy.io
import numpy
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.mlab as mlab
import imageproc.main as imgproc
import random

DATA_PATH = "/Users/ryan/Downloads/by_class"
LETTERS = [i for i in range(ord('0'), ord('9') + 1)] + \
          [i for i in range(ord('a'), ord('z') + 1)] + \
          [i for i in range(ord('A'), ord('Z') + 1)]

COL_SIZE = 36 			# size used when saving the image
TRAINING_LIMIT = 1000	# number of images for the training set
TESTING_LIMIT = 100		# number of images for the testing set


if __name__ == '__main__':
	for i, letter_ascii in enumerate(LETTERS):
		d = DATA_PATH + "/{0:x}/train_{0:x}".format(letter_ascii)
		imgs = [name for name in os.listdir(d)]
		random.shuffle(imgs)
		print("Processing images for '%s'..." % chr(letter_ascii))
		training_inputs = []
		testing_inputs = []
		num_training_imgs = 0
		num_testing_imgs = 0

		for img in imgs:
			img_path = d + "/{0}".format(img)
			img_arr = imgproc.read_image_grayscale(img_path)
			img_arr = imgproc.preprocess_nist_img(img_arr, COL_SIZE, COL_SIZE)

			# imgproc.display_image(img_arr)

			if num_training_imgs < TRAINING_LIMIT:
				training_inputs.append([j for j in img_arr.flat])
				num_training_imgs += 1
			elif num_testing_imgs < TESTING_LIMIT:
				testing_inputs.append([j for j in img_arr.flat])
				num_testing_imgs += 1
			else:
				break

		training_input_mat = numpy.matrix(training_inputs).reshape((num_training_imgs, COL_SIZE ** 2))
		testing_input_mat = numpy.matrix(testing_inputs).reshape((num_testing_imgs, COL_SIZE ** 2))

		data_dict = {'X': training_input_mat}
		print("Saving training data...")
		scipy.io.savemat("data/trainingdata_%d.mat" % letter_ascii, data_dict,
			             do_compression=True)
		print("Saving testing data...")
		data_dict = {'X': testing_input_mat}
		scipy.io.savemat("data/testingdata_%d.mat" % letter_ascii, data_dict,
			             do_compression=True)
		print("Finished saving data.")
