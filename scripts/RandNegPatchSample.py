import Image
import ImageDraw
import ImageChops
import sys
from json import load
from random import randint
from random import sample
from shapely.geometry import Polygon
from shapely.geometry import Point
from shapely.geometry import box
from shapely.affinity import rotate
from shapely.affinity import translate
from os import listdir, rename
from os.path import isfile, join, splitext, basename

'''USAGE: python RandPatchSample input_dir output_dir number_of_samples percent_test'''

def random_reserve_for_test(filenames, percent, output_dir):
	'''Set aside a portion of the dataset for test.  
	Save the label file in a different directory and remove from the list'''
	#Pick a random sample population
	sample_size = len(filenames) * (percent * 0.01)
	sample_size = round(sample_size)
	sample_size = int(sample_size)
	test_filenames = sample(filenames, sample_size)
	
	#move the annotation files to the test output directory
	for filename in test_filenames:
		output_filename = basename(filename)	#Remove the path, leave the name
		print "Reserving %s for testing" % output_filename
		output_filename  = join(output_dir, 'test', output_filename)
		rename(filename, output_filename)
	
	#remove the reserved sample from the file names
	filenames = [file for file in filenames if file not in test_filenames]
	return filenames
	
def output_patch(image, rotation, crop_box, output_path):
	'''Perform cropping and rotation to get the output image, then save it.  
	Also bump up samples *4 by rotating around right angles and saving them too.'''
	image = image.rotate(rotation, resample=Image.BICUBIC, expand=True)
	image = image.crop(crop_box)
	
	#Store an image at each rotation
	for rotation in [0]:
		output_image = image.copy()
		output_image = output_image.rotate(rotation, expand=False)
		filename = "%s_%s.jpg" % (output_path, rotation)
		print filename
		output_image.save(filename, format='JPEG', quality=100)
		
def directory_paths(input_dir):
	'''Returns the absolute path to all the files in a directory'''
	#Get all the files in dir
	files = [ file for file in listdir(input_dir) if isfile(join(input_dir, file)) ]
	files = map(lambda x: input_dir + x, files)  #Add the directory to the path
	return files

PATCH_SIZE = 200
i = 0

try:
	'''Get command line args'''
	input_dir = sys.argv[1]
	output_dir = sys.argv[2]
	samples_count = int(sys.argv[3])
except:
	print "USAGE: python RandNegPatchSample input_dir output_dir number_of_samples"
	exit()

#Get the file paths to read from
files = directory_paths(input_dir)

for file in files:
	#open up the electron micrograph image
	image = Image.open(file)

	i += 1
	
	#Loop for each sample required
	for j in range(samples_count):							
		#pick a random box in the image
		x = randint(0, image.size[0] - PATCH_SIZE)
		y = randint(0, image.size[1] - PATCH_SIZE)
		candidate_patch_coords = (x, y, x + PATCH_SIZE, y + PATCH_SIZE)
		candidate_patch = box(*candidate_patch_coords)

		#Construct the file name
		filename, ext = splitext(file)
		filename = basename(filename)
		filename = '%s_%s_%s' % (filename, i, j)
		output_path = join(output_dir, filename)
		
		#Output
		output_patch(image, 0, candidate_patch_coords, output_path)
	
		