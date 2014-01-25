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
from RandPatchSample import directory_paths, output_patch

'''USAGE: python RandPatchSample input_dir output_dir number_of_samples percent_test'''

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
		output_patch(image, 0, candidate_patch_coords, output_path, output_rotations=[0])
	
		