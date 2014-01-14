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

'''USAGE: python PolyToRandPatch input_dir output_dir number_of_samples percent_test'''

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

def rounded_bbox(polygon):
	'''Round the bbox to ints'''
	bbox = polygon.bounds
	bbox = map(round, bbox)
	bbox = map(int, bbox)
	return bbox

def polygon2image(polygon, output_size, mode='1'):
	'''Convert a vector polygon into a raster image'''
	copy = Image.new(mode, output_size, 'black')	#Blank canvas
	draw = ImageDraw.Draw(copy)					#Draw the blank canvas
	x, y = polygon.exterior.xy					#Get all the vertex coordinates
	x = map(int, x)								
	y = map(int, y)								
	draw.polygon(zip(x, y), fill='white')		#Draw the polygon
	return copy
	
def subtract_polygon_from_image(polygon, image):
	'''Sets all pixels inside the polygon to 0'''
	polygon = polygon2image(polygon, image.size, 'RGB')
	image = ImageChops.subtract(image, polygon)
	return image

def total_golgi_area(json):
	'''Get the total area of all of the Golgi annotations'''
	total_area = 0

	for file in json:
		polygons = file['shapes']

		for polygon in polygons:
			#Check the polygon is labelled as a golgi
			if polygon['label'] == 'golgi':
				#Make the list of points into a Shapely Polygon object
				polygon = Polygon(polygon['points'])
				total_area += polygon.area
			
	return total_area

def samples_required(total_samples, golgi_area, total_golgi_area):
	'''The samples taken from each Golgi is proportional to its share 
	of the total area of all Golgi'''
	samples = total_samples * (golgi_area / total_golgi_area)
	samples = round(samples)
	return int(samples)
	
def output_patch(image, rotation, crop_box, output_path):
	'''Perform cropping and rotation to get the output image, then save it.  
	Also bump up samples *4 by rotating around right angles and saving them too.'''
	image = image.rotate(rotation, resample=Image.BICUBIC, expand=True)
	image = image.crop(crop_box)
	
	#Store an image at each rotation
	for rotation in [0, 90, 180, 270]:
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
	
def batch_open_deserialise(file_list):
	files = []

	#read in all the files
	for file in file_list:
		file = open(file, 'r')
		file = load(file)	#Deserialise the JSON
		files.append(file)
		
	return files


PATCH_SIZE = 200
i = 0

try:
	'''Get command line args'''
	input_dir = sys.argv[1]
	output_dir = sys.argv[2]
	number_of_samples = int(sys.argv[3])
	percent_test = int(sys.argv[4])
except:
	print "USAGE: python PolyToRandPatch input_dir output_dir number_of_samples percent_test"
	exit()

#Get the file paths to read from
annotation_files = directory_paths(input_dir)

#Reserve portion of the data for testing by removing it and moving the files to output_dir
annotation_files = random_reserve_for_test(annotation_files, percent_test, output_dir)

#Open all the files and read the JSON
files = batch_open_deserialise(annotation_files)

#Find the total area of all golgi
total_area = total_golgi_area(files)

for file in files:
	#open up the electron micrograph image
	image = Image.open(file['imagePath'])
	neg_image = image.copy()	#Store a copy to put everything BUT Golgi
	polygons = file['shapes']

	for polygon in polygons:
		polygon_label = polygon['label']
	
		#Make the list of points into a Shapely Polygon object
		polygon = Polygon(polygon['points'])
		
		#Remove the golgi from the negative examples
		neg_image = subtract_polygon_from_image(polygon, neg_image)
		
		#Check the polygon is labelled as a golgi
		if  polygon_label != 'golgi':
			continue
		i += 1
		
		#Crop to the region of interest for efficiency
		bbox = rounded_bbox(polygon)
		golgi_crop = image.crop(bbox)
		
		#Shift the polygon to be over the golgi in golgi_crop image
		polygon = translate(polygon, xoff=-bbox[0], yoff=-bbox[1])
		
		#Calculate the samples required from this example Golgi
		samples_count = samples_required(number_of_samples, polygon.area, total_area)
		print "Taking %s from %s " % (samples_count, file['imagePath'])
		
		#Rasterise the polygon, so we can rotate it using 
		#the same function used to rotate the output image
		polygon = polygon2image(polygon, golgi_crop.size)		
		
		#Loop for each sample required
		for j in range(samples_count):			
			#loop until a patch fully inside the polygon is found
			output = False
			attempts = 0
			while(not output):
				
				#Stop attempting to get a sample if the Golgi is too small
				attempts += 1
				if attempts > 3000:
					print "Skipping %s: cannot find a valid sample" % file['imagePath']
					break
				
				#apply random rotation
				rotation = randint(0, 359)
				mask = polygon.copy()
				mask = mask.rotate(rotation, resample=Image.BICUBIC, expand=True)
					
				#pick a random box in the image
				x = randint(0, mask.size[0] - PATCH_SIZE)
				y = randint(0, mask.size[1] - PATCH_SIZE)
				candidate_patch_coords = (x, y, x + PATCH_SIZE, y + PATCH_SIZE)
				candidate_patch = box(*candidate_patch_coords)

				#and then cut it out
				crop_mask = mask.crop(candidate_patch_coords)
				
				#then test if the patch contains entirely golgi
				if crop_mask.histogram()[-1] == PATCH_SIZE**2:
					#if yes - output the patch
					output = True
					
					#Construct the file name
					filename, ext = splitext(file['imagePath'])
					filename = basename(filename)
					filename = '%s_%s_%s' % (filename, i, j)
					output_path = join(output_dir, 'pos', filename)
					
					#Output
					output_patch(golgi_crop, rotation, candidate_patch_coords, output_path)
	
	#Output the negative images with all the Golgis removed
	neg_image_path = join(output_dir, "neg", filename + ".jpg")
	neg_image.save(neg_image_path, format='JPEG', quality=100)
		