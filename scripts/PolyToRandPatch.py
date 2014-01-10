from json import load
from random import randint
from shapely.geometry import Polygon
from shapely.geometry import Point
from shapely.geometry import box
from shapely.affinity import rotate
from shapely.affinity import translate
import Image
import ImageDraw
import sys
from os import listdir
from os.path import isfile, join, splitext, basename


'''USAGE: python PolyToRandPatch input_dir output_dir number_of_samples'''

def rounded_bbox(polygon):
	'''Round the bbox to ints'''
	bbox = polygon.bounds
	bbox = map(round, bbox)
	bbox = map(int, bbox)
	return bbox

def polygon2image(polygon, output_size):
	'''Convert a vector polygon into a raster image'''
	copy = Image.new('1', output_size, 'black')	#Blank canvas
	draw = ImageDraw.Draw(copy)					#Draw the blank canvas
	x, y = polygon.exterior.xy					#Get all the vertex coordinates
	x = map(int, x)								
	y = map(int, y)								
	draw.polygon(zip(x, y), fill='white')		#Draw the polygon
	return copy

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

PATCH_SIZE = 200
i = 0

input_dir = sys.argv[1]
output_dir = sys.argv[2]
number_of_samples = int(sys.argv[3])

#Get all the files in dir
annotation_files = [ file for file in listdir(input_dir) if isfile(join(input_dir, file)) ]
annotation_files = map(lambda x: input_dir + x, annotation_files)  #Add the directory to the path

files = []

#read in all the lif annotation files
for file in annotation_files:
	file = open(file, 'r')
	file = load(file)	#Deserialise the JSON
	files.append(file)
	
total_area = total_golgi_area(files)

for file in files:
	#open up the electron micrograph image
	image = Image.open(file['imagePath'])
	polygons = file['shapes']

	for polygon in polygons:
		#Check the polygon is labelled as a golgi
		if polygon['label'] != 'golgi':
			continue
		
		#Make the list of points into a Shapely Polygon object
		polygon = Polygon(polygon['points'])
		i += 1
		
		#Crop to the region of interest for efficiency
		bbox = rounded_bbox(polygon)
		golgi_crop = image.crop(bbox)
		
		#Shift the polygon to be over the golgi in golgi_crop image
		polygon = translate(polygon, xoff=-bbox[0], yoff=-bbox[1])
		
		#Calculate the samples required from this example Golgi
		samples_count = samples_required(number_of_samples, polygon.area, total_area)
		
		#Rasterise the polygon, so we can rotate it using 
		#the same function used to rotate the output image
		polygon = polygon2image(polygon, golgi_crop.size)		
		
		#Loop for each sample required
		for j in range(samples_count):			
			#loop until a patch fully inside the polygon is found
			output = False;
			while(not output):
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
					filename = join(output_dir, filename)
					
					#Output
					output_patch(golgi_crop, rotation, candidate_patch_coords, filename)
					