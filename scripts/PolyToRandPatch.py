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

#Round the bbox to ints
def rounded_bbox(polygon):
	bbox = polygon.bounds
	bbox = map(round, bbox)
	bbox = map(int, bbox)
	return bbox

#Convert a vector polygon into a raster image
def polygon2image(polygon, output_size):
	copy = Image.new('1', output_size, 'black')	#Blank canvas
	draw = ImageDraw.Draw(copy)					#Draw the blank canvas
	x, y = polygon.exterior.xy					#Get all the vertex coordinates
	x = map(int, x)								
	y = map(int, y)								
	draw.polygon(zip(x, y), fill='white')		#Draw the polygon
	return copy

PATCH_SIZE = 200
i = 0

input_filename = sys.argv[1]
number_of_samples = int(sys.argv[2])

#read in all the points from lif annotation file
file = open(input_filename, 'r')
file = load(file)	#Deserialise the JSON

#open up the electron micrograph image
image = Image.open(file['imagePath'])
polygons = file['shapes']

#Make the list of points into a Shapely Polygon object
for polygon in polygons:
	polygon = Polygon(polygon['points'])
	i += 1
	
	#Crop to the region of interest for efficiency
	bbox = rounded_bbox(polygon)
	golgi_crop = image.crop(bbox)
	
	#Shift the polygon to be over the golgi in golgi_crop image
	polygon = translate(polygon, xoff=-bbox[0], yoff=-bbox[1])
	
	#Loop for each sample required
	for j in range(number_of_samples):
		output_path = '%s_%s_%s.jpg' % (file['imagePath'], i, j)
		
		#loop until a patch fully inside the polygon is found
		output = False;
		while(not output):
			#apply random rotation
			rotation = randint(0, 359)
			mask = polygon2image(polygon, golgi_crop.size)
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
				output_patch = golgi_crop.copy()
				output_patch = output_patch.rotate(rotation, resample=Image.BICUBIC, expand=True)
				output_patch = output_patch.crop(candidate_patch_coords)
				output_patch.save(output_path, format='JPEG', quality=100)
