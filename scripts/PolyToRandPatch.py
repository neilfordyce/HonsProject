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
	
def polygon_overlay(image, polygon, output_path):
	overlay = image.copy()
	draw = ImageDraw.Draw(overlay)
	x, y = polygon.exterior.xy
	x = map(int, x)
	y = map(int, y)
	draw.polygon(zip(x, y), fill='wheat')
	blend_image = Image.blend(overlay, image, 0.5)
	blend_image.save(output_path, quality=100)
	
def polygon2image(polygon, output_size):
	copy = Image.new('1', output_size, 'black')
	draw = ImageDraw.Draw(copy)
	x, y = polygon.exterior.xy
	x = map(int, x)
	y = map(int, y)
	draw.polygon(zip(x, y), fill='white')
	return copy

PATCH_SIZE = 200

input_filename = sys.argv[1]
number_of_samples = int(sys.argv[2])

#read in all the points from lif file
file = open(input_filename, 'r')
file = load(file)

image = Image.open(file['imagePath'])
polygons = file['shapes']
i = 0

#Make the list of points into a Shapely Polygon object
for polygon in polygons:
	polygon_e = Polygon(polygon['points'])
	i += 1
	
	bbox = rounded_bbox(polygon_e)
	golgi_crop = image.crop(bbox)
	
	#Shift the polygon to be over the golgi in golgi_crop image
	polygon_e = translate(polygon_e, xoff=-bbox[0], yoff=-bbox[1])
	
	#Loop for each sample required
	for j in range(number_of_samples):
		output_path = '%s_%s_%s.jpg' % (file['imagePath'], i, j)
		
		#loop until a patch fully inside the polygon is found
		output = False;
		while(not output):
			#apply random rotation
			rotation = randint(0, 359)
			mask = polygon2image(polygon_e, golgi_crop.size)
			mask = mask.rotate(rotation, resample=Image.BICUBIC, expand=True)
			
			#image_center_point = map(lambda x: round(x/2), image.size)
			#image_center_point = Point(image_center_point)
			#polygon = rotate(polygon_e, rotation, origin='center')
			#polygon = polygon_e
			
			#Get polygon bounding box
			#bbox = rounded_bbox(polygon)
		
			#pick a point at random in the polygon's bbox
			mask.size[0]
			#x_range = (bbox[0], bbox[2])
			#y_range = (bbox[1], bbox[3])
			#x = randint(*x_range)
			#y = randint(*y_range)
			
			x = randint(0, mask.size[0])
			y = randint(0, mask.size[1])
			
			
			point = Point(x, y)
			
			#and test if it is contained in the polygon
			#if polygon.contains(point):
			#if yes - plot a box around the point
			candidate_patch_coords = (x, y, x + PATCH_SIZE, y + PATCH_SIZE)
			candidate_patch = box(*candidate_patch_coords)
			#candidate_patch = rotate(candidate_patch, rotation, origin=image_center_point)
			
			crop_mask = mask.crop(candidate_patch_coords)
			
			#then test if the patch is contained in the polygon
			#print crop_mask.histogram()
			#if polygon.contains(candidate_patch):
			if crop_mask.histogram()[-1] == PATCH_SIZE**2:
			
				#if yes - output the patch
				print "center at %s" % point
				print "created %s" % output_path
				output = True
				output_patch = golgi_crop.copy()
				output_patch = output_patch.rotate(rotation, resample=Image.BICUBIC, expand=True)
				
				#polygon_overlay(output_patch, polygon, ".\output.bmp")
				
				#output_patch.save("%s_00_%s_%s" % (output_path, rotation, ".jpg"), format='JPEG', quality=100)
				output_patch = output_patch.crop(candidate_patch_coords)
				output_patch.save(output_path, format='JPEG', quality=100)
