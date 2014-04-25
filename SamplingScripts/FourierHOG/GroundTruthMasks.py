from RandPatchSample import directory_paths, batch_open_deserialise, polygon2image
import Image
from shapely.geometry import Polygon
from os import listdir, rename
from os.path import isfile, join, splitext, basename
import sys

'''USAGE: python GroundTruthMasks input_dir output_dir'''

def fix_path(filepath):
	'''Sometime the annotation tool messes up the file path, pointing it to itself(i.e. the annotation file) instead of the electron micrograph image file. This should fix that'''
	filename, ext = splitext(filepath)
	filename = basename(filename)
	filepath = join("C:/Users/Neil/SkyDrive/University/HonoursProject/annotated_images/golgi/", filename + ".jpg")	#Force the file ext. to .jpg.  
	return filepath
	
if __name__ == '__main__':
	#Check params are correct	
	try:
		'''Get command line args'''
		input_dir = sys.argv[1]
		output_dir = sys.argv[2]
	except:
		print "USAGE: python GroundTruthMasks input_dir output_dir"
		exit()
		
	#Output fill colours for different types of annotations
	annotation2fill = {	'golgi': 'white',
						'ambiguous': 'blue'} #'white'}

	#Get the file paths to read from
	annotation_files = directory_paths(input_dir)

	#Open all the files and read the JSON
	files = batch_open_deserialise(annotation_files)

	#For all the ground truths
	for file in files:
		polygons = file['shapes']
		file['imagePath'] = fix_path(file['imagePath'])	#Force the file ext. to .jpg.  Sometimes it's stored wrong and needs this correction. 
		print "opening " + file['imagePath']
		image = Image.open(file['imagePath'])
		
		polygon_images = []
		
		#For each ground truth annotation
		for polygon in polygons:	
			polygon_label = polygon['label']
		
			#Make the list of points into a Shapely Polygon object
			polygon = Polygon(polygon['points'])
			
			#Convert the polygon into an image
			polygon_image = polygon2image(polygon, 
											image.size, 
											mode='RGB', 
											polygon_fill=annotation2fill[polygon_label])
			polygon_images.append(polygon_image)
		
		#Blend all of the polygon images together
		output_image = polygon_images[0]

		for polygon_image in polygon_images[1:]:
			#Create a mask so we only change the part of the image being added
			mtx = (1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0)
			mask = polygon_image.convert("L", mtx)

			output_image = Image.composite(polygon_image,
											output_image, 
											mask)	

		#Construct the file name
		filename, ext = splitext(file['imagePath'])
		filename = basename(filename)
		output_path = join(output_dir, filename + ext)
		
		#Output
		print output_path
		output_image.save(output_path, format='JPEG', quality=100)
