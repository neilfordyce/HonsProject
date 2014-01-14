from PolyToRandPatch import batch_open_deserialise, directory_paths
from shapely.geometry import Polygon
from shapely.geometry import box
import sys

try:
	input_dir = sys.argv[1]
except:
	print "USAGE: python Accuracy input_dir"
	exit()
	
#Get the file paths to read from
annotation_files = directory_paths(input_dir)
#Open all the files and read the JSON
annotation_files = batch_open_deserialise(annotation_files)

for file in annotation_files:
	fp = 0
	tp = 0
	golgi_polygons = file['shapes']

	for golgi_polygon in golgi_polygons:
		#Check the polygon is labelled as a golgi
		if  golgi_polygon['label'] != 'golgi':
			continue
	
		#Make the list of points into a Shapely Polygon object
		golgi_polygon = Polygon(polygon['points'])
		
		for detected_patch in classifier_output:
			detected_patch = box(detected_patch)
			
			if golgi_polygon.contains(detected_patch):
				tp += 1
				
	total_detections = len(classifier_output)
	fp = total_detections - tp
	
	print "TP: %s" % tp	
	print "FP: %s" % fp
	
	