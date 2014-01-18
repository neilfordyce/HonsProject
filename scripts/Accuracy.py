from cv2 import CascadeClassifier, imread
from PolyToRandPatch import batch_open_deserialise, directory_paths
from shapely.geometry import Polygon
from shapely.geometry import box
import sys

try:
	cascade_xml_path = sys.argv[1]
	test_annotation_dir = sys.argv[2]
except:
	print "USAGE: python Accuracy cascade_xml_path test_annotation_dir"
	exit()

try:
	#Open up the cascade classifier
	cascade = CascadeClassifier(cascade_xml_path)
except:
	print "Error reading cascade xml"
	exit()

try:
	#Get the file paths to read from
	detector_output_files = directory_paths(test_annotation_dir)

	#Open all the files and read the JSON
	annotation_files = batch_open_deserialise([detector_output_files])
except:
	print "Error reading annotation files"
	exit()

#For each of the hand marked test files
for annotation_file in annotation_files:	
	#Read the image corresponding to the annotation
	img = imread(annotation_file['imagePath'])
	classifier_output = cascade.detectMultiScale(img)	#Run the detector
	
	total_detections = len(classifier_output)
	fp = 0
	tp = 0
	
	#Open all the files and read the JSON
	golgi_polygons = annotation_file['shapes']

	for golgi_polygon in golgi_polygons:
		#Make the list of points into a Shapely Polygon object
		golgi_polygon = Polygon(polygon['points'])
		
		for detected_rect in classifier_output:
			#Convert the detected rectangle into a Shapely box 
			detected_rect = box(*detected_rect)
			
			#Test if the detected rectangle is contained in the golgi region
			if golgi_polygon.contains(detected_rect):
				#Check the polygon is labelled as a golgi
				if  golgi_polygon['label'] == 'golgi':
					tp += 1	
				else:
					total_detections -= 1	#Ignore detections in ambiguous regions						
	
	fp = total_detections - tp
	
	print "TP: %s" % tp	
	print "FP: %s" % fp
	
	