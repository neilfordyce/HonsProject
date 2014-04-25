from cv2 import CascadeClassifier, imread, rectangle, imwrite
from RandPatchSample import batch_open_deserialise, directory_paths
from shapely.geometry import Polygon
from shapely.geometry import box
from time import time
import sys
from GroundTruthMasks import fix_path

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


#Get the file paths to read from
detector_output_files = directory_paths(test_annotation_dir)

#Open all the files and read the JSON
annotation_files = batch_open_deserialise(detector_output_files)

#For each of the hand marked test files
for annotation_file in annotation_files:	
	#Read the image corresponding to the annotation
	img = imread(fix_path(annotation_file['imagePath']))
	print "detecting... "
	classifier_output = cascade.detectMultiScale(img, 
												minNeighbors=3,
												minSize=(175, 175), #Constrain scale within a range around training scale
												maxSize=(225, 225))	#Run the detector
	
	total_detections = len(classifier_output)
	fp = 0
	tp = 0
	
	#Open all the files and read the JSON
	golgi_polygons = annotation_file['shapes']

	for golgi_polygon in golgi_polygons:
		#Make the list of points into a Shapely Polygon object
		golgi_polygon_shape = Polygon(golgi_polygon['points'])
		
		for detected_rect in classifier_output:
			(x1, y1, w, h) = detected_rect
			rectangle(img, (x1, y1),  (x1+w, y1+h), (0, 255, 0), thickness=3)
			
			#Convert the detected rectangle into a Shapely box 
			detected_rect = box(x1, y1, x1+w, y1+h)
			
			#Test if the detected rectangle is contained in the golgi region
			if golgi_polygon_shape.intersects(detected_rect):
				#Check the polygon is labelled as a golgi
				#if  golgi_polygon['label'] == 'golgi':
				tp += 1	
				#else:
				#	total_detections -= 1	#Ignore detections in ambiguous regions		
	
	millis = int(time() * 1000) 
	print "output_%s.jpg" % millis
	imwrite("output_%s.jpg" % millis, img)
	fp = total_detections - tp
	
	print "TP: %s" % tp	
	print "FP: %s" % fp
	
	