import cv2 as cv
import numpy as np

# Distance constants
KNOWN_DISTANCE = 29  # INCHES
PERSON_WIDTH = 11 # INCHES
MOBILE_WIDTH = 2  # INCHES
BOTTLE_WIDTH = 1.8 # INCHES

# Object detector constant
CONFIDENCE_THRESHOLD = 0.4
NMS_THRESHOLD = 0.3

# Colors for object detected
COLORS = [(255, 0, 0), (255, 0, 255), (0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]
GREEN = (0, 255, 0)

# Defining fonts
FONTS = cv.FONT_HERSHEY_COMPLEX

# Getting class names from classes.txt file
class_names = []
with open("classes.txt", "r") as f:
    class_names = [cname.strip() for cname in f.readlines()]

# Setting up OpenCV net
yoloNet = cv.dnn.readNet('yolov4-tiny.weights', 'yolov4-tiny.cfg')

yoloNet.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
yoloNet.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)

model = cv.dnn_DetectionModel(yoloNet)
model.setInputParams(size=(416, 416), scale=1 / 255, swapRB=True)

# Object detector function/method
def object_detector(image):
    classes, _, boxes = model.detect(image, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    # Creating an empty list to add objects data
    data_list = []
    for (classid, _, box) in zip(classes, _, boxes):
        # Define color of each object based on its class id
        color = COLORS[int(classid) % len(COLORS)]

        label = "%s" % class_names[int(classid)]

        cv.putText(image, label, (box[0], box[1] - 14), FONTS, 1, color, 2)

        # Getting the data
        # 1: class name, 2: object width in pixels, 3: position where to draw text (distance)
        if label in ["person", "cell phone", "bottle"]:
            data_list.append([label, box[2], (box[0], box[1] - 2)])
        # If you want to include more classes, add more 'elif' statements here
        # Returning list containing the object data
    return data_list

# Focal length finder function
def focal_length_finder(measured_distance, real_width, width_in_rf):
    focal_length = (width_in_rf * measured_distance) / real_width
    return focal_length

# Distance finder function
def distance_finder(focal_length, real_object_width, width_in_frame):
    distance = (KNOWN_DISTANCE * real_object_width) / width_in_frame
    return distance

# Reading the reference image for each class
ref_person = cv.imread('ReferenceImages/image14.jpg')
ref_mobile = cv.imread('ReferenceImages/image4.jpg')
ref_bottle = cv.imread('ReferenceImages/image1.jpg')

# Detecting objects in the reference images
person_data = object_detector(ref_person)
mobile_data = object_detector(ref_mobile)
bottle_data = object_detector(ref_bottle)

# Extracting object width in reference frames
person_width_in_rf = person_data[0][1]
mobile_width_in_rf = mobile_data[0][1]
bottle_width_in_rf = bottle_data[0][1]

# Calculating focal lengths
focal_person = focal_length_finder(KNOWN_DISTANCE, PERSON_WIDTH, person_width_in_rf)
focal_mobile = focal_length_finder(KNOWN_DISTANCE, MOBILE_WIDTH, mobile_width_in_rf)
focal_bottle = focal_length_finder(KNOWN_DISTANCE, BOTTLE_WIDTH, bottle_width_in_rf)

# OpenCV window properties
cv.namedWindow("frame", cv.WND_PROP_FULLSCREEN)
cv.setWindowProperty("frame", cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)

cap = cv.VideoCapture(0)
while True:
    ret, frame = cap.read()

    data = object_detector(frame)
    for d in data:
        if d[0] == 'person':
            distance = distance_finder(focal_person, PERSON_WIDTH, d[1])
            x, y = d[2]
        elif d[0] == 'cell phone':
            distance = distance_finder(focal_mobile, MOBILE_WIDTH, d[1])
            x, y = d[2]
        elif d[0] == 'bottle':
            distance = distance_finder(focal_bottle, BOTTLE_WIDTH, d[1])
            x, y = d[2]
        cv.putText(frame, f' {round(distance, 2)} meters', (x + 5, y + 13), FONTS, 0.8, GREEN, 2)

    cv.imshow('frame', frame)

    key = cv.waitKey(1)
    if key == ord('q'):
        break

cv.destroyAllWindows()
cap.release()
