import cv2 as cv
import numpy as np

# Distance constants
KNOWN_DISTANCE = 34  # INCHES
PERSON_WIDTH = 16  # INCHES
MOBILE_WIDTH = 2.8  # INCHES

# Object detector constant
CONFIDENCE_THRESHOLD = 0.4
NMS_THRESHOLD = 0.3

# Colors for object detected
COLORS = [(255, 0, 0), (255, 0, 255), (0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]
BRIGHT_BLUE = (255, 191, 0)  # Bright blue color
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
yoloNet.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA_FP16)

model = cv.dnn_DetectionModel(yoloNet)
model.setInputParams(size=(416, 416), scale=1 / 255, swapRB=True)

# Object detector function/method
def object_detector(image):
    classes, _, boxes = model.detect(image, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    # Creating empty list to add objects data
    data_list = []
    for (classid, _, box) in zip(classes, _, boxes):
        # Define color of each object based on its class id
        color = COLORS[int(classid) % len(COLORS)]

        # Draw rectangle on object
        cv.rectangle(image, box, color, 2)

        # Getting the data
        # 1: class name, 2: object width in pixels, 3: position where to draw text (distance)
        if int(classid) == 0:  # Person class id
            data_list.append([class_names[int(classid)], box[2], (box[0], box[1] - 2)])
        elif int(classid) == 67:  # Mobile phone class id
            data_list.append([class_names[int(classid)], box[2], (box[0], box[1] - 2)])
        # If you want to include more classes, add more 'elif' statements here
        # Returning list containing the object data
    return data_list

def focal_length_finder(measured_distance, real_width, width_in_rf):
    focal_length = (width_in_rf * measured_distance) / real_width
    return focal_length

# Distance finder function
def distance_finder(focal_length, real_object_width, width_in_frame):
    distance = (KNOWN_DISTANCE * real_object_width) / width_in_frame
    return distance

# Reading the reference image from dir
ref_person = cv.imread('ReferenceImages/image14.jpg')
ref_mobile = cv.imread('ReferenceImages/image4.jpg')

mobile_data = object_detector(ref_mobile)
mobile_width_in_rf = mobile_data[0][1] 
person_data = object_detector(ref_person)
person_width_in_rf = person_data[0][1]  

print(f"Person width in pixels: {person_width_in_rf}, mobile width in pixels: {mobile_width_in_rf}")

# Finding focal length
focal_person = focal_length_finder(KNOWN_DISTANCE, PERSON_WIDTH, person_width_in_rf)
focal_mobile = focal_length_finder(KNOWN_DISTANCE, MOBILE_WIDTH, mobile_width_in_rf)

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
        cv.putText(frame, f'{d[0]}', (x + 5, y - 30), FONTS, 1, BRIGHT_BLUE, 2)
        cv.putText(frame, f'{round(distance, 2)} meters', (x + 5, y - 10), FONTS, 0.7, BRIGHT_BLUE, 2)

    cv.imshow('frame', frame)

    key = cv.waitKey(1)
    if key == ord('q'):
        break

cv.destroyAllWindows()
cap.release()
