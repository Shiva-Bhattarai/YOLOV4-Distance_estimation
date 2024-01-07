import cv2 as cv

# Setting confidence and non-maximum suppression thresholds
CONFIDENCE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.5

# Colors for bounding boxes
COLORS = [(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]
PINK = (147, 20, 255)

# Font style for displaying text
fonts = cv.FONT_HERSHEY_COMPLEX

# Reading object class names from a file
class_names = []
with open("classes.txt", "r") as f:
    class_names = [cname.strip() for cname in f.readlines()]

# Setting up YOLO (You Only Look Once) model for object detection
yoloNet = cv.dnn.readNet('yolov4-tiny.weights', 'yolov4-tiny.cfg')
yoloNet.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
yoloNet.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA_FP16)

model = cv.dnn_DetectionModel(yoloNet)
model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)

# Function for detecting and displaying objects in an image
def ObjectDetector(image):
    # Detect objects in the image
    classes, scores, boxes = model.detect(image, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)

    # Process each detected object
    for (classid, score, box) in zip(classes, scores, boxes):
        # Choose a color for the bounding box
        color = COLORS[int(classid) % len(COLORS)]
        
        # Create a label with the object class name and confidence score
        label = "%s : %f" % (class_names[int(classid)], score)

        # Draw bounding box and display label
        cv.rectangle(image, box, color, 2)
        cv.putText(image, label, (box[0], box[1]-10), fonts, 0.5, color, 2)

# Setting up the camera
camera = cv.VideoCapture(0)
counter = 0
capture = False
number = 0

while True:
    # Capture a frame from the camera
    ret, frame = camera.read()
    original = frame.copy()

    # Perform object detection on the frame
    ObjectDetector(frame)

    # Display the original frame
    cv.imshow('original', original)

    # Display capturing message for a limited time after 'c' key press
    if capture and counter < 10:
        counter += 1
        cv.putText(frame, f"Capturing Img No: {number}", (30, 30), fonts, 0.6, PINK, 2)
    else:
        counter = 0

    # Display the frame with detected objects
    cv.imshow('frame', frame)

    # Check for key presses
    key = cv.waitKey(1)

    # Start capturing reference images when 'c' key is pressed
    if key == ord('c'):
        capture = True
        number += 1
        cv.imwrite(f'ReferenceImages/image{number}.jpg', original)
    # Quit the program when 'q' key is pressed
    elif key == ord('q'):
        break

# Release the camera and close all windows
cv.destroyAllWindows()
