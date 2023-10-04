# ----- IMPORT LIBRARIES ----- #
# Built-ins
import time
import warnings

# Image handling and detection
import cv2
from ultralytics import YOLO
import supervision as sv

# External required modules
from grabscreen import grab_screen


warnings.filterwarnings('ignore')

# ----- GLOBAL VARIABLES ----- #
current_class = ""
current_class_id = -1
window_name = "yolov8"

# Create a single window
cv2.namedWindow(window_name)


def set_detections(model, detections, _id):
    global current_class, current_class_id

    current_class = model.model.names[_id]
    # print(current_class)
    current_class_id = _id

    return detections[detections.class_id == _id]


# ----- MAIN SCRIPT ----- #
def main():
    global current_class, current_class_id  # Use the global variable

    # Add extra time for app switching
    print("Starting app...")
    for i in range(5, 0, -1):
        print(f"{i}...")
        time.sleep(1)

    # Define the model
    model = YOLO("yolov8l.pt").cuda()

    box_annotator = sv.BoxAnnotator(
        thickness=1,
        text_thickness=1,
        text_scale=0.5
    )

    # Main program loop
    while True:
        # Get the desktop screen of size 800x630
        screen = grab_screen(region=(0, 40, 800, 630))

        # Convert from BGR to RGB for correct coloring
        screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)

        # Detection from model
        result = model(screen, agnostic_nms=True, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        if current_class_id != -1:
            detections = set_detections(model, detections, current_class_id)

        # Update the current class based on the key pressed
        k = cv2.waitKey(10)
        if k in range(48, 58):
            detections = set_detections(model, detections, k - 48)
            print(current_class)
        elif k in range(65, 91):
            detections = set_detections(model, detections, k - 55)
            print(current_class)
        elif k in range(97, 123):
            detections = set_detections(model, detections, k - 60)
            print(current_class)

        labels = [
            f"{model.model.names[class_id]} {confidence:0.2f}"
            for confidence, class_id in zip(detections.confidence.tolist(), detections.class_id.tolist())
        ]

        frame = box_annotator.annotate(
            scene=screen,
            detections=detections,
            labels=labels
        )

        # Display the screen with the current class in the same window
        cv2.imshow(window_name, screen)

        # 'Esc' character to exit and break the loop
        if k == 27:
            cv2.destroyAllWindows()
            break


# Run main()
if __name__ == '__main__':
    main()
