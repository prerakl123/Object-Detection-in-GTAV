# ----- IMPORT LIBRARIES ----- #
# Built-ins
import time
import warnings

# Image handling and detection
import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv

# External required modules
from grabscreen import grab_screen
from keys import NUM_ALL, CHAR_CAPS_ALL, CHAR_LOWERCASE_ALL

warnings.filterwarnings('ignore')

# ----- GLOBAL VARIABLES ----- #
current_class = ""
current_class_id = -1
window_name = "yolov8"

# Create a single window
cv2.namedWindow(window_name)


def show_incremental_keybind_info():
    # Display
    img = np.zeros((480, 640, 3), np.uint8)
    put_text(img, org=(155, 40), text="To cycle through objects:")
    put_text(img, org=(201, 80), text="Space | Cycle Next")
    put_text(img, org=(143, 120), text="Backspace | Cycle Previous")
    put_text(img, org=(262, 160), text="A | Show All")
    put_text(img, org=(233, 200), text="Esc | Close")

    # Display the image
    cv2.imshow("Info", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def put_text(
        image, text,
        org=(10, 20),
        font=cv2.FONT_HERSHEY_SIMPLEX,
        fontscale=0.8,
        color=(255, 255, 255),
        thickness=2, **kwargs
):
    cv2.putText(image, text, org, font, fontscale, color, thickness, **kwargs)


def set_detections(model, detections, _id):
    global current_class, current_class_id

    current_class = model.model.names[_id]
    # print(current_class)
    current_class_id = _id

    return detections[detections.class_id == _id]


def incremental_object_detection(k, model, detections):
    global current_class, current_class_id

    if k == 32:  # For Space key
        if current_class_id == 79:
            current_class_id = -1
            current_class = ''
        else:
            return set_detections(model, detections, current_class_id + 1)
    elif k == 8:  # For Backspace key
        if current_class_id == -1:
            return set_detections(model, detections, 79)
        else:
            return set_detections(model, detections, current_class_id - 1)
    elif k == 97:
        current_class_id = -1
        current_class = ''
    return detections


def key_based_object_detection(k, model, detections):
    if k in NUM_ALL:
        return set_detections(model, detections, k - 48)
    elif k in CHAR_CAPS_ALL:
        return set_detections(model, detections, k - 55)
    elif k in CHAR_LOWERCASE_ALL:
        return set_detections(model, detections, k - 60)
    return detections


# ----- MAIN SCRIPT ----- #
def main(class_cycle_method=incremental_object_detection):
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
        detections = class_cycle_method(k, model, detections)

        labels = [
            f"{model.model.names[class_id]} {confidence:0.2f}"
            for confidence, class_id in zip(detections.confidence.tolist(), detections.class_id.tolist())
        ]

        frame = box_annotator.annotate(
            scene=screen,
            detections=detections,
            labels=labels
        )
        put_text(screen, current_class, (15, 25), fontscale=1, color=(0, 0, 0))

        # Display the screen with the current class in the same window
        cv2.imshow(window_name, screen)

        # 'Esc' character to exit and break the loop
        if k == 27:
            cv2.destroyAllWindows()
            break


# Run main()
if __name__ == '__main__':
    show_incremental_keybind_info()
    main()
