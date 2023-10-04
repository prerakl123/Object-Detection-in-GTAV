# ----- IMPORT LIBRARIES ----- #
# Built-ins
import time

# Image handling and detection
import cv2
from ultralytics import YOLO
import supervision as sv

# External required modules
from grabscreen import grab_screen


# ----- MAIN SCRIPT ----- #
def main():
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

    # Time calculation for testing model runtime
    # last_time = time.time()

    # Main program loop
    while True:
        # Get the desktop screen of size 800x630
        screen = grab_screen(region=(0, 40, 800, 630))

        # Convert from BGR to RGB for correct colouring
        screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)

        # detection from model
        result = model(screen, agnostic_nms=True, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        # detections = detections[detections.class_id == 0]

        labels = [
            f"{model.model.names[class_id]} {confidence:0.2f}"
            for confidence, class_id in zip(detections.confidence.tolist(), detections.class_id.tolist())
        ]

        frame = box_annotator.annotate(
            scene=screen,
            detections=detections,
            labels=labels
        )

        # Show the screen
        cv2.imshow("yolov8", screen)

        # print('{} seconds'.format(time.time() - last_time))
        # last_time = time.time()

        # 'Esc' character to exit and break the loop
        if cv2.waitKey(30) == 27:
            cv2.destroyAllWindows()
            break


# Run main()
if __name__ == '__main__':
    main()
