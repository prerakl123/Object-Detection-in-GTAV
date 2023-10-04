import time

import cv2

from grabscreen import grab_screen

last_time = time.time()
while True:
    screen = grab_screen(region=(0, 40, 800, 630))
    screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
    cv2.imshow("yolov8", screen)

    print('{} seconds'.format(time.time() - last_time))
    last_time = time.time()

    if cv2.waitKey(30) == 27:
        cv2.destroyAllWindows()
        break
