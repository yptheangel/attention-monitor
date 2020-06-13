import cv2
import time
def main():
    cap = cv2.VideoCapture(0)
    frame_rate = 3
    prev = 0

    while (True):

        time_elapsed = time.time() - prev
        res, image = cap.read()

        if time_elapsed > 1. / frame_rate:
            prev = time.time()

            cv2.imshow('frame', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # Do something with your image here.


if __name__ == '__main__':
    main()