import argparse
import time
import uuid
from datetime import datetime

import boto3
import dlib
import zmq
from PIL import Image
from imutils import face_utils
import json

from facepose import Facepose
from utils import *
# from live_plotter import data_writter_initialze, data_writter_write
from utils import eye_aspect_ratio, rec_to_roi_box
from zeromq.SerializingContext import SerializingContext

# face detection and landmark
p = "../model/shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)

# aws
awsRegion = "ap-southeast-1"
inputStream = "kinesis-attention-stream"
kinesis = boto3.client('kinesis', region_name=awsRegion)

# zmq
context = SerializingContext()
socket = context.socket(zmq.PUB)

# socket.connect("tcp://10.10.10.163:5555")
socket.connect("tcp://localhost:5555")

id=123

# def send_record():
#     # threading.Timer(5.0, send_record).start()
#     # if len(records) != 0:
#     put_response = kinesis.put_records(StreamName=inputStream, Records=records)
#
#     print("sending record...")
#     print(put_response)

def main(userid, host):
    socket.connect("tcp://" + host + ":5555")

    cap = cv2.VideoCapture(0)
    # testvideo=r"C:\Users\choowilson\Desktop\data_center\attention-monitor\jingzhi.mp4"
    # cap = cv2.VideoCapture(testvideo)

    blinkCount = 0
    yawnCount = 0

    lostFocusCount = 0
    lostFocusDuration = 0
    focusTimer = None

    faceNotPresentDuration = 0
    faceTimer = None

    yawning = False
    eyeClosed = False
    lostFocus = False

    record = None
    records = []

    # control FPS
    frame_rate_use = 5
    prev = 0

    facepose = Facepose()

    shape = None
    yaw_predicted, pitch_predicted, roll_predicted=None, None, None


    while cap.isOpened():
        # fps_count_start_time = time.time()
        ret, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.flip(frame, 1)
        frame_display = frame.copy()
        gray = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2GRAY)

        # control FPS
        time_elapsed = time.time() - prev
        if time_elapsed > 1. / frame_rate_use:
            prev = time.time()

            rects = detector(gray, 0)

            if len(rects) == 0:
                if faceTimer == None:
                    faceTimer = time.time()
                faceNotPresentDuration += time.time() - faceTimer;
                faceTimer = time.time();

            for (i, rect) in enumerate(rects):
                faceTimer = None

                shape = predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)

                leftEye = shape[36:42]
                # print(f"leftEye{leftEye}")
                rightEye = shape[42:48]
                # print(f"righti{rightEye}")
                leftEAR = eye_aspect_ratio(leftEye)
                rightEAR = eye_aspect_ratio(rightEye)
                # average the eye aspect ratio together for both eyes
                ear = (leftEAR + rightEAR) / 2.0
                # print(ear)
                mar = mouth_aspect_ratio(shape[60:69])
                # print(mar)

                if ear < 0.15:
                    eyeClosed = True
                if ear > 0.15 and eyeClosed:
                    blinkCount += 1
                    eyeClosed = False

                if mar > 0.4:
                    # cv2.putText(frame_display, "Yawning! ", (10, 90), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.0,
                    #             color=(255, 0, 0), thickness=2)
                    yawning = True
                if mar < 0.2 and yawning:
                    yawnCount += 1
                    yawning = False

                # Draw circle to show landmarks
                # for idx, (x, y) in enumerate(shape):
                #     if idx in range(36, 48):
                #         cv2.circle(frame_display, (x, y), 2, (0, 255, 0), -1)
                #     elif idx in range(60, 68):
                #         cv2.circle(frame_display, (x, y), 2, (0, 0, 255), -1)
                # Uncomment if you want to visualize all other landmarks
                # else:
                #     cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)

                roi_box, center_x, center_y = rec_to_roi_box(rect)

                roi_img = crop_img(frame, roi_box)

                img = Image.fromarray(roi_img)

                (yaw_predicted, pitch_predicted, roll_predicted) = facepose.predict(img)

                # print(yaw_predicted.item(), pitch_predicted.item(), roll_predicted.item())
                if yaw_predicted.item() < -30 or yaw_predicted.item() > 30:
                    lostFocus = True
                    if focusTimer == None:
                        focusTimer = time.time()

                    lostFocusDuration += time.time() - focusTimer;
                    focusTimer = time.time();
                if (yaw_predicted.item() > -30 and yaw_predicted.item() < 30) and lostFocus:
                    lostFocusCount += 1
                    lostFocus = False
                    focusTimer = None


                # prepare to put records in kinesis
                ###################################################################################################
                record = {
                    'id': str(userid),
                    'sortKey': str(uuid.uuid1()),
                    'timestamp': datetime.now().timestamp(),
                    'yaw': yaw_predicted.item(),
                    'pitch': pitch_predicted.item(),
                    'roll': roll_predicted.item(),
                    'ear': ear,
                    'blink_count': blinkCount,
                    'mar': mar,
                    'yawn_count': yawnCount,
                    'lost_focus_count': lostFocusCount,
                    'lost_focus_duration': lostFocusDuration,
                    'face_not_present_duration': faceNotPresentDuration
                }
                print(record)
                records.append({'Data': bytes(json.dumps(record), 'utf-8'), 'PartitionKey': str(id)})

                if len(records) >= 10:
                    put_response = kinesis.put_records(StreamName=inputStream, Records=records)
                    time.sleep(0.5)
                    print("sending record...")
                    print(put_response)
                    records = []
                ###################################################################################################

            data = {
                'id': str(userid),
                'record': record
            }
            frame_stream = cv2.resize(frame.copy(), (0, 0), fx=0.5, fy=0.5)
            publish(frame_stream, data)
            record = None

        cv2.putText(frame_display, "Blink Count: " + str(blinkCount), (10, 20), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.5, color=(255, 0, 0), thickness=1)
        cv2.putText(frame_display, "Yawn Count: " + str(yawnCount), (10, 40), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.5, color=(255, 0, 0), thickness=1)

        cv2.putText(frame_display, "Lost Focus Count: " + str(lostFocusCount), (10, 70),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.5, color=(255, 0, 0), thickness=1)
        cv2.putText(frame_display, "Lost Focus Duration: " + str(round(lostFocusDuration)), (10, 90),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.5, color=(255, 0, 0), thickness=1)

        cv2.putText(frame_display, "Face Not Present Duration: " + str(round(faceNotPresentDuration)), (10, 110),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.5, color=(255, 0, 0), thickness=1)

        if shape is not None and len(list(rects)) is not 0:
            rect = list(rects)[0]
            draw_border(frame_display,  (rect.left(),rect.top()), (rect.left()+rect.width(), rect.top()+rect.height()), (255,255,255), 1, 10, 20)

            draw_axis(frame_display, yaw_predicted.item(), pitch_predicted.item(), roll_predicted.item(),
                      tdx=int(center_x), tdy=int(center_y), size=100)
            for idx, (x, y) in enumerate(shape):
                cv2.circle(frame_display, (x, y), 2, (255, 255, 0), -1)
                if idx in range(36, 48):
                    cv2.circle(frame_display, (x, y), 2, (255, 0, 255), -1)
                elif idx in range(60, 68):
                    cv2.circle(frame_display, (x, y), 2, (0, 255, 255), -1)

        cv2.imshow('frame', cv2.cvtColor(frame_display, cv2.COLOR_RGB2BGR))

        # print("FPS: ", 1.0 / (time.time() - fps_count_start_time))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def publish(image, data):
    if image.flags['C_CONTIGUOUS']:
        # if image is already contiguous in memory just send it
        socket.send_array(image, data, copy=False)
    else:
        # else make it contiguous before sending
        image = np.ascontiguousarray(image)
        socket.send_array(image, data, copy=False)

def draw_border(img, pt1, pt2, color, thickness, r, d):
    x1,y1 = pt1
    x2,y2 = pt2

    # Top left
    cv2.line(img, (x1 + r, y1), (x1 + r + d, y1), color, thickness)
    cv2.line(img, (x1, y1 + r), (x1, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)

    # Top right
    cv2.line(img, (x2 - r, y1), (x2 - r - d, y1), color, thickness)
    cv2.line(img, (x2, y1 + r), (x2, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)

    # Bottom left
    cv2.line(img, (x1 + r, y2), (x1 + r + d, y2), color, thickness)
    cv2.line(img, (x1, y2 - r), (x1, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)

    # Bottom right
    cv2.line(img, (x2 - r, y2), (x2 - r - d, y2), color, thickness)
    cv2.line(img, (x2, y2 - r), (x2, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='This is a attention monitor program')
    parser.add_argument('--userid', help='user id', required=True)
    parser.add_argument('--host', help='host ip', default="localhost")

    args = parser.parse_args()
    userid = args.userid
    host = args.host

    main(userid, host)
    # single_image_test()
