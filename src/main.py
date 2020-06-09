import dlib
import cv2
import torchvision
from imutils import face_utils
import torch
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
import hopenet
from utils import *
from scipy.spatial import distance as dist

from timeit import default_timer as timer
from threading import Thread
from queue import Queue
from plotter import MetricsMonitor

p = "../model/shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)

model = hopenet.Hopenet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 66)
saved_state_dict = torch.load('../model/hopenet_robust_alpha1.pkl', map_location="cpu")
model.load_state_dict(saved_state_dict)
model.eval()

transformations = transforms.Compose([transforms.Resize(224),
                                      transforms.CenterCrop(224), transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

idx_tensor = [idx for idx in range(66)]
idx_tensor = torch.FloatTensor(idx_tensor)

def eye_aspect_ratio(eye):
    # compute the euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    # compute the euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates
    C = dist.euclidean(eye[0], eye[3])
    # compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)
    # return the eye aspect ratio
    return ear

def mouth_aspect_ratio(mouth):
    A = dist.euclidean(mouth[2], mouth[6])
    B = dist.euclidean(mouth[0], mouth[4])
    mar = A  / B
    return mar

def comp_vision(q1,q2,q3,q4,q5):
    mar_stream = q1.get()
    ear_stream = q2.get()
    yaw_stream = q3.get()
    pitch_stream = q4.get()
    roll_stream = q5.get()

    cap = cv2.VideoCapture(0)
    cap.set(3, 720)
    cap.set(4, 1280)

    blinkCount = 0
    yawnCount = 0
    yawning=False
    eyeClosed=False

    while True:
        start = timer()
        ret, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.flip(frame,1)

        gray = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2GRAY)

        rects = detector(gray, 0)

        for (i, rect) in enumerate(rects):
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            inter1 = timer()
            # print(f"dlib face landmark: {(inter1-start):.2f}")

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

            ear_stream=np.append(ear_stream[1:],ear)
            q2.put(ear_stream)
            mar_stream=np.append(mar_stream[1:],mar)
            q1.put(mar_stream)

            if ear < 0.15:
                eyeClosed=True
            if ear > 0.15 and eyeClosed:
                blinkCount+=1
                eyeClosed=False
            cv2.putText(frame,"Blink Count: "+str(blinkCount),(10,30),fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.0, color=(255,0,0), thickness=2)
            
            if mar > 0.4:
                cv2.putText(frame,"Yawning! ",(10,90),fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.0, color=(255,0,0), thickness=2)
                yawning = True

            if mar < 0.2 and yawning:
                yawnCount+=1
                yawning=False
            cv2.putText(frame,"Yawn Count: "+str(yawnCount),(10,60),fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.0, color=(255,0,0), thickness=2)
            
            # Draw circle to show landmarks
            for idx,(x, y) in enumerate(shape):
                if idx in range(36,48):
                    cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
                elif idx in range(60,68):
                    cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)
                # Uncomment if you want to visualize all other landmarks
                # else:
                #     cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)

        try:
            roi_box, center_x, center_y = rec_to_roi_box(rect)

            roi_img = crop_img(frame, roi_box)

            img = Image.fromarray(roi_img)

            # Transform
            img = transformations(img)
            img_shape = img.size()
            img = img.view(1, img_shape[0], img_shape[1], img_shape[2])

            with torch.no_grad():
                yaw, pitch, roll = model(img)

                yaw_predicted = F.softmax(yaw, dim=1)
                pitch_predicted = F.softmax(pitch, dim=1)
                roll_predicted = F.softmax(roll, dim=1)
                # Get continuous predictions in degrees.
                yaw_predicted = torch.sum(yaw_predicted.data.view(-1) * idx_tensor) * 3 - 99
                pitch_predicted = torch.sum(pitch_predicted.view(-1) * idx_tensor) * 3 - 99
                roll_predicted = torch.sum(roll_predicted.view(-1) * idx_tensor) * 3 - 99

            # plot_pose_cube(frame, yaw_predicted, pitch_predicted, roll_predicted, tdx=int(center_x), tdy=int(center_y),
            #                size=100)

            draw_axis(frame, yaw_predicted.numpy(), pitch_predicted.numpy(), roll_predicted.numpy(), tdx=int(center_x), tdy=int(center_y), size=100)
            yaw_stream=np.append(yaw_stream[1:],float(yaw_predicted))
            q3.put(yaw_stream)
            pitch_stream=np.append(pitch_stream[1:],float(pitch_predicted))
            q4.put(pitch_stream)
            roll_stream=np.append(roll_stream[1:],float(roll_predicted))
            q5.put(roll_stream)

        except Exception as e:
            print(e)
            print("Cannot detect a face!")

        cv2.imshow('frame', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        end = timer()
        # print(f"FPS: {1/(end-start):.2f}")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def single_image_test():
    image = cv2.imread("../image/lena.jpg")

    rects = detector(image, 0)
    rect=rects[0]
    roi_box, center_x, center_y = rec_to_roi_box(rect)

    cv2.circle(image, (int(center_x), int(center_y)), 2, (0, 255, 0), -1)
    cv2.rectangle(image, (int(roi_box[0]),int(roi_box[1])), (int(roi_box[2]),int(roi_box[3])),(0, 255, 0))

    roi_img = crop_img(image, roi_box)
    img = Image.fromarray(roi_img)

    # Transform
    img = transformations(img)
    img_shape = img.size()
    img = img.view(1, img_shape[0], img_shape[1], img_shape[2])

    yaw, pitch, roll = model(img)

    yaw_predicted = F.softmax(yaw)
    pitch_predicted = F.softmax(pitch)
    roll_predicted = F.softmax(roll)

    # Get continuous predictions in degrees.
    yaw_predicted = torch.sum(yaw_predicted.data.view(-1) * idx_tensor) * 3 - 99
    pitch_predicted = torch.sum(pitch_predicted.view(-1) * idx_tensor) * 3 - 99
    roll_predicted = torch.sum(roll_predicted.view(-1) * idx_tensor) * 3 - 99

    plot_pose_cube(image, yaw_predicted, pitch_predicted, roll_predicted, tdx = int(center_x), tdy= int(center_y), size = 100)
    # draw_axis(image, yaw_predicted, pitch_predicted, roll_predicted, tdx = int(center_x), tdy= int(center_y), size = 100)

    cv2.imshow("OpenCV Image Reading", image)

    cv2.waitKey(0)

def monitor(q1,q2,q3,q4,q5):
    monitor_app = MetricsMonitor()
    monitor_app.stream(q1,q2,q3,q4,q5)
    monitor_app.animation()

if __name__ == '__main__':
    # event = Event()
    q1 = Queue()
    q2 = Queue()
    q3 = Queue()
    q4 = Queue()
    q5 = Queue()
    q1.put(np.ones(60))
    q2.put(np.ones(60))
    q3.put(np.zeros(60))
    q4.put(np.zeros(60))
    q5.put(np.zeros(60))

    t1 = Thread(name='Computer Vision Thread', target=comp_vision, args=(q1,q2,q3,q4,q5))
    t1.start()
    monitor(q1,q2,q3,q4,q5)
    t1.join()

    # single_image_test()




    
    

