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

import vispy_realtime_test

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
    # compute the euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates
    A = dist.euclidean(mouth[2], mouth[6])
    # compute the euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates
    B = dist.euclidean(mouth[0], mouth[4])
    # compute the eye aspect ratio
    mar = A  / B
    # return the eye aspect ratio
    return mar

def main():
    cap = cv2.VideoCapture(0)
    blinkCount = 0
    yawnCount = 0
    yawning=False
    eyeClosed=False
    # canvas = vispy_realtime_test.Canvas(1, 1, 1000, 0)

    while True:

        ret, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.flip(frame,1)

        gray = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2GRAY)

        rects = detector(gray, 0)

        for (i, rect) in enumerate(rects):
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
                eyeClosed=True
            if ear > 0.15 and eyeClosed:
                blinkCount+=1
                eyeClosed=False
            cv2.putText(frame,"Blink Count: "+str(blinkCount),(10,30),fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.0, color=(255,0,0), thickness=2)
            
            if mar > 0.4:
                cv2.putText(frame,"Yawning! ",(10,90),fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.0, color=(255,0,0), thickness=2)
                yawning = True
                # canvas.set_data(1)

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
        
        except Exception as e:
            print(e)
            print("Cannot detect a face!")

        cv2.imshow('frame', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
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

if __name__ == '__main__':
    main()
    # single_image_test()




    
    

