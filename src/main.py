import dlib
import cv2
import torchvision
from imutils import face_utils
import torch
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F

from src import hopenet
from src.utils import *

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

def main():
    cap = cv2.VideoCapture(0)

    while (True):
        ret, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.flip(frame,1)

        gray = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2GRAY)

        rects = detector(gray, 0)

        for (i, rect) in enumerate(rects):
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            for (x, y) in shape:
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

        roi_box, center_x, center_y = rec_to_roi_box(rect)

        roi_img = crop_img(frame, roi_box)

        img = Image.fromarray(roi_img)

        # Transform
        img = transformations(img)
        img_shape = img.size()
        img = img.view(1, img_shape[0], img_shape[1], img_shape[2])

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

        draw_axis(frame, yaw_predicted, pitch_predicted, roll_predicted, tdx=int(center_x), tdy=int(center_y), size=100)

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




    
    

