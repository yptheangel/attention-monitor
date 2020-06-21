

# face pose detection
import torch
import torchvision
from torchvision.transforms import transforms
import torch.nn.functional as F

import hopenet

class Facepose:
    def __init__(self):
        self.model = hopenet.Hopenet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 66)
        saved_state_dict = torch.load('../model/hopenet_robust_alpha1.pkl', map_location="cpu")
        self.model.load_state_dict(saved_state_dict)
        self.model.eval()

        self.transformations = transforms.Compose([transforms.Resize(224),
                                      transforms.CenterCrop(224), transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        self.idx_tensor = torch.FloatTensor([idx for idx in range(66)])

    def predict(self, img):
        # Transform
        img = self.transformations(img)
        img_shape = img.size()
        img = img.view(1, img_shape[0], img_shape[1], img_shape[2])

        with torch.no_grad():
            yaw, pitch, roll = self.model(img)

            yaw_predicted = F.softmax(yaw, dim=1)
            pitch_predicted = F.softmax(pitch, dim=1)
            roll_predicted = F.softmax(roll, dim=1)
            # Get continuous predictions in degrees.
            yaw_predicted = torch.sum(yaw_predicted.data.view(-1) * self.idx_tensor) * 3 - 99
            pitch_predicted = torch.sum(pitch_predicted.view(-1) * self.idx_tensor) * 3 - 99
            roll_predicted = torch.sum(roll_predicted.view(-1) * self.idx_tensor) * 3 - 99

        return yaw_predicted, pitch_predicted, roll_predicted