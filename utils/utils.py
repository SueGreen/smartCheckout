import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def get_model(num_classes, model_path, device):
    # load the model and the trained weights
    model = get_detection_model(num_classes + 1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
    return model


def get_detection_model(num_classes=2):
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def text_label(df, index):
    return df.loc[df['labels'] == index, 'supercategory'].iloc[0]