import time
import warnings
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from albumentations import BboxParams, Compose, OneOf, Resize, RGBShift, \
    RandomBrightnessContrast, Normalize, ShiftScaleRotate, Blur, ColorJitter, RandomGamma, \
    Cutout
from albumentations.augmentations.bbox_utils import convert_bbox_from_albumentations
from albumentations.augmentations.transforms import Equalize
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import auc
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from tqdm.notebook import tqdm

warnings.filterwarnings("ignore")
seed = 42
sns.set()


class ProductDataset(Dataset):
    def __init__(self, data_dir, images_info_df, dataset_format, augment=None, transform=None):
        self.data_dir = data_dir
        self.data = images_info_df
        self.augment = augment
        self.transform = transform
        self.mapping = self.data.groupby(['labels', 'supercategory']).size().reset_index().rename(columns={0: 'count'})
        self.dataset_format = dataset_format

    def text_label(self, index):
        return self.mapping.loc[self.mapping['labels'] == index, 'supercategory'].iloc[0]

    def cut_bboxes(self, image, bboxes, env_size=0.2):
        [x, y, w, h] = bboxes
        im_h, im_w, im_c = image.shape
        greater = False
        if x + w > im_w:
            if x > im_w:
                greater = True
            else:
                w = im_w - x
        if y + h > im_h:
            if y > im_h:
                greater = True
            else:
                h = im_h - y
        if greater:
            return image, np.array([0, 0, im_w, im_h])
        bboxes = np.array([x, y, w, h])
        return image, bboxes

    def convert_format(self, res):
        xmin, ymin, width, height = res["bboxes"][0], res["bboxes"][1], \
                                    res["bboxes"][2], res["bboxes"][3]
        xmax, ymax = xmin + width, ymin + height
        img_h, img_w = res['image'].shape[0], res['image'].shape[1]
        res["bboxes"] = convert_bbox_from_albumentations(bbox=(xmin / img_w,
                                                               ymin / img_h,
                                                               xmax / img_w,
                                                               ymax / img_h),
                                                         target_format=self.dataset_format,
                                                         rows=img_h, cols=img_w, check_validity=True)
        return res

    def __getitem__(self, i, apply_augmentations=True,
                    apply_transformations=True):
        record = self.data.iloc[i]
        image = cv2.imread(str(self.data_dir / record['file_name']), 1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        bboxes = np.array(record['bboxes'][1:-1].split(',')).astype('float')
        image, bboxes = self.cut_bboxes(image, bboxes)
        labels = np.array(record['labels']).astype('int64')
        res = {"image": image,
               "bboxes": torch.tensor(bboxes, dtype=torch.float),
               "labels": torch.tensor(np.array([labels])),
               }
        # convert bounding boxes to 'pascal_voc' format from initial
        res = self.convert_format(res)
        res["bboxes"] = torch.tensor(res["bboxes"]).reshape((1, -1))
        if self.augment and apply_augmentations:
            res = self.augment(**res)
        if self.transform and apply_transformations:
            res = self.transform(**res)
        return res["image"], {
            "boxes": torch.tensor(res["bboxes"], dtype=torch.float),
            "labels": torch.tensor(res["labels"], dtype=torch.int64),
        }

    def __len__(self):
        return len(self.data)

    # Visualize dataset
    def get_grid_size(self, imgs_num, nrows, ncols):
        from math import ceil
        if nrows is None and ncols is None:
            nrows = 1
            ncols = imgs_num
        elif nrows is None:
            nrows = ceil(imgs_num / ncols)
        elif ncols is None:
            ncols = ceil(imgs_num / nrows)
        return nrows, ncols

    def plot_detection_boxes(self, img, detection_boxes, detection_labels,
                             dataset_format=None, verbose=False):
        if verbose:
            print(f'Dislay in {dataset_format} format')
        # `coco` format: `(x_min, y_min, width, height)`
        # `pascal_voc`format:`(x_min, y_min, x_max, y_max)`
        # `yolo` format :`(x, y, width, height)`
        for box, label in zip(detection_boxes, detection_labels):
            if dataset_format == 'coco':
                # round values only for visualization (opencv requirement)
                xmin, ymin, width, height = int(box[0]), int(box[1]), \
                                            int(box[2]), int(box[3])
                img = cv2.rectangle(np.array(img), (xmin, ymin + height),
                                    (xmin + width, ymin), (0, 255, 0), 3)
            elif dataset_format == 'pascal_voc':
                xmin, ymin, xmax, ymax = int(box[0]), int(box[1]), \
                                         int(box[2]), int(box[3])
                img = cv2.rectangle(np.array(img), (xmin, ymax), (xmax, ymin),
                                    (0, 255, 0), 3)
            else:
                raise ValueError(f'Please specify a format of bounding boxes as '
                                 f'dataset_format parameter. Currently supported '
                                 f'formats are "coco" and "pascal_voc"')
            scale = 0.3  # (0,1] to change text size relative to the image
            fontScale = min(img.shape[1], img.shape[0]) / (150 / scale)
            img = cv2.putText(img, text=str(label), org=(xmin, ymin),
                              fontFace=3, fontScale=fontScale, color=(0, 0, 0))
        return img

    def plot_images(self, imgs, names=None, axs=None, show=True, nrows=None, ncols=None,
                    figsize=(16, 8), mode=None, detection_boxes=None,
                    detection_labels=None, dataset_format=None, pic_name=None):
        nrows, ncols = self.get_grid_size(len(imgs), nrows, ncols)

        if axs is None:
            fig, axs = plt.subplots(ncols=ncols, nrows=nrows, figsize=figsize)
        if nrows == 1 and ncols == 1:
            if mode == 'detection':
                imgs[0] = self.plot_detection_boxes(imgs[0], detection_boxes,
                                                    detection_labels,
                                                    dataset_format=dataset_format)
            axs.imshow(imgs[0])
            axs.set_axis_off()
            if names and len(names) > 0:
                axs.set_title(names[0], fontsize=15)
            if pic_name:
                pass
                # axs.imsave(f'{pic_name}.jpg', imgs[0])
        elif nrows == 1 or ncols == 1:
            for j, ax in enumerate(axs):
                if mode == 'detection':
                    if ncols == 1:
                        imgs[j] = self.plot_detection_boxes(imgs[j],
                                                            [detection_boxes[j]],
                                                            [detection_labels[j]],
                                                            dataset_format=dataset_format)
                    elif nrows == 1:
                        imgs[j] = self.plot_detection_boxes(imgs[j],
                                                            [detection_boxes[j]],
                                                            [detection_labels[j]],
                                                            dataset_format=dataset_format)
                ax.imshow(imgs[j])
                ax.set_axis_off()
                if names and j < len(names):
                    ax.set_title(names[j], fontsize=15)
        else:
            for j, ax in enumerate(axs):
                for k, sub_ax in enumerate(ax):
                    image_id = j * ncols + k
                    sub_ax.set_axis_off()
                    if image_id < len(imgs):
                        if mode == 'detection':
                            imgs[image_id] = self.plot_detection_boxes(imgs[image_id],
                                                                       [detection_boxes[image_id]],
                                                                       [detection_labels[image_id]],
                                                                       dataset_format=dataset_format)
                        sub_ax.imshow(imgs[image_id])
                        if names and image_id < len(names):
                            sub_ax.set_title(names[image_id], fontsize=15)
        if show:
            plt.show()

    def show_dataset(self, sample_size=3, n=6, show_augmentations=True):
        idx = [np.random.choice(np.arange(self.__len__())) for i in range(sample_size)]
        imgs = []
        names = ['original']
        detection_boxes = []
        detection_labels = []

        for index in idx:
            item = self.__getitem__(index, apply_augmentations=False,
                                    apply_transformations=False)
            img = item[0]
            annotations_true = item[1]
            true_boxes, true_labels = list(annotations_true['boxes'].numpy()[0]), \
                                      list(annotations_true['labels'].numpy())[0]
            label_names = self.text_label(true_labels)
            imgs.append(img)
            detection_boxes.append(true_boxes)
            detection_labels.append(label_names)

            if show_augmentations:
                for _ in range(n - 1):
                    augmented = self.__getitem__(index,
                                                 apply_augmentations=True,
                                                 apply_transformations=False)
                    img_augmented = augmented[0]
                    if len(augmented) > 1 and len(augmented[1]["boxes"]) > 0:
                        annotations = augmented[1]
                        boxes_augmented = list(augmented[1]["boxes"].numpy()[0])
                        labels_augmented = list(augmented[1]["labels"].numpy())[0]
                        label_names = self.text_label(labels_augmented)

                        imgs.append(img_augmented)
                        detection_boxes.append(boxes_augmented)
                        detection_labels.append(label_names)
        if show_augmentations:
            aug = ['augmented' for _ in range(n - 1)]
            names = names + aug
        self.plot_images(imgs, nrows=sample_size, ncols=n, mode='detection',
                         names=names, detection_boxes=detection_boxes,
                         detection_labels=detection_labels,
                         dataset_format=self.dataset_format)

    def collate_fn(self, batch):
        return tuple(zip(*batch))


class Trainer():
    def __init__(self):
        pass

    def intersection_over_union(self, dt_bbox, gt_bbox):
        """
        Intersection over Union between two bboxes
        :param dt_bbox: list or numpy array of size (4,) [x0, y0, x1, y1], i.e. xmin, ymin, xmax, ymax
        :param gt_bbox: list or numpy array of size (4,) [x0, y0, x1, y1]
        :return : intersection over union
        """

        intersection_bbox = np.array(
            [
                max(dt_bbox[0], gt_bbox[0]),
                max(dt_bbox[1], gt_bbox[1]),
                min(dt_bbox[2], gt_bbox[2]),
                min(dt_bbox[3], gt_bbox[3]),
            ]
        )

        intersection_area = max(intersection_bbox[2] - intersection_bbox[0], 0) * max(
            intersection_bbox[3] - intersection_bbox[1], 0
        )
        area_dt = (dt_bbox[2] - dt_bbox[0]) * (dt_bbox[3] - dt_bbox[1])
        area_gt = (gt_bbox[2] - gt_bbox[0]) * (gt_bbox[3] - gt_bbox[1])

        union_area = area_dt + area_gt - intersection_area

        iou = intersection_area / union_area
        return iou

    def evaluate_sample(self, target_pred, target_true, iou_threshold=0.5):
        gt_bboxes = target_true["boxes"].numpy()  # ground truth
        gt_labels = target_true["labels"].numpy()

        dt_bboxes = target_pred["boxes"].numpy()  # detected
        dt_labels = target_pred["labels"].numpy()

        dt_scores = target_pred["scores"].numpy()

        results = []
        for detection_id in range(len(dt_labels)):
            dt_bbox = dt_bboxes[detection_id, :]
            dt_label = dt_labels[detection_id]
            dt_score = dt_scores[detection_id]

            detection_result_dict = {"score": dt_score}

            max_IoU = 0
            max_gt_id = -1
            for gt_id in range(len(gt_labels)):
                gt_bbox = gt_bboxes[gt_id, :]
                gt_label = gt_labels[gt_id]

                if gt_label != dt_label:
                    continue

                iou_current = self.intersection_over_union(dt_bbox, gt_bbox)
                if iou_current > max_IoU:
                    max_IoU = iou_current
                    max_gt_id = gt_id

            if max_gt_id >= 0 and max_IoU >= iou_threshold:
                detection_result_dict["TP"] = 1
                gt_labels = np.delete(gt_labels, max_gt_id, axis=0)
                gt_bboxes = np.delete(gt_bboxes, max_gt_id, axis=0)

            else:
                detection_result_dict["TP"] = 0

            results.append(detection_result_dict)

        return results

    def evaluate(self, model, test_loader, device='cuda'):
        results = []
        model.eval()
        nbr_boxes = 0
        num_batches = len(test_loader)

        with torch.no_grad():
            for images, targets_true in tqdm(test_loader, total=num_batches):
                images = torch.stack(images).to(device)
                targets_true = [{k: v.to(device) for k, v in t.items()} for t in targets_true]

                targets_pred = model(images)

                targets_true = [
                    {k: v.cpu().float() for k, v in t.items()} for t in targets_true
                ]
                targets_pred = [
                    {k: v.cpu().float() for k, v in t.items()} for t in targets_pred
                ]

                for i in range(len(targets_true)):
                    target_true = targets_true[i]
                    target_pred = targets_pred[i]
                    nbr_boxes += target_true["labels"].shape[0]

                    results.extend(self.evaluate_sample(target_pred, target_true))
        results = sorted(results, key=lambda k: k["score"], reverse=True)

        acc_TP = np.zeros(len(results))
        acc_FP = np.zeros(len(results))
        recall = np.zeros(len(results))
        precision = np.zeros(len(results))

        if results is not None:
            if results[0]["TP"] == 1:
                acc_TP[0] = 1
            else:
                acc_FP[0] = 1

        for i in range(1, len(results)):
            acc_TP[i] = results[i]["TP"] + acc_TP[i - 1]
            acc_FP[i] = (1 - results[i]["TP"]) + acc_FP[i - 1]

            precision[i] = acc_TP[i] / (acc_TP[i] + acc_FP[i])
            recall[i] = acc_TP[i] / nbr_boxes

        return auc(recall, precision)

    def train(self, model, optimizer, train_dataloader, device, scheduler=None):
        model.train()
        running_loss = 0
        num_batches = len(train_dataloader)

        pbar = tqdm(enumerate(train_dataloader), total=num_batches)
        for i, (images, targets) in pbar:
            optimizer.zero_grad()

            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()  # Cyclic LRs after each batch
            running_loss += loss.item()
            pbar.set_description_str(f'Iteration #{i}. Loss: {running_loss / (i + 1):.3f}',
                                     refresh=False)
        train_loss = running_loss / len(train_dataloader.dataset)
        return train_loss

    def save_model(self, model, model_save_path):
        torch.save(model.state_dict(), Path(model_save_path))

    def get_detection_model(self, num_classes, pretrained=True):
        model = fasterrcnn_resnet50_fpn(pretrained=pretrained, min_size=256, max_size=448)
        # get the number of input features for the classifier
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        # replace the pre-trained head with a new one
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        return model

    ## Code for model performance evaluation and training
    def visualize(self, train_data_loader):
        images, targets, image_ids = next(iter(train_data_loader))
        images = list(image for image in images)
        targets = [{k: v for k, v in t.items()} for t in targets]
        for i in range(1):
            boxes = targets[i]['boxes'].cpu().numpy().astype(np.int32)
            sample = images[i].permute(1, 2, 0).cpu().numpy()
            fig, ax = plt.subplots(1, 1, figsize=(15, 12))
            for box in boxes:
                cv2.rectangle(sample,
                              (box[0], box[1]),
                              (box[2], box[3]),
                              (220, 0, 0), 3)
            ax.set_axis_off()
            plt.imshow(sample)
            plt.show()


def main():
    # ### Read and preprocessed the data
    data_path = Path('../data')
    csv_path = Path('../data/rpc_train_dataframe_super_tiny.csv')
    data = pd.read_csv(csv_path)
    num_classes = data['labels'].nunique()
    # # data = pd.read_csv(csv_path, index_col=0)
    # # data.rename(columns={'supercategory_id': 'labels'}, inplace=True)
    # # data.rename(columns={'bbox': 'bboxes'}, inplace=True)
    # # data.set_index('image_id', inplace=True)
    train_data, val_data = train_test_split(data, test_size=0.1, shuffle=True, random_state=seed)

    image_height, image_width = 128, 128
    dataset_format = 'pascal_voc'
    bbox_params = BboxParams(format=dataset_format,
                             min_visibility=0.3,
                             label_fields=['labels'])
    train_augment = Compose([
        OneOf([
            RGBShift(r_shift_limit=15, g_shift_limit=15,
                     b_shift_limit=15, p=0.5),
            RandomBrightnessContrast(brightness_limit=0.1,
                                     contrast_limit=0.05),
            Equalize(p=0.1),
            RandomGamma(p=0.5),
            ColorJitter(p=0.5),
        ], p=0.9),
        OneOf([
            ShiftScaleRotate(shift_limit=0.1625,
                             scale_limit=0.1,
                             rotate_limit=20,
                             interpolation=1,
                             border_mode=4, p=0.9),
        ], p=0.6),

        OneOf([
            Blur(blur_limit=3, p=0.1),
            Cutout(num_holes=8, max_h_size=5,
                   max_w_size=5, fill_value=0, p=0.3),

        ], p=0.2),
    ],
        bbox_params=bbox_params)
    transform = Compose([
        Resize(image_height, image_width),
        Normalize(mean=(0, 0, 0), std=(1, 1, 1)),
        ToTensorV2(),
    ], bbox_params=bbox_params)
    train_dataset = ProductDataset(data_path / 'train2019', train_data,
                                   augment=train_augment, transform=transform, dataset_format=dataset_format)
    val_dataset = ProductDataset(data_path / 'train2019', val_data,
                                 augment=None, transform=transform, dataset_format=dataset_format)
    # train_dataset.show_dataset(sample_size=4, n=3)
    # val_dataset.show_dataset(sample_size=2, n=1)

    BATCH_SIZE = 2
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2,
        collate_fn=train_dataset.collate_fn)
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2,
        collate_fn=val_dataset.collate_fn)

    # print(len(train_dataloader), len(train_dataset))
    # print(len(val_dataloader), len(val_dataset))

    ## Train a model
    trainer = Trainer()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"device is {device}")
    model = trainer.get_detection_model(num_classes + 1, pretrained=True).to(device)
    params = [v for k, v in model.named_parameters() if v.requires_grad]
    max_lr = 1e-1
    optimizer = torch.optim.SGD(params, lr=max_lr, momentum=0.9, weight_decay=0.0005)

    n_epochs = 5
    experiment_n = 1

    for epoch in range(n_epochs):
        start = time.time()
        train_loss = trainer.train(model, optimizer, train_dataloader, device, scheduler=None)
        print(f"Epoch #{epoch} loss: {train_loss}")
        end = time.time()
        print(f"Epoch {epoch} took {(end - start) / 60} minutes")
    trainer.save_model(model, f'saved_models/{experiment_n}fasterrcnn_resnet50_fpn_{n_epochs}_epoch.pth')


main()
