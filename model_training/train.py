import warnings
from pathlib import Path

import warnings
from pathlib import Path

import pandas as pd
import seaborn as sns
import torch
from albumentations import BboxParams, Compose, OneOf, Resize, RGBShift, \
    RandomBrightnessContrast, Normalize, ShiftScaleRotate, Blur, ColorJitter, RandomGamma, \
    Cutout
from albumentations.augmentations.transforms import Equalize
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from detector import ProductDataset, ProductDetectionTrainer

warnings.filterwarnings("ignore")
seed = 42
sns.set()


def main():
    # Read and preprocessed the data
    data_path = Path('../data')
    csv_path = Path('../data/rpc_train_dataframe_super_tiny.csv')
    data = pd.read_csv(csv_path)
    num_classes = data['labels'].nunique()
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

    BATCH_SIZE = 2
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2,
        collate_fn=train_dataset.collate_fn)
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2,
        collate_fn=val_dataset.collate_fn)

    # Train the model
    trainer = ProductDetectionTrainer()
    experiment_n = 1
    max_lr = 1e-1
    n_epochs = 3
    path_to_model_new = Path(f'saved_models/{experiment_n}fasterrcnn_resnet50_fpn_{n_epochs}_epoch.pth')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = trainer.get_detection_model(num_classes + 1, pretrained=True)
    params = [v for k, v in model.named_parameters() if v.requires_grad]
    optimizer = torch.optim.SGD(params, lr=max_lr, momentum=0.9, weight_decay=0.0005)
    num_batches_train = len(train_dataloader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=max_lr, epochs=n_epochs,
                                                    steps_per_epoch=num_batches_train)
    model = model.to(device)
    best_val_auc = 0.0

    for epoch in range(n_epochs):
        model.eval()
        val_auc = trainer.evaluate(model, val_dataloader, device=device)
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            trainer.save_model(model, path_to_model_new)
        print(f'Epoch {epoch}. AUC ON VALIDATION: {round(float(val_auc), 4)}')
        model.train()
        train_loss = trainer.train(model, optimizer, train_dataloader, device=device, scheduler=scheduler)


main()
