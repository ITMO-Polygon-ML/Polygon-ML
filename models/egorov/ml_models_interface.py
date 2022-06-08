import os
from typing import Tuple

import cv2

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from utils.engine import train_one_epoch, evaluate
from utils.datasets import MRIClassificationDataset
from utils.transforms import get_transform


DEFAULT_MODEL_WEIGHT = './models/class_segment_model.pth'


class MRISegmentationClassificationNN:
    def __init__(self, image_size, batch_size,
                 epoch=2, weight_path=None, device='cuda', **kwargs):

        self.epoch: int = epoch
        self.batch_size: int = batch_size
        self.image_size: Tuple[int] = image_size

        self.weights_path: str = weight_path

        self.device = device
        self.color_mode = kwargs.get('color_mode', 'grayscale')
        self.history = None

        self.train_data_gen = lambda df: MRIClassificationDataset(
            df, get_transform(train=True)
        )
        self.val_data_gen = lambda df: MRIClassificationDataset(
            df, get_transform(train=False)
        )

    def construct_model(self):
        # Если появится torchvision, можно поменять

        def weights_init(m):
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform(m.weight.data)

        weight_path = self.weights_path or DEFAULT_MODEL_WEIGHT
        model = torch.jit.load(weight_path).to(self.device)
        # reset the weights
        model.apply(weights_init)

        return model

    def _read_df(self, df_path):
        return pd.read_csv(df_path, sep='\t')
    
    def create_data_iterator(self, generator, df_path):
        df = self._read_df(df_path)
        dataset = generator(df)

        data_loader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=self.batch_size, 
            collate_fn=lambda batch: tuple(zip(*batch))
        )

        return data_loader

    def get_callbacks(self) -> list:
        return []

    def fit(self, train_df_path, val_df_path=None):
        # Data preparation and increase the amount of data set
        gen_train = self.create_data_iterator(self.train_data_gen, train_df_path)

        gen_val = None
        if val_df_path:
            gen_val = self.create_data_iterator(self.val_data_gen, val_df_path)

        model = self.construct_model()

        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=0.005,
                                    momentum=0.9, weight_decay=0.0005)

        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                    step_size=3,
                                                    gamma=0.1)
        
        try:
            for epoch in range(self.epoch):
                train_one_epoch(model, optimizer, gen_train, self.device, epoch, 10)
                lr_scheduler.step()
                if gen_val:
                    evaluate(model, gen_val, device=self.device)
        except Exception as e:
            raise Exception(f'Fitting error: {e}')

        if self.weights_path is None:
            self.weights_path = f'weights_{hash(gen_train)}_{self.epoch}.h5'

        model.save(self.weights_path)

        return self

    def _get_tumor_mask_and_class(self, img_path, model):
        img_original = cv2.imread(img_path)
        img, _ = get_transform(train=False)(img_original, {})
        
        with torch.no_grad():
            prediction = model([img.to(self.device)])

        overall_desease = None
        overall_score = 0.5
        total_mask = np.zeros_like(img_original)[:, :, 0]

        for label, mask, score in zip(prediction[0]['labels'], prediction[0]['masks'], prediction[0]['scores']):
            if label.item() != 2 and score.item() > 0.15 and (not overall_desease or label.item() == overall_desease):
                mask = mask[0].mul(255).byte().cpu().numpy()
                mask = cv2.threshold(mask, 100, 255, cv2.THRESH_BINARY)[1]
                overall_desease = label.item()
                overall_score = max(overall_score, score.item())

                total_mask[mask == 255] = 255

        return {'mask': total_mask, 'label': overall_desease or 0, 'proba': overall_score}

    def evaluate(self, test_df_path, predict_column: str = 'predict', proba_column='proba'):
        if self.weights_path is None or not os.path.exists(self.weights_path):
            raise Exception(f'Weights were not found by path: {self.weights_path}')
        
        df = self._read_df(test_df_path)
        model = self.construct_model()

        predict = []
        proba = []

        for image_path in df['image_path']:
            result = self._get_tumor_mask_and_class(image_path, model)
            predict.append(result['label'])
            proba.append(result['proba'])

            # можно сюда добавить маски: result['mask']

        return {predict_column: predict, proba_column: proba}
