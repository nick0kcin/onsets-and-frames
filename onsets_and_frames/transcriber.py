"""
A rough translation of Magenta's Onsets and Frames implementation [1].

    [1] https://github.com/tensorflow/magenta/blob/master/magenta/models/onsets_frames_transcription/model.py
"""

import torch
import torch.nn.functional as F
from torch import nn
from typing import Dict, Tuple

from .lstm import BiLSTM
from .mel import melspectrogram


class ConvStack(nn.Module):
    def __init__(self, input_features, output_features):
        super().__init__()

        # input is batch_size * 1 channel * frames * input_features
        self.cnn = nn.Sequential(
            # layer 0
            nn.Conv2d(1, output_features // 16, (3, 3), padding=1),
            nn.BatchNorm2d(output_features // 16),
            nn.ReLU(),
            # layer 1
            nn.Conv2d(output_features // 16, output_features // 16, (3, 3), padding=1),
            nn.BatchNorm2d(output_features // 16),
            nn.ReLU(),
            # layer 2
            nn.MaxPool2d((1, 2)),
            nn.Dropout(0.25),
            nn.Conv2d(output_features // 16, output_features // 8, (3, 3), padding=1),
            nn.BatchNorm2d(output_features // 8),
            nn.ReLU(),
            # layer 3
            nn.MaxPool2d((1, 2)),
            nn.Dropout(0.25),
        )
        self.fc = nn.Sequential(
            nn.Linear((output_features // 8) * (input_features // 4), output_features),
            nn.Dropout(0.5)
        )

    def forward(self, mel):
        x = mel.view(mel.size(0), 1, mel.size(1), mel.size(2))
        x = self.cnn(x)
        x = x.transpose(1, 2).flatten(-2)
        x = self.fc(x)
        return x


class OnsetsAndFrames(nn.Module):
    def __init__(self, input_features, output_features, model_complexity=48, use_mel: bool = True):
        super().__init__()
        self.use_mel = use_mel

        model_size = model_complexity * 16
        sequence_model = lambda input_size, output_size: BiLSTM(input_size, output_size // 2)

        self.onset_stack = nn.Sequential(
            ConvStack(input_features, model_size),
            sequence_model(model_size, model_size),
            nn.Linear(model_size, output_features),
            nn.Sigmoid()
        )
        self.offset_stack = nn.Sequential(
            ConvStack(input_features, model_size),
            sequence_model(model_size, model_size),
            nn.Linear(model_size, output_features),
            nn.Sigmoid()
        )
        self.frame_stack = nn.Sequential(
            ConvStack(input_features, model_size),
            nn.Linear(model_size, output_features),
            nn.Sigmoid()
        )
        self.combined_stack = nn.Sequential(
            sequence_model(output_features * 3, model_size),
            nn.Linear(model_size, output_features),
            nn.Sigmoid()
        )
        # self.velocity_stack = nn.Sequential(
        #     ConvStack(input_features, model_size),
        #     nn.Linear(model_size, output_features)
        # )

    def forward(self, mel):
        onset_pred = self.onset_stack(mel)
        offset_pred = self.offset_stack(mel)
        activation_pred = self.frame_stack(mel)
        combined_pred = torch.cat([onset_pred.detach(), offset_pred.detach(), activation_pred], dim=-1)
        frame_pred = self.combined_stack(combined_pred)
        # velocity_pred = self.velocity_stack(mel)
        return onset_pred, offset_pred, activation_pred, frame_pred # , velocity_pred

    def run_on_batch(self, batch):
        audio_label = batch['audio']
        onset_label = batch['onset']
        offset_label = batch['offset']
        frame_label = batch['frame']
        # velocity_label = batch['velocity']

        if self.use_mel:
            mel = melspectrogram(audio_label.reshape(-1, audio_label.shape[-1])[:, :-1]).transpose(-1, -2)
        else:
            mel = audio_label
        onset_pred, offset_pred, _, frame_pred = self(mel)

        predictions = {
            'onset': onset_pred.reshape(*onset_label.shape),
            'offset': offset_pred.reshape(*offset_label.shape),
            'frame': frame_pred.reshape(*frame_label.shape),
            # 'velocity': velocity_pred.reshape(*velocity_label.shape)
        }
        #
        # losses = {
        #     'loss/onset': F.binary_cross_entropy(predictions['onset'], onset_label),
        #     'loss/offset': F.binary_cross_entropy(predictions['offset'], offset_label),
        #     'loss/frame': F.binary_cross_entropy(predictions['frame'], frame_label),
        #     # 'loss/velocity': self.velocity_loss(predictions['velocity'], velocity_label, onset_label)
        # }

        return self.loss(batch, predictions)

    def velocity_loss(self, velocity_pred, velocity_label, onset_label):
        denominator = onset_label.sum()
        if denominator.item() == 0:
            return denominator
        else:
            return (onset_label * (velocity_label - velocity_pred) ** 2).sum() / denominator

    def loss(self, batch: Dict[str, torch.Tensor], predict: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        seq_len = (batch["gt_shift"][:, 1] - batch["gt_shift"][:, 0]).max()
        gt_shift = batch["gt_shift"][0, 0], batch["gt_shift"][0, 1]
        if "shift" in batch:
            if (batch["shift"] > 0).sum() > 0:
                losses_dict = {
                    'loss/onset': sum([F.binary_cross_entropy(predict["onset"][i, index:index + seq_len, :],
                                          batch["onset"][i, index:index + seq_len, :]) for i, index in enumerate(batch["shift"])]),
                    'loss/offset': sum([F.binary_cross_entropy(predict["offset"][i, index:index + seq_len, :],
                                                                  batch["offset"][i, index:index + seq_len, :]) for i, index in enumerate(batch["shift"])]),
                    'loss/frame': sum([F.binary_cross_entropy(predict["frame"][i, index:index + seq_len, :],
                                                                 batch["frame"][i, index:index + seq_len, :])
                                          for i, index in enumerate(batch["shift"])])
                    # 'velocity': velocity_pred.reshape(*velocity_label.shape)
                }

                predictions = {
                    'onset': torch.stack([predict["onset"][i, index:index + seq_len, :] for i, index in enumerate(batch["shift"])]),
                    'offset': torch.stack([predict["offset"][i, index:index + seq_len, :] for i, index in enumerate(batch["shift"])]),
                    'frame': torch.stack([predict["frame"][i, index:index + seq_len, :] for i, index in enumerate(batch["shift"])]),
                    'path': batch["path"]
                    # 'loss/velocity': self.velocity_loss(predictions['velocity'], velocity_label, onset_label)
                }
            else:
                batch_onset = batch["onset"][:, gt_shift[0]:gt_shift[1], :]
                onset_weight = (batch_onset > 0.5).float() * 0.999 + (batch_onset < 0.5).float() * 0.001
                batch_offset = batch["offset"][:, gt_shift[0]:gt_shift[1], :]
                offset_weight = (batch_offset > 0.5).float() * 0.999 + (batch_offset < 0.5).float() * 0.001
                batch_frame = batch["frame"][:, gt_shift[0]:gt_shift[1], :]
                frame_weight = (batch_frame > 0.5).float() * 0.9 + (batch_frame < 0.5).float() * 0.1
                losses_dict = {
                    'loss/onset': F.binary_cross_entropy(predict["onset"][:, gt_shift[0]:gt_shift[1], :],
                                                         batch_onset, weight=onset_weight),
                    'loss/offset': F.binary_cross_entropy(predict["offset"][:, gt_shift[0]:gt_shift[1], :],
                                                          batch_offset, weight=offset_weight),
                    'loss/frame': F.binary_cross_entropy(predict["frame"][:, gt_shift[0]:gt_shift[1], :],
                                                         batch_frame, weight=frame_weight)
                    # 'velocity': velocity_pred.reshape(*velocity_label.shape)
                }

                predictions = {
                    'onset': predict["onset"][:, gt_shift[0]:gt_shift[1], :],
                    'offset': predict["offset"][:, gt_shift[0]:gt_shift[1], :],
                    'frame': predict["frame"][:, gt_shift[0]:gt_shift[1], :],
                    'path': batch["path"]
                    # 'loss/velocity': self.velocity_loss(predictions['velocity'], velocity_label, onset_label)
                }
        else:
            losses = torch.zeros((predict["frame"].shape[0], predict["frame"].shape[1] - seq_len + 1, 3), device=predict["frame"].device)
            for shift in range(0, predict["frame"].shape[1] - seq_len + 1, 2):
                for i in range(predict["frame"].shape[0]):
                    batch_onset =  batch["onset"][i, shift: shift + seq_len, :] if batch["full_gt"][i] else batch["onset"][i, gt_shift[0]: gt_shift[1], :]
                    onset_weight = (batch_onset > 0.5).float() * 0.999 + (batch_onset < 0.5).float() * 0.001
                    batch_offset = batch["offset"][i, shift: shift + seq_len, :] if batch["full_gt"][i] else batch["offset"][i, gt_shift[0]: gt_shift[1], :]
                    offset_weight = (batch_offset > 0.5).float() * 0.999 + (batch_offset < 0.5).float() * 0.001
                    batch_frame = batch["frame"][i, shift: shift + seq_len, :]  if batch["full_gt"][i] else batch["frame"][i, gt_shift[0]: gt_shift[1], :]
                    frame_weight = (batch_frame > 0.5).float() * 0.9 + (batch_frame < 0.5).float() * 0.1
                    losses[i, shift, 0] = F.binary_cross_entropy(predict["frame"][i, shift: shift + seq_len, :],
                                                                batch_frame, weight=frame_weight)
                    losses[i, shift, 1] = F.binary_cross_entropy(predict["onset"][i, shift: shift + seq_len, :],
                                                            batch_onset, weight=onset_weight
                                                                 )
                    losses[i, shift, 2] = F.binary_cross_entropy(predict["offset"][i, shift: shift + seq_len, :],
                                                                 batch_offset, weight=offset_weight)
            minimum, idx = torch.min(losses.sum(-1)[:, ::2], dim=-1)
            losses_dict = {
                'loss/onset': losses[torch.range(0, idx.shape[0] - 1).long(), idx, 1].sum(),
                'loss/offset': losses[torch.range(0, idx.shape[0] - 1).long(), idx, 2].sum(),
                'loss/frame': losses[torch.range(0, idx.shape[0] - 1).long(), idx, 0].sum(),
                # 'loss/velocity': self.velocity_loss(predictions['velocity'], velocity_label, onset_label)
            }
            predictions = {
                'onset': torch.stack([predict["onset"][i, 2 * index:2 * index+seq_len, :] for i, index in enumerate(idx)]),
                'offset': torch.stack([predict["offset"][i, 2 * index:2 * index+seq_len, :] for i, index in enumerate(idx)]),
                'frame': torch.stack([predict["frame"][i, 2 * index:2 * index+seq_len, :] for i, index in enumerate(idx)]),
                'shift': idx,
                'path': batch["path"]
                # 'velocity': velocity_pred.reshape(*velocity_label.shape)
            }
        return predictions, losses_dict



