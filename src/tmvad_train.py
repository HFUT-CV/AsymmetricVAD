import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR

from model import AsymmetricVAD
from tmvad_test import test
from tmvad_option import parser
from utils.dataset import TMVADDataset, load_tmvad_segments
from utils.tools import get_batch_label, get_prompt_text


def clas_m(logits, labels, lengths, device):
    instance_logits = torch.zeros(0, device=device)
    labels = labels / torch.sum(labels, dim=1, keepdim=True)
    for index in range(logits.shape[0]):
        top_k = int(lengths[index] / 16 + 1)
        values, _ = torch.topk(logits[index, :lengths[index]], k=top_k, dim=0)
        instance_logits = torch.cat([instance_logits, values.mean(0, keepdim=True)])
    return -torch.mean(torch.sum(labels * F.log_softmax(instance_logits, dim=1), dim=1))


def clas_binary(logits, labels, lengths, device):
    instance_logits = torch.zeros(0, device=device)
    targets = 1 - labels[:, 0]
    probabilities = torch.sigmoid(logits).reshape(logits.shape[0], logits.shape[1])
    for index in range(logits.shape[0]):
        top_k = int(lengths[index] / 16 + 1)
        values, _ = torch.topk(probabilities[index, :lengths[index]], k=top_k)
        instance_logits = torch.cat([instance_logits, values.mean().view(1)])
    return F.binary_cross_entropy(instance_logits, targets)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def train(model, normal_loader, anomaly_loader, test_loader, args, device):
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = MultiStepLR(optimizer, args.scheduler_milestones, args.scheduler_rate)
    label_map = {'Normal': 'normal', 'Abnormal': 'abnormal'}
    prompt_text = get_prompt_text(label_map)
    category_text_features = torch.tensor(np.load(args.category_text_path)).float().to(device)
    best_ap = -1

    if args.use_checkpoint and os.path.exists(args.checkpoint_path):
        checkpoint = torch.load(args.checkpoint_path, map_location=device)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        load_result = model.load_state_dict(state_dict, strict=False)
        print('checkpoint missing keys:', load_result.missing_keys)
        print('checkpoint unexpected keys:', load_result.unexpected_keys)
        if isinstance(checkpoint, dict) and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        best_ap = checkpoint.get('ap', best_ap) if isinstance(checkpoint, dict) else best_ap

    os.makedirs(os.path.dirname(args.model_path) or '.', exist_ok=True)

    for epoch in range(args.max_epoch):
        model.train()
        normal_iter, anomaly_iter = iter(normal_loader), iter(anomaly_loader)
        for _ in range(min(len(normal_loader), len(anomaly_loader))):
            normal_features, normal_text, normal_audio, normal_labels, normal_lengths = next(normal_iter)
            anomaly_features, anomaly_text, anomaly_audio, anomaly_labels, anomaly_lengths = next(anomaly_iter)
            features = torch.cat([normal_features, anomaly_features]).to(device)
            text_features = torch.cat([normal_text, anomaly_text]).to(device)
            audio_features = torch.cat([normal_audio, anomaly_audio]).to(device)
            lengths = torch.cat([normal_lengths, anomaly_lengths]).to(device)
            labels = get_batch_label(list(normal_labels) + list(anomaly_labels), prompt_text, label_map).to(device)
            text_features_ori, logits1, logits2 = model(
                features, None, lengths,
                text_features=text_features, category_text_features=category_text_features, audio=audio_features
            )
            loss1 = clas_binary(logits1, labels, lengths, device)
            loss2 = clas_m(logits2, labels, lengths, device)
            normal_text_prompt = F.normalize(text_features_ori[0], dim=-1)
            diversity = sum(torch.abs(normal_text_prompt @ F.normalize(text_features_ori[i], dim=-1))
                            for i in range(1, text_features_ori.shape[0])) / 1
            loss = loss1 + args.lambda_cls * loss2 + diversity * 1e-1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
        _, ap = test(model, test_loader, args, category_text_features, device)
        if ap > best_ap:
            best_ap = ap
            torch.save(model.state_dict(), args.model_path)
        print(f'epoch: {epoch + 1}, AP: {ap:.6f}, best AP: {best_ap:.6f}')


if __name__ == '__main__':
    args = parser.parse_args()
    setup_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    label_map = {'Normal': 'normal', 'Abnormal': 'abnormal'}
    train_dataset = TMVADDataset(args.data_root, args.visual_length, False, args.train_list)
    test_dataset = TMVADDataset(args.data_root, args.visual_length, True, args.test_list)
    normal_dataset = [sample for sample in train_dataset.samples if sample[1] == 'Normal']
    anomaly_dataset = [sample for sample in train_dataset.samples if sample[1] == 'Abnormal']
    normal_dataset_obj = TMVADDataset(args.data_root, args.visual_length, False, args.train_list)
    normal_dataset_obj.samples = normal_dataset
    anomaly_dataset_obj = TMVADDataset(args.data_root, args.visual_length, False, args.train_list)
    anomaly_dataset_obj.samples = anomaly_dataset
    normal_loader = DataLoader(normal_dataset_obj, args.batch_size, shuffle=True, drop_last=True)
    anomaly_loader = DataLoader(anomaly_dataset_obj, args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    model = AsymmetricVAD(2, args.embed_dim, args.visual_length, args.visual_width,
                    args.visual_head, args.visual_layers, args.attn_window,
                    device,
                    similarity_threshold=args.similarity_threshold,
                    lookback_steps=args.lookback_steps,
                    sie_window_radius=args.sie_window_radius,
                    sie_topk_ratio=args.sie_topk_ratio,
                    sie_sigma=args.sie_sigma,
                    sie_alpha_init=args.sie_alpha_init,
                    sie_epsilon_audio_init=args.sie_epsilon_audio_init,
                    sie_lambda_audio_init=args.sie_lambda_audio_init,
                    uncertainty_threshold=args.uncertainty_threshold)
    train(model, normal_loader, anomaly_loader, test_loader, args, device)