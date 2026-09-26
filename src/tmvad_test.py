import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from model import AsymmetricVAD
from tmvad_option import parser
from utils.dataset import TMVADDataset, load_tmvad_segments
from utils.tools import get_batch_mask
from utils.xd_detectionMAP import getDetectionMAP as dmAP


def test(model, test_loader, args, category_text_features, device):
    model.to(device).eval()
    predictions, targets = [], []
    element_logits2_stack = []
    video_ids = [path.stem for path, _, _, _ in test_loader.dataset.samples]
    segments, _ = load_tmvad_segments(args.label_root, video_ids)
    gtlabels = [['abnormal'] * len(video_segments) for video_segments in segments]
    with torch.no_grad():
        for item, video_segments in zip(test_loader, segments):
            visual = item[0].squeeze(0)
            text_features = item[1].squeeze(0)
            audio_features = item[2].squeeze(0)
            length = int(item[4])
            visual = visual.to(device)
            chunk_lengths = torch.tensor([min(args.visual_length, length - i)
                                          for i in range(0, length, args.visual_length)], dtype=torch.long, device=device)
            if visual.ndim == 2:
                visual = visual.unsqueeze(0)
            padding_mask = get_batch_mask(chunk_lengths.cpu(), args.visual_length).to(device)
            _, logits1, logits2 = model(
                visual, padding_mask, chunk_lengths,
                text_features=text_features, category_text_features=category_text_features, audio=audio_features
            )
            logits1 = logits1.reshape(-1, logits1.shape[-1])[:length]
            logits2 = logits2.reshape(-1, logits2.shape[-1])[:length]
            probs = logits2.softmax(dim=-1)
            score = probs[:, 1].cpu().numpy()
            predictions.extend(np.repeat(score, 16))
            element_logits2 = np.repeat(probs[:, 1:2].detach().cpu().numpy(), 16, axis=0)
            element_logits2_stack.append(element_logits2)
            frame_target = np.zeros(length * 16, dtype=np.int32)
            for start, end in video_segments:
                frame_target[max(0, start):min(length * 16, end)] = 1
            targets.extend(frame_target)
    predictions = np.asarray(predictions)
    targets = np.asarray(targets)
    auc = roc_auc_score(targets, predictions)
    ap = average_precision_score(targets, predictions)
    dmap, iou = dmAP(element_logits2_stack, segments, gtlabels, classlist=['abnormal'])
    for i in range(len(iou)):
        print(f'mAP@{iou[i]:.1f} = {dmap[i]:.2f}%')
    print(f'average mAP: {sum(dmap) / len(dmap):.2f}%')
    print(f'AUC: {auc:.6f}, AP: {ap:.6f}')
    return auc, ap


if __name__ == '__main__':
    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dataset = TMVADDataset(args.data_root, args.visual_length, True, args.test_list)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
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
    checkpoint = torch.load(args.model_path, map_location=device)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    load_result = model.load_state_dict(state_dict, strict=False)
    if load_result.missing_keys or load_result.unexpected_keys:
        print('checkpoint missing keys:', load_result.missing_keys)
        print('checkpoint unexpected keys:', load_result.unexpected_keys)
    category_text_features = torch.tensor(np.load(args.category_text_path)).float().to(device)
    test(model, loader, args, category_text_features, device)