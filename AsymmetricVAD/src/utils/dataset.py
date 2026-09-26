import numpy as np
import torch
import torch.utils.data as data
import pandas as pd
import utils.tools as tools
from pathlib import Path
import xml.etree.ElementTree as ET

class XDDataset(data.Dataset):
    def __init__(self, clip_dim: int, file_path: str, test_mode: bool, label_map: dict):
        self.df = pd.read_csv(file_path)
        self.clip_dim = clip_dim
        self.test_mode = test_mode
        self.label_map = label_map
        
    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        clip_feature = np.load(self.df.loc[index]['path'])
        if self.test_mode == False:
            clip_feature, clip_length = tools.process_feat(clip_feature, self.clip_dim)
        else:
            clip_feature, clip_length = tools.process_split(clip_feature, self.clip_dim)

        clip_feature = torch.tensor(clip_feature)
        clip_label = self.df.loc[index]['label']
        return clip_feature, clip_label, clip_length


class TMVADDataset(data.Dataset):
    def __init__(self, root: str, clip_dim: int, test_mode: bool, file_path: str = None):
        split = 'test' if test_mode else 'train'
        self.samples = []
        if file_path:
            table = pd.read_csv(file_path)
            for row in table.itertuples(index=False):
                values = row._asdict()
                video_path = Path(values['path'])
                caption_path = Path(values.get('caption_path', video_path))
                audio_path = Path(values.get('audio_path', video_path))
                self.samples.append((video_path, values['label'], caption_path, audio_path))
        else:
            video_root = Path(root) / split / 'videos'
            for label, folder in (('Normal', 'normal'), ('Abnormal', 'abnormal')):
                for path in sorted((video_root / folder).glob('*.npy')):
                    modality_folder = 'nor' if folder == 'normal' else 'abn'
                    caption_path = Path(root) / split / 'captions' / modality_folder / path.name
                    audio_path = Path(root) / split / 'audio' / modality_folder / path.name
                    self.samples.append((path, label, caption_path, audio_path))
        self.clip_dim = clip_dim
        self.test_mode = test_mode

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label, caption_path, audio_path = self.samples[index]
        clip_feature = np.load(path)
        text_feature = np.load(caption_path)
        audio_feature = np.load(audio_path)
        if self.test_mode:
            clip_feature, clip_length = tools.process_split(clip_feature, self.clip_dim)
            text_feature, _ = tools.process_split(text_feature, self.clip_dim)
            audio_feature, _ = tools.process_split(audio_feature, self.clip_dim)
        else:
            clip_feature, clip_length = tools.process_feat(clip_feature, self.clip_dim)
            text_feature, _ = tools.process_feat(text_feature, self.clip_dim)
            audio_feature, _ = tools.process_feat(audio_feature, self.clip_dim)
        return (
            torch.tensor(clip_feature),
            torch.tensor(text_feature),
            torch.tensor(audio_feature),
            label,
            clip_length,
        )


def load_tmvad_segments(label_root: str, video_ids):
    segments, labels = [], []
    for video_id in video_ids:
        xml_path = Path(label_root) / f'{video_id}.xml'
        video_segments, video_labels = [], []
        if xml_path.exists():
            root = ET.parse(xml_path).getroot()
            for segment in root.findall('segment'):
                start = segment.findtext('start')
                end = segment.findtext('end')
                if start is None or end is None:
                    continue
                type_node = next((node for node in segment if node.tag.startswith('type')), None)
                video_segments.append((int(start), int(end)))
                video_labels.append(type_node.text if type_node is not None else 'abnormal')
        segments.append(video_segments)
        labels.append(video_labels)
    return segments, labels