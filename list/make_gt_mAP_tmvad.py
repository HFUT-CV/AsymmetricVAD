import argparse
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd


def read_segments(label_root: Path, video_id: str):
    xml_path = label_root / f'{video_id}.xml'
    if not xml_path.exists():
        return [], []

    root = ET.parse(xml_path).getroot()
    segments, labels = [], []
    for segment in root.findall('segment'):
        start = segment.findtext('start')
        end = segment.findtext('end')
        if start is None or end is None:
            continue
        type_node = next((node for node in segment if node.tag.startswith('type')), None)
        label = type_node.text.strip() if type_node is not None and type_node.text else 'abnormal'
        segments.append([int(start), int(end)])
        labels.append(label)
    return segments, labels


def build_ground_truth(test_list, label_root, output_dir, frames_per_feature=16):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    table = pd.read_csv(test_list)
    frame_ground_truth = []
    segment_ground_truth = []
    label_ground_truth = []

    for row in table.itertuples(index=False):
        feature_path = Path(row.path)
        video_id = feature_path.stem
        segments, labels = read_segments(Path(label_root), video_id)
        feature_count = np.load(feature_path, mmap_mode='r').shape[0]
        frame_labels = np.zeros(feature_count * frames_per_feature, dtype=np.float32)
        for start, end in segments:
            frame_labels[max(0, start):min(frame_labels.size, end)] = 1.0
        frame_ground_truth.extend(frame_labels)
        segment_ground_truth.append(segments)
        label_ground_truth.append(labels)

    np.save(output_dir / 'gt_tmvad.npy', np.asarray(frame_ground_truth, dtype=np.float32))
    np.save(
        output_dir / 'gt_segment_tmvad.npy',
        np.asarray(segment_ground_truth, dtype=object),
        allow_pickle=True,
    )
    np.save(
        output_dir / 'gt_label_tmvad.npy',
        np.asarray(label_ground_truth, dtype=object),
        allow_pickle=True,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create TMVAD ground truth for temporal mAP')
    parser.add_argument('--test-list', default='list/tmvad_test.csv')
    parser.add_argument('--label-root', default='data/labels')
    parser.add_argument('--output-dir', default='list')
    parser.add_argument('--frames-per-feature', default=16, type=int)
    args = parser.parse_args()
    build_ground_truth(args.test_list, args.label_root, args.output_dir, args.frames_per_feature)
