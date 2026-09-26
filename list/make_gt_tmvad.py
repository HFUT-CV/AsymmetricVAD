import argparse
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd


def read_segments(xml_path):
    if not xml_path.exists():
        return [], []
    root = ET.parse(xml_path).getroot()
    segments, labels = [], []
    for node in root.findall('segment'):
        start, end = node.findtext('start'), node.findtext('end')
        if start is None or end is None:
            continue
        type_node = next((child for child in node if child.tag.startswith('type')), None)
        segments.append([int(start), int(end)])
        labels.append(type_node.text if type_node is not None else 'abnormal')
    return segments, labels


def build_ground_truth(test_list, label_root, output_prefix, frames_per_feature=16):
    table = pd.read_csv(test_list)
    frame_labels, all_segments, all_segment_labels = [], [], []
    for row in table.itertuples(index=False):
        video_id = Path(row.path).stem
        segments, labels = read_segments(Path(label_root) / f'{video_id}.xml')
        feature_count = np.load(row.path, mmap_mode='r').shape[0]
        target = np.zeros(feature_count * frames_per_feature, dtype=np.int32)
        for start, end in segments:
            target[max(0, start):min(target.size, end)] = 1
        frame_labels.extend(target)
        all_segments.append(segments)
        all_segment_labels.append(labels)
    output_prefix = Path(output_prefix)
    np.save(output_prefix.with_name('gt_tmvad.npy'), np.asarray(frame_labels))
    np.save(output_prefix.with_name('gt_segment_tmvad.npy'), np.asarray(all_segments, dtype=object), allow_pickle=True)
    np.save(output_prefix.with_name('gt_label_tmvad.npy'), np.asarray(all_segment_labels, dtype=object), allow_pickle=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create TMVAD ground truth arrays')
    parser.add_argument('--test-list', default='list/tmvad_test.csv')
    parser.add_argument('--label-root', default='data/labels')
    parser.add_argument('--output-prefix', default='list/gt_tmvad')
    parser.add_argument('--frames-per-feature', default=16, type=int)
    args = parser.parse_args()
    build_ground_truth(args.test_list, args.label_root, args.output_prefix, args.frames_per_feature)