import argparse
from pathlib import Path

import pandas as pd
import xml.etree.ElementTree as ET


def build_list(data_root, split, output):
    root = Path(data_root) / split / 'videos'
    rows = []
    for label, folder in (('Abnormal', 'abnormal'), ('Normal', 'normal')):
        for path in sorted((root / folder).glob('*.npy')):
            modality_folder = 'nor' if folder == 'normal' else 'abn'
            caption_path = Path(data_root) / split / 'captions' / modality_folder / path.name
            audio_path = Path(data_root) / split / 'audio' / modality_folder / path.name
            rows.append({'path': str(path.resolve()), 'label': label,
                         'caption_path': str(caption_path.resolve()),
                         'audio_path': str(audio_path.resolve())})
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=['path', 'label', 'caption_path', 'audio_path']).to_csv(output, index=False)


def read_xml_segments(label_root, video_id):
    xml_path = Path(label_root) / f'{video_id}.xml'
    if not xml_path.exists():
        return []
    root = ET.parse(xml_path).getroot()
    segments = []
    for node in root.findall('segment'):
        start = node.findtext('start')
        end = node.findtext('end')
        if start is None or end is None:
            continue
        type_node = next((child for child in node if child.tag.startswith('type')), None)
        label = type_node.text if type_node is not None and type_node.text else 'abnormal'
        segments.append((label, int(start), int(end)))
    return segments


def build_annotation_lists(data_root, label_root, output_dir):
    output_dir = Path(output_dir)
    test_root = Path(data_root) / 'test' / 'videos'
    train_names, test_names = [], []
    for folder in ('abnormal', 'normal'):
        train_names.extend(sorted((Path(data_root) / 'train' / 'videos' / folder).glob('*.npy')))
        test_names.extend(sorted((test_root / folder).glob('*.npy')))

    def relative_names(paths):
        return [path.relative_to(Path(data_root)).as_posix() for path in paths]

    (output_dir / 'Anomaly_Train_tmvad.txt').write_text(
        '\n'.join(relative_names(train_names)) + '\n', encoding='utf-8'
    )
    (output_dir / 'Anomaly_Test_tmvad.txt').write_text(
        '\n'.join(relative_names(test_names)) + '\n', encoding='utf-8'
    )

    annotations = []
    multiclass_annotations = []
    temporal_annotations = []
    for path in test_names:
        video_id = path.stem
        segments = read_xml_segments(label_root, video_id)
        if not segments:
            continue
        flat_segments = ' '.join(f'{start} {end}' for _, start, end in segments)
        annotations.append(f'{video_id} {flat_segments}')
        multiclass_annotations.append(
            f'{video_id} ' + ' '.join(f'{label} {start} {end}' for label, start, end in segments)
        )
        temporal_annotations.append(
            f'{video_id} abnormal ' + ' '.join(f'{start} {end}' for _, start, end in segments)
        )

    (output_dir / 'annotations_tmvad.txt').write_text(
        '\n'.join(annotations) + '\n', encoding='utf-8'
    )
    (output_dir / 'annotations_multiclasses_tmvad.txt').write_text(
        '\n'.join(multiclass_annotations) + '\n', encoding='utf-8'
    )
    (output_dir / 'Temporal_Anomaly_Annotation_tmvad.txt').write_text(
        '\n'.join(temporal_annotations) + '\n', encoding='utf-8'
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create TMVAD AsymmetricVAD CSV lists')
    parser.add_argument('--data-root', default='data/TMVAD')
    parser.add_argument('--train-output', default='list/tmvad_train.csv')
    parser.add_argument('--test-output', default='list/tmvad_test.csv')
    parser.add_argument('--label-root', default='data/labels')
    parser.add_argument('--list-dir', default='list')
    args = parser.parse_args()
    build_list(args.data_root, 'train', args.train_output)
    build_list(args.data_root, 'test', args.test_output)
    build_annotation_lists(args.data_root, args.label_root, args.list_dir)