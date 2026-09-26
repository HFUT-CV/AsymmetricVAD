# AsymmetricVAD

A weakly supervised Video Anomaly Detection model. It achieves frame-level anomaly localization and classification through asymmetric visual-text prompt learning, cross-modal fusion, and audio-guided enhancement.

## Supported Datasets

- **XD-Violence**: https://roc-ng.github.io/XD-Violence/ (7 classes, including normal/fighting/shooting/riot/abuse/car accident/explosion)
- **TMVAD**: To be supplemented (multimodal video, subtitles, and audio, 22 classes)

## Environment Dependencies

- Python 3.8+
- PyTorch
- numpy / pandas / scikit-learn

## Directory Structure

```
├── src/
│   ├── model.py                 # Backbone model AsymmetricVAD
│   ├── models/                  # SIE / cross-modal fusion / backtracking control modules
│   ├── utils/                   # Datasets, losses, utility functions
│   ├── xd_train.py / xd_test.py      # XD-Violence training/testing
│   └── tmvad_train.py / tmvad_test.py # TMVAD training/testing
└── list/                        # Data lists and annotation generation scripts
```

## Usage

### 1. Prepare Data

Generate the training/testing lists and GT annotation files according to `list/make_list_xd.py` or `list/make_list_tmvad.py`.

### 2. Training (XD-Violence)

```bash
cd src
python xd_train.py
python tmvad_train.py
```

### 3. Testing (XD-Violence)

```bash
python xd_test.py --model-path model/model_xd.pth
python tmvad_test.py --model-path model/model_tmvad.pth
```

The main hyperparameters can be adjusted in `src/xd_option.py` / `src/tmvad_option.py` (learning rate, batch size, SIE and backtracking mechanism-related parameters, etc.).

## Evaluation Metrics

- XD-Violence: frame-level AUC, AP, and mAP@IoU
- TMVAD: frame-level AUC, AP, and mAP@IoU
