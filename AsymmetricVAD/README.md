# AsymmetricVAD

弱监督视频异常检测(Video Anomaly Detection)模型。通过非对称的视觉-文本提示学习、跨模态融合与音频引导增强,实现视频帧级别的异常定位与分类。

## 支持数据集

- **XD-Violence**:https://roc-ng.github.io/XD-Violence/(7 类,含正常/斗殴/枪击/暴乱/虐待/车祸/爆炸)
- **TMVAD**:待补充(视频、字幕、音频多模态,22 类)

## 环境依赖

- Python 3.8+
- PyTorch
- numpy / pandas / scikit-learn

## 目录结构

```
├── src/
│   ├── model.py                 # 主干模型 AsymmetricVAD
│   ├── models/                  # SIE / 跨模态融合 / 回溯控制模块
│   ├── utils/                   # 数据集、损失、工具函数
│   ├── xd_train.py / xd_test.py      # XD-Violence 训练/测试
│   └── tmvad_train.py / tmvad_test.py # TMVAD 训练/测试
└── list/                        # 数据列表与标注生成脚本
```

## 使用

### 1. 准备数据

按 `list/make_list_xd.py` 或 `list/make_list_tmvad.py` 生成训练/测试列表及 GT 标注文件。

### 2. 训练(XD-Violence)

```bash
cd src
python xd_train.py
python tmvad_train.py
```

### 3. 测试(XD-Violence)

```bash
python xd_test.py --model-path model/model_xd.pth
python tmvad_test.py --model-path model/model_tmvad.pth
```


主要超参数可在 `src/xd_option.py` / `src/tmvad_option.py` 中调整(学习率、批次大小、SIE 与回溯机制相关参数等)。

## 评估指标

- XD-Violence:帧级 AUC、AP 以及 mAP@IoU
- TMVAD:帧级 AUC、AP 以及 mAP@IoU
