# Anode Defect Detector

This repository contains an automated pipeline to detect **anode cracks** from battery CT scans using two complementary algorithms:

* **PatchCore** for unsupervised anomaly detection
* **RF-DETR** (Region-Free DEtection TRansformer) for supervised object detection

---

## Project Structure

```bash
.
├── .gitignore
├── README.md
├── config.yaml
├── download_and_setup_models.py        # Download pretrained checkpoints
├── example_patchcore.py                # Inference runner for PatchCore
├── example_rf_detr.py                  # Inference runner for RF-DETR
├── install_env.sh                      # Shell script to install dependencies
├── models/                             # Folder to store model weights
│   ├── .gitkeep
│   ├── patchcore/
│   │   └── nnscorer_search_index.faiss
│   │   └── patchcore_params.pkl
│   └── rf_detr/
│       └── checkpoint_best_total.pth
├── requirements.txt                   # Python dependencies
├── src/
│   ├── inference/
│   │   ├── inference_patchcore.py
│   │   └── inference_rf_detr.py
│   └── patchcore_inspection/          # Forked PatchCore implementation
│       ├── bin/
│       ├── models/
│       ├── setup.py, *.sh, requirements.txt
│       └── src/, test/
```

---

## PatchCore Architecture

PatchCore is an unsupervised anomaly detection method:

* Extracts multi-scale features using Resnet50
* Compresses features via coreset sampling
* During inference, compares patch-level embeddings against normal data with nearest-neighbor search (FAISS)
* Outputs anomaly score heatmap + bounding boxes via DBSCAN clustering

![PatchCore Architecture](https://raw.githubusercontent.com/amazon-science/patchcore-inspection/main/images/architecture.png)

---

## RF-DETR Architecture

RF-DETR is a supervised object detection transformer without anchor regions:

* Utilizes a Swin Transformer backbone to extract image features
* Employs a transformer decoder with learnable object queries to detect objects without predefined anchor regions
* Uses a shared MLP head to predict bounding box coordinates and class labels
* Trains end-to-end using set-based Hungarian matching with a combination of classification and localization losses

![RF-DETR Architecture](https://github.com/facebookresearch/detectron2/raw/main/docs/tutorials/assets/detr.png)

---

## Installation & Setup

### Step 1: Clone the repo

```bash
git clone -b deploy https://github.com/RUTILEA/anode-defect-detector.git
cd anode-defect-detector
```

### Step 2: Install the environment

```bash
chmod +x install_env.sh
./install_env.sh
```

### Step 3: Download pre-trained models

```bash
python download_and_setup_models.py
```

You should see the following under `models/`:

```
models/
├── patchcore/
│   ├── nnscorer_search_index.faiss
│   └── patchcore_params.pkl
└── rf_detr/
    └── checkpoint_best_total.pth
```

---

## Dataset Configuration

Update `config.yaml`:

```yaml
output_dir_model_rf_detr: models/rf_detr
output_dir_model_patchcore: models/patchcore
output_inference_dir: results
input_dir: data/raw/input_data
export_ai_csv_files: false

roi_config:
  - x_min: 127
    x_max: 294
    y_min: 448
    y_max: 576
    name: "left"
  - x_min: 733
    x_max: 900
    y_min: 448
    y_max: 576
    name: "right"

rf_detr_checkpoint: checkpoint_best_total.pth
class_label: anode crack
patchcore_threshold: 2.5
rf_detr_threshold: 0.8
resize_size: [168, 128]
filter_index_range:
  min: 215
  max: 222
```

---

## 🔧 Configuration Guide for `config.yaml`

Before running inference, make sure to update the following paths in your `config.yaml` file to match your local directories.

### 🔹 Path to Input Images

This is the path to the folder that contains your battery CT scan images for inference.

```yaml
input_dir : data/raw/input_data        # ← Update this to your own image folder
```

### 🔹 Path to Output Images

This is the output path that contains the inference results.

```yaml
output_inference_dir: results              # ← Update this if you want outputs stored elsewhere
```

---

## Run Inference

### PatchCore

```bash
python example_patchcore.py
```

### RF-DETR

```bash
python example_rf_detr.py
```

---

## Output Directory

```
results/
├── inference_patchcore/
│   ├── anomaly/<battery_id>/<axis>/...
│   └── normal/<battery_id>/<axis>/...
└── inference_rf_detr/
    ├── anomaly/<battery_id>/<axis>/...
    └── normal/<battery_id>/<axis>/...
```

Each folder contains annotated `.png` images and optionally CSVs for detected defects.

---



# アノード欠陥検出器

このリポジトリは、以下の2つのアルゴリズムを使用してバッテリーのCTスキャンから**アノードの亀裂**を検出する自動化パイプラインを提供します：

* **PatchCore**：教師なし異常検出
* **RF-DETR**（Region-Free DEtection TRansformer）：教師ありオブジェクト検出

---

## プロジェクト構成

```bash
.
├── config.yaml                      # 設定ファイル（入力・出力パス、閾値など）
├── install_env.sh                  # 環境構築用スクリプト
├── download_and_setup_models.py   # モデルチェックポイントのダウンロード
├── example_patchcore.py           # PatchCore用推論スクリプト
├── example_rf_detr.py             # RF-DETR用推論スクリプト
├── models/                        # モデル保存用フォルダ
│   ├── patchcore/
│   │   ├── nnscorer_search_index.faiss
│   │   └── patchcore_params.pkl
│   └── rf_detr/
│       └── checkpoint_best_total.pth
├── src/                           # ソースコード
│   ├── inference/                # 推論モジュール
│   └── patchcore_inspection/    # PatchCore実装のフォーク
```

---

## PatchCoreの概要

PatchCoreは教師なしの異常検出手法であり、以下のように動作します：

- ResNet50によるマルチスケール特徴抽出
- コアセットサンプリングによる特徴の圧縮
- 正常データとの類似性をFAISSを用いて推論時に計算
- 異常スコアのヒートマップとDBSCANによるバウンディングボックスを出力

---

## RF-DETRの概要

RF-DETRは、アンカー領域を使用せずにオブジェクトを検出する教師ありトランスフォーマーです：

- Swin Transformerをバックボーンに使用して特徴を抽出
- オブジェクトクエリを用いたトランスフォーマーデコーダで領域を予測
- MLPヘッドでバウンディングボックスとラベルを出力
- ハンガリアンマッチングに基づいた損失でエンドツーエンドに学習

---

## インストール手順

### ステップ 1: クローン

```bash
git clone -b deploy https://github.com/RUTILEA/anode-defect-detector.git
cd anode-defect-detector
```

### ステップ 2: 環境構築

```bash
chmod +x install_env.sh
./install_env.sh
```

### ステップ 3: モデルのダウンロード

```bash
python download_and_setup_models.py
```

`models/` フォルダが以下のようになります：

```
models/
├── patchcore/
│   ├── nnscorer_search_index.faiss
│   └── patchcore_params.pkl
└── rf_detr/
    └── checkpoint_best_total.pth
```

---

## config.yaml の設定

```yaml
output_dir_model_rf_detr: models/rf_detr
output_dir_model_patchcore: models/patchcore
output_inference_dir: results
input_dir: data/raw/input_data
export_ai_csv_files: false

roi_config:
  - x_min: 127
    x_max: 294
    y_min: 448
    y_max: 576
    name: "left"
  - x_min: 733
    x_max: 900
    y_min: 448
    y_max: 576
    name: "right"

rf_detr_checkpoint: checkpoint_best_total.pth
class_label: アノードの亀裂
patchcore_threshold: 2.5
rf_detr_threshold: 0.8
resize_size: [168, 128]
filter_index_range:
  min: 215
  max: 222
```

## 🔧 `config.yaml` の設定ガイド

推論を実行する前に、`config.yaml` ファイル内の以下のパスをローカルのディレクトリに合わせて更新してください。

### 🔹 入力画像へのパス

推論に使用するバッテリー CT スキャン画像を保存するフォルダへのパスです。

```yaml
input_dir : data/raw/input_data # ←これを自分の画像フォルダ
``` に更新します。

### 🔹 出力画像へのパス

推論結果を出力するパスです。

```yaml
output_inference_dir: results # ←出力を別の場所に保存したい場合はこれを更新する
```
---


## 推論実行

### PatchCore

```bash
python example_patchcore.py
```

### RF-DETR

```bash
python example_rf_detr.py
```

---

## 出力ディレクトリ構成

```
results/
├── inference_patchcore/
│   ├── anomaly/
│   └── normal/
└── inference_rf_detr/
    ├── anomaly/
    └── normal/
```

---
