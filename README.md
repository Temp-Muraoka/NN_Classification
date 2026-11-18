# Neural Network Classification with PyTorch

## Overview

PyTorchを用いたニューラルネットワークによる多クラス分類の実装です。

Dry Bean Datasetを使用し、全結合ニューラルネットワーク（Multi-Layer Perceptron: MLP）による分類モデルを構築します。

基本的なモデル構築・学習に加え、Optunaによるハイパーパラメータ探索、およびStratified K-Foldによる交差検証を実装しています。

## Objectives

- PyTorchを用いたニューラルネットワークの構築・学習
- 隠れ層数・ユニット数・Dropout率などのモデル構造の変更
- Optunaによるハイパーパラメータ最適化
- Early Stoppingによる過学習の抑制
- Stratified K-Foldによる交差検証を用いたモデル評価

## Dataset

### Dry Bean Dataset

乾燥豆の形状・寸法情報から、豆の品種を分類するデータセットです。

| Item | Description |
|---|---|
| Number of samples | 13,611 |
| Number of features | 16 |
| Number of classes | 7 |
| Feature type | Numerical |
| Target | Bean variety |

### Classification Classes

以下の7種類の豆を分類します。

- Seker
- Barbunya
- Bombay
- Cali
- Horoz
- Sira
- Dermason

### Features

豆の形状・寸法に関する16個の特徴量を使用します。

主な特徴量：

- Area
- Perimeter
- MajorAxisLength
- MinorAxisLength
- AspectRation
- Eccentricity
- ConvexArea
- EquivDiameter
- Extent
- Solidity
- roundness
- Compactness
- ShapeFactor1～4

## Implementations

### 1. Basic Neural Network

File: `NN_Classification_pytorch.py`

PyTorchを用いた全結合ニューラルネットワークによる多クラス分類です。

主な実装内容：

- Neural Networkモデルの定義
- 隠れ層数・各層のユニット数の設定
- ReLUによる活性化
- Dropoutによる正則化
- Cross Entropy Lossによる損失計算
- Adamによるパラメータ最適化
- ミニバッチ学習
- GPU / CPUの実行デバイス切り替え
- 学習曲線の可視化
- Early Stopping

### 2. Hyperparameter Optimization with Optuna

File: `NN_Classification_optuna.py`

Optunaを用いて、ニューラルネットワークのハイパーパラメータを探索します。

主な探索対象：

- 隠れ層数
- 各隠れ層のユニット数
- Dropout率

検証データにおけるLossを目的関数として、ハイパーパラメータを探索します。

### 3. Optuna with Stratified K-Fold Cross Validation

File: `NN_Classification_optuna_Kfold.py`

Stratified K-Foldによる交差検証をOptunaのハイパーパラメータ探索に組み合わせた実装です。

主な実装内容：

- Stratified K-Fold（5分割）
- 各Foldにおけるモデルの学習・評価
- Foldごとの評価Loss・Accuracyの集計
- 交差検証結果に基づくハイパーパラメータ探索

各Foldで訓練データと検証データを分割し、訓練データに対して標準化をFitした後、検証データに適用する構成としています。

### 4. Google Colab

File: `Colab_run.ipynb`

Google Colab上で実行するためのNotebookです。

## Model Architecture

本プロジェクトでは、隠れ層数や各層のユニット数を変更可能な全結合ニューラルネットワークを実装しています。

基本構成：

Input Layer → Hidden Layers（Linear + ReLU + Dropout）→ Output Layer

出力層では7クラスの分類に対応し、Cross Entropy Lossを用いて学習します。

## Preprocessing

- 重複データの確認・削除
- Label Encodingによる目的変数の数値化
- StandardScalerによる特徴量の標準化
- 訓練データ・検証データの分割
- ミニバッチ形式への変換

## Training

- Framework：PyTorch
- Loss Function：Cross Entropy Loss
- Optimizer：Adam
- Activation Function：ReLU
- Regularization：Dropout
- Early Stopping：Validation Lossに基づく学習停止
- Hyperparameter Optimization：Optuna

## Environment

- Python
- PyTorch
- Optuna
- scikit-learn
- pandas
- NumPy
- Matplotlib

環境構築用ファイル：

- `environment.yml`

Condaを利用する場合：

```bash
conda env create -f environment.yml
conda activate my_env
```

## How to Run

リポジトリをクローンします。

```bash
git clone https://github.com/Temp-Muraoka/NN_Classification.git
cd NN_Classification
```

実行例：

```bash
python NN_Classification_pytorch.py
python NN_Classification_optuna.py
python NN_Classification_optuna_Kfold.py
```

データセットは以下のディレクトリに配置してください。

```text
data/
└── Dry_Bean_Dataset/
    └── Dry_Bean_Dataset.xlsx
```

## Evaluation

モデルの評価には、主に以下の指標を使用します。

- Accuracy
- Cross Entropy Loss
- Training / Validation Loss

交差検証では、各Foldの評価結果を集計し、モデル性能を確認します。

## Results

学習結果については、今後以下を整理・掲載予定です。

- 学習曲線（Training / Validation Loss）
- Accuracy
- Optunaによる探索履歴
- 交差検証における各Foldの評価結果

※実際の実行結果を確認したうえで数値を追記します。

## Notes

本リポジトリは個人学習・自主開発を目的としたものです。

独自に定義したニューラルネットワークの構造、学習ループ、評価処理と、PyTorchなどの既存ライブラリを利用した部分を組み合わせて実装しています。

なお、本リポジトリでは独立した最終テストデータによる汎化性能評価は実施していません。記載する評価結果は、検証データまたは交差検証による結果です。