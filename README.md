# 内容
## NN_Regression_pytorch
ニューラルネットワークを用いた他クラス分類
pytorchを使用

## NN_Regression_optuna
Optuna(最適化ライブラリ)を用いたハイパーパラメータ探索


# Dataset
Dry Bean Dataset（乾燥豆データセット）

• サンプル数（行数） 13,611粒（欠損値や重複データはなし）。
• 特徴量（説明変数） 16個の数値データ（12個の次元・サイズ指標と、4個の形状指標）。
• 目的変数（Class） 7種類の豆の品種名（文字列データ）。

## 分類クラス
• Seker（丸みを帯びた品種）
• Barbunya（うずら豆に近い品種）
• Bombay（非常に大粒で他と区別しやすい品種）
• Cali（Barbunyaと特徴が似ており誤分類されやすい品種）
• Horoz（細長い形状の品種）
• Sira（Dermasonと特徴が似ている品種）
• Dermason（最もサンプル数が多い標準的な品種）

## 特徴量feature

|カテゴリ|	特徴量名（英語）|	概要・意味
|---|---|---|
|サイズ・次元（12個）|	Area, Perimeter|	豆の領域の面積（ピクセル数）と外周の長さ|
|	|MajorAxisLength, MinorAxisLength|	豆にフィットさせた楕円の長軸と短軸の長さ|
|	|AspectRation, Eccentricity|	アスペクト比（長軸÷短軸）と離心率（楕円の扁平度）|
|	|ConvexArea, EquivDiameter|	凸包（豆を囲む最小の凸多角形）の面積と、同じ面積を持つ円の直径|
|	|Extent, Solidity|	バウンディングボックス（外接矩形）に対する面積比と、凸包に対する面積比|
|	|roundness, Compactness|	真円度とコンパクトさ（円にどれだけ近いかの指標）|
|形状係数（4個）|	ShapeFactor1 ～ ShapeFactor4|	長軸・短軸・面積などを組み合わせて算出された独自の形状指標|
