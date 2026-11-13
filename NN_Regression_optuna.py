"""
Optunaを用いたハイパーパラメータチューニング
NN_Regression_pytorch.pyのモデルを対象

Colab上での実行を想定
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import time
import copy

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score

from torch.utils.data import DataLoader, TensorDataset

import optuna
import os


# ============================================================
# Setting
# ============================================================
TRIAL_N = 10 #探索回数


# HIDDEN_LAYERS = [32, 16]

# LEARNING_RATE = 0.001

EPOCHS = 100 #Earlystoppingするので、十分大きい数

# DROPOUT_RATE = 0.0

# Early Stopping
PATIENCE = 15       # 検証Lossが改善しないことを許容するエポック数
MIN_DELTA = 0.0     # 改善とみなす最小のLoss減少量



# =====================================================================
# デバイス設定と Google Drive 保存先の設定
# =====================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用中の実行デバイス: {device}")

# Google Drive 上の保存用ディレクトリを作成
save_dir = "/content/drive/MyDrive/optuna_results"
os.makedirs(save_dir, exist_ok=True)

# データベース（.db）と CSV の保存パスを設定
db_path = os.path.join(save_dir, "optuna_study.db")
csv_path = os.path.join(save_dir, "trials_history.csv")


# ============================================================
# Model Definition
# ============================================================

class NeuralNetwork(nn.Module):

    def __init__(self,
                 input_dim,
                 output_dim,
                 hidden_layers,
                 dropout_rate=0.0
                 ):

       
        """
        Args:
            input_dim (int): 入力特徴量の次元数
            hidden_layers (list of int): 各中間層のノード数のリスト (例: [64, 32, 16])
            output_dim (int): 出力次元数
        """

        super().__init__() #親クラス(NN.module))の初期化(必須)


        # Sequential の箱を作る
        self.network = nn.Sequential()
        
        # .append() で順々に追加していく
        prev_dim = input_dim
        for hidden_dim in hidden_layers:
            self.network.append(nn.Linear(prev_dim, hidden_dim))
            self.network.append(nn.ReLU())

            if dropout_rate > 0:
                self.network.append(nn.Dropout(dropout_rate))

            prev_dim = hidden_dim
            
        # 3. 最後に出力層を追加
        self.network.append(nn.Linear(prev_dim, output_dim))
        # self.network.append(nn.Sigmoid()) # 2値分類の場合


    def forward(self, x):
        return self.network(x)


# ==========================================
# データ読み込み・前処理
# ==========================================
# Excel形式で保存されている場合
df = pd.read_excel("./data/Dry_Bean_Dataset/Dry_Bean_Dataset.xlsx")

# 削除対象となる重複行を抽出（デフォルトでは最初の1件を残し、2件目以降がTrueになる）
duplicates = df[df.duplicated()]

# 重複データが存在する場合のみ中身と件数をprint表示
if len(duplicates) > 0:
    print(f"削除対象の重複データ {len(duplicates)} 件を削除します")
    # print(duplicates)
else:
    print("重複データはありませんでした。")

# 重複を削除して更新
df = df.drop_duplicates()


X = df.drop(columns=["Class"])
Y = df["Class"]

# 目的変数を数値化（0～6）
encoder = LabelEncoder()
Y = encoder.fit_transform(Y)

# クラス数
OUTPUT_DIM = len(encoder.classes_)



# NumPy配列に変換
X = X.values.astype(np.float32)
Y = Y.astype(np.int64)


print(f"Number of samples: {len(X)}")
print(f"Number of features: {X.shape[1]}")
print("Classes:", encoder.classes_)
print("Number of classes:", OUTPUT_DIM)


# 訓練データと検証データに分割（8:2）
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y,
    test_size=0.2,
    random_state=42,
    stratify=Y
)


# 標準化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Tensorに変換
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
Y_train_tensor = torch.tensor(Y_train, dtype=torch.long)

X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
Y_test_tensor = torch.tensor(Y_test, dtype=torch.long)

# DataLoader
# 入力、出力をペアにする。画像など型が違ってもペアにできる
train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, Y_test_tensor)

# ミニバッチに分ける（ジェネレータとして使う）。shuffle=True：1epochごとにデータシャッフル
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# ==========================================
# optunaによるハイパーパラメータ最適化
# ==========================================


start_time = time.perf_counter()

print("--- ハイパーパラメータ探索開始 ---")


def objective(trial):

    # --------------------------------------
    # 探索したいハイパーパラメータの設定
    # --------------------------------------
    
    # 中間層の「層数」を決める（例: 1〜4層）
    n_layers = trial.suggest_int('n_layers', 1, 4)
    
    # 各層の「ノード数」をループで決定してリストを作成する
    hidden_dims_triral = []
    prev_dim = 256

    for i in range(n_layers):
        # 層ごとに個別のアディショナルパラメータ（hidden_dim_0, hidden_dim_1...）として定義
        # 例: 16〜256の間で、16の倍数（step=16）や2の累乗で探す
        # 次の層は前の層のサイズ（prev_dim）を超えないように上限を設定
        dim = trial.suggest_int(f'hidden_dim_{i}', 16, prev_dim, step=16)
        hidden_dims_triral.append(dim)
        prev_dim = dim  # 上限を更新
        
    # ドロップアウト率を決める（例: 0.0〜0.5）
    drop_rate_trial = trial.suggest_float('drop_rate', 0.0, 0.5, step=0.1)
    
    # 学習率を決める（対数スケール log=True で 1e-4 〜 1e-2 を探す）
    lr_trial = trial.suggest_float('lr', 1e-4, 1e-2, log=True)

    # ============================================================
    # Initialize Model
    # ============================================================

    model = NeuralNetwork(
                        input_dim=X_train.shape[1],
                        output_dim=OUTPUT_DIM,
                        hidden_layers=hidden_dims_triral,
                        dropout_rate=drop_rate_trial
                        ).to(device)

    # print("\nModel Structure:")
    # print(model)

    # Loss Function and Optimizer

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr_trial)


    # ==========================================
    # 学習ループ
    # ==========================================


    # Early Stopping用の変数
    best_valid_loss = float("inf")
    best_model_state = copy.deepcopy(model.state_dict())
    patience_counter = 0


    for epoch in range(EPOCHS):

        # ----- Training -----
        model.train()

        train_loss = 0.0

        all_predictions = []
        all_targets = []

        for batch_X, batch_y in train_loader:

            # .to(device)
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()

            outputs = model(batch_X)

            loss = criterion(outputs, batch_y)

            loss.backward()

            optimizer.step()

            train_loss += loss.item() * batch_X.size(0)

            predictions = outputs.argmax(dim=1)

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_targets.extend(
                batch_y.cpu().numpy()
            )

        train_loss /= len(train_loader.dataset)

        train_acc = accuracy_score(
            all_targets,
            all_predictions
        )

        # ----- Validation -----
        model.eval()

        valid_loss = 0.0
        all_predictions = []
        all_targets = []

        with torch.no_grad():

            for batch_X, batch_y in test_loader:

                # テストデータも .to(device)
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)

                outputs = model(batch_X)

                loss = criterion(outputs, batch_y)

                valid_loss += loss.item() * batch_X.size(0)

                predictions = outputs.argmax(dim=1)
        
                all_predictions.extend(
                    predictions.cpu().numpy()
                )
        
                all_targets.extend(
                    batch_y.cpu().numpy()
                )
        
            valid_loss /= len(train_loader.dataset)
        
            valid_acc = accuracy_score(
                all_targets,
                all_predictions
            )



        # ----- Early Stopping & Best Metrics Tracking -----
        if valid_loss < best_valid_loss - MIN_DELTA:
            best_valid_loss = valid_loss
            best_epoch = epoch
            best_train_loss = train_loss
            best_train_acc = train_acc
            best_valid_acc = valid_acc
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print(f"Early Stopping: Epoch {epoch + 1}")
            break

    # 各トライアルの最後に、最も良かったepochでのLossとAccを表示
    print(
        f"Trial Best [Epoch {best_epoch + 1}] "
        f"Train Loss: {best_train_loss:.4f} | Valid Loss: {best_valid_loss:.4f} | "
        f"Train Acc: {best_train_acc:.4f} | Valid Acc: {best_valid_acc:.4f}"
    )

    return best_valid_loss

# Optuna Study の作成または既存 DB からの読み込み
study = optuna.create_study(
    study_name="subrutin_nn_tuning",
    # storage=f"sqlite:///{db_path}",  # Google Drive 上の SQLite DB
    # load_if_exists=True,             # ★重要: DBが存在すれば自動で続きから再開
    direction="minimize",            # 目的関数（Lossなど）を最小化する場合
    pruner=optuna.pruners.MedianPruner()  # 枝刈り用（任意）
)

print(f"現在完了している Trial 数: {len(study.trials)}")

# 探索の実行
study.optimize(objective, n_trials=TRIAL_N)

# 完了後に結果を CSV としても Google Drive に出力
df = study.trials_dataframe()
df.to_csv(csv_path, index=False, encoding="utf-8-sig")


print("\nハイパーパラメータチューニング完了！")
print(f"ベスト評価値: {study.best_value:.4f}")
print("ベストパラメータ:")
for key, val in study.best_params.items():
    print(f"  - {key}: {val}")
print(f"履歴保存先: {csv_path}")



elapsed_time = time.perf_counter() - start_time

print(f"処理時間: {elapsed_time:.2f}秒")



# # ==========================================
# # 最も良いパラメータでのモデル構築・再学習と保存
# # ==========================================
# print("\n--- ベストパラメータによるモデルの再学習と保存 ---")

# best_params = study.best_params
# best_n_layers = best_params['n_layers']
# best_hidden_layers = [best_params[f'hidden_dim_{i}'] for i in range(best_n_layers)]
# best_drop_rate = best_params['drop_rate']
# best_lr = best_params['lr']

# # ベストモデルのインスタンス化
# best_model = NeuralNetwork(
#                             input_dim=X_train.shape[1],
#                             output_dim=OUTPUT_DIM,
#                             hidden_layers=best_hidden_layers,
#                             dropout_rate=best_drop_rate
#                             ).to(device)

# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(best_model.parameters(), lr=best_lr)

# # 全訓練データを用いてベストパラメータで再学習（必要に応じてエポック数を調整してください）
# best_model.train()
# for epoch in range(EPOCHS):
#     for batch_X, batch_y in train_loader:
#         batch_X, batch_y = batch_X.to(device), batch_y.to(device)
#         optimizer.zero_grad()
#         outputs = best_model(batch_X)
#         loss = criterion(outputs, batch_y)
#         loss.backward()
#         optimizer.step()
    

# # 状態や前処理スケーラーを含めて保存
# torch.save(
#     {
#         "model_state_dict": best_model.state_dict(),
#         "input_dim": X.shape[1],
#         "hidden_dims": best_hidden_layers,
#         "output_dim": OUTPUT_DIM,
#         "drop_rate": best_drop_rate,
#         "classes": encoder.classes_,
#         "scaler_mean": scaler.mean_,
#         "scaler_scale": scaler.scale_,
#     },
#     "./data/Dry_Bean_Dataset/dry_bean_model.pth"
# )

# print(f"ベストモデルを保存しました")

# elapsed_time = time.perf_counter() - start_time
# print(f"総処理時間: {elapsed_time:.2f}秒")

