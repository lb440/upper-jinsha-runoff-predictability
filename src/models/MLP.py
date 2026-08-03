import argparse
import numpy as np
from typing import Optional
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import warnings
import os
import json
import copy
from model_shared_config import (
    FEATURE_COLS,
    FILE_PATH,
    TARGET_COL,
    TEST_END_DATE,
    TIME_COL,
    TRAIN_END_DATE,
    VAL_END_DATE,
    get_split_indices,
)
from baseline_shared_utils import build_prediction_frame, save_flood_analysis, save_json

warnings.filterwarnings("ignore")

# ==================== 配置 ====================
def parse_args():
    parser = argparse.ArgumentParser(description="Run the MLP model with a fixed 90-day window.")
    parser.add_argument("--horizon", type=int, choices=[1, 3, 7, 15], default=15)
    parser.add_argument("--kim", type=int, default=90)
    parser.add_argument("--random-seed", type=int, default=222)
    parser.add_argument("--compute-device", choices=["cuda", "cpu", "auto"], default="cuda")
    parser.add_argument("--save-path", type=str, default=None)
    parser.add_argument("--param-path", type=str, default=None)
    return parser.parse_args()


ARGS = parse_args()
HORIZON = ARGS.horizon
kim = ARGS.kim
save_path = os.path.normpath(ARGS.save_path or f"C:/MLP-{HORIZON}")
param_path = ARGS.param_path or f"C:/MLP-BO-{HORIZON}/mlp_best_params.json"

os.makedirs(save_path, exist_ok=True)

RANDOM_SEED = ARGS.random_seed
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

resolved_device = ARGS.compute_device
if resolved_device == "auto":
    resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
if resolved_device == "cuda" and not torch.cuda.is_available():
    raise RuntimeError("CUDA was requested for MLP.py but no GPU is available.")

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

device = torch.device(resolved_device)
print(f"使用设备: {device}")

# ==================== 数据参数 ====================
train_end_date = TRAIN_END_DATE
val_end_date = VAL_END_DATE
test_end_date = TEST_END_DATE

max_epochs = 100
patience = 10
min_delta = 1e-4
weight_decay = 1e-4

scheduler_factor = 0.5
scheduler_patience = 5
scheduler_min_lr = 1e-6

# ==================== 读取最优参数 ====================

print("\n读取最优参数...")

with open(param_path, "r", encoding="utf-8") as f:
    best_params = json.load(f)

mlp_hidden = best_params["mlp_hidden"]
batch_size = best_params["batch_size"]
dropout = best_params["dropout"]
lr = best_params["lr"]

print("正式运行参数：")
print(f"mlp_hidden  = {mlp_hidden}")
print(f"batch_size  = {batch_size}")
print(f"dropout     = {dropout}")
print(f"lr          = {lr}")

# ==================== 数据加载 ====================
print("\n加载数据...")

try:
    data = pd.read_csv(FILE_PATH, encoding="utf-8")
except:
    data = pd.read_csv(FILE_PATH, encoding="gbk")

data.columns = data.columns.str.strip()

required_cols = [TIME_COL, TARGET_COL] + FEATURE_COLS
data = data.dropna(subset=required_cols).reset_index(drop=True)

data[TIME_COL] = pd.to_datetime(data[TIME_COL])

data = data[
    (data[TIME_COL] >= "2006-01-01") &
    (data[TIME_COL] <= "2020-12-31")
    ].reset_index(drop=True)

data[TARGET_COL] = data[TARGET_COL].clip(lower=0)

for col in FEATURE_COLS:
    if "Q" in col:
        data[col] = data[col].clip(lower=0)

time = data[TIME_COL]
features = data[FEATURE_COLS].values.astype(np.float32)
runoff = data[TARGET_COL].values.astype(np.float32)

print(f"数据加载完成：{len(data)} 行")

# ==================== 构建滑动窗口 ====================
print("\n构建滑动窗口...")

valid_samples = len(runoff) - kim - HORIZON + 1

X, y, out_idx = [], [], []

for i in range(valid_samples):
    X.append(features[i:i + kim])
    y.append(runoff[i + kim + HORIZON - 1])
    out_idx.append(i + kim + HORIZON - 1)

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.float32).reshape(-1, 1)

data_time = time.iloc[out_idx].reset_index(drop=True)

# ==================== 划分训练集和验证集 ====================
train_idx, val_idx, test_idx = get_split_indices(
    data_time,
    train_end_date,
    val_end_date,
    test_end_date,
)
X_train, y_train = X[train_idx], y[train_idx]
X_val, y_val = X[val_idx], y[val_idx]
X_test, y_test = X[test_idx], y[test_idx]

time_train = data_time.iloc[train_idx].reset_index(drop=True)
time_val = data_time.iloc[val_idx].reset_index(drop=True)
time_test = data_time.iloc[test_idx].reset_index(drop=True)

M_train, M_val, M_test = len(train_idx), len(val_idx), len(test_idx)
print(f"独立测试集：{M_test} 样本")

print(f"训练集：{M_train} 样本")
print(f"验证集：{M_val} 样本")

# ==================== 归一化 ====================
print("\n数据归一化...")

feature_dim = len(FEATURE_COLS)

scalers_x = {}
X_train_norm = np.zeros_like(X_train)
X_val_norm = np.zeros_like(X_val)
X_test_norm = np.zeros_like(X_test)

for i in range(feature_dim):
    scaler = MinMaxScaler()
    scaler.fit(X_train[:, :, i].reshape(-1, 1))

    X_train_norm[:, :, i] = scaler.transform(
        X_train[:, :, i].reshape(-1, 1)
    ).reshape(M_train, kim)

    X_val_norm[:, :, i] = scaler.transform(
        X_val[:, :, i].reshape(-1, 1)
    ).reshape(M_val, kim)

    X_test_norm[:, :, i] = scaler.transform(
        X_test[:, :, i].reshape(-1, 1)
    ).reshape(M_test, kim)

    scalers_x[i] = scaler

y_scaler = MinMaxScaler()
y_train_norm = y_scaler.fit_transform(y_train)
y_val_norm = y_scaler.transform(y_val)
y_test_norm = y_scaler.transform(y_test)


def serialize_minmax_scaler(scaler):
    return {
        "feature_range": tuple(scaler.feature_range),
        "min_": scaler.min_.tolist(),
        "scale_": scaler.scale_.tolist(),
        "data_min_": scaler.data_min_.tolist(),
        "data_max_": scaler.data_max_.tolist(),
        "data_range_": scaler.data_range_.tolist(),
        "n_features_in_": int(scaler.n_features_in_),
        "n_samples_seen_": int(scaler.n_samples_seen_)
    }

X_train_tensor = torch.FloatTensor(X_train_norm).to(device)
y_train_tensor = torch.FloatTensor(y_train_norm).to(device)
X_val_tensor = torch.FloatTensor(X_val_norm).to(device)
y_val_tensor = torch.FloatTensor(y_val_norm).to(device)
X_test_tensor = torch.FloatTensor(X_test_norm).to(device)
y_test_tensor = torch.FloatTensor(y_test_norm).to(device)

train_loader = DataLoader(
    TensorDataset(X_train_tensor, y_train_tensor),
    batch_size=batch_size,
    shuffle=True,
    generator=torch.Generator().manual_seed(RANDOM_SEED)
)


# ==================== MLP模型 ====================
class MLPModel(nn.Module):
    def __init__(self, input_dim, seq_len, hidden_dim, dropout):
        super().__init__()

        flattened_dim = input_dim * seq_len
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),   # 直接输出
        )

    def forward(self, x):
        return self.net(x)


# ==================== 评价指标 ====================
def calculate_nse(true_vals, pred_vals):
    true_vals = np.array(true_vals)
    pred_vals = np.array(pred_vals)

    denominator = np.sum((true_vals - np.mean(true_vals)) ** 2)

    if denominator == 0:
        return np.nan

    numerator = np.sum((true_vals - pred_vals) ** 2)
    return 1 - numerator / denominator


# 新增KGE指标计算
def calculate_kge(true_vals, pred_vals):
    true_vals = np.array(true_vals)
    pred_vals = np.array(pred_vals)

    mean_true = np.mean(true_vals)
    mean_pred = np.mean(pred_vals)
    std_true = np.std(true_vals)
    std_pred = np.std(pred_vals)

    # 皮尔逊相关系数
    r = np.corrcoef(true_vals, pred_vals)[0, 1]
    # 标准差比
    alpha = std_pred / (std_true + 1e-8)
    # 均值比
    beta = mean_pred / (mean_true + 1e-8)

    kge = 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)
    return kge


def calculate_metrics(true_vals, pred_vals):
    true_vals = np.array(true_vals)
    pred_vals = np.array(pred_vals)
    eps = 1e-8  # 防止除零

    mse = mean_squared_error(true_vals, pred_vals)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(true_vals, pred_vals)
    r2 = r2_score(true_vals, pred_vals)
    nse = calculate_nse(true_vals, pred_vals)

    # 新增水文指标
    bias = np.mean(pred_vals - true_vals)  # 偏差
    mape = np.mean(np.abs((pred_vals - true_vals) / (true_vals + eps))) * 100  # 平均绝对百分比误差
    kge = calculate_kge(true_vals, pred_vals)  # Kling-Gupta效率系数
    # 皮尔逊相关系数
    r = np.corrcoef(true_vals, pred_vals)[0, 1]

    return {
        "MSE": float(mse),
        "RMSE": float(rmse),
        "MAE": float(mae),
        "R2": float(r2),
        "R": float(r),
        "NSE": float(nse),
        "MAPE(%)": float(mape),
        "Bias": float(bias),
        "KGE": float(kge)
    }


# ==================== 正式训练 ====================
print("\n开始正式训练...")

model = MLPModel(
    input_dim=len(FEATURE_COLS),
    seq_len=kim,
    hidden_dim=mlp_hidden,
    dropout=dropout
).to(device)

criterion = nn.MSELoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=lr,
    weight_decay=weight_decay
)

scheduler = ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=scheduler_factor,
    patience=scheduler_patience,
    min_lr=scheduler_min_lr,
    verbose=True
)

best_val_loss = float("inf")
best_state = copy.deepcopy(model.state_dict())
no_improve = 0

train_losses = []
val_losses = []

for epoch in range(max_epochs):
    model.train()
    running_loss = 0.0

    for x_batch, y_batch in train_loader:
        pred = model(x_batch)
        loss = criterion(pred, y_batch)

        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        running_loss += loss.item() * x_batch.shape[0]

    train_loss = running_loss / M_train

    model.eval()
    with torch.no_grad():
        val_pred = model(X_val_tensor)
        val_loss = criterion(val_pred, y_val_tensor).item()

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    scheduler.step(val_loss)

    if val_loss < best_val_loss - min_delta:
        best_val_loss = val_loss
        best_state = copy.deepcopy(model.state_dict())
        no_improve = 0
    else:
        no_improve += 1

    print(
        f"Epoch [{epoch + 1}/{max_epochs}] "
        f"Train Loss: {train_loss:.6f} | "
        f"Val Loss: {val_loss:.6f}"
    )

    if no_improve >= patience:
        print(f"\n早停触发：第 {epoch + 1} 轮")
        break

# ==================== 加载最佳模型 ====================
model.load_state_dict(best_state)
model.eval()

# ==================== 训练集预测 ====================
print("\n进行训练集预测...")
with torch.no_grad():
    pred_train_norm = model(X_train_tensor).cpu().numpy()

y_train_pred = y_scaler.inverse_transform(pred_train_norm).flatten()
y_train_true = y_train.flatten()
y_train_pred = np.clip(y_train_pred, 0, None)

train_metrics = calculate_metrics(y_train_true, y_train_pred)

print("\n训练集评价指标：")
for k, v in train_metrics.items():
    print(f"{k}: {v:.4f}")

# ==================== 验证集预测 ====================
print("\n进行验证集预测...")

with torch.no_grad():
    pred_val_norm = model(X_val_tensor).cpu().numpy()

y_val_pred = y_scaler.inverse_transform(pred_val_norm).flatten()
y_val_true = y_val.flatten()

# 负值修正
y_val_pred = np.clip(y_val_pred, 0, None)

# ==================== 计算指标 ====================
val_metrics = calculate_metrics(y_val_true, y_val_pred)

with torch.no_grad():
    pred_test_norm = model(X_test_tensor).cpu().numpy()

y_test_pred = y_scaler.inverse_transform(pred_test_norm).flatten()
y_test_true = y_test.flatten()
y_test_pred = np.clip(y_test_pred, 0, None)

test_metrics = calculate_metrics(y_test_true, y_test_pred)

print("\n验证集评价指标：")
for k, v in val_metrics.items():
    print(f"{k}: {v:.4f}")

print("\n独立测试集评价指标：")
for k, v in test_metrics.items():
    print(f"{k}: {v:.4f}")

# ==================== 保存模型 ====================
model_save_path = os.path.join(save_path, "best_mlp_model.pth")

torch.save({
    "model_state_dict": model.state_dict(),
    "model_type": "MLP",
    "input_dim": len(FEATURE_COLS),
    "hidden_dim": mlp_hidden,
    "dropout": dropout,
    "kim": kim,
    "horizon": HORIZON,
    "feature_cols": FEATURE_COLS,
    "target_col": TARGET_COL,
    "x_scalers": {
        FEATURE_COLS[i]: serialize_minmax_scaler(scalers_x[i])
        for i in range(feature_dim)
    },
    "y_scaler": serialize_minmax_scaler(y_scaler),
    "val_metrics": val_metrics,
    "test_metrics": test_metrics,
    "metrics": test_metrics
}, model_save_path)

print(f"\n模型已保存至: {model_save_path}")

# ==================== 保存预测结果 ====================
pred_df = pd.DataFrame({
    "Date": time_test,
    "True_Q_shigu": y_test_true,
    "Pred_Q_shigu": y_test_pred,
    "Error": y_test_pred - y_test_true
})

pred_csv_path = os.path.join(save_path, "prediction_test.csv")
pred_df.to_csv(pred_csv_path, index=False, encoding="utf-8-sig")

print(f"预测结果已保存至: {pred_csv_path}")

# ==================== 保存指标（TXT文件） ====================
metrics_path = os.path.join(save_path, "metrics.txt")
with open(metrics_path, "w", encoding="utf-8") as f:
    keys_to_write = ["R", "NSE", "RMSE", "MAE", "MAPE(%)", "Bias", "KGE"]
    f.write("训练集指标：\n")
    for k in keys_to_write:
        f.write(f"{k}: {train_metrics[k]:.4f}\n")
    f.write("\n验证集指标：\n")
    for k in keys_to_write:
        f.write(f"{k}: {val_metrics[k]:.4f}\n")
print(f"评价指标已保存至: {metrics_path}")

# ==================== 可视化 ====================
with open(metrics_path, "a", encoding="utf-8") as f:
    f.write("\n独立测试集指标：\n")
    for k in keys_to_write:
        f.write(f"{k}: {test_metrics[k]:.4f}\n")

validation_metrics_snapshot = dict(val_metrics)
time_val = time_test
y_val_true = y_test_true
y_val_pred = y_test_pred
val_metrics = test_metrics

plt.rcParams["font.sans-serif"] = ["SimSun"]
plt.rcParams["axes.unicode_minus"] = False

# 训练损失曲线
fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(train_losses, label="训练损失")
ax.plot(val_losses, label="验证损失")

ax.set_xlabel("Epoch")
ax.set_ylabel("MSE Loss")
ax.set_title("正式模型训练过程")
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(save_path, "train_val_loss.png"), dpi=300)
plt.close()

# 独立测试集预测对比图
fig, ax = plt.subplots(figsize=(14, 5))

ax.plot(time_val, y_val_true, label="实测值", linewidth=1.5)
ax.plot(time_val, y_val_pred, label="预测值", linewidth=1.5)

ax.set_xlabel("日期")
ax.set_ylabel("Q_shigu")
ax.set_title(f"MLP独立测试集预测结果：Horizon={HORIZON}天，NSE={val_metrics['NSE']:.4f}")
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(save_path, "test_prediction.png"), dpi=300)
plt.close()

prediction_df = build_prediction_frame(time_test, y_test_true, y_test_pred)
flood_save_dir = save_flood_analysis(
    output_dir=save_path,
    prediction_df=prediction_df,
    horizon=HORIZON,
    dir_name=f"flood_analysis_{HORIZON}d",
    run_label="MLP",
)
save_json(
    os.path.join(save_path, "mlp_run_metadata.json"),
    {
        "model_name": "MLP",
        "run_type": "main",
        "random_seed": RANDOM_SEED,
        "compute_device": device.type,
        "kim": kim,
        "horizon": HORIZON,
        "feature_cols": FEATURE_COLS,
        "param_path": param_path,
        "save_path": save_path,
        "train_end_date": str(train_end_date.date()),
        "val_end_date": str(val_end_date.date()),
        "test_end_date": str(test_end_date.date()),
        "train_metrics": train_metrics,
        "val_metrics": validation_metrics_snapshot,
        "test_metrics": test_metrics,
        "flood_analysis_dir": str(flood_save_dir),
    },
)
print(f"Updated flood-season analysis saved to: {flood_save_dir}")
raise SystemExit(0)

print("训练损失曲线已保存")
print("独立测试集预测图已保存")

print("\n正式运行完成！")

# ==================== 汛期洪水性能分析（与Transformer完全一致） ====================
print("\n" + "=" * 60)
print("汛期洪水性能分析（测试集按年分析）")
print("=" * 60)

# 汛期定义（与Transformer相同）
FLOOD_SEASON_MONTHS = [6, 7, 8, 9, 10]

# 构建测试集 DataFrame（使用 MLP 已存在的变量）
test_analysis_df = pd.DataFrame({
    "Date": time_test,
    "True": y_test_true,
    "Pred": y_test_pred
})
test_analysis_df["Year"] = test_analysis_df["Date"].dt.year
test_analysis_df["Month"] = test_analysis_df["Date"].dt.month

# 获取汛期数据
flood_df = test_analysis_df[test_analysis_df["Month"].isin(FLOOD_SEASON_MONTHS)].copy()
flood_years = sorted(flood_df["Year"].unique())

print(f"汛期月份：{FLOOD_SEASON_MONTHS}")
print(f"涉及年份：{flood_years}")

# ==================== 洪水性能指标计算函数（与Transformer完全相同） ====================
def calculate_flood_metrics(true_vals: np.ndarray, pred_vals: np.ndarray, dates: pd.Series) -> dict:
    true = np.array(true_vals)
    pred = np.array(pred_vals)

    # 基础指标
    r = np.corrcoef(true, pred)[0, 1] if len(true) > 1 else np.nan
    rmse = np.sqrt(mean_squared_error(true, pred))
    mae = mean_absolute_error(true, pred)
    mape = np.mean(np.abs((true - pred) / (true + 1e-8))) * 100
    bias = np.mean(pred - true)

    ss_res = np.sum((true - pred) ** 2)
    ss_tot = np.sum((true - np.mean(true)) ** 2)
    nse = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    alpha_kge = np.std(pred) / (np.std(true) + 1e-8)
    beta_kge = np.mean(pred) / (np.mean(true) + 1e-8)
    kge = 1 - np.sqrt((r - 1) ** 2 + (alpha_kge - 1) ** 2 + (beta_kge - 1) ** 2)

    # 洪峰指标
    peak_true_idx = np.argmax(true)
    peak_pred_idx = np.argmax(pred)
    peak_true_val = true[peak_true_idx]
    peak_pred_val = pred[peak_pred_idx]
    peak_abs_error = peak_pred_val - peak_true_val
    peak_rel_error = (peak_pred_val - peak_true_val) / (peak_true_val + 1e-8) * 100

    # 峰现时间
    dates_array = dates.reset_index(drop=True)
    peak_true_date = dates_array.iloc[peak_true_idx]
    peak_pred_date = dates_array.iloc[peak_pred_idx]
    peak_time_error = (peak_pred_date - peak_true_date).days

    # 洪量
    volume_true = np.sum(true)
    volume_pred = np.sum(pred)
    volume_rel_error = (volume_pred - volume_true) / (volume_true + 1e-8) * 100

    # 洪峰合格判定
    peak_qualified = abs(peak_rel_error) <= 20.0

    return {
        "NSE": nse,
        "KGE": kge,
        "R": r,
        "RMSE": rmse,
        "MAE": mae,
        "MAPE(%)": mape,
        "Bias": bias,
        "洪峰实测值(m³/s)": peak_true_val,
        "洪峰预测值(m³/s)": peak_pred_val,
        "洪峰绝对误差(m³/s)": peak_abs_error,
        "洪峰相对误差(%)": peak_rel_error,
        "实测峰现日期": str(peak_true_date.date()),
        "预测峰现日期": str(peak_pred_date.date()),
        "峰现时间误差(天)": peak_time_error,
        "实测汛期总量(m³/s·d)": volume_true,
        "预测汛期总量(m³/s·d)": volume_pred,
        "洪量相对误差(%)": volume_rel_error,
        "洪峰合格(≤20%)": "合格" if peak_qualified else "不合格"
    }

# ==================== 逐年汛期分析并保存 ====================
flood_save_dir = os.path.join(save_path, f"flood_analysis_{HORIZON}d")
os.makedirs(flood_save_dir, exist_ok=True)

all_flood_metrics = {}

for year in flood_years:
    year_flood = flood_df[flood_df["Year"] == year].copy()

    if len(year_flood) == 0:
        print(f"  {year}年汛期无数据，跳过")
        continue

    true_year = year_flood["True"].values
    pred_year = year_flood["Pred"].values
    dates_year = year_flood["Date"]

    metrics = calculate_flood_metrics(true_year, pred_year, dates_year)
    all_flood_metrics[year] = metrics

    # 保存TXT
    txt_path = os.path.join(flood_save_dir, f"flood_metrics_{year}_H{HORIZON}d.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write(f"MLP 汛期洪水性能分析\n")
        f.write(f"年份：{year}  |  汛期：{FLOOD_SEASON_MONTHS[0]}-{FLOOD_SEASON_MONTHS[-1]}月\n")
        f.write(f"预见期：{HORIZON} 天  |  样本数：{len(year_flood)}\n")
        f.write("=" * 60 + "\n\n")

        f.write("---------- 基础统计指标 ----------\n")
        for k in ["NSE", "KGE", "R", "RMSE", "MAE", "MAPE(%)", "Bias"]:
            f.write(f"  {k}: {metrics[k]:.4f}\n")

        f.write("\n---------- 洪峰指标 ----------\n")
        f.write(f"  洪峰实测值: {metrics['洪峰实测值(m³/s)']:.2f} m³/s\n")
        f.write(f"  洪峰预测值: {metrics['洪峰预测值(m³/s)']:.2f} m³/s\n")
        f.write(f"  洪峰绝对误差: {metrics['洪峰绝对误差(m³/s)']:.2f} m³/s\n")
        f.write(f"  洪峰相对误差: {metrics['洪峰相对误差(%)']:.3f}%\n")
        f.write(f"  洪峰合格判定(≤20%): {metrics['洪峰合格(≤20%)']}\n")

        f.write("\n---------- 峰现时间 ----------\n")
        f.write(f"  实测峰现日期: {metrics['实测峰现日期']}\n")
        f.write(f"  预测峰现日期: {metrics['预测峰现日期']}\n")
        f.write(f"  峰现时间误差: {metrics['峰现时间误差(天)']} 天\n")

        f.write("\n---------- 洪量指标 ----------\n")
        f.write(f"  实测汛期总量: {metrics['实测汛期总量(m³/s·d)']:.2f} m³/s·d\n")
        f.write(f"  预测汛期总量: {metrics['预测汛期总量(m³/s·d)']:.2f} m³/s·d\n")
        f.write(f"  洪量相对误差: {metrics['洪量相对误差(%)']:.2f}%\n")

    print(f"  {year}年汛期：NSE={metrics['NSE']:.4f}, KGE={metrics['KGE']:.4f}, "
          f"洪峰误差={metrics['洪峰相对误差(%)']:.3f}%, "
          f"峰现误差={metrics['峰现时间误差(天)']}天, "
          f"洪量误差={metrics['洪量相对误差(%)']:.2f}%")

    # 保存CSV
    csv_path = os.path.join(flood_save_dir, f"flood_pred_{year}_H{HORIZON}d.csv")
    year_flood_save = year_flood[["Date", "True", "Pred"]].copy()
    year_flood_save.columns = ["Date", "True_Q_shigu", "Pred_Q_shigu"]
    year_flood_save["Error"] = year_flood_save["Pred_Q_shigu"] - year_flood_save["True_Q_shigu"]
    year_flood_save["Rel_Error(%)"] = (
        year_flood_save["Error"] / (year_flood_save["True_Q_shigu"] + 1e-8) * 100
    )
    year_flood_save.to_csv(csv_path, index=False, encoding="utf-8-sig")

    # 逐年绘图
    plt.figure(figsize=(12, 5))
    plt.plot(year_flood["Date"], true_year, label="实测值", linewidth=1.5, color="black")
    plt.plot(year_flood["Date"], pred_year, label="预测值", linewidth=1.5, color="red", alpha=0.85)

    peak_true_idx_local = np.argmax(true_year)
    peak_pred_idx_local = np.argmax(pred_year)
    plt.scatter(dates_year.iloc[peak_true_idx_local], true_year[peak_true_idx_local],
                color="black", s=80, zorder=5, marker="^", label=f"实测洪峰 {true_year[peak_true_idx_local]:.1f}")
    plt.scatter(dates_year.iloc[peak_pred_idx_local], pred_year[peak_pred_idx_local],
                color="red", s=80, zorder=5, marker="v", label=f"预测洪峰 {pred_year[peak_pred_idx_local]:.1f}")

    plt.xlabel("日期")
    plt.ylabel("Q_shigu (m³/s)")
    plt.title(f"{year}年汛期预测对比 (H={HORIZON}d, NSE={metrics['NSE']:.4f}, 洪峰误差={metrics['洪峰相对误差(%)']:.3f}%)")
    plt.legend(loc="upper right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    fig_path = os.path.join(flood_save_dir, f"flood_plot_{year}_H{HORIZON}d.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()

print(f"\n逐年汛期分析结果已保存至: {flood_save_dir}")

# ==================== 汛期汇总指标 ====================
all_flood_true = flood_df["True"].values
all_flood_pred = flood_df["Pred"].values
all_flood_dates = flood_df["Date"]
overall_flood_metrics = calculate_flood_metrics(all_flood_true, all_flood_pred, all_flood_dates)

# 汇总TXT
summary_txt_path = os.path.join(flood_save_dir, f"flood_summary_H{HORIZON}d.txt")
with open(summary_txt_path, "w", encoding="utf-8") as f:
    f.write("=" * 60 + "\n")
    f.write("MLP 汛期洪水性能汇总\n")
    f.write(f"预见期：{HORIZON} 天  |  汛期月份：{FLOOD_SEASON_MONTHS}\n")
    f.write(f"分析年份：{flood_years}\n")
    f.write("=" * 60 + "\n\n")

    f.write("==================== 整体汛期指标 ====================\n")
    for k in ["NSE", "KGE", "R", "RMSE", "MAE", "MAPE(%)", "Bias"]:
        f.write(f"  {k}: {overall_flood_metrics[k]:.4f}\n")
    f.write(f"  洪峰相对误差: {overall_flood_metrics['洪峰相对误差(%)']:.3f}%\n")
    f.write(f"  洪量相对误差: {overall_flood_metrics['洪量相对误差(%)']:.3f}%\n\n")

    f.write("==================== 逐年汛期指标对比 ====================\n")
    f.write(f"{'年份':<6}{'NSE':<10}{'KGE':<10}{'R':<10}{'RMSE':<10}"
            f"{'洪峰误差(%)':<14}{'峰现误差(天)':<14}{'洪量误差(%)':<14}{'洪峰合格':<10}\n")
    f.write("-" * 100 + "\n")
    for year in flood_years:
        m = all_flood_metrics[year]
        f.write(f"{year:<6}{m['NSE']:<10.4f}{m['KGE']:<10.4f}{m['R']:<10.4f}{m['RMSE']:<10.2f}"
                f"{m['洪峰相对误差(%)']:<14.2f}{m['峰现时间误差(天)']:<14d}{m['洪量相对误差(%)']:<14.2f}"
                f"{m['洪峰合格(≤20%)']:<10}\n")

    qualified_count = sum(1 for y in flood_years if all_flood_metrics[y]["洪峰合格(≤20%)"] == "合格")
    f.write(f"\n洪峰预报合格率：{qualified_count}/{len(flood_years)} = {qualified_count/len(flood_years)*100:.1f}%\n")

print(f"汛期汇总报告已保存至: {summary_txt_path}")

# ==================== 多年汛期指标对比柱状图 ====================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
years_str = [str(y) for y in flood_years]

# NSE
nse_vals = [all_flood_metrics[y]["NSE"] for y in flood_years]
axes[0, 0].bar(years_str, nse_vals, color="steelblue", edgecolor="black")
axes[0, 0].axhline(y=0.5, color="red", linestyle="--", alpha=0.7, label="NSE=0.5")
axes[0, 0].set_title("逐年汛期 NSE")
axes[0, 0].set_ylabel("NSE")
axes[0, 0].legend()
axes[0, 0].grid(axis="y", alpha=0.3)

# 洪峰误差
peak_errs = [all_flood_metrics[y]["洪峰相对误差(%)"] for y in flood_years]
colors_peak = ["green" if abs(e) <= 20 else "red" for e in peak_errs]
axes[0, 1].bar(years_str, peak_errs, color=colors_peak, edgecolor="black")
axes[0, 1].axhline(y=20, color="red", linestyle="--", alpha=0.7)
axes[0, 1].axhline(y=-20, color="red", linestyle="--", alpha=0.7)
axes[0, 1].set_title("逐年汛期洪峰相对误差")
axes[0, 1].set_ylabel("洪峰相对误差 (%)")
axes[0, 1].grid(axis="y", alpha=0.3)

# 峰现时间
time_errs = [all_flood_metrics[y]["峰现时间误差(天)"] for y in flood_years]
axes[1, 0].bar(years_str, time_errs, color="orange", edgecolor="black")
axes[1, 0].axhline(y=0, color="black", linestyle="-", alpha=0.5)
axes[1, 0].set_title("逐年汛期峰现时间误差")
axes[1, 0].set_ylabel("峰现时间误差 (天)")
axes[1, 0].grid(axis="y", alpha=0.3)

# 洪量误差
vol_errs = [all_flood_metrics[y]["洪量相对误差(%)"] for y in flood_years]
colors_vol = ["green" if abs(e) <= 20 else "red" for e in vol_errs]
axes[1, 1].bar(years_str, vol_errs, color=colors_vol, edgecolor="black")
axes[1, 1].axhline(y=20, color="red", linestyle="--", alpha=0.7)
axes[1, 1].axhline(y=-20, color="red", linestyle="--", alpha=0.7)
axes[1, 1].set_title("逐年汛期洪量相对误差")
axes[1, 1].set_ylabel("洪量相对误差 (%)")
axes[1, 1].grid(axis="y", alpha=0.3)

plt.suptitle(f"MLP 汛期洪水性能对比 (H={HORIZON}d)", fontsize=14)
plt.tight_layout()
summary_fig_path = os.path.join(flood_save_dir, f"flood_summary_plot_H{HORIZON}d.png")
plt.savefig(summary_fig_path, dpi=300)
plt.close()
print(f"汛期汇总对比图已保存至: {summary_fig_path}")

print(f"\n汛期洪水性能分析全部完成！结果保存在: {flood_save_dir}")
