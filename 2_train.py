import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
import os
import joblib
from scipy.fft import rfft, rfftfreq
from scipy.signal import welch

# --- 特征提取函数 ---
def extract_features(window, sample_rate=250):
    features = []

    for col in window.columns:
        data = window[col].values

        # --- 时域特征 ---
        features.extend([
            np.mean(data),
            np.std(data),
            np.max(data),
            np.min(data),
            np.median(data),
            np.sum(np.abs(data)),        # MAV
            np.sqrt(np.mean(data**2)),   # RMS
        ])

        # --- 频域特征 ---
        # 使用 Welch 方法估计功率谱密度
        freqs, psd = welch(data, fs=sample_rate, nperseg=len(data))

        # 总能量（PSD 积分）
        total_power = np.trapz(psd, freqs)

        # MNF: Mean Frequency
        mean_freq = np.sum(freqs * psd) / np.sum(psd)

        # MDF: Median Frequency
        cumulative_power = np.cumsum(psd)
        median_freq = freqs[np.where(cumulative_power >= cumulative_power[-1] / 2)[0][0]]

        # Peak Frequency
        peak_freq = freqs[np.argmax(psd)]

        features.extend([
            mean_freq,
            median_freq,
            total_power,
            peak_freq
        ])

    return features

# --- 加载训练数据集 ---
df = pd.read_csv('emg_training_data.csv')
print(df)

# --- 分离通道与标签 ---
feature_columns = [col for col in df.columns if col not in ['Time', 'label']]
X_raw = df[feature_columns]
y = df['label']

# --- 滑动窗口特征提取 ---
window_size = 100   # 例如：100个样本（取决于采样率，如250Hz则为0.4秒）
step_size = 50      # 每次移动50个样本（50%重叠）

X_features = []
y_labels = []

for start in range(0, len(X_raw) - window_size, step_size):
    end = start + window_size
    window = X_raw.iloc[start:end]  # [100 rows x 4 columns]; iloc函数：通过行号来取行数据
    label = y.iloc[start:end].mode()[0]  # 取窗口中最常见的标签
    feats = extract_features(window)
    X_features.append(feats)
    y_labels.append(label)

# --- 构建训练集 ---
X = np.array(X_features)
y = np.array(y_labels)

# --- 划分训练/测试集 ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 训练分类器 ---
clf = RandomForestClassifier(n_estimators=100, random_state=42)     # 随机森林分类

# from xgboost import XGBClassifier
# clf = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)  # XGBoost (极端梯度提升)
#
# from lightgbm import LGBMClassifier
# clf = LGBMClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42) # LightGBM — 更快的 GBDT 替代
#
# from catboost import CatBoostClassifier     # 安装库失败
# clf = CatBoostClassifier(iterations=100, learning_rate=0.1, depth=5, verbose=0, random_state=42)    # CatBoost, 对类别数据非常友好

# from sklearn.neural_network import MLPClassifier
# clf = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=300, random_state=42)    # MLP (多层感知机/神经网络)

clf.fit(X_train, y_train)

# --- 保存训练模型 ---
joblib.dump(clf, 'emg_rf_model.pkl')
print("Trained model is saved as: emg_rf_model.pkl")

# --- 模型评估 ---
y_pred = clf.predict(X_test)

print("分类报告:\n", classification_report(y_test, y_pred))
print("混淆矩阵:\n", confusion_matrix(y_test, y_pred))

# 可选：交叉验证分数
scores = cross_val_score(clf, X, y, cv=5)
print("5折交叉验证平均准确率: {:.2f}%".format(np.mean(scores) * 100))
