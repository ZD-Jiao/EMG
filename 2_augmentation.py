# 对手势数据集emg_training_data.csv进行 数据增强，调训练模型的鲁棒性
import pandas as pd
import numpy as np
import random

input_file = 'emg_training_data.csv'
output_file = 'emg_augmented_data.csv'

scaling_factors = [0.8, 1.2]  # 幅值缩放
window_size = 100             # 每个窗口长度
stride = 50                   # 滑动窗口步长
num_augments = 3              # 每个窗口生成几条增强样本

zero_out_prob = 0.5        # 每条样本有 50% 概率执行通道置零
max_zero_channels = 3      # 最多置零 3 个通道

# === 加载数据 ===
df = pd.read_csv(input_file, sep=',')  # 你的是tab分隔符？
df.columns = df.columns.str.strip()
channel_cols = [col for col in df.columns if col.startswith('Ch')]
label_col = 'label'
time_col = 'Time'
print(df.columns.tolist())

augmented_rows = []

# === 滑窗遍历 ===
for start in range(0, len(df) - window_size + 1, stride):
    window = df.iloc[start:start+window_size].copy()
    label = window[label_col].mode()[0]  # 取窗口中最多的标签

    for _ in range(num_augments):
        augmented = window.copy()

        # --- 幅值缩放 ---
        factor = random.uniform(*scaling_factors)
        augmented[channel_cols] = augmented[channel_cols] * factor

        # --- 通道交换 ---
        if random.random() < 0.5:
            shuffled = channel_cols.copy()
            random.shuffle(shuffled)    # 打乱 shuffled 列表中通道的顺序
            augmented[channel_cols] = augmented[shuffled].values

        # --- 随机通道置零 ---
        if random.random() < zero_out_prob:
            zero_channels = random.sample(channel_cols, k=random.randint(1, max_zero_channels))
            for ch in zero_channels:
                augmented[ch] = 0.0001  # 赋值 0.0 train.py会报错
        
        # --- 时间扰动（可选） ---
        if time_col in df.columns:
            delta = random.uniform(0, 0.01)
            augmented[time_col] += delta

        # --- 统一标签 ---
        augmented[label_col] = label

        augmented_rows.append(augmented)

# === 保存 ===
df_aug = pd.concat(augmented_rows, ignore_index=True)   # augmented_rows 纵向拼接
df_total = pd.concat([df, df_aug], ignore_index=True)

df_total.to_csv(output_file, index=False)
print(f"Augmentation is done, {len(df_aug)} EMG data is added")


