import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter # SciPy 的信号处理库

print("🧪 正在加载科学计算库 (Pandas + SciPy)...")

# 1. 模拟数据生成 (Data Generation)
# 并在数据中加入随机噪声 (Gaussian Noise)
np.random.seed(42) # 固定随机种子，保证每次运行结果一样
x = np.linspace(0, 20, 200)
true_signal = np.sin(x) + 0.5 * x # 真实信号：正弦波 + 线性趋势
noise = np.random.normal(0, 0.5, size=len(x)) # 高斯噪声
noisy_signal = true_signal + noise

# 2. 使用 Pandas 封装数据 (Data Engineering)
# 这是数据分析的标准操作，用 DataFrame 管理数据
df = pd.DataFrame({
    'Time': x,
    'Noisy_Data': noisy_signal,
    'True_Signal': true_signal
})

# 打印一下 DataFrame 的前几行看看 (就像 SQL 里的 LIMIT 5)
print("-" * 30)
print("📊 Pandas DataFrame 预览:")
print(df.head())
print("-" * 30)

# 3. 使用 SciPy 进行信号处理 (Signal Processing)
# Savitzky-Golay 滤波器：一种平滑数据的强大算法
print("⚙️ 正在调用 SciPy 进行滤波处理...")
# window_length=15 (窗口长度), polyorder=3 (多项式阶数)
df['Filtered_Signal'] = savgol_filter(df['Noisy_Data'], window_length=15, polyorder=3)

# 4. 可视化对比 (Visualization)
print("🎨 正在绘制分析图表...")
plt.figure(figsize=(12, 6))

# 画散点图：带噪声的原始数据
plt.scatter(df['Time'], df['Noisy_Data'], color='lightgray', label='Noisy Input', s=15)

# 画线：真实的信号（理论值）
plt.plot(df['Time'], df['True_Signal'], color='green', linestyle='--', label='True Signal', alpha=0.6)

# 画线：SciPy 修复后的信号
plt.plot(df['Time'], df['Filtered_Signal'], color='red', linewidth=2, label='SciPy Filtered')

plt.title("SciPy Signal Processing Test on M3 iPad", fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)

plt.show()

print("✅ 测试通过！Pandas 和 SciPy 运行正常。")
