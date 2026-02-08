import torch
import time

# 矩阵大小保持 6000 (足以让显卡出力)
N = 6000
LOOPS = 20  # 循环跑 20 次，让时间足够长

print(f"🏎️ 准备开始耐久测试 (每轮跑 {LOOPS} 次)...")
print(f"📊 请现在打开 Termius 的 btop 准备观察！")
print("-" * 40)

# --- 1. CPU 耐久跑 ---
print("🐢 CPU 选手准备... (3秒后开始，请盯着 btop 的 CPU 区域)")
time.sleep(3)
print("🏁 CPU 开始起跑！(你会看到 CPU 占用飙升)")

t0 = time.time()
a_cpu = torch.randn(N, N, device="cpu")
b_cpu = torch.randn(N, N, device="cpu")

for i in range(LOOPS):
    c = torch.matmul(a_cpu, b_cpu)
    print(f"  CPU 跑完第 {i + 1}/{LOOPS} 圈...")

print(f"🛑 CPU 休息。耗时: {time.time() - t0:.2f} 秒")
print("-" * 40)

# --- 2. GPU 耐久跑 ---
if torch.backends.mps.is_available():
    device = torch.device("mps")

    print("🚀 GPU (M4) 选手准备... (3秒后开始，请盯着 btop 的 GPU/Proc 区域)")
    time.sleep(3)
    print("🏁 GPU 开始起跑！(你会看到进度条刷得飞快)")

    # 预先加载数据到显存，测试纯计算速度
    a_gpu = torch.randn(N, N, device=device)
    b_gpu = torch.randn(N, N, device=device)

    # 预热一次
    torch.matmul(a_gpu, b_gpu)

    t0 = time.time()
    for i in range(LOOPS):
        c = torch.matmul(a_gpu, b_gpu)
        torch.mps.synchronize()  # 强制同步，确保每一步都算完了
        print(f"  🚀 GPU 跑完第 {i + 1}/{LOOPS} 圈...")

    print(f"🛑 GPU 完赛。耗时: {time.time() - t0:.2f} 秒")
else:
    print("❌ 没检测到 GPU")