# Numpy Cyber Snake

使用 **Python 3.10 + numpy** 实现的炫酷贪吃蛇。

- 默认使用 `tkinter` 窗口渲染（适合 IDE 直接运行）
- 若 Tk 不可用且在真实终端中运行，会自动 fallback 到 `curses` 终端版

## 安装

```bash
cd /Users/luochongpeng/Code/python_learning/numpy_cyber_snake
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 运行

```bash
python src/main.py
```

## 操作

- `WASD` / 方向键：移动
- `Shift`（窗口版）或 `Space`（终端版）：加速
- `R`：死亡后重开
- `Esc` / `Q`：退出
