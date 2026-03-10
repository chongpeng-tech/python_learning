# Auto Learning Snake (Torch + NumPy)

这个项目是一个用 `torch + numpy` 实现的 DQN 贪吃蛇，支持训练时实时可视化。

## 1) 安装依赖

在项目根目录执行：

```bash
pip install -r codex_project/auto_learning_snake/requirements.txt
```

> 说明：可视化使用的是 Python 自带 `tkinter`，一般无需额外安装。

## 2) 训练（带实时可视化）

```bash
python codex_project/auto_learning_snake/snake_rl.py train --episodes 600 --render
```

可选参数：

- `--render-every 1`：每隔多少个 episode 渲染一次（默认 1）
- `--save dqn_snake.pt`：模型保存路径

## 3) 仅训练（不显示窗口，更快）

```bash
python codex_project/auto_learning_snake/snake_rl.py train --episodes 600
```

## 4) 用训练好的模型演示

```bash
python codex_project/auto_learning_snake/snake_rl.py play --model dqn_snake.pt
```

关闭可视化窗口即可退出演示。

## 训练时你会看到什么

- 实时网格窗口：蛇、食物、当前 episode、分数、epsilon
- 终端日志：每 10 局打印一次 `score / best / reward / eps`
