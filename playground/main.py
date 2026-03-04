import cv2
import numpy as np
from pathlib import Path

# 输出目录：放到当前脚本所在目录下的 opencv_demo/
out_dir = Path.cwd() / "opencv_demo"
out_dir.mkdir(parents=True, exist_ok=True)

# 1) 生成一张测试图（不依赖相册/网络）
img = np.zeros((360, 640, 3), dtype=np.uint8)  # 黑底
cv2.putText(img, "Hello OpenCV on Pyto!", (30, 120),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
cv2.circle(img, (480, 220), 60, (255, 0, 0), 3)
cv2.rectangle(img, (40, 180), (260, 320), (0, 200, 255), 3)

# 2) 灰度 + 边缘检测
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 80, 160)

# 3) 保存文件
p1 = out_dir / "original.png"
p2 = out_dir / "gray.png"
p3 = out_dir / "edges.png"

cv2.imwrite(str(p1), img)
cv2.imwrite(str(p2), gray)
cv2.imwrite(str(p3), edges)

print("✅ OpenCV works! Files saved:")
print(" -", p1)
print(" -", p2)
print(" -", p3)

# 4) 再打印一下版本，确认确实是 opencv-python
print("OpenCV version:", cv2.__version__)