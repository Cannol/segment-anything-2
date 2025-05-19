import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
import pandas as pd
import os
from PIL import Image

# 创建一个图形窗口和坐标轴
fig, (ax,ax2) = plt.subplots(1,2)
img_path = "/data/LaSOT/LaSOTTest/LaSOTTest/airplane-13/img"
imgs = os.listdir(img_path)
imgs.sort()
img = Image.open(os.path.join(img_path, imgs[0]))
data = pd.read_csv("data1.csv")
x = data["X"]
y = data["Y"]
# ax.figure(figsize=(100,50))
img = img.resize((100, 100))
aximg = ax2.imshow(img, aspect="auto")

# 创建散点图
scat = ax.scatter(x[0], y[0], c="b", s=5)

# 创建线图
line2 = ax.plot(x[0], y[0])[0]

# 设置坐标轴范围和标签
ax.set(xlim=[min(x), max(x)], ylim=[min(y), max(y)])

# 添加图例
ax.legend()


def update(frame):
    xx = x[:frame]
    yy = y[:frame]

    # 更新散点图
    new_data = np.stack([xx, yy]).T
    # 更新散点图中每个点的位置
    scat.set_offsets(new_data)
    ax.set_title("Frame {}".format(frame+1))
    img = Image.open(os.path.join(img_path, imgs[frame]))
    aximg.set_data(img)

    # 更新线图
    line2.set_xdata(x[:frame])
    line2.set_ydata(y[:frame])

    return (scat, line2)


# 创建动画
# frames为数值表示动画的总帧数，即每次更新参数传入当前帧号
ani = animation.FuncAnimation(fig=fig, func=update, frames=len(x), interval=20)

# 显示图形
plt.show()