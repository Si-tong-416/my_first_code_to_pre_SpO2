# import cv2
# import numpy as np
#
# # 加载图像
# image_path = '123.jpg'  # 替换为你要处理的图片路径
# image = cv2.imread(image_path)
#
# # 检查图像是否加载成功，如果没有则显示错误
# if image is None:
#     print("Error loading image.")
# else:
#     # 将BGR图像转换为RGB通道，因为OpenCV默认是BGR格式
#     bgr_image = image
#     rgb_image = bgr_image[:, :, ::-1]
#
#     # 分离RGB通道
#     b_channel, g_channel, r_channel = cv2.split(rgb_image)
#
#     # 创建一个新的图像来显示每个通道
#     b_channel_vis = cv2.cvtColor(b_channel, cv2.COLOR_GRAY2BGR)
#     g_channel_vis = cv2.cvtColor(g_channel, cv2.COLOR_GRAY2BGR)
#     r_channel_vis = cv2.cvtColor(r_channel, cv2.COLOR_GRAY2BGR)
#
#     # 显示原始图像和每个通道
#     titles = ['Blue Channel', 'Green Channel', 'Red Channel']
#     images = [b_channel_vis, g_channel_vis, r_channel_vis]
#     for i in range(len(images)):
#         cv2.imshow(titles[i], images[i])
#         cv2.waitKey(0)  # 等待用户按键，按任意键继续
# # 关闭所有窗口
# cv2.destroyAllWindows()

import matplotlib.pyplot as plt
from skimage import io
import numpy as np

img = io.imread('123.jpg')

red_channel = img[:, :, 0]
green_channel = img[:, :, 1]
blue_channel = img[:, :, 2]

red_counts =np.bincount(red_channel.ravel(), minlength=256)
green_counts =np.bincount(green_channel.ravel(), minlength=256)
blue_counts = np.bincount(blue_channel.ravel(), minlength=256)

red_percentages = red_counts / float(red_channel.size) * 100
green_percentages = green_counts / float(green_channel.size) * 100
blue_percentages = blue_counts / float(blue_channel.size) * 100

# 绘制链状图
fig,(ax1, ax2, ax3)=plt.subplots(1, 3)

ax1.pie(red_percentages,colors=[(1, 0, 0, i/255)for i in range(256)])
ax1.set_title('Red channel')

ax2.pie(green_percentages,colors=[(0, 1, 0, i/255)for i in range(256)])
ax2.set_title('Green channel')

ax3.pie(blue_percentages,colors=[(0, 0, 1, i/255)for i in range(256)])
ax3.set_title('Blue channel')

plt.show()
