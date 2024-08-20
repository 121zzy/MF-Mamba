import cv2
import os

# 定义原始标签图像文件夹和目标灰度图像文件夹
original_folder = '/exp/home/username/data/DSIFN256/test/label'
target_folder = '/exp/home/username/data/DSIFN256/test/label1'

# 确保目标文件夹存在
if not os.path.exists(target_folder):
    os.makedirs(target_folder)

# 遍历原始标签图像文件夹中的图像文件
for filename in os.listdir(original_folder):
    # 检查文件是否为图像文件
    if filename.endswith(('.png', '.jpg', '.jpeg')):
        # 读取原始标签图像
        original_image = cv2.imread(os.path.join(original_folder, filename))
        
        # 将彩色图像转换为灰度图像
        gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        
        # 保存灰度图像到目标文件夹中，保持相同的文件名
        target_filename = os.path.join(target_folder, filename)
        cv2.imwrite(target_filename, gray_image)
