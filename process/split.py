from PIL import Image
import os

# 输入和输出的文件夹路径
input_dir = "/exp/home/username/data/LEVIR128"
output_dir = "/exp/home/username/data/LEVIR64"

# 新图像的尺寸
new_size = (64,64)

# 遍历train/test/val文件夹
for split in ['train', 'test', 'val']:
    split_dir = os.path.join(input_dir, split)
    output_split_dir = os.path.join(output_dir, split)
    
    # 遍历A/B/label文件夹
    for subfolder in ['A', 'B', 'label']:
        subfolder_dir = os.path.join(split_dir, subfolder)
        output_subfolder_dir = os.path.join(output_split_dir, subfolder)
        
        # 创建输出文件夹
        os.makedirs(output_subfolder_dir, exist_ok=True)
        
        # 遍历文件夹中的图像文件
        for filename in os.listdir(subfolder_dir):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                # 打开图像文件
                image_path = os.path.join(subfolder_dir, filename)
                image = Image.open(image_path)
                
                # 将图像剪切为四份
                width, height = image.size
                cropped_images = []
                for i in range(2):
                    for j in range(2):
                        left = i * width // 2
                        top = j * height // 2
                        right = (i + 1) * width // 2
                        bottom = (j + 1) * height // 2
                        cropped_image = image.crop((left, top, right, bottom))
                        cropped_images.append(cropped_image)
                
                # 保存剪切后的图像
                for idx, cropped_image in enumerate(cropped_images):
                    output_filename = filename.split('.')[0] + f"_{idx+1}.jpg"  # 添加后缀区分剪切后的图像
                    output_path = os.path.join(output_subfolder_dir, output_filename)
                    cropped_image.save(output_path)

print("Image resizing completed.")
