import os
import random
import shutil
# 這段程式碼會將資料集分割成訓練集和驗證集，並將它們存儲在指定的資料夾中
def split_dataset(source_dir, target_dir='data', val_ratio=0.2):
    random.seed(42)  # 為了讓每次結果一樣

    classes = os.listdir(source_dir)

    for cls in classes:
        cls_path = os.path.join(source_dir, cls)
        if not os.path.isdir(cls_path):
            continue

        images = [f for f in os.listdir(cls_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        random.shuffle(images)

        val_count = int(len(images) * val_ratio)
        val_images = images[:val_count]
        train_images = images[val_count:]

        for split, split_images in [('train', train_images), ('val', val_images)]:
            split_dir = os.path.join(target_dir, split, cls)
            os.makedirs(split_dir, exist_ok=True)

            for img_name in split_images:
                src_path = os.path.join(cls_path, img_name)
                dst_path = os.path.join(split_dir, img_name)
                shutil.copy(src_path, dst_path)

        print(f"Class '{cls}': {len(train_images)} train / {len(val_images)} val")

if __name__ == '__main__':
    split_dataset('all_data', target_dir='data', val_ratio=0.2)
