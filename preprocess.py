import os
from PIL import Image

# 設定目標資料夾
input_folder = 'pixiv_artist_48631'   # 輸入資料夾
output_folder = 'clear_{}'.format(input_folder)  # 輸出資料夾
#output_folder = 'output_dataset'  # 輸出資料夾
os.makedirs(output_folder, exist_ok=True)  # 如果沒有這個資料夾則創建

# 設定長寬比範圍，3:4 大約是 0.75 ~ 0.8
min_ratio = 0.6
max_ratio = 0.9

def crop_image_to_3_4(image):
    width, height = image.size
    
    # 計算3:4比例的裁切區域
    target_ratio = 3 / 4
    if width / height > target_ratio:
        # 寬度過大，裁掉左右
        new_width = int(height * target_ratio)
        left = (width - new_width) // 2
        right = left + new_width
        top = 0
        bottom = height
    else:
        # 高度過大，裁掉上下
        new_height = int(width / target_ratio)
        top = (height - new_height) // 2
        bottom = top + new_height
        left = 0
        right = width

    # 裁切圖片並回傳
    return image.crop((left, top, right, bottom))

def process_images(input_folder, output_folder):
    for filename in os.listdir(input_folder):
        filepath = os.path.join(input_folder, filename)
        
        # 只處理圖片格式
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            try:
                with Image.open(filepath) as img:
                    width, height = img.size
                    aspect_ratio = width / height
                    
                    # 檢查比例是否在範圍內
                    if min_ratio <= aspect_ratio <= max_ratio:
                        print(f"處理圖片: {filename}（比例 {aspect_ratio:.2f}）")
                        
                        # 裁切並儲存新圖片
                        cropped_img = crop_image_to_3_4(img)
                        new_filename = f"cropped_{filename}"
                        new_filepath = os.path.join(output_folder, new_filename)
                        cropped_img.save(new_filepath)
                    else:
                        print(f"跳過圖片: {filename}（比例 {aspect_ratio:.2f}）")
            except Exception as e:
                print(f"處理圖片 {filename} 時出錯: {e}")

process_images(input_folder, output_folder)
