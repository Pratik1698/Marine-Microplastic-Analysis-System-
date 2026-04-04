import pandas as pd
import os
import shutil

print("🌊 Converting CSV to YOLOv8 Format...")

# 1. Read your uploaded CSV
df = pd.read_csv('_annotations.csv')

# 2. Create the exact folder structure YOLO requires
os.makedirs('yolo_dataset/images/train', exist_ok=True)
os.makedirs('yolo_dataset/labels/train', exist_ok=True)

# 3. Convert coordinates and create .txt files
images_processed = 0
for filename, group in df.groupby('filename'):
    # The CSV luckily contains the exact width and height of every image
    img_w = group['width'].iloc[0]
    img_h = group['height'].iloc[0]
    
    # Create the corresponding .txt file for the image
    txt_filename = filename.rsplit('.', 1)[0] + '.txt'
    txt_filepath = os.path.join('yolo_dataset/labels/train', txt_filename)
    
    with open(txt_filepath, 'w') as f:
        for _, row in group.iterrows():
            # YOLO requires normalized (0.0 to 1.0) center coordinates
            class_id = 0 # 0 means "Microplastic"
            x_center = ((row['xmin'] + row['xmax']) / 2) / img_w
            y_center = ((row['ymin'] + row['ymax']) / 2) / img_h
            box_w = (row['xmax'] - row['xmin']) / img_w
            box_h = (row['ymax'] - row['ymin']) / img_h
            
            # Write the YOLO format string
            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}\n")
            
    # Move the actual image file into the YOLO images folder
    if os.path.exists(filename):
        shutil.copy(filename, os.path.join('yolo_dataset/images/train', filename))
        images_processed += 1

# 4. Create the YAML configuration file that tells YOLO how to learn
yaml_content = """
path: ./yolo_dataset  # dataset root dir
train: images/train  # train images (relative to 'path')
val: images/train    # using train for validation just for hackathon prototyping

# Classes
names:
  0: Microplastic
"""
with open('microplastics.yaml', 'w') as f:
    f.write(yaml_content)

print(f"\n🎉 SUCCESS! Converted annotations for {images_processed} images.")
print("Created 'microplastics.yaml' configuration file.")
print("You are now ready to train YOLOv8!")