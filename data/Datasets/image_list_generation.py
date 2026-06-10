import os

base_dir = os.path.dirname(os.path.abspath(__file__))
train_root = os.path.join(base_dir, 'faceImage')
output_list = os.path.join(base_dir, 'faceImage.txt')

# 허용할 확장자 목록
VALID_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff', '.tif')

groups = sorted(os.listdir(train_root))
label_map = {g: i for i, g in enumerate(groups)}
print("Label map:", label_map)

with open(output_list, 'w', encoding='utf-8') as f:
    for group in groups:
        group_path = os.path.join(train_root, group)
        for img_name in os.listdir(group_path):
            if img_name.lower().endswith(VALID_EXTENSIONS):
                rel_path = os.path.join(group, img_name)
                f.write(f'{rel_path}\t{label_map[group]}\n')

print(f'Done: {output_list}')


