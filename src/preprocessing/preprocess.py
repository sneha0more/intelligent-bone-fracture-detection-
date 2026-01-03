import os
import cv2
import json
from tqdm import tqdm

# Step 2: Resize all images to target size
def resize_images(input_folder, output_folder, target_size=(512, 512)):
    os.makedirs(output_folder, exist_ok=True)
    print(f"Resizing images from {input_folder} to {target_size}...")
    for img_name in tqdm(os.listdir(input_folder)):
        input_path = os.path.join(input_folder, img_name)
        output_path = os.path.join(output_folder, img_name)

        img = cv2.imread(input_path)
        if img is None:
            print(f"Warning: Skipping unreadable image {img_name}")
            continue

        resized_img = cv2.resize(img, target_size)
        cv2.imwrite(output_path, resized_img)

    print(f"Resized images saved to {output_folder}")


# Step 3: Adjust bounding boxes in COCO annotation files
def update_coco_annotations(json_path, output_path, target_size=(512, 512)):
    with open(json_path, 'r') as f:
        coco = json.load(f)

    print(f"Updating annotation file {json_path} to new image size {target_size}...")

    for img in coco['images']:
        original_w = img['width']
        original_h = img['height']

        scale_x = target_size[0] / original_w
        scale_y = target_size[1] / original_h

        img['width'], img['height'] = target_size

        for ann in [a for a in coco['annotations'] if a['image_id'] == img['id']]:
            ann['bbox'][0] *= scale_x
            ann['bbox'][1] *= scale_y
            ann['bbox'][2] *= scale_x
            ann['bbox'][3] *= scale_y

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(coco, f)

    print(f"Updated annotations saved to {output_path}")


# MAIN
if __name__ == "__main__":
    # Define your paths
    raw_image_folder = "../raw_data/images"
    processed_image_folder = "../processed_data/images"
    target_size = (512, 512)

    train_ann_in = "../raw_data/train_annotations.coco.json"
    valid_ann_in = "../raw_data/valid_annotations.coco.json"

    train_ann_out = "../processed_data/annotations/train_resized.json"
    valid_ann_out = "../processed_data/annotations/valid_resized.json"

    # Step 2: Resize images
    resize_images(raw_image_folder, processed_image_folder, target_size)

    # Step 3: Update annotations
    update_coco_annotations(train_ann_in, train_ann_out, target_size)
    update_coco_annotations(valid_ann_in, valid_ann_out, target_size)

    print("All preprocessing steps completed successfully!")
