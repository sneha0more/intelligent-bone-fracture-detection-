import json
import os
import argparse

def convert(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[2]) / 2.0
    y = (box[1] + box[3]) / 2.0
    w = box[2] - box[0]
    h = box[3] - box[1]
    return x * dw, y * dh, w * dw, h * dh

def convert_coco_to_yolo(coco_annotation_file, output_dir, image_dir):
    os.makedirs(output_dir, exist_ok=True)

    with open(coco_annotation_file, 'r') as f:
        data = json.load(f)

    images = {img['id']: img for img in data['images']}
    annotations = data['annotations']

    label_map = {cat['id']: idx for idx, cat in enumerate(data['categories'])}

    for ann in annotations:
        img = images[ann['image_id']]
        img_width = img['width']
        img_height = img['height']
        box = ann['bbox']
        x_min, y_min, width, height = box
        x_max = x_min + width
        y_max = y_min + height

        bbox = (x_min, y_min, x_max, y_max)
        yolo_box = convert((img_width, img_height), bbox)

        label_file = os.path.join(output_dir, f"{os.path.splitext(img['file_name'])[0]}.txt")
        with open(label_file, "a") as f_out:
            f_out.write(f"{label_map[ann['category_id']]} {' '.join(map(str, yolo_box))}\n")

    print(f"Conversion completed! YOLO annotations saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert COCO annotation to YOLO format.")
    parser.add_argument('--annotation', required=True, help='Path to COCO annotation JSON')
    parser.add_argument('--output', required=True, help='Output directory for YOLO txt files')
    parser.add_argument('--image-dir', required=True, help='Path to images directory')

    args = parser.parse_args()

    convert_coco_to_yolo(args.annotation, args.output, args.image_dir)
