import os
import torch
import torchvision.transforms as transforms
from xml.etree import ElementTree as ET
from PIL import Image
import albumentations as A
import cv2

# Function to parse the Pascal VOC XML annotation
def parse_voc_annotation(xml_file):

    tree = ET.parse(xml_file)
    root = tree.getroot()
    boxes = []
    labels = []
    
    for obj in root.findall('object'):
        label = obj.find('name').text
        bbox = obj.find('bndbox')
        xmin = float(bbox.find('xmin').text)
        ymin = float(bbox.find('ymin').text)
        xmax = float(bbox.find('xmax').text)
        ymax = float(bbox.find('ymax').text)
        boxes.append([xmin, ymin, xmax, ymax])
        labels.append(label)
    
    return boxes, labels

# Function to save the augmented annotations in Pascal VOC format
def save_voc_annotation(xml_file, boxes, labels, save_path, augmented_image_size):
    
    tree = ET.parse(xml_file)
    root = tree.getroot()

    size_elem = root.find('size')
    width_elem = size_elem.find('width')
    height_elem = size_elem.find('height')
    width_elem.text = str(augmented_image_size[1])
    height_elem.text = str(augmented_image_size[0])

    for obj in root.findall('object'):
        root.remove(obj)
    
    for box, label in zip(boxes, labels):
        xmin, ymin, xmax, ymax = box
        obj_elem = ET.Element('object')
        
        name_elem = ET.Element('name')
        name_elem.text = label
        obj_elem.append(name_elem)
        
        bbox_elem = ET.Element('bndbox')
        xmin_elem = ET.Element('xmin')
        ymin_elem = ET.Element('ymin')
        xmax_elem = ET.Element('xmax')
        ymax_elem = ET.Element('ymax')
        
        xmin_elem.text = str(int(xmin))
        ymin_elem.text = str(int(ymin))
        xmax_elem.text = str(int(xmax))
        ymax_elem.text = str(int(ymax))
        
        bbox_elem.append(xmin_elem)
        bbox_elem.append(ymin_elem)
        bbox_elem.append(xmax_elem)
        bbox_elem.append(ymax_elem)
        obj_elem.append(bbox_elem)
        
        root.append(obj_elem)
    
    tree.write(os.path.join(save_path, "augmented_"+os.path.basename(xml_file)))

# Function to apply augmentation and save the augmented images
def apply_augmentation(image_path, xml_path, save_path):
    transform = A.Compose([
        A.ColorJitter(p=0.5, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        A.augmentations.geometric.rotate.RandomRotate90(p=0.5),
        A.augmentations.geometric.transforms.ShiftScaleRotate(p=0.5),
        A.augmentations.geometric.transforms.Flip(p=0.5),
        A.augmentations.geometric.resize.RandomScale(p=0.5, scale_limit=(-0.2, 1.5)),
        A.augmentations.crops.transforms.BBoxSafeRandomCrop(p=0.3),
        # Add any other augmentation techniques you want to use.
    ], bbox_params=A.BboxParams(format='pascal_voc',  min_area=500, min_visibility=0.1, label_fields=['labels']))

    image = cv2.imread(image_path)
    boxes, labels = parse_voc_annotation(xml_path)

    # Convert box format to Albumentations format
    bboxes = [[box[0], box[1], box[2], box[3]] for box in boxes]
    
    # Apply augmentation
    augmented = transform(image=image, bboxes=bboxes, labels=labels)
    augmented_image = augmented['image']
    augmented_bboxes = augmented['bboxes']
    augmented_labels = augmented['labels']

    # Convert bounding boxes back to Pascal VOC format
    augmented_boxes = [[box[0], box[1], box[0]+box[2], box[1]+box[3]] for box in augmented_bboxes]

    image_dir = os.path.join(save_path, 'images')
    label_dir = os.path.join(save_path, 'labels')

    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    if not os.path.exists(label_dir):
        os.makedirs(label_dir)

    # Save the augmented image and annotation
    image_name = os.path.basename(image_path)
    augmented_image_path = os.path.join(image_dir, f"augmented_{image_name}")
    augmented_image_size = augmented_image.shape[:2]

    cv2.imwrite(augmented_image_path, augmented_image)

    save_voc_annotation(xml_path, augmented_boxes, augmented_labels, label_dir, augmented_image_size)

def apply_augmentation_dir(image_dir, xml_dir, save_dir):
    # Get the list of images and annotations
    image_list = os.listdir(image_dir)
    xml_list = os.listdir(xml_dir)

    # Sort the lists to ensure the order of images and annotations is the same
    image_list.sort()
    xml_list.sort()

    # Apply augmentation to each image and annotation
    for image_name, xml_name in zip(image_list, xml_list):
        image_path = os.path.join(image_dir, image_name)
        xml_path = os.path.join(xml_dir, xml_name)
        try:
            apply_augmentation(image_path, xml_path, save_dir)
            print(f"Augmented {image_name}")
        except Exception as e:
            print(e)
            print(f"Error in processing {image_name}")

# Example usage:
if __name__ == "__main__":
    # image_path = "path/to/image.jpg"
    # xml_path = "path/to/annotation.xml"
    # save_path = "path/to/save/augmented/data"
    
    # apply_augmentation(image_path, xml_path, save_path)

    image_dir = r"D:\NEOM_V2\@dust_cloud-train\imagebb_augment\sample\images"
    xml_dir = r"D:\NEOM_V2\@dust_cloud-train\imagebb_augment\sample\labels"
    save_dir = r"D:\NEOM_V2\@dust_cloud-train\imagebb_augment\sample\augmented"

    apply_augmentation_dir(image_dir, xml_dir, save_dir)
