import argparse
import cv2
import imgaug as ia
import imageio
import numpy as np
import os
import xml.etree.ElementTree as ET
from imgaug import augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from imgaug.augmentables import Keypoint, KeypointsOnImage
from xml.dom import minidom
from seq import get_kps_bbs, augment, save_xml

def main(args):
    seed = args.seed                        # random seed for imgaug
    num_augment = args.num_augment          # 20
    image_path = args.image_path            # input path that contains all the images
    imglab_xml = args.imglab_xml            # xml files
    data_out = args.data_out                # output path
    
    if not os.path.exists(data_out):
        os.mkdir(data_out)
    
    ia.seed(seed)

    xmldoc = minidom.parse(imglab_xml) # After labeling on imgaug
    itemlist = xmldoc.getElementsByTagName('image')
    data = ET.Element('dataset') 
    element1 = ET.SubElement(data, 'images')
    
    for index, s in enumerate(itemlist):
        image_url = itemlist[index].attributes['file'].value
        newPath = os.path.join(image_path, image_url)                        # if image is found
        imageList = xmldoc.getElementsByTagName('part')                  
        box = xmldoc.getElementsByTagName('box')
        part = box[index].getElementsByTagName('part')
        for j in range(num_augment):
            try:
                image = imageio.imread(newPath)
                kps, bbs = get_kps_bbs(part, box, image, index)
                seq = augment()
                image_aug, kps_aug, bbs_aug = seq(image=image, keypoints=kps, bounding_boxes=bbs)

                for i in range(len(kps.keypoints)):
                    kps_before = kps.keypoints[i]
                    kps_after = kps_aug.keypoints[i]

                for i in range(len(bbs.bounding_boxes)):
                    bbs_before = bbs.bounding_boxes[i]
                    bbs_after = bbs_aug.bounding_boxes[i]

                image_after = bbs_aug.draw_on_image(image_aug, size=5)
                image_after_2 = kps_aug.draw_on_image(image_after, size=5)
                img_url = str(image_url + "_augmented_" + str(j) + ".jpg").replace(".jpg_augmented", "_augmented")
                augmented_img = os.path.join(data_out, img_url)

                cv2.imwrite(augmented_img, cv2.cvtColor(image_aug, cv2.COLOR_RGB2BGR))

                box_top = str(int(bbs_aug.bounding_boxes[0].x1))
                box_left = str(int(bbs_aug.bounding_boxes[0].y1))
                box_width = str(int(bbs_aug.bounding_boxes[0].x2))
                box_height = str(int(bbs_aug.bounding_boxes[0].y2))
                bbs_box = [box_top, box_left, box_width, box_height]
                
                new_keypoints = []
                for keypoint in range(0,4):
                    keypoints_x = int(float(kps_aug.keypoints[keypoint].x))
                    keypoints_y = int(float(kps_aug.keypoints[keypoint].y))
                    new_keypoints.append(keypoints_x)
                    new_keypoints.append(keypoints_y)
                        
                save_xml(img_url, bbs_box, new_keypoints, data, element1)
        
            except FileNotFoundError:
                print(image_url + " is not found.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=234, type=int)                                                                                   # 234
    parser.add_argument('--num_augment', help='Number of augmentation', default=1, type=int)                                               # 20
    parser.add_argument('--image_path', help='Where the images are located.', default='in', type=str)                                      # where the images
    parser.add_argument('--imglab_xml', help='xml that contains images/bounding-box/keypoints', default='output_imglab.xml', type=str)     # output_imglab
    parser.add_argument('--data_out', help='Where the images should be stored', default='out', type=str)                                   # output
    
    main(parser.parse_args())
