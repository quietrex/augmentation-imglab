import os
import imgaug as ia
import imageio
import xml.etree.ElementTree as ET
from imgaug import augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from imgaug.augmentables import Keypoint, KeypointsOnImage
from xml.dom import minidom

def augment():
    rotate = iaa.Affine(rotate=(-25, 25))
    sometimes = lambda aug: iaa.Sometimes(0.5, aug)

    seq = iaa.Sequential(
            [
                iaa.Fliplr(0.5), # horizontally flip 50% of all images
                iaa.Flipud(0.2), # vertically flip 20% of all images

                # crop some of the images by 0-10% of their height/width
                sometimes(iaa.Crop(percent=(0, 0.1))),

                iaa.SomeOf((0, 5),
                    [
                        # Convert some images into their superpixel representation,
                        # sample between 20 and 200 superpixels per image, but do
                        # not replace all superpixels with their average, only
                        # some of them (p_replace).
                        sometimes(
                            iaa.Superpixels(
                                p_replace=(0, 1.0),
                                n_segments=(20, 200)
                            )
                        ),

                        # Blur each image with varying strength using
                        # gaussian blur (sigma between 0 and 3.0),
                        # average/uniform blur (kernel size between 2x2 and 7x7)
                        # median blur (kernel size between 3x3 and 11x11).
                        iaa.OneOf([
                            iaa.GaussianBlur((0, 3.0)),
                            iaa.AverageBlur(k=(2, 7)),
                            iaa.MedianBlur(k=(3, 11)),
                            iaa.MotionBlur(angle=(72, 144)),
                        ]),
                        
                        # Search in some images either for all edges or for
                        # directed edges. These edges are then marked in a black
                        # and white image and overlayed with the original image
                        # using an alpha of 0 to 0.7.
                        sometimes(iaa.OneOf([
                            iaa.EdgeDetect(alpha=(0, 0.5)),
                            iaa.DirectedEdgeDetect(
                                alpha=(0, 0.7), direction=(0.0, 1.0)
                            ),
                        ])),

                        # Add gaussian noise to some images.
                        # In 50% of these cases, the noise is randomly sampled per
                        # channel and pixel.
                        # In the other 50% of all cases it is sampled once per
                        # pixel (i.e. brightness change).
                        iaa.AdditiveGaussianNoise(
                            loc=0, scale=(0.0, 0.05*255), per_channel=0.5
                        ),

                        # Either drop randomly 1 to 10% of all pixels (i.e. set
                        # them to black) or drop them on an image with 2-5% percent
                        # of the original size, leading to large dropped
                        # rectangles.
                        iaa.OneOf([
                            iaa.Dropout((0.01, 0.1), per_channel=0.5),
                            iaa.CoarseDropout(
                                (0.03, 0.15), size_percent=(0.02, 0.05),
                                per_channel=0.2
                            ),
                        ]),

                        # Invert each image's channel with 5% probability.
                        # This sets each pixel value v to 255-v.
                        iaa.Invert(0.05, per_channel=True), # invert color channels

                        # Add a value of -10 to 10 to each pixel.
                        iaa.Add((-10, 10), per_channel=0.5),

                        # Change brightness of images (50-150% of original value).
                        iaa.Multiply((0.5, 1.5), per_channel=0.5),

                        # Improve or worsen the contrast of images.
                        iaa.LinearContrast((0.5, 2.0), per_channel=0.5),

                        # Sigmoid contrast
                        iaa.SigmoidContrast(gain=(3, 10), cutoff=(0.4, 0.6)),

                        # Convert each image to grayscale and then overlay the
                        # result with the original with random alpha. I.e. remove
                        # colors with varying strengths.
                        iaa.Grayscale(alpha=(0.0, 1.0))
                    ],
                    # do all of the above augmentations in random order
                    random_order=True
                )
            ],
            # do all of the above augmentations in random order
            random_order=True
        )
    return seq

def get_kps_bbs(part, box, image, index):
    # Key points
    kps = KeypointsOnImage([
        Keypoint(x=int(part[0].attributes['x'].value), y=int(part[0].attributes['y'].value)),
        Keypoint(x=int(part[1].attributes['x'].value), y=int(part[1].attributes['y'].value)),
        Keypoint(x=int(part[2].attributes['x'].value), y=int(part[2].attributes['y'].value)),
        Keypoint(x=int(part[3].attributes['x'].value), y=int(part[3].attributes['y'].value)),
    ], shape=image.shape)

    # Bounding box
    top = int(box[index].attributes['top'].value)
    left = int(box[index].attributes['left'].value)
    width = int(box[index].attributes['width'].value)
    height = int(box[index].attributes['height'].value)

    bbs = BoundingBoxesOnImage([
        BoundingBox(x1=left, y1=top, x2=width+left, y2=height+top)
    ], shape=image.shape)
    
    return kps, bbs

def save_xml(image_url, box, part, data, element1):
    s_elem1 = ET.SubElement(element1, 'image')  
    s_elem1.set('file', image_url) 
    
    # subtag to s_elem1
    s_elem2 = ET.SubElement(s_elem1, 'box') 
    s_elem2.set('top', str(box[0]))
    s_elem2.set('left', str(box[1]))
    s_elem2.set('width', str(box[2]))
    s_elem2.set('height', str(box[3]))

    for i in range(0,4):
        s_elem4 = ET.SubElement(s_elem2, 'part') 
        index = 2 * i
        s_elem4.set('name', str(i))
        s_elem4.set('x', str(part[index]))
        s_elem4.set('y', str(part[index+1]))

    b_xml = minidom.parseString(ET.tostring(data)).toprettyxml(indent="   ")

    # Opening a file under the name `items.xml`, 
    # with operation mode `wb` (write + binary) 
    output_path = os.path.join("output.xml")
    with open(output_path, "wb") as f: 
        f.write(b_xml.encode('utf-8'))
