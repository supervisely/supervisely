#!/usr/bin/python
#
# Reads labels as polygons in JSON format and converts them to instance images,
# where each pixel has an ID that represents the ground truth class and the
# individual instance of that class.
#
# The pixel values encode both, class and the individual instance.
# The integer part of a division by 1000 of each ID provides the class ID,
# as described in labels.py. The remainder is the instance ID. If a certain
# annotation describes multiple instances, then the pixels have the regular
# ID of that class.
#
# Example:
# Let's say your labels.py assigns the ID 26 to the class 'car'.
# Then, the individual cars in an image get the IDs 26000, 26001, 26002, ... .
# A group of cars, where our annotators could not identify the individual
# instances anymore, is assigned to the ID 26.
#
# Note that not all classes distinguish instances (see labels.py for a full list).
# The classes without instance annotations are always directly encoded with
# their regular ID, e.g. 11 for 'building'.
#
# Usage: json2instanceImg.py [OPTIONS] <input json> <output image>
# Options:
#   -h   print a little help text
#   -t   use train IDs
#
# Can also be used by including as a module.
#
# Uses the mapping defined in 'labels.py'.
#
# See also createTrainIdInstanceImgs.py to apply the mapping to all annotations in Cityscapes.
#

# python imports
import os
import sys

import numpy as np

# Image processing
# Check if PIL is actually Pillow as expected
try:
    from PIL import PILLOW_VERSION
except:
    print("Please install the module 'Pillow' for image processing, e.g.")
    print("pip install pillow")
    sys.exit(-1)

try:
    import PIL.Image     as Image
    import PIL.ImageDraw as ImageDraw
except:
    print("Failed to import the image processing packages.")
    sys.exit(-1)


from .annotation import Annotation
from .labels import labels, name2label

# Print the information
def printHelp():
    print('{} [OPTIONS] inputJson outputImg'.format(os.path.basename(sys.argv[0])))
    print('')
    print(' Reads labels as polygons in JSON format and converts them to instance images,')
    print(' where each pixel has an ID that represents the ground truth class and the')
    print(' individual instance of that class.')
    print('')
    print(' The pixel values encode both, class and the individual instance.')
    print(' The integer part of a division by 1000 of each ID provides the class ID,')
    print(' as described in labels.py. The remainder is the instance ID. If a certain')
    print(' annotation describes multiple instances, then the pixels have the regular')
    print(' ID of that class.')
    print('')
    print(' Example:')
    print(' Let\'s say your labels.py assigns the ID 26 to the class "car".')
    print(' Then, the individual cars in an image get the IDs 26000, 26001, 26002, ... .')
    print(' A group of cars, where our annotators could not identify the individual')
    print(' instances anymore, is assigned to the ID 26.')
    print('')
    print(' Note that not all classes distinguish instances (see labels.py for a full list).')
    print(' The classes without instance annotations are always directly encoded with')
    print(' their regular ID, e.g. 11 for "building".')
    print('')
    print('Options:')
    print(' -h                 Print this help')
    print(' -t                 Use the "trainIDs" instead of the regular mapping. See "labels.py" for details.')

# Print an error message and quit
def printError(message):
    print('ERROR: {}'.format(message))
    print('')
    print('USAGE:')
    printHelp()
    sys.exit(-1)

# Convert the given annotation to a label image
def createInstanceImage(annotation, encoding):
    # the size of the image
    size = (annotation.imgWidth, annotation.imgHeight)

    # the background
    if encoding == "ids":
        backgroundId = name2label['unlabeled'].id
    elif encoding == "trainIds":
        backgroundId = name2label['unlabeled'].trainId
    else:
        print("Unknown encoding '{}'".format(encoding))
        return None

    # this is the image that we want to create
    instanceImg = Image.new("I", size, backgroundId)

    # a drawer to draw into the image
    drawer = ImageDraw.Draw(instanceImg)

    # a dict where we keep track of the number of instances that
    # we already saw of each class
    nbInstances = {}
    for labelTuple in labels:
        if labelTuple.hasInstances:
            nbInstances[labelTuple.name] = 0

    unique_ids = {}
    num = 1
    # loop over all objects
    for obj in annotation.objects:
        label   = obj.label
        polygon = obj.polygon

        # if the label is not known, but ends with a 'group' (e.g. cargroup)
        # try to remove the s and see if that works
        # also we know that this polygon describes a group
        isGroup = False
        # if label.endswith('group'):
        #     tmp_label = label[:-len('group')]
        # else:
        #     tmp_label = label

        tmp_label = label


        if not tmp_label in name2label:
            printError( "Label '{}' not known.".format(tmp_label) )


        # the label tuple

        labelTuple = name2label[tmp_label]

        # get the class ID
        if encoding == "ids":
            id = labelTuple.id
        elif encoding == "trainIds":
            id = labelTuple.trainId

        # if this label distinguishs between invidudial instances,
        # make the id a instance ID
        if labelTuple.hasInstances and not isGroup:
            id = id * 1000 + nbInstances[tmp_label]
            nbInstances[tmp_label] += 1

        # If the ID is negative that polygon should not be drawn
        if id < 0:
            continue
        # print(label, id)

        try:
            drawer.polygon(polygon, fill=id)
            if id not in unique_ids:
                unique_ids[id] = (num, label)
                num += 1
        except:
            print("Failed to draw polygon with label {} and id {}: {}".format(label,id,polygon))
            raise


    def convert_pixel(pixel):
        if pixel == 0:
            return 0
        return unique_ids[pixel][0]
    convert_pixel_v = np.vectorize(convert_pixel)

    instanceImg_np = np.array(instanceImg).reshape((-1))
    instanceImg_np = convert_pixel_v(instanceImg_np)
    instanceImg_np = instanceImg_np.reshape((size[1], size[0]))

    result_codes_dict = {}
    for id in unique_ids:
        code, class_name = unique_ids[id]
        result_codes_dict[code] = class_name

    return instanceImg_np, result_codes_dict

# A method that does all the work
# inJson is the filename of the json file
# outImg is the filename of the instance image that is generated
# encoding can be set to
#     - "ids"      : classes are encoded using the regular label IDs
#     - "trainIds" : classes are encoded using the training IDs
def json2instanceImg(json_path, encoding="ids"):
    annotation = Annotation()
    annotation.fromJsonFile(json_path)
    return createInstanceImage(annotation, encoding)

# dataset_path = '../../../big_datasets/images/CITYSCAPES/cityscapes'
# num = 12
# currAnnPath = os.path.join(dataset_path, 'gtFine/train/aachen/aachen_%06d_000019_gtFine_polygons.json' % (num))
# currImgPath = os.path.join(dataset_path, 'pics/img-frames/train/aachen/aachen_%06d_000019_leftImg8bit.png' % (num))
# json2instanceImg(currAnnPath, '../../../result_%d.png' % (num))

