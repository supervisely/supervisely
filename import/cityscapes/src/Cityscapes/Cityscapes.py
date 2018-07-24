import os
from os.path import join
import json

from .cityscapesscripts.labels import cls2col, instance_classes
#from .cityscapesscripts.json2instanceImg import json2instanceImg
from ProjectWriter import ProjectWriter
import shared_utils


class Cityscapes(ProjectWriter):
    type = 'Cityscapes'
    can_ignore_imgs = True

    def __init__(self):
        pass

    @staticmethod
    def convert_json(json_data):
        result = []
        for obj in json_data['objects']:
            cls = obj['label']
            if cls == 'out of roi':
                polygon = obj['polygon'][:5]
                interiors = [obj['polygon'][5:]]
            else:
                polygon = obj['polygon']
                interiors = []
            instance_name = shared_utils.get_random_hash() if cls in instance_classes else ''
            obj = ProjectWriter.get_polygon_obj(cls, polygon, interiors=interiors, instance_name=instance_name)
            result.append(obj)
        gt_size = json_data['imgHeight'], json_data['imgWidth']
        return result, gt_size

    @staticmethod
    def json2image(path):
        img_path = path.replace('/gtFine/', '/leftImg8bit/')
        img_path = img_path.replace('_gtFine_polygons.json', '_leftImg8bit.png')
        return img_path

    @staticmethod
    def json2gt(path):
        return path.replace('_polygons.json', '_color.png')

    def convert_sample(self, sample):
        root, json_name = sample
        dataset_name = os.path.basename(root)
        name = json_name[:-len('.json')]
        name = '_'.join(name.split('_')[:3])
        src_ann_path = join(root, json_name)
        src_img_path = Cityscapes.json2image(src_ann_path)
        json_data = json.load(open(src_ann_path, 'r'))
        objects_data, gt_size = Cityscapes.convert_json(json_data)
        self.save_element(src_img_path, objects_data, gt_size, self.output_dirpath, dataset_name, name)

    def convert(self, input_dirpath, output_dirpath):
        self.convert_pre_start(input_dirpath, output_dirpath)

        if self.ignore_imgs:
            fn = lambda x: Cityscapes.convert_json(json.load(open(x, 'r')))
            self.convert_anns(input_dirpath, output_dirpath,
                              fn=fn,
                              exts=['.json'])
        else:
            json_dirs = join(input_dirpath, 'gtFine')
            samples = []
            for root, dirs, files in os.walk(json_dirs, topdown=True):
                for json_name in files:
                    if json_name.endswith('.json'):
                        dataset_name = os.path.basename(root)
                        if dataset_name in ['berlin', 'bielefeld', 'bonn', 'leverkusen', 'mainz', 'munich']:
                            continue
                        samples.append((root, json_name))

            self.total_elements = len(samples)
            self.output_dirpath = output_dirpath

            ProjectWriter.mp_process(self.convert_sample, samples)

        self.verbose_finish()
        self.save_classes_json(output_dirpath, cls2col, masks=False)