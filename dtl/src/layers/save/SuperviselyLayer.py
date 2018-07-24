# coding: utf-8

from copy import deepcopy

from supervisely_lib import ProjectWriterFS

from Layer import Layer


class SuperviselyLayer(Layer):

    action = 'supervisely'

    layer_settings = {
        "required": ["settings"],
        "properties": {
            "settings": {}
        }
    }

    def __init__(self, config, output_folder, net):
        Layer.__init__(self, config)
        self.output_folder = output_folder
        self.net = net
        self.pr_writer = ProjectWriterFS(output_folder)
        self.net_change_images = self.net.may_require_images()

    def is_archive(self):
        return False

    def process(self, data_el):
        img_desc, ann = data_el

        free_name = self.net.get_free_name(img_desc.get_img_name())
        if self.net_change_images:
            if img_desc.need_write() is True:
                self.pr_writer.write_image(img_desc, free_name)
            else:
                self.pr_writer.copy_image(img_desc, free_name)

        ann_to_save = deepcopy(ann)
        ann_to_save.normalize_figures()
        packed_ann = ann_to_save.pack()
        self.pr_writer.write_ann(img_desc, packed_ann, free_name)

        yield ([img_desc, ann],)
