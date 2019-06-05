# coding: utf-8

import os
import json

from legacy_supervisely_lib.utils.general_utils import generate_random_string

from Layer import Layer
import layers  # to register layers


class Net:
    def __init__(self, graph_desc, input_project_metas, output_folder):
        self.input_project_metas = input_project_metas
        self.layers = []

        if type(graph_desc) is str:
            graph_path = graph_desc

            if not os.path.exists(graph_path):
                raise RuntimeError('No such config file "%s"' % graph_path)
            else:
                graph = json.load(open(graph_path, 'r'))
        else:
            graph = graph_desc

        graph_has_datal = False
        graph_has_savel = False

        for layer_config in graph:
            if 'action' not in layer_config:
                raise RuntimeError('No "action" field in layer "{}".'.format(layer_config))
            action = layer_config['action']
            if action not in Layer.actions_mapping:
                raise RuntimeError('Unrecognized action "{}".'.format(action))
            layer_cls = Layer.actions_mapping[action]
            if layer_cls.type == 'data':
                graph_has_datal = True
                layer = layer_cls(layer_config, input_project_metas)
            elif layer_cls.type == 'processing':
                layer = layer_cls(layer_config)
            elif layer_cls.type == 'save':
                if graph_has_savel:
                    raise ValueError(
                        'Graph has to contain only one save layer. You added at least 2 layers: {!r} and {!r}.'
                        .format(self.save_layer.config, layer_config))
                graph_has_savel = True
                layer = layer_cls(layer_config, output_folder, self)
                self.save_layer = layer
            else:
                raise NotImplementedError()
            self.layers.append(layer)

        if graph_has_datal is False:
            raise RuntimeError("Graph error: missing data layer.")
        if graph_has_savel is False:
            raise RuntimeError("Graph error: missing save layer.")
        if len(self.layers) < 2:
            raise RuntimeError("Graph error: less than two layers.")

        self.check_connections()
        self.flat_out_names = False  # @TODO: move out
        self.annot_archive = None
        self.reset_existing_names()

        self._output_meta = self._calc_result_meta()  # ok now for single save layer

    def get_free_name(self, img_desc):
        name = img_desc.get_img_name()
        new_name = name
        names_in_ds = self.existing_names.get(img_desc.get_res_ds_name(), set())

        if name in names_in_ds:
            new_name = name + '_' + generate_random_string(10)

        names_in_ds.add(new_name)
        self.existing_names[img_desc.get_res_ds_name()] = names_in_ds

        return new_name

    def may_require_images(self):
        # req_layers = [layer for layer in self.layers if layer.requires_image()]
        # return len(req_layers) > 0
        return True

    def check_connections(self, indx=-1):
        if indx == -1:
            for i in range(len(self.layers)):
                if self.layers[i].type == 'data':
                    for layer_ in self.layers:
                        layer_.color = 'not visited'
                    self.src_check_mappings = []
                    self.check_connections(i)
        else:
            color = self.layers[indx].color
            if color == 'visiting':
                raise RuntimeError('Loop in layers structure.')
            if color == 'visited':
                return
            self.layers[indx].color = 'visiting'
            for next_layer_indx in self.get_next_layer_indxs(indx):
                self.check_connections(next_layer_indx)
            self.layers[indx].color = 'visited'

    def get_next_layer_indxs(self, indx, branch=-1):

        #:param indx:
        #:param branch: specify when calling while processing images, do not specify when calling before processing images
        #:return:

        if indx >= len(self.layers):
            raise RuntimeError('Invalid layer index.')
        if branch == -1:  # check class mappings
            if hasattr(self.layers[indx], 'src_check_mappings'):
                for cls in self.layers[indx].src_check_mappings:
                    self.src_check_mappings.append((indx, cls))
            if hasattr(self.layers[indx], 'dst_check_mappings'):
                for i, cls in self.src_check_mappings:
                    for l_name, l in self.layers[indx].dst_check_mappings.items():
                        if cls not in l:
                            raise RuntimeError(
                                'No mapping for class "{}" declared in layer "{}" in "{}" mapping in layer "{}"'.format(
                                    cls, self.layers[i].description(), l_name, self.layers[indx].description()
                                ))

        if self.layers[indx].type == 'save':
            return []

        if branch == -1:
            dsts = self.layers[indx].dsts
        else:
            dsts = [self.layers[indx].dsts[branch]]
        dsts = list(set(dsts) - {Layer.null})

        result = []
        for dst in dsts:
            for i, layer_ in enumerate(self.layers):
                if dst in layer_.srcs:
                    result.append(i)
        return result

    # will construct 'export' archive
    def is_archive(self):
        res = self.save_layer.is_archive()
        return res

    def reset_existing_names(self):
        self.existing_names = {}

    def start(self, data_el):
        img_pr_name = data_el[0].get_pr_name()
        img_ds_name = data_el[0].get_ds_name()

        start_layer_indx = None
        for idx, layer in enumerate(self.layers):
            if layer.type != 'data':
                continue
            if layer.project_name == img_pr_name \
                    and (layer.dataset_name == '*' or layer.dataset_name == img_ds_name):
                start_layer_indx = idx
                break
        if start_layer_indx is None:
            raise RuntimeError('Can not find data layer for the image: {}'.format(data_el))

        output_generator = self.process(start_layer_indx, data_el)
        for output in output_generator:
            yield output

    def push(self, indx, data_el, branch):
        next_layer_indxs = self.get_next_layer_indxs(indx, branch=branch)
        for next_layer_indx in next_layer_indxs:
            for x in self.process(next_layer_indx, data_el):
                yield x

    def process(self, indx, data_el):
        layer = self.layers[indx]
        for layer_output in layer.process_timed(data_el):
            if layer_output is None:
                raise RuntimeError('Layer_output ({}) is None.'.format(layer))

            if len(layer_output) == 3:
                new_data_el = layer_output[:2]
                branch = layer_output[-1]
            elif len(layer_output) == 2:
                new_data_el = layer_output
                branch = 0
            elif len(layer_output) == 1:
                yield layer_output
                continue
            else:
                raise RuntimeError('Wrong number of items in layer output ({}). Got {} items.'.format(
                    layer, len(layer_output)
                ))
            for x in self.push(indx, new_data_el, branch):
                yield x


############################################################################################################
# Process classes begin
############################################################################################################
    def get_save_layer_dest(self):
        return self.save_layer.dsts[0]

    def get_final_project_name(self):
        return self.get_save_layer_dest()

    def get_result_project_meta(self):
        return self._output_meta

    def _calc_result_meta(self):
        cur_level_layers = {layer for layer in self.layers if layer.type == 'data'}
        datalevel_metas = {}
        for data_layer in cur_level_layers:
            input_meta = self.input_project_metas[data_layer.project_name]
            for src in data_layer.srcs:
                datalevel_metas[src] = input_meta

        def get_dest_layers(the_layer):
            return [dest_layer for dest_layer in self.layers if len(set(the_layer.dsts) & set(dest_layer.srcs)) > 0]

        def layer_input_metas_are_calculated(the_layer):
            return all((x in datalevel_metas for x in the_layer.srcs))

        processed_layers = set()
        while len(cur_level_layers) != 0:
            next_level_layers = set()

            for cur_layer in cur_level_layers:
                processed_layers.add(cur_layer)
                # TODO no need for dict here?
                cur_layer_input_metas = {src: datalevel_metas[src] for src in cur_layer.srcs}
                cur_layer_res_meta = cur_layer.make_output_meta(cur_layer_input_metas)

                for dst in cur_layer.dsts:
                    datalevel_metas[dst] = cur_layer_res_meta

                dest_layers = get_dest_layers(cur_layer)
                for next_candidate in dest_layers:
                    if layer_input_metas_are_calculated(next_candidate):
                        next_level_layers.update([next_candidate])

            cur_level_layers = next_level_layers

        if set(processed_layers) != set(self.layers):
            raise RuntimeError('Graph has several connected components. Only one is allowed.')

        result_meta = datalevel_metas[self.get_save_layer_dest()]
        return result_meta

############################################################################################################
# Process classes end
############################################################################################################
