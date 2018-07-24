# coding: utf-8

def get_data_sources(graph_json):
    data_sources = {}
    for layer in graph_json:
        if layer["action"] == "data":
            for src in layer["src"]:
                src_parts = src.split("/")
                src_pr = src_parts[0]
                if src_pr not in data_sources:
                    data_sources[src_pr] = []

                if src_parts[1] == "*":
                    data_sources[src_pr] = "*"

                if data_sources[src_pr] == "*":
                    continue

                data_sources[src_pr].append(src_parts[1])

    for pr in data_sources.keys():
        if data_sources[pr] == "*":
            continue
        data_sources[pr] = list(set(data_sources[pr]))

    return data_sources


def get_res_project_name(graph_json):
    for layer in graph_json:
        if layer["action"] in ["supervisely", "save", "save_masks"]:
            return layer["dst"]
    raise RuntimeError("supervisely save layer not found")
