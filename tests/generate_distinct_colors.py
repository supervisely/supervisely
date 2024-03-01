# generate colors
import gzip
import json

import distinctipy

import supervisely as sly

data = {}
num_cls = 100
fname = f"colors_{num_cls}.json"
for n in range():
    print(n)
    colors = distinctipy.get_colors(n)
    rgb_colors = [distinctipy.get_rgb256(color) for color in colors]
    data[n] = rgb_colors
sly.json.dump_json_file(data, fname)


data = sly.json.load_json_file(fname)
with gzip.open(fname + ".gz", "wt", encoding="UTF-8") as zipfile:
    json.dump(data, zipfile)
with gzip.open(fname + ".gz", "r") as fin:
    data = json.loads(fin.read().decode("utf-8"))
