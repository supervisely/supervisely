import json

import distinctipy

import supervisely as sly

data = {}
for n in range(5):
    print(n)
    colors = distinctipy.get_colors(n)
    rgb_colors = [distinctipy.get_rgb256(color) for color in colors]
    data[n] = rgb_colors
sly.json.dump_json_file(data, "colors.json")

import gzip

data = sly.json.load_json_file("colors.json")
with gzip.open("colors.json.gz", "wt", encoding="UTF-8") as zipfile:
    json.dump(data, zipfile)
with gzip.open("colors.json.gz", "r") as fin:
    data = json.loads(fin.read().decode("utf-8"))
