import supervisely as sly

url = "some url"

sly.fs.download(
    url,
    "abc.jpg",
    headers={
        "User-Agent": "Mozilla/5.0",
    },
)
