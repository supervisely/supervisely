import requests
from requests_toolbelt import MultipartEncoder


if __name__ == '__main__':
    content_dict = {}
    content_dict['image'] = ("big_image.png", open("/workdir/src/big_image.jpg", 'rb'), 'image/*')
    content_dict['mode'] = ("mode", open('/workdir/src/sliding_window_mode_example.json', 'rb'))

    encoder = MultipartEncoder(fields=content_dict)
    response = requests.post("http://0.0.0.0:5000/model/inference", data=encoder, headers={'Content-Type': encoder.content_type})
    print(response.json())
