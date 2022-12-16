from supervisely import Api


api = Api(server_address="http://127.0.0.1:5678", token="a"*128)

try:
    r = api.post("metohd", data={}, retries=7)
    print(r.status_code)
except Exception as e:
    print()
    print(e)

try:
    r = api.get("method", params={}, retries=7)
except Exception as e:
    print()
    print(e)
