# Usage:
# docker run --rm -ti \
#        -v /path-to/model:/sly_task_data/model
#        [model docker image name]
#        python -- /workdir/src/rest_inference.py

from inference import UnetV2SingleImageApplier
import os

from supervisely_lib.worker_api.rpc_servicer import InactiveRPCServicer
from supervisely_lib.nn.inference.rest_server import ModelRest, RestInferenceServer
from supervisely_lib.nn.inference.rest_constants import REST_INFERENCE_PORT


if __name__ == '__main__':
    port = os.getenv(REST_INFERENCE_PORT, '')
    model_deploy = ModelRest(model_applier_cls=UnetV2SingleImageApplier, rpc_servicer_cls=InactiveRPCServicer)
    server = RestInferenceServer(model=model_deploy.serv_instance, name=__name__, port=port)
    server.run()
