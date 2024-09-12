from supervisely.nn.inference import SessionJSON

WORKSPACE_NAME = "Model Benchmark: predictions and differences"
WORKSPACE_DESCRIPTION = "Technical workspace for model benchmarking. Contains predictions and differences between ground truth and predictions."

def try_set_conf_auto(session: SessionJSON, conf: float):
    conf_names = ["conf", "confidence", "confidence_threshold", "confidence_thresh"]
    default = session.get_default_inference_settings()
    for name in conf_names:
        if name in default:
            session.inference_settings[name] = conf
            return True
    return False
