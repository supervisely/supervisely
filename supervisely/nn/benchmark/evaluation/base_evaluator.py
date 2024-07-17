import os
import pickle


class BaseEvaluator:
    def __init__(
            self,
            gt_project_path: str,
            dt_project_path: str,
            result_dir: str = "./evaluation",
        ):
        self.gt_project_path = gt_project_path
        self.dt_project_path = dt_project_path
        self.result_dir = result_dir
        os.makedirs(result_dir, exist_ok=True)

    def evaluate(self):
        raise NotImplementedError()
    
    def get_result_dir(self) -> str:
        return self.result_dir

    def _dump_pickle(self, data, file_path):
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)
