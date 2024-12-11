from typing import Dict, List

from supervisely.nn.benchmark.visualization.widgets.widget import BaseWidget


class RadioGroupWidget(BaseWidget):

    def __init__(
        self,
        name: str,
        radio_group: str,
        switch_keys: List[str],
        default_key: str = None,
    ) -> None:
        super().__init__(name)
        self.radio_group = radio_group
        self.switch_keys = switch_keys
        self.default_key = default_key or switch_keys[0]

    def save_data(self, basepath: str) -> None:
        return

    def get_state(self) -> Dict:
        return {self.radio_group: self.default_key}

    def to_html(self) -> str:
        res = ""
        for switch_key in self.switch_keys:
            res += f"""
            <el-radio v-model="state.{self.radio_group}" label="{switch_key}" style="margin-top: 10px;">
            {switch_key}
            </el-radio>
            """
        return res
