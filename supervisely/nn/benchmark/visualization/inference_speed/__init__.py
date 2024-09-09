from supervisely.nn.benchmark.visualization.inference_speed.speedtest_batch import (
    SpeedtestBatch,
)
from supervisely.nn.benchmark.visualization.inference_speed.speedtest_intro import (
    SpeedtestIntro,
)
from supervisely.nn.benchmark.visualization.inference_speed.speedtest_overview import (
    SpeedtestOverview,
)
from supervisely.nn.benchmark.visualization.inference_speed.speedtest_real_time import (
    SpeedtestRealTime,
)

SPEEDTEST_METRICS = [
    SpeedtestIntro,
    SpeedtestOverview,
    # SpeedtestRealTime,
    SpeedtestBatch,
]
