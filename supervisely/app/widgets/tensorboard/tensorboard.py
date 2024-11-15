# import os
# import socket
# import struct
# import subprocess
# import threading
# import time
# from datetime import datetime
# from typing import Dict, List, Optional, Union

# import crc32c
# from tensorboard.compat.proto.event_pb2 import Event
# from tensorboard.compat.proto.summary_pb2 import Summary
# from tensorboardX import SummaryWriter

# from supervisely.app import DataJson
# from supervisely.app.widgets import Widget


# class Tensorboard(Widget):
#     def __init__(
#         self,
#         logging_config: Dict = None,
#         height: Optional[Union[int, str]] = None,
#         width: Optional[Union[int, str]] = None,
#         widget_id: str = None,
#     ):
#         self._logging_config = logging_config or {
#             "enable": True,
#             "interval": 1,
#             "save_to_file": True,
#             "metrics": ["loss"],
#         }
#         self._height, self._width = self._check_plot_size(height=height, width=width)
#         self._event_file = None
#         self._tb_process = None
#         self._port = 8001  # self._find_free_port()
#         self._log_dir = os.path.join("logs", datetime.now().strftime("%Y%m%d-%H%M%S"))
#         self._writer = SummaryWriter(self._log_dir)
#         super().__init__(widget_id=widget_id, file_path=__file__)
#         self._initialize()

#     @property
#     def writer(self):
#         return self._writer

#     def get_json_data(self):
#         return {
#             "isReady": True,
#             "pathToHtml": f"http://localhost:{self._port}",
#             "width": self._width,
#             "height": self._height,
#         }

#     def get_json_state(self):
#         return {**self._logging_config, "port": self._port, "log_dir": self._log_dir}

#     def _initialize(self):
#         if self._logging_config["enable"]:
#             try:
#                 os.makedirs(self._log_dir, exist_ok=True)
#                 filename = os.path.join(self._log_dir, "events.out.tfevents." + str(time.time()))
#                 self._event_file = open(filename, "wb")
#                 self._start_tensorboard()
#                 DataJson()[self.widget_id] = {
#                     "pathToHtml": f"http://localhost:{self._port}",
#                     "width": self._width,
#                     "height": self._height,
#                     "isReady": True,
#                     "error": None,
#                 }
#             except Exception as e:
#                 DataJson()[self.widget_id] = {"error": str(e), "isReady": False}

#     def _write_event(self, event):
#         """Write event to file with CRC checksum."""
#         data = event.SerializeToString()
#         header = struct.pack("Q", len(data))
#         crc = crc32c.crc32c(header + data)
#         footer = struct.pack("I", crc)

#         if self._event_file:
#             self._event_file.write(header)
#             self._event_file.write(data)
#             self._event_file.write(footer)
#             self._event_file.flush()
#             print(f"Written event with step {event.step}")

#     def add_metrics(self, metrics: Dict, step: int):
#         if not self._logging_config["enable"] or not self._event_file:
#             return

#         for name, value in metrics.items():
#             if name in self._logging_config["metrics"]:
#                 # tensorboardX
#                 self.writer.add_scalar(name, value, step)
#                 self.writer.flush()

#     def _check_plot_size(
#         self,
#         height: Optional[Union[int, str]],
#         width: Optional[Union[int, str]],
#     ) -> tuple[str, str]:
#         """Validate and normalize plot size parameters"""
#         if height is None and width is None:
#             return "800px", "100%"

#         def _check_single_size(size: Optional[Union[int, str]]) -> str:
#             if size is None:
#                 return "800px"
#             elif isinstance(size, int):
#                 return f"{size}px"
#             elif isinstance(size, str):
#                 if size.endswith("px") or size.endswith("%") or size == "auto":
#                     return size
#                 raise ValueError(f"Size must be in pixels or percent, got '{size}'")
#             raise ValueError(f"Size must be int or str, got '{type(size)}'")

#         return _check_single_size(height), _check_single_size(width)

#     def _find_free_port(self) -> int:
#         """Find an available port for TensorBoard server"""
#         with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
#             s.bind(("", 0))
#             s.listen(1)
#             port = s.getsockname()[1]
#         return port

#     def _start_tensorboard(self):
#         """Start TensorBoard server in a subprocess"""
#         self._tb_process = subprocess.Popen(
#             [
#                 "tensorboard",
#                 "--logdir",
#                 self._log_dir,
#                 "--host",
#                 "localhost",
#                 "--port",
#                 str(self._port),
#                 "--load_fast=false",
#             ]
#         )
#         print(f"Started TensorBoard on port {self._port}")

#     def stop(self):
#         """Stop TensorBoard and clean up"""
#         if self._event_file:
#             self._event_file.close()
#         if self._tb_process:
#             self._tb_process.terminate()
#             self._tb_process.wait()
