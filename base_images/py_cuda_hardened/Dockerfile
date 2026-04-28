ARG tag_ref_name

# ============================================================================
# STAGE 1: CUDA Stage
# ============================================================================
FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04 AS cuda

# ============================================================================
# STAGE 2: Runtime Stage (Hardened + CUDA)
# ============================================================================
FROM supervisely/base-py-sdk-hardened:${tag_ref_name} AS runtime

COPY --from=cuda /usr/local/cuda-12.8/targets/x86_64-linux/lib/libcudart.so* /usr/lib/
COPY --from=cuda /usr/local/cuda-12.8/targets/x86_64-linux/lib/libcublas.so* /usr/lib/
COPY --from=cuda /usr/local/cuda-12.8/targets/x86_64-linux/lib/libcublasLt.so* /usr/lib/
COPY --from=cuda /usr/local/cuda-12.8/targets/x86_64-linux/lib/libcufft.so* /usr/lib/
COPY --from=cuda /usr/local/cuda-12.8/targets/x86_64-linux/lib/libcurand.so* /usr/lib/
COPY --from=cuda /usr/local/cuda-12.8/targets/x86_64-linux/lib/libcusolver.so* /usr/lib/
COPY --from=cuda /usr/local/cuda-12.8/targets/x86_64-linux/lib/libcusparse.so* /usr/lib/
COPY --from=cuda /usr/local/cuda-12.8/targets/x86_64-linux/lib/libnvrtc.so* /usr/lib/
COPY --from=cuda /usr/local/cuda-12.8/targets/x86_64-linux/lib/libnvJitLink.so* /usr/lib/

ENV LD_LIBRARY_PATH="/usr/lib:/usr/local/lib" \
	NVIDIA_VISIBLE_DEVICES=all \
	NVIDIA_DRIVER_CAPABILITIES=compute,utility