def _require_pyarrow():
    try:
        import pyarrow
        import pyarrow.parquet
    except Exception as e:
        raise RuntimeError(
            "pyarrow is required to build/restore project snapshots. Please install pyarrow."
        ) from e
    import pyarrow as pa
    import pyarrow.parquet as pq

    return pa, pq