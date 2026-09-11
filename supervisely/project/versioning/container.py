"""The ``.tar.zst`` payload container shared by Parquet-backed project snapshots.

Video and volume snapshots each carry their own copy of this packing code; image
snapshots use this module instead. The two older copies are left alone on purpose -
they sit on the restore path for archives that already exist, and an extraction that
is only *meant* to be behaviour-preserving is not worth that risk in the same change
that introduces a new format.
"""

from __future__ import annotations

import io
import os
import tarfile
from typing import Optional

from supervisely.io.json import load_json_file
from supervisely.project.versioning.schema_fields import VersionSchemaField

MANIFEST_NAME = "manifest.json"

# Parquet's own compression codec. pyarrow defaults to snappy, which was the wrong
# choice here twice over: it compresses this content far worse than zstd (measured on a
# real snapshot, 14.2 MB against 9.9 MB), and the archive used to be zstd-compressed
# again on the way out, so the outer pass spent CPU re-compressing snappy output for
# almost nothing. Compressing once, inside Parquet, is both smaller and faster - and it
# is per column, so reading three columns decompresses three columns.
PARQUET_COMPRESSION = "zstd"

# Magic number of a zstd frame, and of a POSIX tar (at offset 257). A pickle-format
# snapshot starts with 0x80 (pickle protocol 2+), so a snapshot can be identified
# without unpacking it - and both container shapes have to be recognised, because
# archives written before the codec change are zstd-wrapped, as video and volume
# snapshots still are.
_ZSTD_MAGIC = b"\x28\xb5\x2f\xfd"
_TAR_MAGIC = b"ustar"
_TAR_MAGIC_OFFSET = 257

# Enough bytes to see either magic.
SNIFF_SIZE = _TAR_MAGIC_OFFSET + len(_TAR_MAGIC) + 3


def is_zstd_container(data: bytes) -> bool:
    """True when ``data`` starts a zstd frame."""
    return data[:4] == _ZSTD_MAGIC


def is_tar_container(data: bytes) -> bool:
    """True when ``data`` starts an uncompressed tar archive."""
    return data[_TAR_MAGIC_OFFSET : _TAR_MAGIC_OFFSET + len(_TAR_MAGIC)] == _TAR_MAGIC


def is_snapshot_container(data: bytes) -> bool:
    """True for a Parquet-backed snapshot in either container shape, false for a pickle."""
    return is_zstd_container(data) or is_tar_container(data)


def pack_payload_dir(payload_dir: str, tmp_root: str) -> io.BytesIO:
    """Tar ``payload_dir`` into memory.

    Not compressed: the Parquet files inside are already zstd-compressed per column, and
    a second pass over them costs time to save nothing. Unpacking is where it shows -
    5.6x faster on a 200 MB snapshot, because opening one no longer means decompressing
    all of it before reading a single column.
    """
    tar_path = os.path.join(tmp_root, "snapshot.tar")
    with tarfile.open(tar_path, "w") as tar:
        tar.add(payload_dir, arcname=".")

    with open(tar_path, "rb") as f:
        out = io.BytesIO(f.read())
    out.seek(0)
    return out


def _extractall(tar: tarfile.TarFile, payload_dir: str) -> None:
    """Extract with the 'data' filter where the interpreter has one.

    A snapshot is downloaded from Team Files, so it is not trusted input: the filter is
    what stops a crafted archive from writing outside ``payload_dir``. It is the default
    from 3.14 and available from 3.12; on the older interpreters ``python_requires``
    still admits, the keyword may not exist, hence the fallback rather than a version
    check.
    """
    try:
        tar.extractall(payload_dir, filter="data")
    except TypeError:
        tar.extractall(payload_dir)


def unpack_snapshot(snapshot_bytes: bytes, payload_dir: str) -> None:
    """Extract a snapshot archive into ``payload_dir``, compressed or not.

    Archives written before the codec change are zstd-wrapped, and video and volume
    snapshots still are, so both shapes have to be read here.
    """
    if not is_zstd_container(snapshot_bytes):
        with tarfile.open(fileobj=io.BytesIO(snapshot_bytes), mode="r") as tar:
            _extractall(tar, payload_dir)
        return

    import zstd  # imported lazily to avoid a GC-during-module-init crash in some zstd builds

    try:
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(io.BytesIO(snapshot_bytes)) as reader:
            with tarfile.open(fileobj=reader, mode="r|") as tar:
                _extractall(tar, payload_dir)
    except Exception:
        # Fallback: one-shot decompression. Some zstd builds have no streaming reader.
        tar_bytes = zstd.decompress(snapshot_bytes)
        with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r") as tar:
            _extractall(tar, payload_dir)


def read_manifest_schema_version(payload_dir: str) -> Optional[str]:
    """Schema version recorded in the unpacked payload's manifest, if it has one."""
    manifest_path = os.path.join(payload_dir, MANIFEST_NAME)
    if not os.path.isfile(manifest_path):
        return None
    manifest = load_json_file(manifest_path)
    return manifest.get(VersionSchemaField.SCHEMA_VERSION)


def table_meta(name: str, path: str, row_count: int) -> dict:
    return {
        VersionSchemaField.NAME: name,
        VersionSchemaField.PATH: path,
        VersionSchemaField.ROW_COUNT: row_count,
    }


class ParquetTableWriter:
    """Appends rows to a Parquet file in row groups, holding only one group in memory.

    The point of the Parquet container is that neither writing nor reading a snapshot
    has to hold the whole project. Collecting every row into a list first - which is
    what building a single ``Table`` requires - would give that up on the write side,
    and figures are the table where it matters.

    The file is created on the first flush, so a table with no rows leaves no file and
    no manifest entry, matching how the video and volume snapshots record their tables.
    """

    # A row count alone does not bound anything: a figure row carrying a mask is three
    # orders of magnitude heavier than a dataset row, so the buffer is capped by payload
    # bytes as well and whichever limit is reached first flushes it.
    DEFAULT_BATCH_ROWS = 5000
    DEFAULT_BATCH_BYTES = 16 * 1024 * 1024

    def __init__(
        self,
        pa_module,
        parquet_module,
        path: str,
        schema,
        batch_rows: int = DEFAULT_BATCH_ROWS,
        batch_bytes: int = DEFAULT_BATCH_BYTES,
        compression: str = PARQUET_COMPRESSION,
    ):
        self._pa = pa_module
        self._parquet = parquet_module
        self._path = path
        self._schema = schema
        self._batch_rows = batch_rows
        self._batch_bytes = batch_bytes
        self._compression = compression
        self._rows = []
        self._pending_bytes = 0
        self._writer = None
        self._row_count = 0

    def add(self, row: dict) -> None:
        self._rows.append(row)
        self._pending_bytes += sum(len(v) for v in row.values() if isinstance(v, str))
        if len(self._rows) >= self._batch_rows or self._pending_bytes >= self._batch_bytes:
            self._flush()

    def _flush(self) -> None:
        if not self._rows:
            return
        table = self._pa.Table.from_pylist(self._rows, schema=self._schema)
        if self._writer is None:
            self._writer = self._parquet.ParquetWriter(
                self._path, self._schema, compression=self._compression
            )
        self._writer.write_table(table)
        self._row_count += len(self._rows)
        self._rows = []
        self._pending_bytes = 0

    def close(self) -> int:
        """Flush the tail and return the total number of rows written."""
        self._flush()
        if self._writer is not None:
            self._writer.close()
            self._writer = None
        return self._row_count

    @property
    def row_count(self) -> int:
        return self._row_count + len(self._rows)
