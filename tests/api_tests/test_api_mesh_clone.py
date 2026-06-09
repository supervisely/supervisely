# isort:skip_file
# coding: utf-8
"""Manual integration test for server-side mesh annotation cloning.

Fill in the constants below, then run:
    python tests/api_tests/test_api_mesh_clone.py

The script clones annotations from SRC_MESH_IDS to DST_MESH_IDS using
``figures.clone`` (no binary transfer) and prints a diff of figure counts
before and after to verify correctness.
"""
import os
import sys

sdk_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, sdk_path)

import supervisely as sly

# ---------------------------------------------------------------------------
# Fill in these constants before running
# ---------------------------------------------------------------------------

# Dataset that owns the SOURCE meshes.
SRC_DATASET_ID: int = 0  # TODO: set source dataset ID

# Mesh entity IDs to copy annotations FROM.
# The lists must be the same length and ordered: SRC_MESH_IDS[i] -> DST_MESH_IDS[i].
SRC_MESH_IDS: list = [0]  # TODO: e.g. [111, 222, 333]

# Mesh entity IDs to copy annotations TO.
DST_MESH_IDS: list = [0]  # TODO: e.g. [444, 555, 666]

# Project ID of the destination meshes.
# Leave as None to let the SDK resolve it automatically from DST_MESH_IDS[0].
DST_PROJECT_ID: int = 0  # TODO: set or leave None

# ---------------------------------------------------------------------------


def _figure_counts(api: sly.Api, dataset_id: int, mesh_ids: list) -> dict:
    """Return {mesh_id: figure_count} for the given mesh IDs."""
    objects_by_id = api.mesh.object.download(dataset_id, mesh_ids, skip_geometry=True)
    return {mid: len(objects_by_id.get(mid, [])) for mid in mesh_ids}


def _dst_dataset_id(api: sly.Api) -> int:
    """Resolve dataset ID of the first destination mesh."""
    return api.mesh.get_info_by_id(DST_MESH_IDS[0]).dataset_id


def main():
    assert SRC_DATASET_ID, "Set SRC_DATASET_ID"
    assert SRC_MESH_IDS, "Set SRC_MESH_IDS"
    assert DST_MESH_IDS, "Set DST_MESH_IDS"
    assert len(SRC_MESH_IDS) == len(DST_MESH_IDS), (
        f"SRC_MESH_IDS and DST_MESH_IDS must be the same length "
        f"({len(SRC_MESH_IDS)} != {len(DST_MESH_IDS)})"
    )

    api = sly.Api.from_env()

    dst_dataset_id = _dst_dataset_id(api)

    # --- snapshot: figure counts before cloning ---
    print("Figure counts BEFORE copy_batch:")
    src_before = _figure_counts(api, SRC_DATASET_ID, SRC_MESH_IDS)
    dst_before = _figure_counts(api, dst_dataset_id, DST_MESH_IDS)
    for src_id, dst_id in zip(SRC_MESH_IDS, DST_MESH_IDS):
        print(f"  src mesh {src_id}: {src_before[src_id]} figures")
        print(f"  dst mesh {dst_id}: {dst_before[dst_id]} figures")

    # --- run copy_batch ---
    print("\nRunning api.mesh.annotation.copy_batch ...")
    api.mesh.annotation.copy_batch(
        src_dataset_id=SRC_DATASET_ID,
        src_mesh_ids=SRC_MESH_IDS,
        dst_mesh_ids=DST_MESH_IDS,
        dst_project_id=DST_PROJECT_ID,
    )
    print("Done.")

    # --- snapshot: figure counts after cloning ---
    print("\nFigure counts AFTER copy_batch:")
    src_after = _figure_counts(api, SRC_DATASET_ID, SRC_MESH_IDS)
    dst_after = _figure_counts(api, dst_dataset_id, DST_MESH_IDS)
    all_ok = True
    for src_id, dst_id in zip(SRC_MESH_IDS, DST_MESH_IDS):
        expected = src_before[src_id]
        got = dst_after[dst_id] - dst_before[dst_id]
        status = "OK" if got == expected else "MISMATCH"
        if status == "MISMATCH":
            all_ok = False
        print(
            f"  src {src_id} -> dst {dst_id}: "
            f"expected +{expected} figures, got +{got}  [{status}]"
        )
        # src must be untouched
        if src_after[src_id] != src_before[src_id]:
            print(
                f"  WARNING: src mesh {src_id} figure count changed "
                f"({src_before[src_id]} -> {src_after[src_id]})"
            )
            all_ok = False

    print("\nResult:", "ALL OK" if all_ok else "FAILED — see mismatches above")


if __name__ == "__main__":
    main()
