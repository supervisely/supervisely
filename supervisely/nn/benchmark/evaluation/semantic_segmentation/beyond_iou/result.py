import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List
from supervisely.nn.benchmark.evaluation.semantic_segmentation.beyond_iou.table_template import (
    table_template,
)


class Result:
    def __init__(
        self,
        class_names: List[str],
        dataframe: pd.DataFrame,
    ):
        """Class for presenting the results of the error analysis.
        :param class_names: List of strings providing names for class ids 0,...,C.
        :param dataframe: A pandas dataframe holding the statistics for all classes.
        """
        global pickle
        import pickle

        self.class_names = class_names
        self.num_classes = len(self.class_names)
        self.dataframe = dataframe
        # pixel accuracy
        overall_TP = self.dataframe["TP"][: self.num_classes].sum()
        overall_FN = self.dataframe["FN"][: self.num_classes].sum()
        self.pixel_accuracy = overall_TP / (overall_TP + overall_FN)

    @classmethod
    def from_evaluator(cls, evaluator):
        """Alternative constructor
        :param evaluator: An evaluator object which has run the evaluation already.
        :returns: The corresponding Result object.
        """
        dataframe = pd.DataFrame(index=evaluator.class_names)
        for error_name, error_counts in evaluator.results.items():
            if error_name == "unassigned":
                assert (error_counts == 0).all()
                continue
            dataframe[error_name] = error_counts

        dataframe["FP"] = (
            dataframe["FP_boundary"] + dataframe["FP_extent"] + dataframe["FP_segment"]
        )
        dataframe["FN"] = (
            dataframe["FN_boundary"] + dataframe["FN_extent"] + dataframe["FN_segment"]
        )
        dataframe["E_boundary"] = dataframe["FP_boundary"] + dataframe["FN_boundary"]
        dataframe["E_extent"] = dataframe["FP_extent"] + dataframe["FN_extent"]
        dataframe["E_segment"] = dataframe["FP_segment"] + dataframe["FN_segment"]

        union = dataframe["TP"] + dataframe["FP"] + dataframe["FN"]
        dataframe["IoU"] = dataframe["TP"] / union
        dataframe["precision"] = dataframe["TP"] / (dataframe["TP"] + dataframe["FP"])
        dataframe["recall"] = dataframe["TP"] / (dataframe["TP"] + dataframe["FN"])
        dataframe["F1_score"] = 2 / (
            1.0 / dataframe["precision"] + 1.0 / dataframe["recall"]
        )

        dataframe["FP_boundary_oU"] = dataframe["FP_boundary"] / union
        dataframe["FN_boundary_oU"] = dataframe["FN_boundary"] / union
        dataframe["E_boundary_oU"] = dataframe["E_boundary"] / union

        dataframe["FP_extent_oU"] = dataframe["FP_extent"] / union
        dataframe["FN_extent_oU"] = dataframe["FN_extent"] / union
        dataframe["E_extent_oU"] = dataframe["E_extent"] / union

        dataframe["FP_segment_oU"] = dataframe["FP_segment"] / union
        dataframe["FN_segment_oU"] = dataframe["FN_segment"] / union
        dataframe["E_segment_oU"] = dataframe["E_segment"] / union

        dataframe["E_boundary_oU_renormed"] = dataframe["E_boundary"] / (
            dataframe["TP"] + dataframe["E_boundary"]
        )
        dataframe["E_extent_oU_renormed"] = dataframe["E_extent"] / (
            dataframe["TP"] + dataframe["E_boundary"] + dataframe["E_extent"]
        )
        dataframe["E_segment_oU_renormed"] = dataframe["E_segment_oU"]

        with np.errstate(invalid="ignore"):  # avoid warnings for zero-division
            # boundary IoU
            dataframe["boundary_IoU"] = (
                evaluator.boundary_iou_intersection_counts
                / evaluator.boundary_iou_union_counts
            )
            # aggregate classes
            dataframe.loc["mean"] = dataframe.mean(axis=0)

        # dataframe.to_csv(f"{evaluator.result_dir}/result_df.csv", index=True)

        return cls(
            class_names=evaluator.class_names,
            dataframe=dataframe,
        )

    @classmethod
    def from_file(cls, path, verbose=True):
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if verbose:
            print(f"Successfully loaded result from: {path}.")
        return obj

    @classmethod
    def from_csv(cls, path, verbose=True):
        dataframe = pd.read_csv(path, index_col=0)
        class_names = list(dataframe.index[:-1])
        result = cls(class_names=class_names, dataframe=dataframe)
        if verbose:
            print(f"Successfully loaded result from: {path}.")
        return result

    def save(self, path, verbose=True):
        with open(path, "wb") as f:
            pickle.dump(self, f)
        if verbose:
            print(f"Successfully saved result to: {path}.")
        return

    def save_csv(self, path, verbose=True):
        self.dataframe.to_csv(path)
        if verbose:
            print(f"Successfully saved error stats to: {path}.")
        return

    def __str__(self):
        return table_template.format(
            mIoU=self.dataframe.loc["mean"]["IoU"] * 100,
            mE_boundary_oU=self.dataframe.loc["mean"]["E_boundary_oU"] * 100,
            mFP_boundary_oU=self.dataframe.loc["mean"]["FP_boundary_oU"] * 100,
            mFN_boundary_oU=self.dataframe.loc["mean"]["FN_boundary_oU"] * 100,
            mE_boundary_oU_renormed=self.dataframe.loc["mean"]["E_boundary_oU_renormed"]
            * 100,
            mE_extent_oU=self.dataframe.loc["mean"]["E_extent_oU"] * 100,
            mFP_extent_oU=self.dataframe.loc["mean"]["FP_extent_oU"] * 100,
            mFN_extent_oU=self.dataframe.loc["mean"]["FN_extent_oU"] * 100,
            mE_extent_oU_renormed=self.dataframe.loc["mean"]["E_extent_oU_renormed"]
            * 100,
            mE_segment_oU=self.dataframe.loc["mean"]["E_segment_oU"] * 100,
            mFP_segment_oU=self.dataframe.loc["mean"]["FP_segment_oU"] * 100,
            mFN_segment_oU=self.dataframe.loc["mean"]["FN_segment_oU"] * 100,
            mE_segment_oU_renormed=self.dataframe.loc["mean"]["E_segment_oU_renormed"]
            * 100,
            mPrecision=self.dataframe.loc["mean"]["precision"] * 100,
            mRecall=self.dataframe.loc["mean"]["recall"] * 100,
            mF1_score=self.dataframe.loc["mean"]["F1_score"] * 100,
            PixelAcc=self.pixel_accuracy * 100,
            mBoundaryIoU=self.dataframe.loc["mean"]["boundary_IoU"] * 100,
        )

    def make_table(self, path=None):
        table_string = str(self)
        if path is not None:
            with open(path, "w") as f:
                f.write(table_string)
        return table_string
