# mcp_aif360_tools.py
from mcp.server.fastmcp import FastMCP
import pandas as pd
import numpy as np

from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
from aif360.algorithms.preprocessing import Reweighing, DisparateImpactRemover
from aif360.algorithms.postprocessing import (
    CalibratedEqOddsPostprocessing, EqOddsPostprocessing, RejectOptionClassification
)

mcp = FastMCP("AIF360 MCP", stateless_http=True, debug=True)

def _make_bld(X_df, y, sensitive, attr_name, favorable_label=1, unfavorable_label=0):
    df = X_df.copy()
    df[attr_name] = sensitive
    df["label"] = y
    return BinaryLabelDataset(
        favorable_label=favorable_label, unfavorable_label=unfavorable_label,
        df=df, label_names=["label"], protected_attribute_names=[attr_name]
    )

def _groups(attr_name, privileged_values):
    # e.g. [{'sex': 1}] for privileged; unprivileged is everything else
    return ([{attr_name: v} for v in privileged_values],
            None)  # unprivileged inferred by BLD/metrics

@mcp.tool("Compute Fairness Metrics")
def compute_fairness_metrics(
    y_true: list[int], y_pred: list[int], sensitive: list, attr_name: str, privileged_values: list
):
    X = pd.DataFrame(index=range(len(y_true)))  # empty features, just to build BLDs
    bld_true = _make_bld(X, y_true, sensitive, attr_name)
    bld_pred = _make_bld(X, y_pred, sensitive, attr_name)
    priv, _ = _groups(attr_name, privileged_values)

    dm = BinaryLabelDatasetMetric(bld_true, privileged_groups=priv)
    cm = ClassificationMetric(bld_true, bld_pred, privileged_groups=priv)

    # A compact subset for the UI
    result = {
        "overall": {
            "disparate_impact": dm.disparate_impact(),
            "stat_parity_diff": dm.statistical_parity_difference(),
            "eq_opp_diff": cm.equal_opportunity_difference(),
            "eq_odds_diff": cm.average_odds_difference(),
            "ppv_diff": cm.positive_predictive_value_difference(),
        },
        "by_group": {}  # you can loop groups and compute per-slice stats if desired
    }
    return result

@mcp.tool("Reweigh Samples")
def reweigh_samples(
    X: list[dict], y: list[int], sensitive: list, attr_name: str, privileged_values: list
):
    X_df = pd.DataFrame(X)
    bld = _make_bld(X_df, y, sensitive, attr_name)
    priv, unpriv = _groups(attr_name, privileged_values)
    rw = Reweighing(unprivileged_groups=unpriv, privileged_groups=priv).fit(bld)
    return {"sample_weight": rw.transform(bld).instance_weights.tolist()}
