"""
Fairlearn MCP Server
Fairness metrics computation and constraint-based mitigation
"""

from mcp.server.fastmcp import FastMCP
from pydantic import Field
from typing import Dict, List, Any, Optional, Literal
import sys
import os

from fairlearn.metrics import MetricFrame, selection_rate, demographic_parity_difference
from fairlearn.reductions import ExponentiatedGradient, DemographicParity, EqualizedOdds
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd

# Remove port specification for FastAPI mounting
mcp = FastMCP("Fairlearn MCP Server", stateless_http=True, debug=True)

# ------------------------------
# Core Fairlearn Tools
# ------------------------------

@mcp.tool(
    title="Fairness Assessment",
    description="Compute fairness metrics (demographic parity, selection rate) for predictions."
)
def fairness_assessment(
    y_true: List[int] = Field(description="True labels (0/1)"),
    y_pred: List[int] = Field(description="Predicted labels (0/1)"),
    sensitive_features: List[str] = Field(description="Group identifiers (e.g. gender, ethnicity)")
) -> Dict[str, Any]:
    """Run Fairlearn metrics on predictions grouped by sensitive features"""
    try:
        y_true_arr = np.array(y_true)
        y_pred_arr = np.array(y_pred)
        sensitive_features_arr = np.array(sensitive_features)

        # Validate input lengths
        if not (len(y_true_arr) == len(y_pred_arr) == len(sensitive_features_arr)):
            return {"error": "All input arrays must have the same length"}

        # Compute group-wise selection rate
        metric_frame = MetricFrame(
            metrics={"selection_rate": selection_rate},
            y_true=y_true_arr,
            y_pred=y_pred_arr,
            sensitive_features=sensitive_features_arr,
        )

        # Compute overall demographic parity difference
        dp_diff = demographic_parity_difference(
            y_true_arr, y_pred_arr, sensitive_features=sensitive_features_arr
        )

        # Convert numpy types to Python native types for JSON serialization
        group_rates = {}
        for group, rate in metric_frame.by_group["selection_rate"].items():
            group_rates[str(group)] = float(rate)

        return {
            "overall_selection_rate": float(metric_frame.overall["selection_rate"]),
            "group_selection_rates": group_rates,
            "demographic_parity_difference": float(dp_diff),
            "total_samples": len(y_true_arr),
            "unique_groups": list(set(sensitive_features)),
            "interpretation": {
                "dp_difference_threshold": 0.1,
                "concerning": abs(float(dp_diff)) > 0.1,
                "recommendation": "Consider mitigation if DP difference > 0.1"
            }
        }
    except Exception as e:
        return {"error": f"Fairness assessment failed: {str(e)}"}

@mcp.tool(
    title="Advanced Fairness Metrics",
    description="Compute comprehensive fairness metrics including equalized odds, TPR, FPR differences."
)
def advanced_fairness_metrics(
    y_true: List[int] = Field(description="True labels (0/1)"),
    y_pred: List[int] = Field(description="Predicted labels (0/1)"),
    sensitive_features: List[str] = Field(description="Group identifiers")
) -> Dict[str, Any]:
    """Compute advanced fairness metrics beyond basic demographic parity"""
    try:
        y_true_arr = np.array(y_true)
        y_pred_arr = np.array(y_pred)
        sensitive_features_arr = np.array(sensitive_features)

        if not (len(y_true_arr) == len(y_pred_arr) == len(sensitive_features_arr)):
            return {"error": "All input arrays must have the same length"}

        # Define custom metrics
        def true_positive_rate(y_true, y_pred):
            if np.sum(y_true) == 0:
                return np.nan
            return np.sum((y_true == 1) & (y_pred == 1)) / np.sum(y_true == 1)

        def false_positive_rate(y_true, y_pred):
            if np.sum(y_true == 0) == 0:
                return np.nan
            return np.sum((y_true == 0) & (y_pred == 1)) / np.sum(y_true == 0)

        def accuracy(y_true, y_pred):
            return np.mean(y_true == y_pred)

        # Compute comprehensive metrics
        metric_frame = MetricFrame(
            metrics={
                "accuracy": accuracy,
                "selection_rate": selection_rate,
                "tpr": true_positive_rate,
                "fpr": false_positive_rate
            },
            y_true=y_true_arr,
            y_pred=y_pred_arr,
            sensitive_features=sensitive_features_arr,
        )

        # Format results
        results = {
            "overall_metrics": {},
            "group_metrics": {},
            "disparities": {}
        }

        # Overall metrics
        for metric_name, value in metric_frame.overall.items():
            results["overall_metrics"][metric_name] = float(value) if not np.isnan(value) else None

        # Group metrics
        for metric_name in metric_frame.by_group.keys():
            results["group_metrics"][metric_name] = {}
            for group, value in metric_frame.by_group[metric_name].items():
                results["group_metrics"][metric_name][str(group)] = float(value) if not np.isnan(value) else None

        # Compute disparities (max - min for each metric)
        for metric_name in metric_frame.by_group.keys():
            values = [v for v in metric_frame.by_group[metric_name].values() if not np.isnan(v)]
            if values:
                results["disparities"][f"{metric_name}_disparity"] = float(max(values) - min(values))
            else:
                results["disparities"][f"{metric_name}_disparity"] = None

        return results

    except Exception as e:
        return {"error": f"Advanced metrics computation failed: {str(e)}"}

@mcp.tool(
    title="Fairness Mitigation Trainer",
    description="Train a fair classifier using ExponentiatedGradient with fairness constraints."
)
def mitigate_fairness(
    X: List[List[float]] = Field(description="Feature matrix as list of lists"),
    y: List[int] = Field(description="Binary labels (1=approve, 0=reject)"),
    sensitive_features: List[str] = Field(description="Protected attribute values"),
    constraint: str = Field(
        description="Fairness constraint: 'demographic_parity' or 'equalized_odds'", 
        default="demographic_parity"
    ),
    test_size: float = Field(description="Fraction of data for testing", default=0.3),
    random_state: int = Field(description="Random seed for reproducibility", default=42)
) -> Dict[str, Any]:
    """
    Train a fair classifier using ExponentiatedGradient and return metrics + model info.
    """
    try:
        X_arr = np.array(X)
        y_arr = np.array(y)
        sensitive_features_arr = np.array(sensitive_features)

        # Validate inputs
        if len(X_arr) != len(y_arr) or len(y_arr) != len(sensitive_features_arr):
            return {"error": "Feature matrix, labels, and sensitive features must have same length"}

        if len(X_arr) < 10:
            return {"error": "Need at least 10 samples for training"}

        # Train/test split for evaluation
        X_train, X_test, y_train, y_test, sf_train, sf_test = train_test_split(
            X_arr, y_arr, sensitive_features_arr, 
            test_size=test_size, 
            random_state=random_state, 
            stratify=y_arr
        )

        # Base estimator
        base_estimator = LogisticRegression(solver="liblinear", random_state=random_state)

        # Choose fairness constraint
        if constraint.lower() == "equalized_odds":
            constraint_obj = EqualizedOdds()
        else:
            constraint_obj = DemographicParity()

        # Train fairness-constrained model
        mitigator = ExponentiatedGradient(
            estimator=base_estimator, 
            constraints=constraint_obj,
            max_iter=50
        )
        mitigator.fit(X_train, y_train, sensitive_features=sf_train)

        # Evaluate on test set
        y_pred = mitigator.predict(X_test)
        accuracy_post = accuracy_score(y_test, y_pred)

        # Train baseline model for comparison
        baseline = LogisticRegression(solver="liblinear", random_state=random_state)
        baseline.fit(X_train, y_train)
        y_pred_baseline = baseline.predict(X_test)
        accuracy_baseline = accuracy_score(y_test, y_pred_baseline)

        # Compute fairness metrics for both models
        dp_diff_mitigated = demographic_parity_difference(
            y_test, y_pred, sensitive_features=sf_test
        )
        dp_diff_baseline = demographic_parity_difference(
            y_test, y_pred_baseline, sensitive_features=sf_test
        )

        # Extract model parameters (from one of the predictors)
        if hasattr(mitigator, '_pmf_predictor') and hasattr(mitigator._pmf_predictor, 'estimator'):
            model_weights = mitigator._pmf_predictor.estimator.coef_.tolist()
            model_intercept = mitigator._pmf_predictor.estimator.intercept_.tolist()
        else:
            model_weights = None
            model_intercept = None

        return {
            "mitigation_results": {
                "constraint_applied": constraint,
                "accuracy_baseline": float(accuracy_baseline),
                "accuracy_post_mitigation": float(accuracy_post),
                "accuracy_change": float(accuracy_post - accuracy_baseline),
                "dp_difference_baseline": float(dp_diff_baseline),
                "dp_difference_mitigated": float(dp_diff_mitigated),
                "dp_improvement": float(abs(dp_diff_baseline) - abs(dp_diff_mitigated)),
                "samples_train": len(X_train),
                "samples_test": len(X_test)
            },
            "model_info": {
                "weights": model_weights,
                "intercept": model_intercept,
                "n_features": X_arr.shape[1] if len(X_arr.shape) > 1 else 1
            },
            "interpretation": {
                "fairness_improved": abs(float(dp_diff_mitigated)) < abs(float(dp_diff_baseline)),
                "accuracy_cost": float(accuracy_baseline - accuracy_post),
                "recommendation": "Monitor accuracy-fairness tradeoff in production"
            }
        }

    except Exception as e:
        return {"error": f"Mitigation training failed: {str(e)}"}

@mcp.tool(
    title="Generate Fairness Report",
    description="Generate a comprehensive fairness assessment report in markdown format."
)
def generate_fairness_report(
    assessment_results: Dict[str, Any] = Field(description="Results from fairness_assessment tool"),
    mitigation_results: Optional[Dict[str, Any]] = Field(
        description="Optional results from mitigate_fairness tool", default=None
    ),
    context: str = Field(description="Context about the model/use case", default="")
) -> Dict[str, str]:
    """
    Generate a comprehensive fairness report in markdown format.
    """
    try:
        if "error" in assessment_results:
            return {"error": "Cannot generate report from assessment results with errors"}

        # Start building markdown report
        report_lines = [
            "# Fairness Assessment Report",
            "",
            f"**Context:** {context}" if context else "",
            "",
            "## Executive Summary",
            ""
        ]

        # Overall assessment
        dp_diff = assessment_results.get("demographic_parity_difference", 0)
        concerning = assessment_results.get("interpretation", {}).get("concerning", False)
        
        if concerning:
            report_lines.extend([
                f"⚠️ **CONCERNING BIAS DETECTED**: Demographic parity difference of {dp_diff:.3f} exceeds recommended threshold.",
                ""
            ])
        else:
            report_lines.extend([
                f"✅ **ACCEPTABLE FAIRNESS**: Demographic parity difference of {dp_diff:.3f} within acceptable range.",
                ""
            ])

        # Detailed metrics
        report_lines.extend([
            "## Detailed Metrics",
            "",
            f"- **Overall Selection Rate:** {assessment_results.get('overall_selection_rate', 'N/A'):.3f}",
            f"- **Demographic Parity Difference:** {dp_diff:.3f}",
            f"- **Total Samples:** {assessment_results.get('total_samples', 'N/A')}",
            ""
        ])

        # Group breakdown
        if "group_selection_rates" in assessment_results:
            report_lines.extend([
                "## Group-Level Analysis",
                "",
                "| Group | Selection Rate |",
                "|-------|----------------|"
            ])
            
            for group, rate in assessment_results["group_selection_rates"].items():
                report_lines.append(f"| {group} | {rate:.3f} |")
            
            report_lines.append("")

        # Mitigation results if provided
        if mitigation_results and "error" not in mitigation_results:
            mit_res = mitigation_results.get("mitigation_results", {})
            report_lines.extend([
                "## Bias Mitigation Results",
                "",
                f"- **Constraint Applied:** {mit_res.get('constraint_applied', 'N/A')}",
                f"- **Accuracy Before:** {mit_res.get('accuracy_baseline', 'N/A'):.3f}",
                f"- **Accuracy After:** {mit_res.get('accuracy_post_mitigation', 'N/A'):.3f}",
                f"- **Accuracy Cost:** {mit_res.get('accuracy_change', 'N/A'):.3f}",
                f"- **DP Difference Before:** {mit_res.get('dp_difference_baseline', 'N/A'):.3f}",
                f"- **DP Difference After:** {mit_res.get('dp_difference_mitigated', 'N/A'):.3f}",
                f"- **Fairness Improvement:** {mit_res.get('dp_improvement', 'N/A'):.3f}",
                ""
            ])

        # Recommendations
        report_lines.extend([
            "## Recommendations",
            ""
        ])

        if concerning:
            report_lines.extend([
                "1. **Immediate Action Required:** Bias exceeds acceptable thresholds",
                "2. **Apply Mitigation:** Consider demographic parity or equalized odds constraints",
                "3. **Data Review:** Examine training data for representation issues",
                "4. **Stakeholder Review:** Involve domain experts and affected communities"
            ])
        else:
            report_lines.extend([
                "1. **Continue Monitoring:** Maintain regular fairness assessments",
                "2. **Document Compliance:** Keep records for audit purposes",
                "3. **Stakeholder Communication:** Share results with relevant teams"
            ])

        report_lines.extend([
            "",
            "---",
            f"*Report generated by Fairlearn MCP Server*"
        ])

        markdown_report = "\n".join(report_lines)

        return {
            "report_markdown": markdown_report,
            "report_format": "markdown",
            "report_length": len(markdown_report)
        }

    except Exception as e:
        return {"error": f"Report generation failed: {str(e)}"}

# ------------------------------
# Resources
# ------------------------------

@mcp.resource("fairlearn://help", name="Fairlearn MCP Help")
def fairlearn_help() -> str:
    return """
# Fairlearn MCP Server Help

## Available Tools

### fairness_assessment
Compute basic fairness metrics including demographic parity and selection rates.
- Input: predictions, true labels, sensitive features
- Output: group-level metrics and overall assessment

### advanced_fairness_metrics  
Compute comprehensive metrics including TPR, FPR, accuracy disparities.
- Input: predictions, true labels, sensitive features
- Output: detailed metric breakdown with disparities

### mitigate_fairness
Train fair models using ExponentiatedGradient with fairness constraints.
- Input: features, labels, sensitive features, constraint type
- Output: mitigated model with performance comparison

### generate_fairness_report
Create comprehensive markdown reports from assessment results.
- Input: assessment results, optional mitigation results
- Output: formatted markdown report

## Workflow

1. Run `fairness_assessment` on your model predictions
2. If bias detected, use `mitigate_fairness` to train fair model
3. Generate comprehensive report with `generate_fairness_report`
4. Monitor and iterate as needed

## Constraint Types

- **demographic_parity**: Equal selection rates across groups
- **equalized_odds**: Equal TPR and FPR across groups

## Interpretation

- DP difference > 0.1: Concerning bias requiring attention
- DP difference < 0.05: Generally acceptable
- Monitor accuracy-fairness tradeoffs in mitigation
"""

# ------------------------------
# Prompts
# ------------------------------

@mcp.prompt(
    name="fairlearn.interpret_results",
    description="Interpret Fairlearn assessment results and provide recommendations."
)
def interpret_results(
    assessment_results: str,
    business_context: str = ""
) -> str:
    return f"""
You are a fairness expert analyzing ML model assessment results. 

Assessment Results:
{assessment_results}

Business Context:
{business_context}

Please provide:
1. Clear interpretation of the fairness metrics
2. Assessment of bias severity (none/moderate/severe)
3. Specific recommendations for next steps
4. Explanation of any tradeoffs to consider
5. Timeline suggestions for remediation if needed

Keep your response practical and actionable for a technical team.
"""

@mcp.prompt(
    name="fairlearn.explain_constraints",
    description="Explain different fairness constraints and when to use them."
)
def explain_constraints(
    use_case: str,
    protected_attributes: str = ""
) -> str:
    return f"""
Explain fairness constraints for this use case:

Use Case: {use_case}
Protected Attributes: {protected_attributes}

For each constraint type (demographic parity, equalized odds), explain:
1. What it measures
2. When it's most appropriate
3. Potential tradeoffs
4. Implementation considerations

Recommend which constraint would be best for this specific use case and why.
"""
