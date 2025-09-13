
"""
MCP Server Template
"""

from mcp.server.fastmcp import FastMCP
from pydantic import Field


mcp = FastMCP("Echo Server", port=3000, stateless_http=True, debug=True)





    # fairness_assessment tool

from fairlearn.metrics import MetricFrame, selection_rate, demographic_parity_difference
import numpy as np

@mcp.tool(
    title="Fairness Assessment",
    description="Compute fairness metrics (demographic parity, selection rate) for predictions."
)
def fairness_assessment(
    y_true: list = Field(description="True labels"),
    y_pred: list = Field(description="Predicted labels"),
    sensitive_features: list = Field(description="Group identifiers (e.g. gender, ethnicity)")
) -> dict:
    """Run Fairlearn metrics on predictions grouped by sensitive features"""
    try:
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        sensitive_features = np.array(sensitive_features)

        # Compute group-wise selection rate
        metric_frame = MetricFrame(
            metrics={"selection_rate": selection_rate},
            y_true=y_true,
            y_pred=y_pred,
            sensitive_features=sensitive_features,
        )

        # Compute overall demographic parity difference
        dp_diff = demographic_parity_difference(y_true, y_pred, sensitive_features=sensitive_features)

        return {
            "overall_selection_rate": float(metric_frame.overall["selection_rate"]),
            "group_selection_rates": metric_frame.by_group["selection_rate"].to_dict(),
            "demographic_parity_difference": float(dp_diff)
        }
    except Exception as e:
        return {"error": str(e)}

@mcp.tool(
    title="Fairness Audit Note",
    description="Generate an audit-ready PDF with fairness metrics, definitions, coverage, and mitigation recommendations."
)
def generate_audit_note(assessment_results: dict) -> str:
    """
    Creates a structured PDF report (path returned).
    """
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet
    
    file_path = "/tmp/fairness_audit_note.pdf"
    doc = SimpleDocTemplate(file_path)
    styles = getSampleStyleSheet()
    story = []
    
    story.append(Paragraph("Fairness Audit Report", styles["Title"]))
    story.append(Spacer(1, 12))
    story.append(Paragraph(f"Disparate Impact: {assessment_results['disparate_impact']:.2f}", styles["Normal"]))
    story.append(Paragraph(f"TPR Gap: {assessment_results['tpr_gap']:.2%}", styles["Normal"]))
    story.append(Paragraph("Definitions: DI < 0.8 indicates potential disparate impact; TPR gap > 5% signals EO violation.", styles["Normal"]))
    story.append(Paragraph("Recommendations: Consider threshold optimization or retraining with ExponentiatedGradient mitigation.", styles["Normal"]))
    
    doc.build(story)
    return file_path

from fairlearn.reductions import ExponentiatedGradient, DemographicParity, EqualizedOdds
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd

@mcp.tool(
    title="Fairness Mitigation Trainer",
    description="Retrains a credit approval model with fairness constraints (Demographic Parity or Equalized Odds)."
)
def mitigate_fairness(
    X: list = Field(description="Feature matrix as list of lists"),
    y: list = Field(description="Binary labels (1=approve, 0=reject)"),
    sensitive_features: list = Field(description="Protected attribute values (gender/age group/etc.)"),
    constraint: str = Field(description="Fairness constraint to apply: 'demographic_parity' or 'equalized_odds'", default="demographic_parity")
) -> dict:
    """
    Trains a fair classifier using ExponentiatedGradient and returns new metrics + coefficients.
    """
    try:
        X = np.array(X)
        y = np.array(y)
        sensitive_features = np.array(sensitive_features)

        # Train/test split for evaluation
        X_train, X_test, y_train, y_test, sf_train, sf_test = train_test_split(
            X, y, sensitive_features, test_size=0.3, random_state=42, stratify=y
        )

        # Base estimator
        base_estimator = LogisticRegression(solver="liblinear")

        # Choose fairness constraint
        if constraint == "equalized_odds":
            constraint_obj = EqualizedOdds()
        else:
            constraint_obj = DemographicParity()

        # Train fairness-constrained model
        mitigator = ExponentiatedGradient(estimator=base_estimator, constraints=constraint_obj)
        mitigator.fit(X_train, y_train, sensitive_features=sf_train)

        # Evaluate
        y_pred = mitigator.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        # Return results
        return {
            "accuracy_post_mitigation": float(acc),
            "model_weights": mitigator._pmf_predictor.estimator.coef_.tolist(),
            "intercept": mitigator._pmf_predictor.estimator.intercept_.tolist(),
            "constraint": constraint,
            "note": "Model trained with fairness constraints; monitor accuracy-fairness tradeoff."
        }

    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
