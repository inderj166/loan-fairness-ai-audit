"""
loan_fairness_report.py

A lightweight, single-file ethical AI project: generates a synthetic loan
dataset, loads it into SQLite, runs SQL fairness queries, trains a
"gender-blind" approval model, audits it for disparate impact, and writes
everything into ONE self-contained HTML report (charts are embedded
directly as base64 images -- no separate image files, no web server,
just open the .html file in any browser).

Run:
    pip install -r requirements.txt
    python loan_fairness_report.py

Then open fairness_report.html in your browser.
"""

import base64
import io
import sqlite3

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------------
# 1. Generate synthetic data
# ---------------------------------------------------------------------
def build_dataset(n: int = 400, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    gender = rng.choice(["Male", "Female"], size=n, p=[0.5, 0.5])

    # A realistic gender pay gap: gender itself is never used as a model
    # feature, but income is -- and income is correlated with gender here,
    # so it can act as a proxy for a "gender-blind" model.
    income_mean = np.where(gender == "Female", 32000, 55000)
    annual_income = np.clip(rng.normal(income_mean, 12000), 15000, None)
    credit_score = np.clip(rng.normal(650, 60, size=n), 300, 850)
    existing_debt = np.clip(rng.normal(8000, 4000, size=n), 0, None)

    z = -9 + 0.00007 * annual_income + 0.012 * credit_score - 0.00015 * existing_debt
    prob_approved = 1 / (1 + np.exp(-z))
    approved = rng.binomial(1, np.clip(prob_approved, 0, 1))

    return pd.DataFrame({
        "gender": gender,
        "annual_income": annual_income.round(0),
        "credit_score": credit_score.round(0),
        "existing_debt": existing_debt.round(0),
        "approved": np.where(approved == 1, "Yes", "No"),
    })


# ---------------------------------------------------------------------
# 2. Load into SQLite and run the SQL audit queries
# ---------------------------------------------------------------------
def build_database(df: pd.DataFrame, db_path: str = "loan_data.db") -> None:
    conn = sqlite3.connect(db_path)
    conn.execute("DROP TABLE IF EXISTS loan_applicants")
    conn.execute("""
        CREATE TABLE loan_applicants (
            applicant_id   INTEGER PRIMARY KEY AUTOINCREMENT,
            gender         TEXT NOT NULL,
            annual_income  REAL NOT NULL,
            credit_score   REAL NOT NULL,
            existing_debt  REAL NOT NULL,
            approved       TEXT NOT NULL
        )
    """)
    df.to_sql("loan_applicants", conn, if_exists="append", index=False)
    conn.commit()
    conn.close()


def run_sql_queries(db_path: str = "loan_data.db") -> dict:
    conn = sqlite3.connect(db_path)
    overall = pd.read_sql(
        "SELECT ROUND(100.0*SUM(CASE WHEN approved='Yes' THEN 1 ELSE 0 END)/COUNT(*),2) AS approval_rate_pct, COUNT(*) AS total FROM loan_applicants",
        conn,
    )
    by_gender = pd.read_sql(
        """SELECT gender, COUNT(*) AS applicants,
                  ROUND(100.0*SUM(CASE WHEN approved='Yes' THEN 1 ELSE 0 END)/COUNT(*),2) AS approval_rate_pct
           FROM loan_applicants GROUP BY gender""",
        conn,
    )
    income_by_gender = pd.read_sql(
        "SELECT gender, ROUND(AVG(annual_income),0) AS avg_income FROM loan_applicants GROUP BY gender",
        conn,
    )
    by_bracket = pd.read_sql(
        """
        SELECT
            CASE
                WHEN annual_income < 30000 THEN '1. Under 30k'
                WHEN annual_income < 50000 THEN '2. 30k-50k'
                WHEN annual_income < 70000 THEN '3. 50k-70k'
                ELSE '4. 70k+'
            END AS income_bracket,
            gender,
            COUNT(*) AS applicants,
            ROUND(100.0*SUM(CASE WHEN approved='Yes' THEN 1 ELSE 0 END)/COUNT(*),2) AS approval_rate_pct
        FROM loan_applicants
        GROUP BY income_bracket, gender
        ORDER BY income_bracket, gender
        """,
        conn,
    )
    conn.close()
    return {"overall": overall, "by_gender": by_gender, "income_by_gender": income_by_gender, "by_bracket": by_bracket}


# ---------------------------------------------------------------------
# 3. Train a "gender-blind" model and audit it for fairness
# ---------------------------------------------------------------------
def train_and_audit(df: pd.DataFrame) -> dict:
    features = ["annual_income", "credit_score", "existing_debt"]
    X = df[features]
    y = (df["approved"] == "Yes").astype(int)
    gender = df["gender"]

    X_train, X_test, y_train, y_test, gender_train, gender_test = train_test_split(
        X, y, gender, test_size=0.3, random_state=42, stratify=y
    )
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    results = pd.DataFrame({"prediction": y_pred, "group": gender_test.values})
    rates = results.groupby("group")["prediction"].mean()
    male_rate, female_rate = rates.get("Male", 0), rates.get("Female", 0)
    lower, higher = min(male_rate, female_rate), max(male_rate, female_rate)
    ratio = lower / higher if higher > 0 else float("nan")

    return {
        "accuracy": accuracy,
        "male_rate": male_rate,
        "female_rate": female_rate,
        "disparate_impact_ratio": ratio,
        "passes_four_fifths_rule": ratio >= 0.8,
    }


# ---------------------------------------------------------------------
# 4. Chart helper: render a matplotlib figure straight to a base64
#    <img> tag, so the HTML report is a single self-contained file
# ---------------------------------------------------------------------
def fig_to_base64_img() -> str:
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close()
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    return f'<img src="data:image/png;base64,{encoded}" style="max-width:600px;">'


def make_approval_chart(by_gender: pd.DataFrame) -> str:
    plt.figure(figsize=(5, 4))
    plt.bar(by_gender["gender"], by_gender["approval_rate_pct"], color=["#9b59b6", "#3498db"])
    plt.ylabel("Approval rate (%)")
    plt.title("Loan Approval Rate by Gender")
    return fig_to_base64_img()


def make_income_chart(income_by_gender: pd.DataFrame) -> str:
    plt.figure(figsize=(5, 4))
    plt.bar(income_by_gender["gender"], income_by_gender["avg_income"], color=["#9b59b6", "#3498db"])
    plt.ylabel("Average annual income ($)")
    plt.title("Average Income by Gender\n(the proxy variable)")
    return fig_to_base64_img()


# ---------------------------------------------------------------------
# 5. Build the HTML report
# ---------------------------------------------------------------------
def build_html_report(sql_results: dict, audit: dict, chart1: str, chart2: str, out_path: str = "fairness_report.html") -> None:
    overall = sql_results["overall"].iloc[0]
    by_gender_rows = "".join(
        f"<tr><td>{r.gender}</td><td>{r.applicants}</td><td>{r.approval_rate_pct}%</td></tr>"
        for r in sql_results["by_gender"].itertuples()
    )
    bracket_rows = "".join(
        f"<tr><td>{r.income_bracket}</td><td>{r.gender}</td><td>{r.applicants}</td><td>{r.approval_rate_pct}%</td></tr>"
        for r in sql_results["by_bracket"].itertuples()
    )

    verdict = (
        "❌ FAILS the four-fifths rule" if not audit["passes_four_fifths_rule"]
        else "✅ Passes the four-fifths rule"
    )
    verdict_color = "#c0392b" if not audit["passes_four_fifths_rule"] else "#27ae60"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Ethical AI Fairness Report — Loan Approval Model</title>
<style>
    body {{ font-family: -apple-system, Segoe UI, Arial, sans-serif; max-width: 900px; margin: 40px auto; padding: 0 20px; color: #222; line-height: 1.5; }}
    h1 {{ border-bottom: 3px solid #2c3e50; padding-bottom: 10px; }}
    h2 {{ color: #2c3e50; margin-top: 40px; }}
    table {{ border-collapse: collapse; width: 100%; margin: 15px 0; }}
    th, td {{ border: 1px solid #ddd; padding: 8px 12px; text-align: left; }}
    th {{ background-color: #2c3e50; color: white; }}
    tr:nth-child(even) {{ background-color: #f7f7f7; }}
    .verdict {{ font-size: 1.3em; font-weight: bold; color: {verdict_color}; padding: 15px; border: 2px solid {verdict_color}; border-radius: 8px; display: inline-block; }}
    .note {{ background: #fff8dc; border-left: 4px solid #e67e22; padding: 12px 16px; margin: 20px 0; }}
    .charts {{ display: flex; gap: 20px; flex-wrap: wrap; }}
</style>
</head>
<body>

<h1>⚖️ Ethical AI Fairness Report — Loan Approval Model</h1>
<p>A lightweight audit combining <strong>SQL</strong> (data queries), <strong>Python</strong> (model training + fairness metrics), and this <strong>HTML</strong> report to demonstrate responsible AI evaluation on a synthetic loan approval dataset.</p>

<h2>1. Overall Portfolio Summary (SQL query)</h2>
<table>
<tr><th>Total Applicants</th><th>Overall Approval Rate</th></tr>
<tr><td>{int(overall.total)}</td><td>{overall.approval_rate_pct}%</td></tr>
</table>

<h2>2. Approval Rate by Gender (SQL query)</h2>
<table>
<tr><th>Gender</th><th>Applicants</th><th>Approval Rate</th></tr>
{by_gender_rows}
</table>
<div class="charts">
{chart1}
{chart2}
</div>

<h2>3. Approval Rate by Income Bracket and Gender (SQL query)</h2>
<p>Checking whether the gap narrows once income is controlled for -- this helps distinguish "the model is biased" from "the model is reflecting an income gap that itself may need addressing."</p>
<table>
<tr><th>Income Bracket</th><th>Gender</th><th>Applicants</th><th>Approval Rate</th></tr>
{bracket_rows}
</table>

<h2>4. Fairness Audit of a "Gender-Blind" Model (Python + scikit-learn)</h2>
<p>A logistic regression model was trained to predict loan approval using only <code>annual_income</code>, <code>credit_score</code>, and <code>existing_debt</code> — <strong>gender was never given to the model as an input.</strong> Model accuracy on the held-out test set: <strong>{audit['accuracy']:.1%}</strong>.</p>

<table>
<tr><th>Metric</th><th>Value</th></tr>
<tr><td>Male approval rate (model predictions)</td><td>{audit['male_rate']:.1%}</td></tr>
<tr><td>Female approval rate (model predictions)</td><td>{audit['female_rate']:.1%}</td></tr>
<tr><td>Disparate Impact Ratio</td><td>{audit['disparate_impact_ratio']:.3f}</td></tr>
</table>

<p class="verdict">{verdict}</p>

<div class="note">
<strong>Why this matters:</strong> The EEOC's "four-fifths rule" treats a selection-rate ratio below 0.8 between groups as evidence of disparate impact. This model never saw gender as an input feature, yet it can still fail this rule -- because <code>annual_income</code> is correlated with gender in this dataset (reflecting a real, well-documented gender pay gap pattern), and the model learned that correlation as a proxy. <strong>Excluding a protected attribute from a model's features does not, by itself, guarantee the model is fair.</strong>
</div>

<h2>5. Takeaway</h2>
<p>This is exactly the kind of audit a responsible AI/data team should run before deploying any model that affects people's access to credit, jobs, or other opportunities — not just checking whether the model is accurate, but checking <em>who</em> it's accurate for, and whether it treats groups equitably.</p>

</body>
</html>"""

    with open(out_path, "w") as f:
        f.write(html)
    print(f"Report written to {out_path} -- open it in your browser to view.")


# ---------------------------------------------------------------------
def main():
    print("Generating synthetic loan applicant data...")
    df = build_dataset()

    print("Building SQLite database and running SQL queries...")
    build_database(df)
    sql_results = run_sql_queries()

    print("Training a gender-blind model and running the fairness audit...")
    audit = train_and_audit(df)

    print("Rendering charts...")
    chart1 = make_approval_chart(sql_results["by_gender"])
    chart2 = make_income_chart(sql_results["income_by_gender"])

    print("Writing HTML report...")
    build_html_report(sql_results, audit, chart1, chart2)

    print("\nDone. Open fairness_report.html in your browser to see the full report.")


if __name__ == "__main__":
    main()
