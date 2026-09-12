# Ethical AI Fairness Audit — Loan Approval Model

A lightweight, single-purpose project combining **SQL**, **Python**, and **HTML** to demonstrate a core responsible-AI skill: checking whether a machine learning model discriminates against a protected group — even when that group's attribute was never given to the model as an input.

No server, no dashboard app to run — the output is one self-contained HTML report you open directly in a browser.

## What it does

1. **Generates** a synthetic loan applicant dataset (income, credit score, debt, gender, approval outcome) with a realistic, documented bias pattern: a gender pay gap in income.
2. **Loads it into SQLite** and runs SQL queries (`loan_fairness_audit.sql`) to check approval rates by gender and by income bracket.
3. **Trains a "gender-blind" model** in Python (`scikit-learn`) that never sees gender as a feature.
4. **Audits the model** using the EEOC's "four-fifths rule" (Disparate Impact Ratio) — a standard fairness metric.
5. **Writes everything into one HTML report** (`fairness_report.html`) with embedded charts — no separate image files, nothing to break.

## The finding

The model never uses gender as an input, but it still fails the fairness check: **Male approval rate ~90% vs Female approval rate ~28% (Disparate Impact Ratio 0.31, well below the 0.8 threshold)**. This happens because `annual_income` — a completely reasonable, legitimate-looking model feature — is correlated with gender in this dataset (reflecting a real gender pay gap pattern), and the model learns that correlation as a proxy.

**The lesson:** excluding a protected attribute from a model's features does not, by itself, make the model fair.

## Files

- `loan_fairness_report.py` — the entire pipeline: data generation, SQL, model training, fairness audit, HTML report generation
- `loan_fairness_audit.sql` — the SQL schema and queries, viewable/runnable on their own
- `fairness_report.html` — a sample generated report (open directly in your browser)
- `requirements.txt`

## How to run it

```bash
pip install -r requirements.txt
python loan_fairness_report.py
```

Then open `fairness_report.html` in any browser. That's it — no other setup.

## Why this is worth knowing (for interviews)

- **Why a proxy variable (income) instead of using gender directly?** Because that's the realistic failure mode. No serious company ships a model that uses a protected attribute directly anymore — the real risk today is a model that looks compliant on paper but discriminates through a correlated feature.
- **Why check approval rate by income bracket too?** To separate two different problems: "the model treats similarly-qualified people differently based on gender" versus "the model reflects a real income gap between groups that itself needs a business/policy conversation, not just a model fix." Both matter, but they call for different responses.
- **Why one HTML file instead of a dashboard?** Simplicity. A recruiter or reviewer can open the file directly with zero setup — no server, no installed packages needed just to view the result.

## License

MIT — feel free to fork and adapt for your own portfolio.
