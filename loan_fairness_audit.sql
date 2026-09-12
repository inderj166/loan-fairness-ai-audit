-- ============================================================
-- loan_fairness_audit.sql
-- Schema + audit queries for a small loan-approval fairness check.
-- Run this against loan_data.db after running loan_fairness_report.py
-- (or open loan_data.db in any SQLite browser and run these queries
-- directly).
-- ============================================================

-- Schema (also created automatically by the Python script -- shown
-- here so the database design is visible and reviewable on its own).
CREATE TABLE IF NOT EXISTS loan_applicants (
    applicant_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    gender         TEXT NOT NULL CHECK (gender IN ('Male', 'Female')),
    annual_income  REAL NOT NULL,
    credit_score   REAL NOT NULL,
    existing_debt  REAL NOT NULL,
    approved       TEXT NOT NULL CHECK (approved IN ('Yes', 'No'))
);

-- 1. Overall approval rate
SELECT
    ROUND(100.0 * SUM(CASE WHEN approved = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 2) AS approval_rate_pct,
    COUNT(*) AS total_applicants
FROM loan_applicants;

-- 2. Approval rate by gender -- the core fairness question
SELECT
    gender,
    COUNT(*) AS applicants,
    ROUND(100.0 * SUM(CASE WHEN approved = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 2) AS approval_rate_pct
FROM loan_applicants
GROUP BY gender;

-- 3. Is average income different by gender? (checking for a proxy
--    variable that a "gender-blind" model could still pick up on)
SELECT
    gender,
    ROUND(AVG(annual_income), 0) AS avg_income,
    ROUND(AVG(credit_score), 0) AS avg_credit_score
FROM loan_applicants
GROUP BY gender;

-- 4. Approval rate by income bracket, split by gender -- checks
--    whether the gap persists even among similarly-qualified applicants
SELECT
    CASE
        WHEN annual_income < 30000 THEN 'Under 30k'
        WHEN annual_income < 50000 THEN '30k-50k'
        WHEN annual_income < 70000 THEN '50k-70k'
        ELSE '70k+'
    END AS income_bracket,
    gender,
    COUNT(*) AS applicants,
    ROUND(100.0 * SUM(CASE WHEN approved = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 2) AS approval_rate_pct
FROM loan_applicants
GROUP BY income_bracket, gender
ORDER BY income_bracket, gender;
