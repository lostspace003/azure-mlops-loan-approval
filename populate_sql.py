# populate_sql.py
import pyodbc
import random
import numpy as np
from config import get_sql_connection_string

# Set seed for reproducible data across all learners
random.seed(42)
np.random.seed(42)

conn_str = get_sql_connection_string()
conn = pyodbc.connect(conn_str)
cursor = conn.cursor()

# Drop table if it already exists, then create
cursor.execute("""
    IF OBJECT_ID('LoanApplications', 'U') IS NOT NULL
        DROP TABLE LoanApplications
""")

cursor.execute("""
    CREATE TABLE LoanApplications (
        ApplicationID INT IDENTITY(1,1) PRIMARY KEY,
        Age INT,
        Income FLOAT,
        LoanAmount FLOAT,
        CreditScore INT,
        EmploymentYears INT,
        DebtToIncomeRatio FLOAT,
        NumCreditLines INT,
        Education VARCHAR(20),
        MaritalStatus VARCHAR(20),
        HasMortgage BIT,
        HasDependents BIT,
        LoanPurpose VARCHAR(30),
        PreviousDefault BIT,
        Approved BIT  -- Target column
    )
""")

# Generate 1000 rows of realistic loan data
education_levels = ['HighSchool', 'Bachelor', 'Master', 'PhD']
marital_statuses = ['Single', 'Married', 'Divorced']
loan_purposes = ['Home', 'Car', 'Education', 'Business', 'Personal']

for i in range(1000):
    age = random.randint(21, 65)
    income = round(random.uniform(25000, 200000), 2)
    loan_amount = round(random.uniform(5000, 500000), 2)
    credit_score = random.randint(300, 850)
    employment_years = random.randint(0, 40)
    dti = round(random.uniform(0.05, 0.80), 2)
    num_credit_lines = random.randint(0, 15)
    education = random.choice(education_levels)
    marital = random.choice(marital_statuses)
    has_mortgage = random.randint(0, 1)
    has_dependents = random.randint(0, 1)
    loan_purpose = random.choice(loan_purposes)
    previous_default = random.randint(0, 1)

    # Realistic approval logic
    score = 0
    score += 2 if credit_score > 700 else (1 if credit_score > 600 else 0)
    score += 2 if income > 75000 else (1 if income > 50000 else 0)
    score += 1 if dti < 0.35 else 0
    score += 1 if employment_years > 3 else 0
    score -= 2 if previous_default == 1 else 0
    score += 1 if loan_amount / max(income, 1) < 3 else 0
    approved = 1 if score >= 4 else 0

    cursor.execute("""
        INSERT INTO LoanApplications
        (Age, Income, LoanAmount, CreditScore, EmploymentYears,
         DebtToIncomeRatio, NumCreditLines, Education, MaritalStatus,
         HasMortgage, HasDependents, LoanPurpose, PreviousDefault, Approved)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, age, income, loan_amount, credit_score, employment_years,
       dti, num_credit_lines, education, marital,
       has_mortgage, has_dependents, loan_purpose, previous_default, approved)

conn.commit()
cursor.execute("SELECT COUNT(*) FROM LoanApplications")
print(f"Total rows inserted: {cursor.fetchone()[0]}")
cursor.execute("SELECT AVG(CAST(Approved AS FLOAT)) FROM LoanApplications")
print(f"Approval rate: {cursor.fetchone()[0]:.2%}")
conn.close()
