import pandas as pd
import numpy as np
from faker import Faker
from datetime import datetime, timedelta
import random

# ----------------------------
# Setup
# ----------------------------
fake = Faker()
np.random.seed(42)
random.seed(42)

NUM_RECORDS = 2000

cities = [
    "Lahore", "Karachi", "Islamabad",
    "Rawalpindi", "Faisalabad",
    "Multan", "Peshawar", "Quetta"
]

subscription_types = ["Basic", "Standard", "Premium"]
payment_methods = ["Credit Card", "Debit Card", "Bank Transfer", "JazzCash", "EasyPaisa"]
device_types = ["Android", "iPhone", "Web"]

data = []

# ----------------------------
# Generate Data
# ----------------------------
for i in range(NUM_RECORDS):

    customer_id = f"CUST{1000+i}"

    age = np.random.randint(18, 65)

    gender = np.random.choice(["Male", "Female"])

    city = np.random.choice(cities)

    subscription = np.random.choice(
        subscription_types,
        p=[0.5, 0.35, 0.15]
    )

    # Spending depends on subscription
    if subscription == "Basic":
        monthly_spending = round(np.random.uniform(10, 30), 2)
    elif subscription == "Standard":
        monthly_spending = round(np.random.uniform(30, 70), 2)
    else:
        monthly_spending = round(np.random.uniform(70, 150), 2)

    tenure = np.random.randint(1, 61)

    purchases = max(
        1,
        int(np.random.normal(tenure * 1.5, 5))
    )

    support_requests = np.random.randint(0, 11)

    login_frequency = np.random.randint(0, 31)

    satisfaction = np.random.randint(1, 6)

    payment = np.random.choice(payment_methods)

    device = np.random.choice(device_types)

    auto_renew = np.random.choice(
        ["Yes", "No"],
        p=[0.7, 0.3]
    )

    # Last activity date
    days_ago = np.random.randint(0, 180)
    last_activity = (
        datetime.today() - timedelta(days=days_ago)
    ).strftime("%Y-%m-%d")

    # ----------------------------
    # Intelligent Churn Logic
    # ----------------------------
    churn_score = 0

    if satisfaction <= 2:
        churn_score += 3

    if login_frequency < 5:
        churn_score += 3

    if support_requests >= 6:
        churn_score += 2

    if tenure < 8:
        churn_score += 2

    if subscription == "Basic":
        churn_score += 1

    if purchases < 10:
        churn_score += 1

    if auto_renew == "No":
        churn_score += 1

    if monthly_spending > 80:
        churn_score -= 2

    if satisfaction >= 4:
        churn_score -= 2

    if login_frequency > 20:
        churn_score -= 2

    churn = "Yes" if churn_score >= 5 else "No"

    data.append([
        customer_id,
        age,
        gender,
        city,
        subscription,
        monthly_spending,
        tenure,
        purchases,
        support_requests,
        login_frequency,
        last_activity,
        satisfaction,
        payment,
        device,
        auto_renew,
        churn
    ])

# ----------------------------
# Create DataFrame
# ----------------------------
columns = [
    "CustomerID",
    "Age",
    "Gender",
    "City",
    "SubscriptionType",
    "MonthlySpending",
    "Tenure",
    "NumberOfPurchases",
    "CustomerSupportRequests",
    "LoginFrequency",
    "LastActivityDate",
    "SatisfactionScore",
    "PaymentMethod",
    "DeviceType",
    "AutoRenew",
    "ChurnStatus"
]

df = pd.DataFrame(data, columns=columns)

# ----------------------------
# Add Missing Values (2%)
# ----------------------------
for col in [
    "MonthlySpending",
    "SatisfactionScore",
    "LoginFrequency"
]:
    idx = df.sample(frac=0.02, random_state=42).index
    df.loc[idx, col] = np.nan

# ----------------------------
# Add Duplicate Rows
# ----------------------------
duplicates = df.sample(20, random_state=42)
df = pd.concat([df, duplicates], ignore_index=True)

# ----------------------------
# Save CSV
# ----------------------------
df.to_csv("customer_churn_dataset.csv", index=False)

print("Dataset Created Successfully!")
print(df.head())

print("\nShape:", df.shape)

print("\nChurn Distribution:")
print(df["ChurnStatus"].value_counts())