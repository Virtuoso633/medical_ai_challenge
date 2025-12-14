https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset?utm_source=chatgpt.com

download the dataset from the link above and save it in the same directory as this file save in a folder data.

use google colab extension and in that create a notebook there we will do all the analysis.

📊 Part 2: Exploratory Data Analysis (30 pts)
Create a notebook or Python script named analysis.py or analysis.ipynb that performs and outputs the following:

1. Summary Statistics
   Compute and print in a clear tabular form:
   Feature
   Mean
   Std
   Min
   Median
   Max
   % Missing

For all numeric columns: age, bmi, HbA1c_level, blood_glucose_level.
Also report:
Number of males vs females

Smoking history counts

Diabetes prevalence (%)

2. Correlation & Feature Insights
   Compute Pearson correlations between numeric variables and diabetes.
   Create:
   A sorted table of correlations (highest → lowest)

Optional: visualize using seaborn.heatmap or matplotlib

Deliverable: correlation.json with { feature: corr }

3. Risk Group Statistics
   Split dataset into meaningful cohorts and compute diabetes prevalence (% positive):
   Cohort
   Condition
   N
   Diabetes %
   Elderly
   age ≥ 60

Overweight
BMI ≥ 30

Hypertension
hypertension = 1

Heart Disease
heart_disease = 1

High Glucose
blood_glucose_level ≥ 180

Smokers
smoking_history ∈ {current, ever, former}

Save this as risk_groups.csv.

4. Feature Distributions
   Generate and save to /out/plots/:
   Histogram of age, BMI, HbA1c_level, blood_glucose_level

Boxplots of each numeric feature grouped by diabetes

Bar chart of smoking history vs diabetes prevalence

(You can use matplotlib, seaborn, or plotly.express.)
