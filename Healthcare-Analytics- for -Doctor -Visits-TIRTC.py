# Converted from Google Colab/Jupyter Notebook: Healthcare-Analytics- for -Doctor -Visits-TIRTC.ipynb
# Generated from notebook cells; markdown cells are preserved as comments.

# %% Cell 1 (markdown)
# # **Project Overview**
# # **Healthcare Analytics for Doctor Visits**
# # **TIRTC** **Project**

# %% Cell 2 (markdown)
# # **1**. **Import** **Libraries**

# %% Cell 3 (code)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from IPython.display import display

pd.set_option("display.max_columns", None)
pd.set_option("display.float_format", lambda x: f"{x:,.2f}")

sns.set_theme(style="whitegrid")

# %% Cell 4 (markdown)
# # **2**. **Load** **the** **Uploaded** **Dataset**

# %% Cell 5 (code)
# Update this path if running outside Google Colab
file_path = "/content/1776250375-P2-Healthcare Analytics for Doctor Visits.csv"

df = pd.read_csv(file_path)
print(f"Dataset loaded successfully: {df.shape[0]:,} rows and {df.shape[1]} columns")
display(df.head())

# %% Cell 6 (markdown)
# # **Data** **Inspection** **and** **Cleaning**

# %% Cell 7 (markdown)
# ## **3**. **Initial** **Dataset** **Inspection**

# %% Cell 8 (code)
print("Shape:", df.shape)
print("\nColumn names:")
print(df.columns.tolist())

print("\nData types:")
display(df.dtypes.to_frame("dtype"))

print("\nDataset information:")
df.info()

# %% Cell 9 (code)
# First and last observations
display(df.head())
display(df.tail())

# Statistical summary
display(df.describe(include="all").T)

# %% Cell 10 (markdown)
# # **4**. **Data** **Quality** **Check** **and** **Cleaning**

# %% Cell 11 (markdown)
# # The uploaded data contains an Unnamed: 0 column that functions as an index/record identifier rather than an analytical feature. It is removed from the analysis. Missing values, duplicate records, and categorical levels are also checked before analysis.   

# %% Cell 12 (code)
# Work on a copy
data = df.copy()

# Remove CSV index-like column if present
index_like_cols = [c for c in data.columns if c.lower().startswith("unnamed")]
data = data.drop(columns=index_like_cols, errors="ignore")

# Missing values
missing = data.isnull().sum().sort_values(ascending=False)
missing_pct = (data.isnull().mean() * 100).sort_values(ascending=False)
missing_report = pd.DataFrame({"missing_count": missing, "missing_percent": missing_pct})
display(missing_report)

# Duplicate rows
print("Duplicate rows:", data.duplicated().sum())

# Remove exact duplicates
data = data.drop_duplicates().reset_index(drop=True)

print("Shape after cleaning:", data.shape)

# %% Cell 13 (code)
# Inspect categorical variables
categorical_cols = data.select_dtypes(include="object").columns.tolist()
for col in categorical_cols:
    print(f"\n{col}:")
    print(data[col].value_counts(dropna=False))

# Numeric columns
numeric_cols = data.select_dtypes(include=np.number).columns.tolist()
print("\nNumeric columns:", numeric_cols)

# %% Cell 14 (markdown)
# # **5**. **Project** **Problem** **Statement**

# %% Cell 15 (markdown)
# # Healthcare providers and planners need to understand patterns in doctor visits across demographic, socioeconomic, health-status, and healthcare-access variables. This project uses historical visit records to identify descriptive patterns in utilization and highlight factors that may warrant further investigation.

# %% Cell 16 (markdown)
# # **6 . **Key** **Dataset** Variables**
# *   `visits` – recorded number of doctor visits; primary outcome for this analysis.
# *   `gender` – patient gender category.
# *   `age` – age value recorded in the dataset.
# *   `income` – income-related numeric measure.
# *   `illness` – illness count/score recorded for the patient.
# *   `reduced` – reduced-activity measure.
# *   `health` – health-status measure.
# *   `private` – private insurance indicator.
# *   `freepoor` – free-care indicator associated with poverty eligibility.
# *   `freerepat` – free/repatriation-care indicator.
# *   `nchronic` – indicator for chronic condition category.
# *   `lchronic` – indicator for another chronic-condition category.

# %% Cell 17 (markdown)
# # **Exploratory** **Data** **Analysis**

# %% Cell 18 (markdown)
# # **7**. **Univariate** **Analysis** **–** **Gender** **Distribution**

# %% Cell 19 (code)
plt.figure(figsize=(7, 5))
sns.countplot(data=data, x="gender", hue="gender", legend=False)
plt.title("Distribution of Patients by Gender")
plt.xlabel("Gender")
plt.ylabel("Number of Records")
plt.tight_layout()
plt.show()

display(data["gender"].value_counts().rename_axis("gender").reset_index(name="count"))

# %% Cell 20 (markdown)
# # **8**. **Univariate** **Analysis** **–** **Age** **Distribution**

# %% Cell 21 (code)
plt.figure(figsize=(9, 5))
sns.histplot(data["age"], bins=20, kde=True)
plt.title("Distribution of Age")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

display(data["age"].describe().to_frame("age_summary"))

# %% Cell 22 (markdown)
# # **9**. **Univariate** **Analysis** **–** **Doctor** **Visits**

# %% Cell 23 (code)
plt.figure(figsize=(9, 5))
bins = np.arange(data["visits"].min(), data["visits"].max() + 2) - 0.5
sns.histplot(data["visits"], bins=bins, discrete=True)
plt.title("Distribution of Doctor Visits")
plt.xlabel("Recorded Doctor Visits")
plt.ylabel("Number of Records")
plt.tight_layout()
plt.show()

display(data["visits"].describe().to_frame("visits_summary"))

# %% Cell 24 (markdown)
# # **10**. **Univariate** **Analysis** **–** **Income** **Distribution**

# %% Cell 25 (code)
plt.figure(figsize=(9, 5))
sns.histplot(data["income"], bins=20, kde=True)
plt.title("Distribution of Income")
plt.xlabel("Income")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

display(data["income"].describe().to_frame("income_summary"))

# %% Cell 26 (markdown)
# # **11**. **Univariate** **Analysis** **–** **Illness** **Distribution**

# %% Cell 27 (code)
plt.figure(figsize=(9, 5))
bins = np.arange(data["illness"].min(), data["illness"].max() + 2) - 0.5
sns.histplot(data["illness"], bins=bins, discrete=True)
plt.title("Distribution of Illness")
plt.xlabel("Illness Measure")
plt.ylabel("Number of Records")
plt.tight_layout()
plt.show()

display(data["illness"].describe().to_frame("illness_summary"))

# %% Cell 28 (markdown)
# # **12**. **Univariate** **Analysis** **–** **Reduced** **Activity** **Distribution**

# %% Cell 29 (code)
plt.figure(figsize=(9, 5))
bins = np.arange(data["reduced"].min(), data["reduced"].max() + 2) - 0.5
sns.histplot(data["reduced"], bins=bins, discrete=True)
plt.title("Distribution of Reduced Activity")
plt.xlabel("Reduced Activity Measure")
plt.ylabel("Number of Records")
plt.tight_layout()
plt.show()

display(data["reduced"].describe().to_frame("reduced_summary"))

# %% Cell 30 (markdown)
# # **13**. **Univariate** **Analysis** **–** **Health** **Distribution**

# %% Cell 31 (code)
plt.figure(figsize=(9, 5))
bins = np.arange(data["health"].min(), data["health"].max() + 2) - 0.5
sns.histplot(data["health"], bins=bins, discrete=True)
plt.title("Distribution of Health")
plt.xlabel("Health Measure")
plt.ylabel("Number of Records")
plt.tight_layout()
plt.show()

display(data["health"].describe().to_frame("health_summary"))

# %% Cell 32 (markdown)
# ## **14**. **Doctor** **Visits** **by** **Gender**

# %% Cell 33 (code)
gender_analysis = (
    data.groupby("gender")["visits"]
    .agg(["count", "mean", "median", "std", "min", "max"])
    .reset_index()
)
display(gender_analysis)

plt.figure(figsize=(7, 5))
sns.barplot(data=gender_analysis, x="gender", y="mean")
plt.title("Average Recorded Doctor Visits by Gender")
plt.xlabel("Gender")
plt.ylabel("Mean Visits")
plt.tight_layout()
plt.show()

# %% Cell 34 (markdown)
# # **15**. **Doctor** **Visits** **by** **Illness**

# %% Cell 35 (code)
illness_analysis = (
    data.groupby("illness")["visits"]
    .agg(["count", "mean", "median"])
    .reset_index()
    .sort_values("illness")
)
display(illness_analysis)

plt.figure(figsize=(9, 5))
sns.lineplot(data=illness_analysis, x="illness", y="mean", marker="o")
plt.title("Illness Level and Average Doctor Visits")
plt.xlabel("Illness Measure")
plt.ylabel("Mean Visits")
plt.tight_layout()
plt.show()

# %% Cell 36 (code)
plt.figure(figsize=(10, 6))
sns.boxplot(data=data, x="illness", y="visits")
plt.title("Distribution of Doctor Visits by Illness Level")
plt.xlabel("Illness Measure")
plt.ylabel("Doctor Visits")
plt.tight_layout()
plt.show()

# %% Cell 37 (markdown)
# # **16**. **Doctor** **Visits** **by** **Health** **Status**

# %% Cell 38 (code)
health_analysis = (
    data.groupby("health")["visits"]
    .agg(["count", "mean", "median"])
    .reset_index()
    .sort_values("health")
)
display(health_analysis)

plt.figure(figsize=(9, 5))
sns.barplot(data=health_analysis, x="health", y="mean")
plt.title("Average Doctor Visits by Health Measure")
plt.xlabel("Health Measure")
plt.ylabel("Mean Visits")
plt.tight_layout()
plt.show()

# %% Cell 39 (markdown)
# # **17**. **Age** **vs** **Doctor** **Visits**

# %% Cell 40 (code)
plt.figure(figsize=(10, 6))
sns.scatterplot(data=data, x="age", y="visits", hue="gender", alpha=0.55)
plt.title("Age vs. Recorded Doctor Visits by Gender")
plt.xlabel("Age")
plt.ylabel("Doctor Visits")
plt.tight_layout()
plt.show()

print("Pearson correlation between age and visits:",
      data[["age", "visits"]].corr().loc["age", "visits"])

# %% Cell 41 (markdown)
# # **18**. **Distribution** **of** **Visits** **by** **Gender**

# %% Cell 42 (code)
plt.figure(figsize=(8, 5))
sns.boxplot(data=data, x="gender", y="visits", hue="gender", legend=False)
plt.title("Distribution of Doctor Visits by Gender")
plt.xlabel("Gender")
plt.ylabel("Doctor Visits")
plt.tight_layout()
plt.show()

# %% Cell 43 (markdown)
# # **19**. **Healthcare** **Access** **/** **Coverage** **Variables**

# %% Cell 44 (markdown)
#
# ## The following binary/categorical variables are examined descriptively against doctor visits: private, freepoor, freerepat, nchronic, and lchronic.   

# %% Cell 45 (code)
access_cols = ["private", "freepoor", "freerepat", "nchronic", "lchronic"]

access_summary = []
for col in access_cols:
    tmp = data.groupby(col)["visits"].agg(["count", "mean", "median"]).reset_index()
    tmp.insert(0, "variable", col)
    tmp = tmp.rename(columns={col: "category"})
    access_summary.append(tmp)

access_summary = pd.concat(access_summary, ignore_index=True)
display(access_summary)

# %% Cell 46 (code)
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
axes = axes.flatten()

for ax, col in zip(axes, access_cols):
    summary = data.groupby(col)["visits"].mean().reset_index()
    sns.barplot(data=summary, x=col, y="visits", ax=ax)
    ax.set_title(f"Mean Visits by {col}")
    ax.set_xlabel(col)
    ax.set_ylabel("Mean Visits")

# Hide unused subplot
for ax in axes[len(access_cols):]:
    ax.axis("off")

plt.tight_layout()
plt.show()

# %% Cell 47 (markdown)
# ## **20**. **Distribution** **of** **Doctor** **Visits** **by** **Private** **Insurance**

# %% Cell 48 (code)
plt.figure(figsize=(8, 5))
sns.boxplot(data=data, x="private", y="visits", hue="private", legend=False)
plt.title("Distribution of Doctor Visits by Private Insurance")
plt.xlabel("Private Insurance")
plt.ylabel("Doctor Visits")
plt.tight_layout()
plt.show()

# %% Cell 49 (markdown)
# ## **21**. **Distribution** **of** **Doctor** **Visits** **by** **Free** **Poor** **Status**

# %% Cell 50 (code)
plt.figure(figsize=(8, 5))
sns.boxplot(data=data, x="freepoor", y="visits", hue="freepoor", legend=False)
plt.title("Distribution of Doctor Visits by Free Poor Status")
plt.xlabel("Free Poor Status")
plt.ylabel("Doctor Visits")
plt.tight_layout()
plt.show()

# %% Cell 51 (markdown)
# ## **22**. **Distribution** **of** **Doctor** **Visits** **by** **Free** **Repatriation** **Status**

# %% Cell 52 (code)
plt.figure(figsize=(8, 5))
sns.boxplot(data=data, x="freerepat", y="visits", hue="freerepat", legend=False)
plt.title("Distribution of Doctor Visits by Free Repatriation Status")
plt.xlabel("Free Repatriation Status")
plt.ylabel("Doctor Visits")
plt.tight_layout()
plt.show()

# %% Cell 53 (markdown)
# ## **23**. **Distribution** **of** **Doctor** **Visits** **by** **Chronic** **Condition** **(nchronic)**

# %% Cell 54 (code)
plt.figure(figsize=(8, 5))
sns.boxplot(data=data, x="nchronic", y="visits", hue="nchronic", legend=False)
plt.title("Distribution of Doctor Visits by Chronic Condition (nchronic)")
plt.xlabel("Chronic Condition (nchronic)")
plt.ylabel("Doctor Visits")
plt.tight_layout()
plt.show()

# %% Cell 55 (markdown)
# ## **24**. **Distribution** **of** **Doctor** **Visits** **by** **Chronic** **Condition** **(lchronic)**

# %% Cell 56 (code)
plt.figure(figsize=(8, 5))
sns.boxplot(data=data, x="lchronic", y="visits", hue="lchronic", legend=False)
plt.title("Distribution of Doctor Visits by Chronic Condition (lchronic)")
plt.xlabel("Chronic Condition (lchronic)")
plt.ylabel("Doctor Visits")
plt.tight_layout()
plt.show()

# %% Cell 57 (markdown)
# # **25. Multivariate Regression Analysis**

# %% Cell 58 (markdown)
# ## **25.1. Feature Engineering: One-Hot Encoding Categorical Variables**
# Categorical variables need to be converted into a numerical format for regression models. One-hot encoding creates new binary columns for each category, preventing the model from assuming an ordinal relationship where none exists.

# %% Cell 59 (code)
from sklearn.preprocessing import OneHotEncoder

# Identify categorical columns (excluding 'visits' if it were categorical, but it's not)
categorical_features = ['gender', 'private', 'freepoor', 'freerepat', 'nchronic', 'lchronic']

# Initialize OneHotEncoder
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

# Fit and transform the categorical features
encoded_features = encoder.fit_transform(data[categorical_features])

# Create a DataFrame with the encoded features
encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_features))

# Drop original categorical columns and concatenate the encoded ones
data_encoded = pd.concat([data.drop(columns=categorical_features), encoded_df], axis=1)

display(data_encoded.head())

# %% Cell 60 (markdown)
# ## **25.2. Data Splitting: Training and Testing Sets**
# The dataset is split into training and testing sets to evaluate the model's performance on unseen data. This helps in assessing the model's generalization ability and preventing overfitting.

# %% Cell 61 (code)
from sklearn.model_selection import train_test_split

# Define features (X) and target (y)
X = data_encoded.drop(columns=['visits'])
y = data_encoded['visits']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training set size: {X_train.shape[0]} records")
print(f"Test set size: {X_test.shape[0]} records")

# %% Cell 62 (markdown)
# ## **25.3. Model Building: Poisson Regression**
# Since 'visits' is a count variable (non-negative integers), a Poisson regression model is more appropriate than ordinary least squares (OLS) regression. Poisson regression models the logarithm of the expected count as a linear combination of the predictor variables.

# %% Cell 63 (code)
import statsmodels.api as sm

# Add a constant to the independent variables for statsmodels
X_train_sm = sm.add_constant(X_train)
X_test_sm = sm.add_constant(X_test)

# Build and train the Poisson regression model
poisson_model = sm.Poisson(y_train, X_train_sm)
poisson_results = poisson_model.fit(disp=False)

# Display the model summary
print(poisson_results.summary())

# %% Cell 64 (markdown)
# ## **25.4. Model Evaluation**
# Model evaluation involves assessing how well the trained model performs on the test set. For count data, metrics like Mean Absolute Error (MAE), Mean Squared Error (MSE), and R-squared (pseudo R-squared for GLMs) are commonly used. Given the nature of Poisson regression, it's also useful to check the goodness-of-fit using the deviance and Pearson chi-squared statistics relative to their degrees of freedom.

# %% Cell 65 (code)
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Predict on the test set
y_pred_poisson = poisson_results.predict(X_test_sm)

# Calculate evaluation metrics
mae = mean_absolute_error(y_test, y_pred_poisson)
mse = mean_squared_error(y_test, y_pred_poisson)
rmse = np.sqrt(mse)

print(f"Mean Absolute Error (MAE): {mae:,.2f}")
print(f"Mean Squared Error (MSE): {mse:,.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:,.2f}")

# Visualize actual vs. predicted values for a sample
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_poisson, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Actual Visits")
plt.ylabel("Predicted Visits")
plt.title("Actual vs. Predicted Doctor Visits (Poisson Regression)")
plt.tight_layout()
plt.show()

# %% Cell 66 (markdown)
# # **26**. **Income** **and** **Doctor** **Visits**

# %% Cell 67 (code)
plt.figure(figsize=(9, 5))
sns.scatterplot(data=data, x="income", y="visits", alpha=0.5)
plt.title("Income vs. Recorded Doctor Visits")
plt.xlabel("Income")
plt.ylabel("Doctor Visits")
plt.tight_layout()
plt.show()

print("Pearson correlation between income and visits:",
      data[["income", "visits"]].corr().loc["income", "visits"])

# %% Cell 68 (markdown)
# # **27**. **Reduced** **Activity** **and** **Doctor** **Visits**

# %% Cell 69 (code)
plt.figure(figsize=(9, 5))
sns.scatterplot(data=data, x="reduced", y="visits", alpha=0.5)
plt.title("Reduced Activity vs. Recorded Doctor Visits")
plt.xlabel("Reduced Activity Measure")
plt.ylabel("Doctor Visits")
plt.tight_layout()
plt.show()

# %% Cell 70 (markdown)
# # **28**. **Correlation** **Analysis**

# %% Cell 71 (markdown)
# ## Correlation is calculated only for numeric variables. It measures association, not causation.

# %% Cell 72 (code)
corr = data.select_dtypes(include=np.number).corr()

plt.figure(figsize=(10, 7))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Correlation Matrix of Numeric Variables")
plt.tight_layout()
plt.show()

# Correlations with visits
visit_corr = corr["visits"].drop("visits").sort_values(ascending=False)
display(visit_corr.to_frame("correlation_with_visits"))

# %% Cell 73 (markdown)
# # **29. Segment-Level Analysis**

# %% Cell 74 (markdown)
# ## This section combines demographic and health dimensions to identify descriptive differences in average recorded visits.

# %% Cell 75 (code)
segment = (
    data.groupby(["gender", "nchronic"])["visits"]
    .agg(["count", "mean", "median"])
    .reset_index()
    .sort_values("mean", ascending=False)
)
display(segment)

plt.figure(figsize=(9, 5))
sns.barplot(data=segment, x="gender", y="mean", hue="nchronic")
plt.title("Average Visits by Gender and Chronic-Condition Indicator")
plt.xlabel("Gender")
plt.ylabel("Mean Visits")
plt.legend(title="nchronic")
plt.tight_layout()
plt.show()

# %% Cell 76 (code)
segment_lchronic = (
    data.groupby(["gender", "lchronic"])["visits"]
    .agg(["count", "mean", "median"])
    .reset_index()
    .sort_values("mean", ascending=False)
)
display(segment_lchronic)

plt.figure(figsize=(9, 5))
sns.barplot(data=segment_lchronic, x="gender", y="mean", hue="lchronic")
plt.title("Average Visits by Gender and Chronic-Condition Indicator (lchronic)")
plt.xlabel("Gender")
plt.ylabel("Mean Visits")
plt.legend(title="lchronic")
plt.tight_layout()
plt.show()

# %% Cell 77 (markdown)
# # **30. High-Visit Records**

# %% Cell 78 (markdown)
# ## A simple descriptive view of records with comparatively high visit counts is included below. The threshold is based on the 75th percentile rather than an arbitrary clinical definition.

# %% Cell 79 (code)
visit_threshold = data["visits"].quantile(0.75)
high_visit = data[data["visits"] >= visit_threshold].copy()

print(f"75th-percentile visit threshold: {visit_threshold:.2f}")
print(f"Records at/above threshold: {len(high_visit):,} ({len(high_visit)/len(data)*100:.2f}%)")

display(
    high_visit[["visits", "gender", "age", "income", "illness", "reduced", "health",
                "private", "freepoor", "freerepat", "nchronic", "lchronic"]]
    .sort_values("visits", ascending=False)
    .head(10)
)

# %% Cell 80 (markdown)
# ## **31**. **Key** **Findings**   
# Run the cells above and use the generated tables/figures to report the following categories of findings:   
# *   **Dataset profile**: number of records, variables, missing values, and duplicate records.   
# *   **Visit distribution**: typical visit level, spread, and whether a small group of records has substantially higher visit counts.   
# *   **Demographic patterns**: differences in recorded visits across gender and age groups.   
# *   **Health patterns**: relationship between illness, health measures, chronic-condition indicators, and recorded visits.   
# *   **Socioeconomic/access patterns**: differences in recorded visits across income and healthcare-access indicators.   
# *   **Associations**: numeric correlations with visits, interpreted descriptively rather than causally.   

# %% Cell 81 (markdown)
# ## **32**. **Recommendations**   
# Based on the observed patterns, a project report can consider the following data-driven recommendations:   
# *   Monitor patient groups with higher recorded utilization for service-planning purposes.   
# *   Examine chronic-condition and illness-related segments separately when planning healthcare resources.   
# *   Use demographic and socioeconomic segmentation to understand differences in healthcare utilization.   
# *   Investigate unusually high visit counts using patient-level context before drawing conclusions.   
# *   Improve data documentation for coded variables so that analytical interpretations are reproducible.   
# *   For future work, consider statistical modeling such as Poisson/negative-binomial regression for count outcomes, while checking model assumptions and data quality.

# %% Cell 82 (markdown)
# ## **33**. **Conclusion**   
# This project provides an end-to-end descriptive analysis of doctor visits using the uploaded healthcare dataset. The workflow covers data loading, quality checks, exploratory analysis, demographic segmentation, health-related analysis, healthcare-access analysis, correlation analysis, visualizations, findings, and recommendations.   
# The analysis is intended for **healthcare analytics and project-reporting purposes**. Observed associations should not be interpreted as evidence that one variable causes another.

# %% Cell 83 (markdown)
# ## **34**. **Summary Report of Key Findings**
#
# ### **Dataset Profile**
# *   The dataset contains `3,870` unique records after removing duplicates from an initial `5,190` rows and `13` columns.
# *   There are no missing values in the cleaned dataset.
# *   The dataset includes `12` variables covering demographics (gender, age), socioeconomic status (income), health status (illness, reduced, health, nchronic, lchronic), healthcare access (private, freepoor, freerepat), and the primary outcome variable (visits).
#
# ### **Visit Distribution**
# *   The majority of records show `0` doctor visits, with a long tail indicating a small number of individuals with higher visit counts. The mean number of visits is approximately `0.39`, while the median is `0`, suggesting a skewed distribution with many individuals having no visits and a few with multiple visits.
# *   The 75th percentile for visits is `1.00`, meaning 25.58% of the records have `1` or more visits.
#
# ### **Demographic Patterns**
# *   **Gender**: Females tend to have a higher average number of doctor visits (`0.45`) compared to males (`0.32`).
# *   **Age**: Age shows a positive correlation with doctor visits (`0.12`). While the scatter plot shows a wide distribution, there's a slight tendency for higher visits among older individuals.
#
# ### **Health Patterns**
# *   **Illness**: There is a clear positive relationship between the illness measure and average doctor visits. As illness severity increases, the mean number of visits also increases, ranging from `0.13` for `illness=0` to `0.85` for `illness=5`.
# *   **Health Measure**: Similar to illness, higher health measures generally correspond to higher average doctor visits, with a notable peak at health measure `10` (`1.52` mean visits).
# *   **Chronic Conditions (nchronic and lchronic)**: Individuals with chronic conditions (both `nchronic` and `lchronic`) generally have a higher mean number of doctor visits compared to those without chronic conditions.
#
# ### **Socioeconomic/Access Patterns**
# *   **Income**: Income has a slight negative correlation with doctor visits (`-0.08`), suggesting that higher income might be weakly associated with fewer visits, though the relationship is not strong.
# *   **Private Insurance**: Individuals with private insurance (`yes`) have a slightly lower mean number of visits (`0.37`) than those without private insurance (`no`) (`0.40`).
# *   **Free Poor Status (freepoor)**: Patients with `freepoor=yes` have a lower average number of visits (`0.18`) than those with `freepoor=no` (`0.40`).
# *   **Free Repatriation Status (freerepat)**: Patients with `freerepat=yes` have a significantly higher average number of visits (`0.61`) compared to those with `freerepat=no` (`0.33`).
#
# ### **Associations (Numeric Variables)**
# *   **Reduced Activity (`reduced`)**: Shows the strongest positive correlation with doctor visits (`0.40`), indicating that individuals with higher reduced activity measures tend to have more doctor visits.
# *   **Illness (`illness`)**: Has a positive correlation with visits (`0.19`).
# *   **Health (`health`)**: Shows a positive correlation with visits (`0.15`).
# *   **Age (`age`)**: Shows a positive correlation with visits (`0.12`).
# *   **Income (`income`)**: Shows a weak negative correlation with visits (`-0.08`).
#
# ### **Multivariate Regression Analysis (Poisson Regression)**
# *   The Poisson regression model shows that `illness`, `reduced`, and `age` are statistically significant predictors of doctor visits (low p-values).
# *   `income` also shows some significance (`P>|z|=0.057`).
# *   The model indicates that `illness` and `reduced` have a strong positive association with `visits` (positive coefficients), while `income` has a negative association.
# *   The model fit resulted in a Pseudo R-squared of `0.1460`, indicating that the model explains a modest portion of the variance in doctor visits.
# *   The evaluation metrics are: Mean Absolute Error (MAE): `0.53`, Mean Squared Error (MSE): `0.71`, Root Mean Squared Error (RMSE): `0.84`.

