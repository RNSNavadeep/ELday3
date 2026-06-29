Titanic Dataset — Data Preprocessing Pipeline
A end-to-end data preprocessing workflow applied to the classic Titanic dataset, preparing raw passenger data for machine learning.
---
📋 Steps Performed
1. Data Exploration
Loaded the dataset and inspected its structure — first few rows, column data types, and a general summary — to understand what we're working with before touching anything.
2. Handling Missing Values
Used `SimpleImputer` to fill in gaps intelligently:
Age → filled with the mean (continuous numeric column)
Cabin & Embarked → filled with the most frequent value (categorical columns)
3. Dropping Irrelevant Columns
Removed columns that add noise or are too unique to be useful features:
`Name`, `Ticket` — unique per passenger, no predictive value
`Cabin` — too sparse even after imputation
4. Encoding Categorical Variables
Converted text categories into numbers the model can understand:
Sex → `OneHotEncoder` with `drop="first"` to avoid multicollinearity
Embarked → Full one-hot encoding into three port columns (`S`, `C`, `Q`)
5. Feature Scaling
Applied `StandardScaler` to normalize Age and Fare — bringing them onto the same scale so no single feature dominates during model training.
6. Outlier Detection & Removal
Used Z-score analysis to catch extreme values:
Computed Z-scores for `Age` and `Fare`
Removed rows where |Z-score| > 3 (beyond 3 standard deviations from the mean)
Visualized the impact with before/after boxplots
---
🛠️ Libraries Used
Library	Purpose
`pandas`	Data loading and manipulation
`numpy`	Numerical operations
`scikit-learn`	Imputation, encoding, scaling
`scipy`	Z-score based outlier detection
`matplotlib`	Boxplot visualizations
---
✅ Output
A clean, fully numeric, scaled, and outlier-free dataframe — ready to be fed into any machine learning model.
