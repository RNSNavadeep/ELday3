Housing Price Prediction — Linear Regression

A end-to-end Machine Learning project that predicts house prices based on various features using Linear Regression, built on the Housing dataset.


📁 Dataset


File: Housing.csv
Target Column: price
Features: Area, bedrooms, bathrooms, stories, parking, and other housing attributes



🔄 Workflow

1. 🔍 Data Exploration

Loaded the dataset and performed initial checks:


Displayed the first 5 rows to understand structure
Checked for null values — ensuring no missing data issues
Checked for duplicate rows — ensuring data integrity


2. 🏷️ Label Encoding

Identified all categorical (object) columns and converted them into numeric format using LabelEncoder — since Linear Regression requires all inputs to be numerical.

pythonobjcol = [x for x in df.columns if df[x].dtype == "object"]
le = LabelEncoder()
for i in objcol:
    df[i] = le.fit_transform(df[i])

3. 📊 Exploratory Data Analysis (EDA)

Three visualizations were used to understand relationships in the data:


Correlation Heatmap — identified which features are most related to price
Price vs Area Line Plot — visual trend between house size and price
Pairplot — pairwise relationships across all features
Regression Plot — scatter plot with a regression line between area and price


4. ✂️ Train-Test Split

Split the data into 80% training and 20% testing sets:

pythonxtr, xte, ytr, yte = train_test_split(x, y, test_size=0.2, random_state=42)

5. 🤖 Model Training

Trained a Linear Regression model on the training data:

pythonlr = LinearRegression()
lr.fit(xtr, ytr)

6. 📉 Model Evaluation

Evaluated the model using four key metrics:

MetricDescriptionMAEMean Absolute Error — average of absolute differencesMSEMean Squared Error — penalizes large errors moreRMSERoot MSE — same unit as target, easier to interpretR² ScoreHow well the model explains variance in price (1.0 = perfect)

7. 📈 Prediction Plot

Plotted predicted prices vs actual prices to visually assess model performance:


🔴 Red line → Predicted values
🟢 Green line → Actual values



🛠️ Libraries Used

LibraryPurposepandasData loading and manipulationnumpyNumerical computationsmatplotlibLine plots and visualizationsseabornHeatmap, pairplot, regression plotscikit-learnEncoding, splitting, model training, evaluation


🧠 What is Linear Regression?

Linear Regression finds the best fit line through the data that minimizes the error between predicted and actual values. It assumes a linear relationship between features and the target:

price = m1×area + m2×bedrooms + ... + c

Where m = coefficients (slopes) and c = intercept.


✅ Conclusion

This project demonstrates a complete ML pipeline — from raw data to a trained and evaluated regression model — predicting house prices based on features like area, bedrooms, and more. The evaluation metrics and prediction plot together give a clear picture of how well the model performs.
