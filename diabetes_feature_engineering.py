import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Reading and inspecting dataset.
main_df = pd.read_csv('Datasets/data.csv')
df = main_df.copy()

df.head()
df.shape
df.info()
df.isnull().sum()
df.describe([0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]).T


# Function for split features with types.
def grab_col_names(dataframe, cat_th=10, car_th=20):
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes in ['object', 'category']]
    num_but_cat = [col for col in dataframe.columns if
                   dataframe[col].nunique() < cat_th and dataframe[col].dtypes not in ['object', 'category']]
    cat_but_car = [col for col in dataframe.columns if
                   dataframe[col].nunique() > car_th and dataframe[col].dtypes in ['object', 'category']]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes not in ['object', 'category']]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print("Observations:", dataframe.shape[0])
    print("Variables:", len(dataframe.columns))
    print("cat_cols:", len(cat_cols))
    print("num_cols:", len(num_cols))
    print("cat_but_car:", len(cat_but_car))
    print("num_but_cat:", len(num_but_cat))

    return cat_cols, num_cols, cat_but_car


cat_cols, num_cols, cat_but_car = grab_col_names(df)

# Inspecting new lists.
print(cat_cols)
print(num_cols)
for col in num_cols:
    print(col, df[col].nunique())

# Correlation of each feature.
corr = df.corr()
sns.heatmap(corr, cmap='Blues', annot=True, fmt='.3f')
plt.show()

# Finding missing values. (missing values filled with ('0')
df.isnull().sum()
df.describe().T
# Glucose, BloodPressure, SkinThickness, Insulin, BMI has missing values (0)

df.loc[df['Glucose'] == 0, "Glucose"] = np.nan
df.loc[df['BloodPressure'] == 0, "BloodPressure"] = np.nan
df.loc[df['SkinThickness'] == 0, "SkinThickness"] = np.nan
df.loc[df['Insulin'] == 0, "Insulin"] = np.nan
df.loc[df['BMI'] == 0, "BMI"] = np.nan


# Function for analyzing all missing values.
def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end='\n')
    if na_name:
        return na_columns


# Getting features with missing values.
na_cols = missing_values_table(df, na_name=True)


# Function for analyzing features with missing values with the target variable.
def missing_vs_target(dataframe, target, na_columns):
    temp_df = dataframe.copy()
    for col in na_columns:
        temp_df[col + '_NA_FLAG'] = np.where(temp_df[col].isnull(), 1, 0)
    na_flags = temp_df.loc[:, temp_df.columns.str.contains('_NA_')].columns
    for col in na_flags:
        print(pd.DataFrame({"TARGET_MEAN": temp_df.groupby(col)[target].mean(),
                            "Count": temp_df.groupby(col)[target].count()}), end="\n\n\n")


missing_vs_target(df, 'Outcome', na_cols)

# To fill missing values, extracting new feature.
df.loc[(df['Age'] < 17), 'AgeGroup'] = "Children"
df.loc[(df['Age'] < 31) & (16 < df['Age']), 'AgeGroup'] = "YoungAdult"
df.loc[(df['Age'] < 46) & (30 < df['Age']), 'AgeGroup'] = "MiddleAgedAdult"
df.loc[(df['Age'] < 65) & (45 < df['Age']), 'AgeGroup'] = "OldAgedAdult"
df.loc[(64 < df['Age']), 'AgeGroup'] = "Elder"


# Function for filling missing values.
def fill_missing_values(dataframe, col):
    dataframe[col] = dataframe[col].fillna(dataframe.groupby("AgeGroup")[col].transform("mean"))


for col in na_cols:
    fill_missing_values(df, col)

df.isnull().sum()
df.describe().T


# Function for defining outlier threshold.
def outlier_thresholds(dataframe, variable, q3_val=0.75, q1_val=0.25):
    q3 = dataframe[variable].quantile(q3_val)
    q1 = dataframe[variable].quantile(q1_val)
    iqr = q3 - q1
    upper_bound = q3 + (1.5 * iqr)
    lower_bound = q1 - (1.5 * iqr)
    return upper_bound, lower_bound


# Function for checking outlier values.
def check_outlier(dataframe, variable, q3_val=0.75, q1_val=0.25):
    upper_bound, lower_bound = outlier_thresholds(dataframe, variable, q3_val, q1_val)
    return dataframe[(dataframe[variable] > upper_bound) | (dataframe[variable] < lower_bound)].any(axis=None)


outliers = []
for col in num_cols:
    if check_outlier(df, col, q3_val=0.95, q1_val=0.05):
        outliers.append(col)


# Function for replacing outliers with thresholds.
def replace_with_thresholds(dataframe, variable, q3_val=0.75, q1_val=0.25):
    upper_bound, lower_bound = outlier_thresholds(dataframe, variable, q3_val, q1_val)
    dataframe.loc[dataframe[variable] > upper_bound, variable] = upper_bound
    dataframe.loc[dataframe[variable] < lower_bound, variable] = lower_bound


for col in outliers:
    replace_with_thresholds(df, col, q3_val=0.95, q1_val=0.05)

# Extracting new features.
df.head()

# Glucose
df.loc[(df['Glucose'] < 61), "GlucoseType"] = "HypoGlycemia"
df.loc[(60 < df['Glucose']) & (df['Glucose'] < 181), "GlucoseType"] = "Normal"
df.loc[(180 < df['Glucose']), "GlucoseType"] = "HyperGlycemia"

# BloodPressure
df.loc[(df['BloodPressure'] < 81), "BloodPressureType"] = "Optimal"
df.loc[(80 < df['BloodPressure']) & (df['BloodPressure'] < 86), "BloodPressureType"] = "Normal"
df.loc[(85 < df['BloodPressure']) & (df['BloodPressure'] < 90), "BloodPressureType"] = "HighNormal"
df.loc[(89 < df['BloodPressure']) & (df['BloodPressure'] < 100), "BloodPressureType"] = "Stage_1"
df.loc[(99 < df['BloodPressure']) & (df['BloodPressure'] < 110), "BloodPressureType"] = "Stage_2"
df.loc[(109 < df['BloodPressure']), "BloodPressureType"] = "Stage_3"

# BMI
df.loc[(df['BMI'] < 18.6), "BMIType"] = "Underweight"
df.loc[(18.5 < df['BMI']) & (df['BMI'] < 25), "BMIType"] = "Normal"
df.loc[(24.9 < df['BMI']) & (df['BMI'] < 30), "BMIType"] = "Overweight"
df.loc[(29.9 < df['BMI']) & (df['BMI'] < 35), "BMIType"] = "Class_1_Obesity"
df.loc[(34.9 < df['BMI']) & (df['BMI'] < 40), "BMIType"] = "Class_2_Obesity"
df.loc[(39.9 < df['BMI']), "BMIType"] = "Class_3_Obesity"

# Age
# Age feature extraction has already been done.

df.info()
cat_cols, num_cols, cat_but_car = grab_col_names(df)

# Selecting features for encoding.
cols_to_encode = [col for col in df.columns if (df[col].dtypes in ['object', 'category']) & (df[col].nunique() < 10)]


# Function for one-hot encoding.
def one_hot_encoder(dataframe, col, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=col, drop_first=drop_first, dtype=int)
    return dataframe


df = one_hot_encoder(df, cols_to_encode)

# Scaling non-object features.
scaler = StandardScaler()
for col in num_cols:
    df[col] = scaler.fit_transform(df[[col]])
df.head()

# Modeling.
y = df['Outcome']
X = df.drop("Outcome", axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
log_model = LogisticRegression().fit(X_train, y_train)
y_log_pred = log_model.predict(X_test)
accuracy_score(y_log_pred, y_test)

# GridSearchCV for best parameters.
log_model.get_params()
log_params = {'solver': ['liblinear', 'lbfgs']}
log_best = GridSearchCV(log_model, log_params, cv=5, n_jobs=-1, verbose=1).fit(X_train, y_train)
print(log_best.best_params_)

log_params = {'solver': ['liblinear'],
              'penalty': ['l1', 'l2']}
log_best = GridSearchCV(log_model, log_params, cv=5, n_jobs=-1, verbose=1).fit(X_train, y_train)
print(log_best.best_params_)

# Final model.
log_final = log_model.set_params(**log_best.best_params_).fit(X_train, y_train)
y_log_pred = log_model.predict(X_test)
accuracy_score(y_log_pred, y_test)
