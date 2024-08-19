import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV

df = pd.read_csv('dataset/Car details v3.csv')

df.head()

df.shape

df.describe().T

df.drop(['torque'], axis=1, inplace=True)
df.head()

df.duplicated().sum()


df = df.drop_duplicates()
df.shape

df.isnull().sum()

df.dropna(axis=0, inplace=True)

df.shape

df['age'] = 2024 - df['year']
df.drop(['year'], axis=1, inplace=True)

df['brand'] = df['name'].str.split(' ').str.get(0)
df.drop(['name'],axis=1,inplace=True)

first_column = df.pop('brand')
df.insert(0, 'brand', first_column)

df.head()

def remove_unit_and_convert(df, col_name, to_type=float):
    df[col_name] = df[col_name].apply(lambda x: str(x).split(' ')[0])
    df[col_name] = pd.to_numeric(df[col_name], errors='coerce').astype(to_type)
    return df

df = remove_unit_and_convert(df, 'mileage', float)
df = remove_unit_and_convert(df, 'engine', int)
df = remove_unit_and_convert(df, 'max_power', float)

df.head()

df.info()

df.dropna(axis=0, inplace=True)

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Plot 1: 5 Brand dengan Jumlah Kendaraan Terbanyak
top_5_brands = df['brand'].value_counts().nlargest(5)
sns.barplot(x=top_5_brands.index, y=top_5_brands.values, ax=axes[0, 0])
axes[0, 0].set_title('5 Brand dengan Jumlah Kendaraan Terbanyak')
axes[0, 0].set_xlabel('Brand')
axes[0, 0].set_ylabel('Jumlah Kendaraan')

# Plot 2: Distribusi Jenis Bahan Bakar
sns.countplot(x='fuel', data=df, ax=axes[0, 1])
axes[0, 1].set_title('Distribusi Jenis Bahan Bakar')
axes[0, 1].set_xlabel('Jenis Bahan Bakar')
axes[0, 1].set_ylabel('Jumlah Kendaraan')

# Plot 3: Distribusi Jenis Transmisi
sns.countplot(x='transmission', data=df, ax=axes[0, 2])
axes[0, 2].set_title('Distribusi Jenis Transmisi')
axes[0, 2].set_xlabel('Jenis Transmisi')
axes[0, 2].set_ylabel('Jumlah Kendaraan')

# Plot 4: Distribusi Seller Type
sns.countplot(x='seller_type', data=df, ax=axes[1, 0])
axes[1, 0].set_title('Distribusi Seller Type')
axes[1, 0].set_xlabel('Jenis Seller Type')
axes[1, 0].set_ylabel('Jumlah Kendaraan')

# Plot 5: Distribusi Owner
sns.countplot(x='owner', data=df, ax=axes[1, 1])
axes[1, 1].set_title('Distribusi Owner')
axes[1, 1].set_xlabel('Jenis Owner')
axes[1, 1].set_ylabel('Jumlah Kendaraan')
axes[1, 1].tick_params(axis='x', rotation=25)

fig.delaxes(axes[1, 2])
plt.tight_layout()

plt.show()

fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))

columns = ['selling_price', 'km_driven', 'mileage', 'engine', 'max_power', 'age']
titles = ["Selling Price", "Total KM Driven", "Fuel Efficiency in KM per litre",
          "Engine CC", "Brake Horse Power(BHP)", "Age of Car"]

for i, (column, title) in enumerate(zip(columns, titles)):
    row, col = divmod(i, 3)
    axes[row, col].hist(df[column], bins=20, color='blue', edgecolor='black')
    axes[row, col].set_title(title)  
    axes[row, col].set_xlabel(column.replace('_', ' ').title())
    axes[row, col].set_ylabel('Frequency') 

plt.tight_layout()

fig.suptitle("Distribution of Numerical Data", fontsize=16, y=1.02)

plt.show()


fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))

columns = ['selling_price', 'km_driven', 'mileage', 'engine', 'max_power', 'age']
titles = ["Selling Price", "Total KM Driven", "Fuel Efficiency in KM per litre",
          "Engine CC", "BHP", "Age of Car"]

for i, (column, title) in enumerate(zip(columns, titles)):
    row, col = divmod(i, 3)
    axes[row, col].boxplot(df[column].dropna())
    axes[row, col].set_title(title)
    axes[row, col].set_xlabel(column.replace('_', ' ').title())

plt.tight_layout()

fig.suptitle("Distribution of Numerical Data", fontsize=16, y=1.02)
plt.show()

fig, ax = plt.subplots(figsize=(10,6))
correlation_matrix = df.corr(numeric_only=True)
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

sns.heatmap(
    correlation_matrix,
    annot=True,
    mask=mask,
    cmap="coolwarm",
    center=0,
    fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

sns.pairplot(df)

df_clean = df.copy()

df_clean = df_clean[
    (df_clean['selling_price'] < 2500000) & 
    (df_clean['km_driven'] < 300000) & 
    (~df_clean['fuel'].isin(['CNG', 'LPG'])) & 
    (df_clean['mileage'].between(5, 35)) & 
    (df_clean['max_power'] < 300)
]

# Transformasi logaritma
df_clean['selling_price'] = np.log(df_clean['selling_price'])
df_clean['max_power'] = np.log(df_clean['max_power'])
df_clean['age'] = np.log(df_clean['age'])

df_clean.head()

X = df_clean.drop('selling_price', axis=1)
y = df_clean['selling_price']

numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), numerical_features),
        ('cat', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
        ]), categorical_features)
    ])

def create_pipeline(model):
    return Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])

models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100)
}

results = []
for model_name, model in models.items():
    pipeline = create_pipeline(model)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    results.append({
        'Model': model_name,
        'MSE': round(mse, 2),
        'R^2': round(r2, 2)
    })

results_df = pd.DataFrame(results)
results_df

predictions = {}
for model_name, model in models.items():
    pipeline = create_pipeline(model)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    predictions[model_name] = y_pred

plt.figure(figsize=(15, 10))

for model_name, y_pred in predictions.items():
    plt.subplot(2, 2, 1)
    plt.scatter(y_test, y_pred, label=model_name, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    plt.xlabel('Actual Selling Price')
    plt.ylabel('Predicted Selling Price')
    plt.title('Actual vs Predicted Selling Price')
    plt.legend()
    
    plt.subplot(2, 2, 2)
    residuals = y_test - y_pred
    sns.histplot(residuals, kde=True, label=model_name, alpha=0.5)
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.title('Residuals Distribution')
    plt.legend()
    
plt.tight_layout()
plt.show()

param_grids = {
    'Linear Regression': {},
    'Random Forest': {
        'regressor__n_estimators': [50, 100, 200],
        'regressor__max_depth': [None, 10, 20, 30],
        'regressor__min_samples_split': [2, 5, 10],
    },
    'Gradient Boosting': {
        'regressor__n_estimators': [50, 100, 200],
        'regressor__learning_rate': [0.01, 0.1, 0.2],
        'regressor__max_depth': [3, 5, 7],
    }
}

models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(),
    'Gradient Boosting': GradientBoostingRegressor()
}

results = []
best_estimators = {}

for model_name, model in models.items():
    pipeline = create_pipeline(model)
    
    # GridSearchCV
    grid_search = GridSearchCV(pipeline, param_grid=param_grids[model_name], cv=5, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    best_estimators[model_name] = grid_search.best_estimator_
    
    y_pred = grid_search.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    results.append({
        'Model': model_name,
        'Best Params': grid_search.best_params_,
        'MSE': round(mse, 2),
        'R^2': round(r2, 2)
    })

results_df = pd.DataFrame(results)
results_df

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))
i = 0

for model_name, model in best_estimators.items():
    if model_name in ['Random Forest', 'Gradient Boosting']:
        feature_importance = model.named_steps['regressor'].feature_importances_
        features = numerical_features + list(model.named_steps['preprocessor'].transformers_[1][1].named_steps['ordinal'].get_feature_names_out(categorical_features))
        
        sorted_idx = np.argsort(feature_importance)
        axes[i].barh(np.array(features)[sorted_idx], feature_importance[sorted_idx])
        axes[i].set_xlabel('Feature Importance')
        axes[i].set_title(f'Feature Importance for {model_name}')
        i += 1
        
        if i == 2:
            break

plt.tight_layout()
plt.show()

predictions = {}
best_estimators = {}

for model_name, model in models.items():
    pipeline = create_pipeline(model)
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grids[model_name],
        cv=5,
        scoring='neg_mean_squared_error',
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    
    # Menyimpan estimator terbaik dan prediksinya
    best_model = grid_search.best_estimator_
    best_estimators[model_name] = best_model
    y_pred = best_model.predict(X_test)
    predictions[model_name] = y_pred

# Plotting
plt.figure(figsize=(15, 10))

for i, (model_name, y_pred) in enumerate(predictions.items(), 1):
    # Plot actual vs predicted selling price
    plt.subplot(2, 2, 1)
    plt.scatter(y_test, y_pred, label=model_name, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    plt.xlabel('Actual Selling Price')
    plt.ylabel('Predicted Selling Price')
    plt.title('Actual vs Predicted Selling Price')
    plt.legend()
    
    # Plot residuals distribution
    plt.subplot(2, 2, 2)
    residuals = y_test - y_pred
    sns.histplot(residuals, kde=True, label=model_name, alpha=0.5)
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.title('Residuals Distribution')
    plt.legend()

plt.tight_layout()
plt.show()