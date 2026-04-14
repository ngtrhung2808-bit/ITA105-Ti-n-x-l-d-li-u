import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer, PowerTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None
import joblib

# --- Custom Transformers ---

class ToNumericTransformer(BaseEstimator, TransformerMixin):
    """Converts input to numeric, turning errors to NaN."""
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X = pd.DataFrame(X).copy()
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors='coerce')
        return X
    def get_feature_names_out(self, input_features=None):
        return input_features

class OutlierRemover(BaseEstimator, TransformerMixin):
    """Custom transformer to handle outliers by clipping to IQR bounds."""
    def __init__(self, factor=1.5):
        self.factor = factor
        self.lower_bound_ = None
        self.upper_bound_ = None

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        Q1 = X.quantile(0.25)
        Q3 = X.quantile(0.75)
        IQR = Q3 - Q1
        self.lower_bound_ = Q1 - self.factor * IQR
        self.upper_bound_ = Q3 + self.factor * IQR
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        for col in X.columns:
            X[col] = X[col].clip(lower=self.lower_bound_[col], upper=self.upper_bound_[col])
        return X
    def get_feature_names_out(self, input_features=None):
        return input_features

class TimeFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extracts month and quarter from datetime-like strings."""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = pd.Series(X.iloc[:, 0]) # Assume single column
        dates = pd.to_datetime(X, errors='coerce')
        # Fill missing dates with the most frequent if any were invalid
        if dates.isna().any():
            mode_date = dates.mode()[0] if not dates.mode().empty else pd.Timestamp('2000-01-01')
            dates = dates.fillna(mode_date)
            
        df = pd.DataFrame({
            'month': dates.dt.month,
            'quarter': dates.dt.quarter,
            'year': dates.dt.year
        })
        return df
    def get_feature_names_out(self, input_features=None):
        return np.array(['month', 'quarter', 'year'])

# --- Implementation ---

def squeeze_transformer(X):
    if isinstance(X, pd.DataFrame):
        return X.iloc[:, 0]
    if isinstance(X, np.ndarray):
        return X.ravel()
    return X

def build_complete_pipeline():
    # Define column groups
    num_cols = ['LotArea', 'Rooms', 'NoiseFeature', 'HasGarage']
    cat_cols = ['Neighborhood', 'Condition']
    text_col = 'Description'
    time_col = 'SaleDate'

    # 1. Numerical Pipeline: Convert to Numeric -> Impute -> Outlier removal -> Scaling -> Log/Power
    num_pipeline = Pipeline([
        ('tonumeric', ToNumericTransformer()),
        ('imputer', SimpleImputer(strategy='median')),
        ('outlier', OutlierRemover()),
        ('scaler', StandardScaler()),
        ('yeo_johnson', PowerTransformer(method='yeo-johnson'))
    ])

    # 2. Categorical Pipeline: Impute -> One-hot
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # 3. Text Pipeline: TF-IDF
    text_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='')),
        ('squeezer', FunctionTransformer(squeeze_transformer, validate=False)),
        ('tfidf', TfidfVectorizer(stop_words='english', max_features=100))
    ])

    # 4. Time Pipeline: Extract (month, quarter...) -> Interpolate missing
    time_pipeline = Pipeline([
        ('extractor', TimeFeatureExtractor()),
        ('imputer', SimpleImputer(strategy='mean'))
    ])

    # Combine into ColumnTransformer
    preprocessor = ColumnTransformer([
        ('num', num_pipeline, num_cols),
        ('cat', cat_pipeline, cat_cols),
        ('text', text_pipeline, [text_col]),
        ('time', time_pipeline, [time_col])
    ])

    return preprocessor

if __name__ == "__main__":
    # Load data
    data_path = r'c:\Users\ADMIN\Downloads\Lab8\Lab8\ITA105_Lab_8.csv'
    df = pd.read_csv(data_path)
    
    X = df.drop('SalePrice', axis=1)
    y = df['SalePrice']

    # --- Bài 1: Build Pipeline ---
    preprocessor = build_complete_pipeline()
    print("Bài 1: Building Pipeline...")
    
    # Smoke test on 10 lines
    demo_data = X.head(10)
    transformed_demo = preprocessor.fit_transform(demo_data)
    print("- Smoke test successful. Transformed shape:", transformed_demo.shape)
    
    # Get feature names
    try:
        feature_names = preprocessor.get_feature_names_out()
        print("- Feature names (first 15):", feature_names[:15])
    except Exception as e:
        print("- Could not extract feature names automatically:", e)

    # --- Bài 2: Test Pipeline ---
    print("\nBài 2: Testing Robustness...")
    
    # 5 test scenarios
    # 1. Full data (already tested in smoke test)
    # 2. Missing data
    missing_df = X.head(5).copy()
    missing_df.loc[0, 'LotArea'] = np.nan
    missing_df.loc[1, 'Description'] = np.nan
    
    # 3. Unseen Categories
    unseen_df = X.head(5).copy()
    unseen_df.loc[0, 'Neighborhood'] = 'NEW_NEIGHBORHOOD'
    
    # 4. Wrong format (string in numerical)
    wrong_format_df = X.head(5).copy().astype(object)
    wrong_format_df.loc[0, 'Rooms'] = 'many'
    
    # 5. Skewed data (just transform existing data)

    for i, test_df in enumerate([X.head(5), missing_df, unseen_df, wrong_format_df]):
        try:
            # For numerical errors in wrong_format, we need to handle during transform or convert before
            # SimpleImputer + OutlierRemover should handle basic cases if they can be forced to float
            # Let's try to convert X to numeric where possible first in a production pipeline
            preprocessor.transform(test_df)
            print(f"- Scenario {i+1} transform: PASSED")
        except Exception as e:
            print(f"- Scenario {i+1} transform: FAILED - {str(e)[:100]}")

    # --- Bài 3: Model Integration ---
    print("\nBài 3: Model Integration & CV...")
    
    models = {
        'LinearRegression': LinearRegression(),
        'RandomForest': RandomForestRegressor(n_estimators=50, random_state=42)
    }

    for name, model in models.items():
        model_pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', model)
        ])
        cv_results = cross_validate(model_pipeline, X, y, cv=5, 
                                   scoring=['neg_root_mean_squared_error', 'neg_mean_absolute_error', 'r2'],
                                   return_train_score=False)
        rmse = -cv_results['test_neg_root_mean_squared_error'].mean()
        mae = -cv_results['test_neg_mean_absolute_error'].mean()
        r2 = cv_results['test_r2'].mean()
        print(f"[{name}] RMSE: {rmse:.2f}, MAE: {mae:.2f}, R2: {r2:.4f}")

    # --- Bài 4: Final Export ---
    print("\nBài 4: Exporting Final Production Model...")
    final_model = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    final_model.fit(X, y)
    
    joblib.dump(final_model, 'house_price_pipeline.joblib')
    print("- Saved to 'house_price_pipeline.joblib'")

    print("\nDone.")
