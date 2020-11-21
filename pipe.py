from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def model_pipeline(model, numerical_features):

    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value=0)), 
        ('scaler', MinMaxScaler(feature_range=(0, 1)))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numerical_transformer, numerical_features), 
    ])

    pipe = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model) 
    ])

    return pipe
