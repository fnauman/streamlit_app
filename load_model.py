from sklearn import linear_model
from sklearn import ensemble


def load_algorithm(model_name, params, SEED=123):
    model = None
    if (model_name == 'Linear Regression'):
        model = linear_model.LinearRegression()
    elif (model_name == 'Lasso'):
        model = linear_model.Lasso(alpha=params['alpha'])
    elif (model_name == 'Random Forest'):
        model = ensemble.RandomForestRegressor(
            n_estimators=params['n_estimators'], 
            max_depth=params['max_depth'], 
            random_state=SEED)

    return model
