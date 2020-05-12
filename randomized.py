import pandas as pd 

from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor

if __name__ == '__main__':

    df = pd.read_csv('./Datasets/felicidad.csv')
    print(df.head(2))

    X = df.drop(['country','rank','score'], axis = 1)
    y = df['score']

    regressor = RandomForestRegressor()

    params = {
        'n_estimators':range(4,16),
        'criterion':['mse','mae'],
        'max_depth':range(2,110)
    }

    rand_est = RandomizedSearchCV(
        regressor,
        params, 
        n_iter = 10,
        cv = 3,
        scoring = 'neg_mean_absolute_error'
    ).fit(X,y)

    print('Estimador: ',rand_est.best_estimator_)
    print('=='*60)
    print('Parametros: ',rand_est.best_params_)
    print('=='*60)
    print(rand_est.predict(X.loc[[0]]))