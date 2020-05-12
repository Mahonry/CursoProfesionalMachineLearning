import pandas as pd 
import numpy as np 

from sklearn.tree import DecisionTreeRegressor

from sklearn.model_selection import(cross_val_score,KFold)

if __name__ == '__main__':

    df = pd.read_csv('./Datasets/felicidad.csv')

    X = df.drop(['country','score'], axis = 1)
    y = df['score']

    model = DecisionTreeRegressor()

    score = cross_val_score(model,
                            X,
                            y,
                            scoring = 'neg_mean_squared_error',
                            cv = 3)
    print(np.abs(np.mean(score)))

    kf = KFold(n_splits = 3, shuffle = True, random_state = 42)

    for train, test in kf.split(df):
        print(train)
        print(test)