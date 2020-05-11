import pandas as pd 
import sklearn 
import matplotlib.pyplot as plt

from sklearn.decomposition import KernelPCA


from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

if __name__ == '__main__':

    df_heart = pd.read_csv('./Datasets/heart.csv')
    print(df_heart.head(5))

    y = df_heart['target']
    X = df_heart.drop(['target'], axis = 1)

    X = StandardScaler().fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

    kpca = KernelPCA(n_components = 4, kernel = 'poly')
    kpca.fit(X_train)

    dt_train = kpca.transform(X_train)
    dt_test = kpca.transform(X_test)

    regresion = LogisticRegression(solver = 'lbfgs')

    regresion.fit(dt_train, y_train)
    print('Score KPCA: ',regresion.score(dt_test, y_test))



    