import pandas as pd 
import sklearn 
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    df_heart = pd.read_csv('./Datasets/heart.csv')

    print(df_heart.head(5))

    y = df_heart['target']
    X = df_heart.drop(['target'], axis = 1)

    # Para PCA necesitamos SIEMPRE normalizar nuestros datos

    X = StandardScaler().fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

    print('X_train: ',X_train.shape,' X_test: ',X_test.shape)
    print('y_train: ',y_train.shape,' y_test: ',y_test.shape)

    #n_components = min(n_muestras,n_features)
    pca = PCA(n_components = 3)
    ipca = IncrementalPCA(n_components = 3, batch_size = 10)

    pca.fit(X_train)
    ipca.fit(X_train)

    #plt.plot(range(len(pca.explained_variance_)), pca.explained_variance_ratio_)
    #plt.show()

    logistic = LogisticRegression(solver = 'lbfgs')

    dt_train = pca.transform(X_train)
    dt_test = pca.transform(X_test)
    
    logistic.fit(dt_train,y_train)
    print('PCA')
    print('X_train post PCA: ',dt_train.shape)
    print('Scores PCA: ', logistic.score(dt_test, y_test))


    dt_train = ipca.transform(X_train)
    dt_test = ipca.transform(X_test)
    
    logistic.fit(dt_train,y_train)
    print('IPCA')
    print('X_train post IPCA: ',dt_train.shape)
    print('Scores IPCA: ', logistic.score(dt_test, y_test))

    