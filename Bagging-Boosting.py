import pandas as pd 

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

if __name__ == '__main__':

    df = pd.read_csv('./Datasets/heart.csv')
    print(df['target'].describe())

    y = df['target']
    X = df.drop(['target'], axis = 1)

    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.35 )

    knn_class = KNeighborsClassifier().fit(X_train, y_train)
    knn_pred = knn_class.predict(X_test)

    print('__'*64)
    print('Knn: ', accuracy_score(knn_pred, y_test))

    bag_class = BaggingClassifier(base_estimator = KNeighborsClassifier(),
                                  n_estimators = 50).fit(X_train, y_train)
    bag_pred = bag_class.predict(X_test)

    print('__'*64)
    print('Bagging: ',accuracy_score(bag_pred, y_test))

    boost = GradientBoostingClassifier(n_estimators = 50).fit(X_train,y_train)
    boost_pred = boost.predict(X_test)

    print('__'*64)
    print('Boosting: ', accuracy_score(boost_pred, y_test))
