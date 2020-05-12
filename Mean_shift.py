import pandas as pd 

from sklearn.cluster import MeanShift


if __name__ == '__main__':

    df = pd.read_csv('./Datasets/candy.csv')

    print(df.head())

    X = df.drop('competitorname', axis = 1)

    meanshift = MeanShift().fit(X)
    print('Numero de clusters: ',max(meanshift.labels_) + 1)
    print('=='*64)

    print('Centros: ',meanshift.cluster_centers_)

    df['meanshift'] = meanshift.labels_

    print('=='*64)
    print(df.head())