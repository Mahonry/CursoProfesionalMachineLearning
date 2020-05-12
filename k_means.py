import pandas as pd 


from sklearn.cluster import MiniBatchKMeans

if __name__ == '__main__':

    df = pd.read_csv('./Datasets/candy.csv')
    print(df.head())

    X = df.drop('competitorname', axis = 1)

    kmeans =  MiniBatchKMeans(n_clusters = 4,
                              batch_size = 8,
                              ).fit(X)
    print('Total de centros: ', len(kmeans.cluster_centers_))
    print('__'*64)
    print(kmeans.predict(X))
    

    df['cluster'] = kmeans.predict(X)

    print(df) 


