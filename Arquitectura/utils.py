import pandas as pd 
import joblib

class Utils():

    def load_from_csv(self, path):
        return pd.read_csv(path)

    def features_target(self,dataset, drop_columns, y):
        X = dataset.drop(drop_columns, axis = 1)
        y = dataset[y]
        return X,y

    def model_export(self,clf,score):
        print(score)
        joblib.dump(clf, './models/best_model.pkl')