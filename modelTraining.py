import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

class ModelTraining:

    def __init__(self):
        pass

    def load_dataset(self,path):
        dataset = pd.read_csv(path)
        return dataset

    def load_model(self,modelName):
        pickle_in = open(modelName,'rb')
        model = pickle.load(pickle_in)
        return model

    def split_data(self,X,y,test_size=0.2):
        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=test_size,random_state=0)
        return X_train,X_test,y_train,y_test

    def rf_model(self,params):
        return RandomForestRegressor(n_estimators= 427,
                                                                    min_samples_split= 100,
                                                                    min_samples_leaf= 5,
                                                                    max_features = 'auto',
                                                                    max_depth= 15)

    def model_fit(self,model,X_train,y_train):
        model.fit(X_train,y_train)
        return model

    def model_predict(self,model,X_test):
        predictions = model.predict(X_test)
        return predictions

    def model_score(self,y_test,predictions):
        mae = mean_absolute_error(y_test,predictions)
        mse = mean_squared_error(y_test,predictions)
        rmse = np.sqrt(mse)
        return mae,mse,rmse