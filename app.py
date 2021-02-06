import uvicorn
import numpy as np
from fastapi import FastAPI
from modelTraining import ModelTraining
from BlackFridaySales import BlackFridaySale

app = FastAPI()
modelTraining = ModelTraining()

@app.post('/train')
def trainModel():
    dataset = modelTraining.load_dataset('processed_train.csv')
    X=dataset.iloc[:,:-1]
    y=dataset.iloc[:,-1]
    X_train,X_test,y_train,y_test = modelTraining.split_data(X,y,0.2)

    model_rf=modelTraining.model_fit(X_train,y_train)

    model = modelTraining.rf_model()

    model_rf = modelTraining.model_fit(model,X_train,y_train)
    predictions = modelTraining.model_predict(model_rf,X_test)
    mae,mse,rmse = modelTraining.model_score(y_test,predictions)

    print ('mean absolute error '.format(mae))
    print('mena squared error'.format(mse))
    print('root mean squared error'.format(rmse))
    return {
        "mae":mae,
        "mse":mse,
        "rmse":rmse
     }

@app.post('/predict')
def predict_price(data:BlackFridaySale):
    Gender = data.Gender
    Age = data.Age
    Occupation = data.Occupation
    City_Category = data.City_Category
    Stay_In_Current_City_Years = data.Stay_In_Current_City_Years
    Marital_Status = data.Marital_Status
    Product_Category_1 = data.Product_Category_1
    Product_Category_2 = data.Product_Category_2
    Product_Category_3 = data.Product_Category_3

    data_new = [[Gender,Age,Occupation,City_Category,Stay_In_Current_City_Years,Marital_Status,Product_Category_1,Product_Category_2,Product_Category_3]]
    model=modelTraining.load_model('rf_model.pkl')
    prediction = modelTraining.model_predict(model,data_new)
    predictionText = 'The black friday sale prediction price is {}'.format(np.round(prediction[0],3))
    return {
        'prediction':predictionText
    }

if __name__ == '__main__':
    uvicorn.run(app,host='127.0.0.1',port=8000)
