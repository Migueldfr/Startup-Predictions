import pickle
import os
import pandas as pd

# Import the model we created
cwd = os.getcwd() + '/model'
cwd

for x in os.listdir(cwd):
    if 'model_20' in x:
        with open('model/'+os.listdir(cwd)[-1], 'rb') as archivo_entrada:
            loaded_model = pickle.load(archivo_entrada)


X_test = pd.read_csv('X_test_precision.csv')

# Predict results 

predictions = loaded_model.predict(X_test)
prediction_proba = loaded_model.predict_proba(X_test) 

predictions = pd.DataFrame(predictions,columns=['Predicciones'])
prediction_proba = pd.DataFrame(prediction_proba.round(3),columns=['Fatal','Exitoso'])

results = pd.concat([X_test,predictions,prediction_proba], axis = 1)

results.to_csv('data/predicciones.csv')





