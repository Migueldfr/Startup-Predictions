import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
import pickle
import datetime
import os
cwd = os.getcwd()


# Import the dataset
cwd = os.getcwd()

startups = pd.read_csv(cwd+'/data/data_for_training.csv')

# Split the dataset along train and test

X = startups[['relationships', 'age_last_milestone_year', 'milestones',
       'Top500', 'age', 'relation10', 'age_first_milestone_year',
       'Has_roundABCD', 'funding_rounds', 'has_roundB', 'milesto_4', 'million']]

y = startups['status']

# Train and test processing

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=5)

# Train de model 

pipe_final = Pipeline(steps=[
                    ('classifier', GradientBoostingClassifier(learning_rate=0.2, max_depth=4, min_samples_split= 7, n_estimators= 90, random_state=60))
])

pipe_final.fit(X_train,y_train)


# Guardar el modelo en un pickle


date = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

with open('model/model_'+ date, 'wb') as archivo_salida:
    pickle.dump(pipe_final , archivo_salida)
