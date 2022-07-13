import pandas as pd
import numpy as np

import actual_modelling
import prep
import misc
import calculus

import wraps


#Tobit Proof of Functionality

df = pd.read_csv('gnormal.csv')
uncensoredDf = df[~df["censored"]].reset_index()
censoredDf = df[df["censored"]].reset_index()
models = [{"BASE_VALUE":1.0,"conts":{"x":[[0,103], [7,103]]},'featcomb':'mult'},{"BASE_VALUE":1.0,"conts":{"x":[[0,0.2], [7,0.2]]},'featcomb':'mult'}]
models = wraps.train_gnormal_models([uncensoredDf, censoredDf], 'y', 500, [10,0.02], models, prints="verbose")
print(models)

#assert(False)

#Logistic Proof of Concept 

df = pd.DataFrame({"cont1":[1,2,3,4,1,2,3,4], "cont2":[1,2,3,4,5,4,3,2], "cat1":['a','a','a','a','b','b','b','b'], "cat2":['c','c','d','d','c','d','e','d'], "y":[0,0,0,1,0,0,0,1]})

cats = ["cat1", "cat2"]
conts = ["cont1", "cont2"]

model = wraps.prep_classifier_model(df, "y", cats, conts)
model = wraps.train_classifier_model(df, "y", 50, 0.1, model)

pred = misc.predict(df,model, "Logit")
print(pred)

wraps.interxhunt_classifier_model(df, "y", cats, conts, model)

#Gamma Proof of Concept

df = pd.DataFrame({"cont1":[1,2,3,4,1,2,3,4], "cont2":[1,2,3,4,5,4,3,4], "cat1":['a','a','a','a','b','b','b','a'], "cat2":['c','c','d','d','c','d','e','d'], "y":[1,2,3,4,5,6,7,8]})

cats = ["cat1", "cat2"]
conts = ["cont1", "cont2"]

models = wraps.prep_gamma_models(df, 'y', cats, conts, 2)
models = wraps.train_gamma_models(df, 'y', 50, [0.2,0.3], models)

pred = misc.predict_models(df, models)
print(pred)

wraps.interxhunt_gamma_models(df, "y", cats, conts, models)

#Gnormal Proof of Concept

models = wraps.gnormalize_gamma_models(models, df, "y", cats, conts, 10)

models = wraps.train_gnormal_models(df, 'y', 1000, [0.001,0.002,0.001], models)
pred = wraps.predict_from_gnormal(df, models)
predErrPct = wraps.predict_error_from_gnormal(df, models)

print(pred, predErr)

wraps.interxhunt_gnormal_models(df,'y',cats,conts,models)

