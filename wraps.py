import pandas as pd
import numpy as np

import actual_modelling
import prep
import misc
import calculus

import copy

ALPHABET="ABCDEFGHIJKLMNOPQRSTUVWXYZ"

def prep_gamma_models(df, resp, cats, conts, N=1, fractions=None, catMinPrev=0.01, contTargetPts=5, edge=0.01, weightCol=None):
 if fractions==None:
  denom = N*(N+1)/2
  fractions = [(N-x)/denom for x in range(N)]
 models = []
 for fraction in fractions:
  model = prep.prep_model(df, resp, cats, conts, catMinPrev, contTargetPts, edge, 1, weightCol)
  model["BASE_VALUE"]*=fraction
  models.append(model)
 return models

def train_gamma_models(df, resp, nrounds, lrs, models, weightCol=None, staticFeats=[], prints="normal"):
 models = actual_modelling.train_models([df], resp, nrounds, lrs, models, weightCol, staticFeats, lras=calculus.addsmoothing_LRAs, lossgrads=[[calculus.Gamma_grad]], links=[calculus.Add_mlink], linkgrads=[[calculus.Add_mlink_grad]*len(models)], pens=None, prints=prints)
 return models

def interxhunt_gamma_models(df, resp, cats, conts, models, silent=False, weightCol=None):
 
 trialModelTemplate = []
 
 for m in range(len(models)):
  df["PredComb_"+str(m)]=misc.predict(df, models[m])
  trialModelTemplate.append({"BASE_VALUE":1, "conts":{"PredComb_"+str(m):[[min(df["PredComb_"+str(m)]),min(df["PredComb_"+str(m)])],[max(df["PredComb_"+str(m)]),max(df["PredComb_"+str(m)])]]}, "featcomb":"mult"})
 
 sugImps=[[]]*len(models)
 sugFeats=[[]]*len(models)
 sugTypes=[[]]*len(models)
 
 for i in range(len(cats)):
  for j in range(i+1, len(cats)):
   if not silent:
    print(cats[i] + " X " + cats[j])
   
   trialModels = copy.deepcopy(trialModelTemplate)
   trialModels = [prep.add_catcat_to_model(trialModel, df, cats[i], cats[j], defaultValue=1) for trialModel in trialModels]
   trialModels = train_gamma_models(df, resp, 1, [1]*len(models), trialModels, weightCol, staticFeats=["PredComb_"+str(m) for m in range(len(models))], prints="silent")
   
   for m in range(len(models)):
    sugFeats[m].append(cats[i]+" X "+cats[j])
    sugImps[m].append(misc.get_importance_of_this_catcat(df, trialModels[m], cats[i]+" X "+cats[j], defaultValue=0))
    sugTypes[m].append("catcat")
 
 for i in range(len(cats)):
  for j in range(len(conts)):
   if not silent:
    print(cats[i] + " X " + conts[j])
   
   trialModels = copy.deepcopy(trialModelTemplate)
   trialModels = [prep.add_catcont_to_model(trialModel, df, cats[i], conts[j], defaultValue=1) for trialModel in trialModels]
   trialModels = train_gamma_models(df, resp, 1, [1]*len(models), trialModels, weightCol, staticFeats=["PredComb_"+str(m) for m in range(len(models))], prints="silent")
   
   for m in range(len(models)):
    sugFeats[m].append(cats[i]+" X "+conts[j])
    sugImps[m].append(misc.get_importance_of_this_catcont(df, trialModels[m], cats[i]+" X "+conts[j], defaultValue=0))
    sugTypes[m].append("catcont")
   
 for i in range(len(conts)):
  for j in range(i+1, len(conts)):
   if not silent:
    print(conts[i] + " X " + conts[j])
   
   trialModels = copy.deepcopy(trialModelTemplate)
   trialModels = [prep.add_contcont_to_model(trialModel, df, conts[i], conts[j], defaultValue=1) for trialModel in trialModels]
   trialModels = train_gamma_models(df, resp, 1, [1]*len(models), trialModels, weightCol, staticFeats=["PredComb_"+str(m) for m in range(len(models))], prints="silent")
   
   for m in range(len(models)):
    sugFeats[m].append(conts[i]+" X "+conts[j])
    sugImps[m].append(misc.get_importance_of_this_contcont(df, trialModels[m], conts[i]+" X "+conts[j], defaultValue=0))
    sugTypes[m].append("contcont")
 
 
 for m in range(len(models)):
  sugDf = pd.DataFrame({"Interaction":sugFeats[m], "Type":sugTypes[m], "Importance":sugImps[m]})
  sugDf = sugDf.sort_values(['Importance'], ascending=False).reset_index()
  sugDf.to_csv("suggestions_"+ALPHABET[m]+".csv")
  

def gnormalize_gamma_models(models, df, resp, cats, conts, startingErrorPercent=20, catMinPrev=0.01, contTargetPts=5, edge=0.01, weightCol=None):
 errModel = prep.prep_model(df, resp, cats, conts, catMinPrev, contTargetPts, edge, 1, weightCol)
 errModel["BASE_VALUE"]=startingErrorPercent*1.25/100
 models.append(errModel)
 return models

def train_gnormal_models(dfs, resp, nrounds, lrs, models, weightCol=None, staticFeats=[], prints="normal"):
 if type(dfs)!=list:
  dfs=[dfs]
 
 models = actual_modelling.train_models(dfs, resp, nrounds, lrs, models, weightCol, staticFeats, lras = calculus.addsmoothing_LRAs_erry[:len(models)-1] + [calculus.default_LRA], lossgrads = [[calculus.gnormal_u_diff, calculus.gnormal_p_diff],[calculus.gnormal_u_diff_censored, calculus.gnormal_p_diff_censored]], links=[calculus.Add_mlink_allbutlast, calculus.Add_mlink_onlylast], linkgrads=[[calculus.Add_mlink_grad]*(len(models)-1)+[calculus.Add_mlink_grad_void], [calculus.Add_mlink_grad_void]*(len(models)-1)+[calculus.Add_mlink_grad]], pens=None, prints=prints)
 
 return models

def interxhunt_gnormal_models(dfs, resp, cats, conts, models, silent=False, weightCol=None):
 
 if type(dfs)!=list:
  dfs=[dfs]
 
 if len(dfs)==2:
  cdf = dfs[0].append(dfs[1]).reset_index()
 else:
  cdf = dfs[0]
 
 trialModelTemplate = []
 
 
 for m in range(len(models)):
  for df in dfs:
   df["PredComb_"+str(m)]=misc.predict(df, models[m])
  
  minever = min([min(df["PredComb_"+str(m)]) for df in dfs])
  maxever = max([max(df["PredComb_"+str(m)]) for df in dfs])
  trialModelTemplate.append({"BASE_VALUE":1, "conts":{"PredComb_"+str(m):[[minever, minever], [maxever, maxever]]}, "featcomb":"mult"})
 
 
 sugImps=[[]]*len(models)
 sugFeats=[[]]*len(models)
 sugTypes=[[]]*len(models)
 
 for i in range(len(cats)):
  for j in range(i+1, len(cats)):
   if not silent:
    print(cats[i] + " X " + cats[j])
   
   trialModels = copy.deepcopy(trialModelTemplate)
   trialModels = [prep.add_catcat_to_model(trialModel, cdf, cats[i], cats[j], defaultValue=1) for trialModel in trialModels]
   trialModels = train_gnormal_models(dfs, resp, 1, [1]*len(models), trialModels, weightCol, staticFeats=["PredComb_"+str(m) for m in range(len(models))], prints="silent")
   
   for m in range(len(models)):
    sugFeats[m].append(cats[i]+" X "+cats[j])
    sugImps[m].append(misc.get_importance_of_this_catcat(cdf, trialModels[m], cats[i]+" X "+cats[j], defaultValue=0))
    sugTypes[m].append("catcat")
 
 for i in range(len(cats)):
  for j in range(len(conts)):
   if not silent:
    print(cats[i] + " X " + conts[j])
   
   trialModels = copy.deepcopy(trialModelTemplate)
   trialModels = [prep.add_catcont_to_model(trialModel, cdf, cats[i], conts[j], defaultValue=1) for trialModel in trialModels]
   trialModels = train_gnormal_models(dfs, resp, 1, [1]*len(models), trialModels, weightCol, staticFeats=["PredComb_"+str(m) for m in range(len(models))], prints="silent")
   
   for m in range(len(models)):
    sugFeats[m].append(cats[i]+" X "+conts[j])
    sugImps[m].append(misc.get_importance_of_this_catcont(cdf, trialModels[m], cats[i]+" X "+conts[j], defaultValue=0))
    sugTypes[m].append("catcont")
   
 for i in range(len(conts)):
  for j in range(i+1, len(conts)):
   if not silent:
    print(conts[i] + " X " + conts[j])
   
   trialModels = copy.deepcopy(trialModelTemplate)
   trialModels = [prep.add_contcont_to_model(trialModel, cdf, conts[i], conts[j], defaultValue=1) for trialModel in trialModels]
   trialModels = train_gnormal_models(dfs, resp, 1, [1]*len(models), trialModels, weightCol, staticFeats=["PredComb_"+str(m) for m in range(len(models))], prints="silent")
   
   for m in range(len(models)):
    sugFeats[m].append(conts[i]+" X "+conts[j])
    sugImps[m].append(misc.get_importance_of_this_contcont(cdf, trialModels[m], conts[i]+" X "+conts[j], defaultValue=0))
    sugTypes[m].append("contcont")
 
 
 for m in range(len(models)-1):
  sugDf = pd.DataFrame({"Interaction":sugFeats[m], "Type":sugTypes[m], "Importance":sugImps[m]})
  sugDf = sugDf.sort_values(['Importance'], ascending=False).reset_index()
  sugDf.to_csv("suggestions_"+ALPHABET[m]+".csv")

def predict_from_gnormal(df, model):
 return misc.predict_models(df, model, calculus.Add_mlink_allbutlast)
 
def predict_error_from_gnormal(df, model):
 return misc.predict_models(df, model, calculus.Add_mlink_onlylast)*100/1.25

def prep_classifier_model(df, resp, cats, conts, catMinPrev=0.01, contTargetPts=5, edge=0.01, weightCol=None):
 model = prep.prep_model(df, resp, cats, conts, catMinPrev, contTargetPts, edge, 0, weightCol)
 model["BASE_VALUE"] = calculus.Logit_delink(model["BASE_VALUE"])
 model["featcomb"] = "addl"
 return model

def train_classifier_model(df, resp, nrounds, lr, model, weightCol=None, staticFeats=[], prints="normal"):
 model = actual_modelling.train_model(df, resp, nrounds, lr, model, weightCol, staticFeats, pen=0, specificPens={}, lossgrad=calculus.Logistic_grad, link = calculus.Logit_link, linkgrad = calculus.Logit_link_grad, prints=prints)
 return model

def interxhunt_classifier_model(df, resp, cats, conts, model, silent=False, weightCol=None):
 df["PredComb"]=misc.predict(df,model)
 
 sugImps=[]
 sugFeats=[]
 sugTypes=[]
 
 for i in range(len(cats)):
  for j in range(i+1, len(cats)):
   trialmodel = {"BASE_VALUE":0, "conts":{"PredComb":[[min(df["PredComb"]),min(df["PredComb"])],[max(df["PredComb"]),max(df["PredComb"])]]}, "featcomb":"addl"}
   prep.add_catcat_to_model(trialmodel, df, cats[i], cats[j], defaultValue=0)
   trialmodel = actual_modelling.train_model(df, "y", 1, 0.2, trialmodel, staticFeats=["PredComb"], link = calculus.Logit_link,  linkgrad = calculus.Logit_link_grad, lossgrad = calculus.Logistic_grad, pen=0, prints="silent")
   
   if not silent:
    print(cats[i]+" X "+cats[j], misc.get_importance_of_this_catcat(df, trialmodel, cats[i]+" X "+cats[j], defaultValue=0))
   
   sugFeats.append(cats[i]+" X "+cats[j])
   sugImps.append(misc.get_importance_of_this_catcat(df, trialmodel, cats[i]+" X "+cats[j], defaultValue=0))
   sugTypes.append("catcat")
 
 for i in range(len(cats)):
  for j in range(len(conts)):
   trialmodel = {"BASE_VALUE":0, "conts":{"PredComb":[[min(df["PredComb"]),min(df["PredComb"])],[max(df["PredComb"]),max(df["PredComb"])]]}, "featcomb":"addl"}
   prep.add_catcont_to_model(trialmodel, df, cats[i], conts[j], defaultValue=0)
   trialmodel = actual_modelling.train_model(df, "y", 1, 0.2, trialmodel, staticFeats=["PredComb"], link = calculus.Logit_link, linkgrad = calculus.Logit_link_grad, lossgrad = calculus.Logistic_grad, pen=0, prints="silent")
   
   if not silent:
    print(cats[i]+" X "+conts[j],misc.get_importance_of_this_catcont(df, trialmodel, cats[i]+" X "+conts[j], defaultValue=0))
   
   sugFeats.append(cats[i]+" X "+conts[j])
   sugImps.append(misc.get_importance_of_this_catcont(df, trialmodel, cats[i]+" X "+conts[j], defaultValue=0))
   sugTypes.append("catcont")
 
 for i in range(len(conts)):
  for j in range(i+1, len(conts)):
   trialmodel = {"BASE_VALUE":0, "conts":{"PredComb":[[min(df["PredComb"]),min(df["PredComb"])],[max(df["PredComb"]),max(df["PredComb"])]]}, "featcomb":"addl"}
   prep.add_contcont_to_model(trialmodel, df, conts[i], conts[j], defaultValue=0)
   trialmodel = actual_modelling.train_model(df, "y", 1, 0.2, trialmodel, staticFeats=["PredComb"], link = calculus.Logit_link, linkgrad = calculus.Logit_link_grad, lossgrad = calculus.Logistic_grad, pen=0, prints="silent")
   
   if not silent:
    print(conts[i]+" X "+conts[j], misc.get_importance_of_this_contcont(df, trialmodel, conts[i]+" X "+conts[j], defaultValue=0))
   
   sugFeats.append(conts[i]+" X "+conts[j])
   sugImps.append(misc.get_importance_of_this_contcont(df, trialmodel, conts[i]+" X "+conts[j], defaultValue=0))
   sugTypes.append("contcont")
 
 sugDf = pd.DataFrame({"Interaction":sugFeats, "Type":sugTypes, "Importance":sugImps})
 sugDf = sugDf.sort_values(['Importance'], ascending=False).reset_index()
 sugDf.to_csv("suggestions.csv")