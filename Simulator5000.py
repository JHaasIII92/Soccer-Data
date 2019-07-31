#---------------------------Imports-------------------------------------#
from soccerData import DataPrep as dp
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from os import system, name
import pyfiglet
#-----------------------------------------------------------------------#
#-----------------------------------------------------------------------#
#-----------------------------------------------------------------------#
def main():
	#-User-Inputs-#
	league = "SWD1"
	asia = True
	oPfn = league  + "/" + league + "_2018_2019_JamesPicks002.csv" # <= output
	lastPWeek = 15
	firstPWeek = 6
	roundSpan =2
	#-Load-Soccer-Data-#
	roundsList = list(range(1, lastPWeek + 1))
	seasonDF = dp.combine(league, roundsList)
	seasonDF = dp.add_form_to_data(seasonDF) # <<<Form<<<
	seasonDF = dp.encode_team_names(seasonDF) # <<<Encode<<<
	seasonDF = dp.add_home_away_form(seasonDF)
	oddsDF = dp.get_odds_sheet(league)
	#---------------------------------------------------------------#
	#------------------Set-Up-&-Execute-First-Run-------------------#
	#-The-Set-Up-#
	weeks = [1]
	train, target = dp.get_train_and_target(league,weeks,seasonDF)
	for i in range(2,firstPWeek):
		weeks.append(i)
		if len(weeks) > roundSpan:
			weeks.pop(0)
		trainTemp, targetTemp = dp.get_train_and_target(league,weeks[::-1],seasonDF)
		train = np.vstack((train, trainTemp))
		target = np.vstack((target, targetTemp))
	predictionSet = dp.get_played_prediction_set(league, firstPWeek, weeks, seasonDF)
	#--------#
	printOut(league, firstPWeek, lastPWeek)
	#-The-Run-#
	predictions = runModel(train, target, predictionSet)
	evaluatedPredictions = evaluatePrediction(predictions)
	#---------Building-The-Output-Array----------------#
	roundDF = seasonDF.loc[seasonDF.Round == firstPWeek]
	roundVector = np.full((predictions.shape[0],1), firstPWeek)
	roudOPTemplate = dp.get_plyed_prediction_OPT(roundDF, oddsDF, league, asia=asia)
	outPut = np.concatenate((roundVector,roudOPTemplate, evaluatedPredictions, predictions),axis=1)
	#----------------------------------------------------#
	#-------------Set-Up-&-Run-Rest-Of-Season------------#
	remainingPWeeks = range(firstPWeek, lastPWeek)
	for i in remainingPWeeks:
		predictionWeek = i + 1
		#-------------------------------------------#
		printOut(league, predictionWeek, lastPWeek)
		#--The-Build--------------------------------------------------------#
		weeks.append(i)
		if len(weeks) > roundSpan:
			weeks.pop(0)
		trainTemp, targetTemp = dp.get_train_and_target(league,weeks[::-1],seasonDF)
		train = np.vstack((train, trainTemp))
		target = np.vstack((target, targetTemp))
		predictionSet = dp.get_played_prediction_set(league, predictionWeek, weeks, seasonDF)
		#--The-Run---------------------------------------------------------------#
		predictions = runModel(train, target, predictionSet)
		evaluatedPredictions = evaluatePrediction(predictions)
		#--UpDate-Output-Array------------------------------#
		roundDF = seasonDF.loc[seasonDF.Round == predictionWeek]
		roundVector = np.full((predictions.shape[0],1), predictionWeek)
		roudOPTemplate = dp.get_plyed_prediction_OPT(roundDF, oddsDF, league, asia=asia)
		tempOutput = np.concatenate((roundVector,roudOPTemplate, evaluatedPredictions, predictions),axis=1)
		outPut = np.vstack((outPut, tempOutput))
	#----------------------------------------------#
	#---Output-To-CSV------------------------------#
	header = getHeader()
	pd.DataFrame(outPut).to_csv(oPfn, header=header)
#-----------------------------------------------------------------------#
#-----------------------------------------------------------------------#
#-----------------------------------------------------------------------#
def runModel(train, target, predictionSet):


	model = keras.Sequential([
keras.layers.Dense(train.shape[0],
                   activity_regularizer=keras.regularizers.l1(0.0001),
                   kernel_regularizer=keras.regularizers.l1(0.0001),
                   activation='relu'),
keras.layers.BatchNormalization(),
keras.layers.Dropout(0.5),
keras.layers.Dense(32,activity_regularizer=keras.regularizers.l1(0.0001),
                   kernel_regularizer=keras.regularizers.l1(0.0001),
                   activation='sigmoid'),
keras.layers.BatchNormalization(),
keras.layers.Dropout(0.5),
keras.layers.Dense(19,activity_regularizer=keras.regularizers.l1(0.0001),
                   kernel_regularizer=keras.regularizers.l1(0.0001),
                   activation='sigmoid'),
keras.layers.Dropout(0.5),
keras.layers.BatchNormalization(),
keras.layers.Dense(8,activity_regularizer=keras.regularizers.l1(0.0001),
                   kernel_regularizer=keras.regularizers.l1(0.0001),
                   activation='sigmoid'),
keras.layers.Dropout(0.5),
keras.layers.BatchNormalization(),
keras.layers.Dense(3, activation='softmax')
])

	model.compile(optimizer='rmsProp',
	loss='logcosh')

	callBack = keras.callbacks.EarlyStopping(monitor="loss",
	 min_delta=0,
	 patience=50)

	callBack2 = keras.callbacks.ReduceLROnPlateau(monitor="loss",
	patience=25,
	min_delta=0)

	model.fit(train,
	target,
	epochs=1000,
	steps_per_epoch=2,
	callbacks=[callBack,callBack2])
	
	predictions = (model.predict(predictionSet))

	return predictions
#-----------------------------------------------------------------------#
#-----------------------------------------------------------------------#
#-----------------------------------------------------------------------#
def evaluatePrediction(prediction):
	#-Change-Favored-Prediction-To-One-#
	#-----------Rest-To-Zero-----------#
	maxloc = np.argmax(prediction, axis=1)
	maxloc = maxloc.reshape(maxloc.shape[0],1)
	home = (maxloc == 0)
	draw = (maxloc == 1)
	away = (maxloc == 2)
	maxloc[home] = 1
	maxloc[draw] = 0
	maxloc[away] = -1

	return maxloc
#-----------------------------------------------------------------------#
#-----------------------------------------------------------------------#
#-----------------------------------------------------------------------#
def getWrongVectorValue(prediction, target):

	predictionLoc = np.argmax(prediction, axis=1)
	targetLoc = np.argmax(target, axis=1)

	numberOfPredictions = prediction.shape[0]

	correct = (predictionLoc == targetLoc).sum()
	wrong = (numberOfPredictions - correct)/numberOfPredictions
	
	return wrong
#-----------------------------------------------------------------------#
#-----------------------------------------------------------------------#
#-----------------------------------------------------------------------#
def printOut(league, PWeek, lastPWeek):
	system('clear')
	result = pyfiglet.figlet_format("SIMULATOR!", font =  "slant" )
	print(result)
	print("\t League =>", league)
	print("\t Running =>", PWeek, "/", lastPWeek)
#-----------------------------------------------------------------------#
#-----------------------------------------------------------------------#
#-----------------------------------------------------------------------#
def getHeader():
	return ["Round", "Date_Time", "League","Home", "Away","Home_Goals", "Away_Goals", "Home_Odds", "Draw_Odds", "Away_Odds","Favored_By_Odds", "Outcome", "Favored_By_Rick","Home_Prediction", "Draw_Prediction", "Away_Prediction"]
#-----------------------------------------------------------------------#
#-----------------------------------------------------------------------#
#-----------------------------------------------------------------------#
main()

	
