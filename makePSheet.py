from soccerData import DataPrep as dp
from Algos import Models as mod
import pandas as pd
import numpy as np 
def main():
	league = "SK1"
	played = True
#	weeks = [10,12,14,16,18,20]
	weeks = [28]
	run(weeks, league, played)
def run(weeks,league,played):
	for week in weeks:
		prediction_week = week
		runs = 10
		opfn = league +"/"+ league + "_Round" + str(prediction_week) + ".xlsx"
		if not played: seasonRounds = list(range(1,prediction_week))
		else: seasonRounds = list(range(1,prediction_week+1))

		seasonDF = dp.combine(league, seasonRounds)
		seasonDF = dp.add_home_away_form(seasonDF=seasonDF)
		seasonDF = dp.add_form_to_data(seasonDF=seasonDF)
		seasonDF = dp.encode_team_names(seasonDF=seasonDF)
    
		print("Just One Week at a time")
		nnOne, lstmOne = setupAndRun(seasonDF, league, 1, prediction_week, runs, False, False)
		print("Rolling All Weeks W/ STD not Skew")
		nnRunThree, lstmRunThree = setupAndRun(seasonDF, league, (prediction_week-1), prediction_week, runs, False,True)
		print("Rolling All Weeks W/ STD Skew")
		nnRunFive, lstmRunFive = setupAndRun(seasonDF, league, (prediction_week-1), prediction_week, runs, True,True)
	
	
		homePointTable, awayPointTable = mod.makePointTable(league, prediction_week, seasonDF)
		poisson = mod.poissonMonte(league, prediction_week, seasonDF)
		Odds = np.zeros((poisson.shape[0],3))

		try:
			fn = league +"/"+ league + "_Round" + str(prediction_week) + ".csv"
			oPDF = pd.read_csv(fn)
		except:
			cols = ["Home","Away"]
			oPDF = seasonDF.loc[seasonDF.Round==prediction_week][::2][cols]
			oPDF["Dates"] = oPDF.index
			index = np.array(range(0,poisson.shape[0]))
			oPDF = oPDF.set_index(index)
		stack=[oPDF.Dates,oPDF.Home, homePointTable, oPDF.Away, awayPointTable, Odds, poisson, nnOne, lstmOne, nnRunThree, lstmRunThree, nnRunFive, lstmRunFive]
		pd.DataFrame(data=pd.np.column_stack(stack), columns=getHeader()).to_excel(opfn)
#-----------------------------------------------------------------------#
#-----------------------------------------------------------------------#
#-----------------------------------------------------------------------#
def setupAndRun(seasonDF, league, rolling, prediction_week, runs, skew, std):
	weeks = [1]
	train, target= dp.get_train_and_target(league,weeks,seasonDF, skew, std)

	for i in range(2,prediction_week):
		weeks.append(i)
		if len(weeks)>rolling:
			p=weeks.pop(0)
		trainT, targetT = dp.get_train_and_target(league,weeks[::-1],seasonDF, skew, std)
		train = np.vstack((train,trainT))
		target = np.vstack((target,targetT))
	predict = dp.get_prediction_set(league, prediction_week, weeks, seasonDF, skew, std)
	nnPredictions =  mod.nural_net(train, target, predict)
	lstmPredictions = mod.GRU(train, target, predict, prediction_week)
	
	nnPredictions = evaluatePrediction(nnPredictions)
	lstmPredictions = evaluatePrediction(lstmPredictions)
	for i in range(runs-1):
		print("\t=>RUN: ", i+2)
		print("\t\t=>running NN...........")
		tempNN = mod.nural_net(train, target, predict)
		tempNN = evaluatePrediction(tempNN)
		nnPredictions = np.hstack((nnPredictions, tempNN))
		print("\t\t=>running LSTM...........")
		tempLSTM = mod.lstm(train, target, predict, prediction_week)
		tempLSTM = evaluatePrediction(tempLSTM)
		lstmPredictions = np.hstack((lstmPredictions, tempLSTM))

	nnWDLTable = []
	lstmWDLTable = []
	for i in range(len(nnPredictions)):

		nnTemp = []
		lstmTemp = []

		nnWinCount = np.count_nonzero(nnPredictions[i]==1)
		nnLossCount = np.count_nonzero(nnPredictions[i]==-1)
		nnDrawCount = np.count_nonzero(nnPredictions[i]==0)

		lstmWinCount = np.count_nonzero(lstmPredictions[i]==1)
		lstmLossCount = np.count_nonzero(lstmPredictions[i]==-1)
		lstmDrawCount = np.count_nonzero(lstmPredictions[i]==0)
    
		nnTemp.append(nnWinCount/runs)
		nnTemp.append(nnDrawCount/runs)
		nnTemp.append(nnLossCount/runs)

		lstmTemp.append(lstmWinCount/runs)
		lstmTemp.append(lstmDrawCount/runs)
		lstmTemp.append(lstmLossCount/runs)
    
		nnWDLTable.append(nnTemp)
		lstmWDLTable.append(lstmTemp)

	return nnWDLTable, lstmWDLTable
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
def getHeader():
	return ['Date_Time','Home','Total_Points_ht','Home_Points_ht','Last_Five_Weeks_ht','Away','Total_Points_ht','Home_Points_ht','Last_Five_Weeks_ht','Win_Odds','Draw_Odds','Loss_Odds','Total_P_Win','Total_P_Draw','Total_P_Loss','Last_Five_P_Win','Last_Five_P_Draw','Last_Five_P_Loss','NNOne_Win','NNOne_Draw','NNOne_Loss','LSTMOne_Win','LSTMOne_Draw','LSTMOne_Loss','NNTwo_Win','NNTwo_Draw','NNTwo_Loss','LSTMTwo_Win','LSTMTwo_Draw','LSTMTwo_Loss','NNThree_Win','NNThree_Draw','NNThree_Loss','LSTMThree_Win','LSTMTHree_Draw','LSTMThree_Loss']



main()
