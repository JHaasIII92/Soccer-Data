import pandas as pd
import numpy as np
from datetime import datetime
from datetime import timedelta
import hashlib

class DataPrep:


	def combine(league, weeks):
		"""Combine rounds sheets into one pandas Data Frame
A column "Round" will be added for indexing
str league: The two to three letter league code
int list weeks: desierd weeks to be added to data frame"""
		#---------Start-DF-With-First-Round-In-List-----------#
		fn = league + "/Rounds/" + league + "Round" + str(weeks[0]) + ".csv"
		df = pd.read_csv(fn)
		df.insert( loc=1, column="Round", value=weeks[0])
		weeks.pop(0)
		#---------Loop-Through-Weeks-&-Add-To-DF--------------#
		for week in weeks:
			fn2 = league + "/Rounds/" + league + "Round" + str(week) + ".csv"
			df2 = pd.read_csv(fn2)
			df2.insert( loc=1, column="Round", value=week)
			df = df.append(df2)
		df.index = pd.to_datetime(df.index)
		
		return df
#-------------------------------------------------------------------------------------------#
#-------------------------------------------------------------------------------------------#
#-------------------------------------------------------------------------------------------#
	def rolling(homeTeamDF, awayTeamDF, roundsDF, skewHA=False):
		"""Rolling is used to normalize training data that incorperates past preformaces
with the teams training week preformance
str league: The two to three letter league code
pd DataFrame seasonDF: can be made with the method "combine" 
int list weeks: training week goes in the first slot and the weeks to be rolled in go in the rest
Returns:
np float array homeAVG: Normalized training data with recpect to home team"""
		#-----Summing-Up-Each-Individual-Team-----#
		meanHomeStats = []
		stdHomeStats = []
		for i, rows in homeTeamDF.iterrows():
			
			if skewHA:
				temp = roundsDF.loc[roundsDF[::2].Team == rows.Team].iloc[:,5:].values.astype(float)
			else:
				temp = roundsDF.loc[roundsDF.Team == rows.Team].iloc[:,5:].values.astype(float)
			
			tempMean = np.mean(temp, axis=0)
			tempSTD = np.std(temp, axis=0)
			meanHomeStats.append(tempMean)
			stdHomeStats.append(tempSTD)
		meanAwayStats = []
		stdAwayStats = []
		for i, rows in awayTeamDF.iterrows():
			if skewHA:
				temp = roundsDF.loc[roundsDF[1::2].Team == rows.Team].iloc[:,5:].values.astype(float)
			else:
				temp = roundsDF.loc[roundsDF.Team == rows.Team].iloc[:,5:].values.astype(float)
			tempMean = np.mean(temp, axis=0)
			tempSTD = np.std(temp, axis=0)
			meanAwayStats.append(tempMean)
			stdAwayStats.append(tempSTD)

		meanHomeStats = np.array(meanHomeStats)
		stdHomeStats = np.array(stdHomeStats)
		meanAwayStats = np.array(meanAwayStats)
		stdAwayStats = np.array(stdAwayStats)

		#-Normalizing-With-Respect-To-Home-Team-#
		summedHomeAwayMean = meanHomeStats + meanAwayStats
		summedHomeAwaySTD = stdHomeStats + stdAwayStats
		homeMeanAVG = meanHomeStats/summedHomeAwayMean
		homeSTDAVG = stdHomeStats/summedHomeAwaySTD
		#---Clean-Up-Normalized-Data---#
		homeMeanAVG = np.nan_to_num(homeMeanAVG)
		bad = (homeMeanAVG> 1.79769313e+308) 
		homeMeanAVG[bad] = 0

		homeSTDAVG = np.nan_to_num(homeSTDAVG)
		bad = (homeSTDAVG> 1.79769313e+308) 
		homeSTDAVG[bad] = 0
		
		return homeMeanAVG, homeSTDAVG
#-------------------------------------------------------------------------------------------#
#-------------------------------------------------------------------------------------------#
#-------------------------------------------------------------------------------------------#
	def get_train_and_target(league,weeks,seasonDF, skewHA=False, addSTD=False):
		"""Use this method to retrive a training set for any week and target
Calls rolling to normalize the training set and returns the target of given
traing week"""
		#------------------------Set-Up---------------------------#
		teams = seasonDF.loc[seasonDF.Round == weeks[0]].iloc[:,2:3]
		homeTeams = teams[::2]
		awayTeams = teams[1::2]
		target = seasonDF.loc[seasonDF.Round == weeks[0]].iloc[:,-4:-1][::2].values
		data = seasonDF[seasonDF.Round.isin(weeks)].iloc[:,2:-5]
		#---------------------Call-Normalizor---------------------#
		homeMeanAVG, homeSTDAVG = DataPrep.rolling( homeTeams, awayTeams, data )
		if addSTD:
			train = np.hstack((homeMeanAVG, homeSTDAVG))
		else:
			train = homeMeanAVG
		return train[:,1:], target
#-------------------------------------------------------------------------------------------#
#-------------------------------------------------------------------------------------------#
#-------------------------------------------------------------------------------------------#
	def get_played_prediction_set(league, predictionWeek, weeks, seasonDF, skewHA=False):
		"""Use this method to make a prediction set 
for a week that has already been played"""
		#------------------------Set-Up---------------------------#
		teams = seasonDF.loc[seasonDF.Round == predictionWeek].iloc[:,2:3]
		homeTeams = teams[::2]
		awayTeams = teams[1::2]
		data = seasonDF[seasonDF.Round.isin(weeks)].iloc[:,2:-5]
		#---------------------Call-Normalizor---------------------#
		homeMeanAVG, homeSTDAVG = DataPrep.rolling( homeTeams, awayTeams, data )
		pSet = np.hstack((homeMeanAVG, homeSTDAVG))
		return pSet[:,1:]
#------------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------------#
	def get_prediction_set(league, predictionWeek, weeks, seasonDF, skewHA=False, addSTD=False):
		"""Use this method to make a prediction set for an upcoming week"""
		#------------------------Set-Up---------------------------#
		try:
			fn = league +"/"+ league + "_Round" + str(predictionWeek) + ".csv"
			homeTeams = pd.read_csv(fn,usecols=[1])
			homeTeams.columns = ["Team"]
			awayTeams = pd.read_csv(fn,usecols=[2])
			awayTeams.columns = ["Team"]
		except:
			homeTeams = pd.DataFrame((seasonDF.loc[seasonDF.Round==predictionWeek][::2].Home))
			homeTeams.columns = ["Team"]
			awayTeams = pd.DataFrame((seasonDF.loc[seasonDF.Round==predictionWeek][::2].Away))
			awayTeams.columns = ["Team"]
		#---------------------Call-Normalizor---------------------#
		data = seasonDF[seasonDF.Round.isin(weeks)].iloc[:,2:-5]
		homeMeanAVG, homeSTDAVG = DataPrep.rolling( homeTeams, awayTeams, data )
		if addSTD:
			pSet = np.hstack((homeMeanAVG, homeSTDAVG))
		else:
			pSet = homeMeanAVG
		return pSet[:,1:]
#------------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------------#     
	def get_odds_sheet(league):
		"""Use this method to get the odds for any specified league 
curently season 2018/2019 only"""
		#-Read-in-Bet-Explorer-CSV-#
		fn = league + "/Odds.csv"
		df = pd.read_csv(fn)
		return df
#------------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------------#
	def get_plyed_prediction_OPT(roundDF, oddsDF, league, asia=False):
		"""Use This meathod to make an output template for a week that has already been played
returns a numpy array """

		roundDF = roundDF[::2]
		templateIndex = []
		templateData = []
		for index, rows in roundDF.iterrows():
			date = index.date()

			if(asia):
				tempDF = oddsDF[str(date + timedelta(days=2)):str(date - timedelta(days=1))]
			else:
				tempDF = oddsDF[str(date)]

			tempDF = tempDF.loc[tempDF.Home==rows.Team]
			templateIndex.append(str(date))
			templateData.append(tempDF.values[0])

		templateIndex = np.array(templateIndex)
		templateIndex = templateIndex.reshape(templateIndex.shape[0],1)
		templateData = np.array(templateData)
	
		template = np.hstack((templateIndex, templateData))

		return template
#------------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------------#
	def add_form_to_data(seasonDF):
		"""This Method gives the option of deriveing and adding the three forms to the season Data Frame. The Form comes form the number of point eraned by the team in each increment"""
		#-------------------------------------------------------------#
		#------------Set-Up-The-Inital-Form-Arrays--------------------#	
		teamsInLeague = seasonDF.loc[seasonDF.Round == 1].Team.shape[0]
		seasonForm = np.zeros((teamsInLeague, 1))
		tenWeekForm = np.zeros((teamsInLeague, 1))
		fiveWeekForm = np.zeros((teamsInLeague, 1))
		#-------------------------------------------------------------#
		#--------------Set-For-Controll-loop--------------------------#	
		rounds = seasonDF.Round.value_counts().index[::-1]
		rounds = np.delete(rounds,0)
		roundsCompleted = [1]
		#-------------------------------------------------------------#
		#----------------------Controll-loop--------------------------#	
		for round_num in rounds:

			num_roundsCompleted = len(roundsCompleted)
			roundTeamOrder = seasonDF.loc[seasonDF.Round == round_num].Team.values
			#------Make-Entier-Season-Form----#
			completedDF = (seasonDF.loc[seasonDF.Round.isin(roundsCompleted)])
			seasonFormTemp = DataPrep.form(completedDF, roundTeamOrder)
			#------Make-Entier-10-Week-Form----#
			if num_roundsCompleted >= 10:
				cut = (10 - roundsCompleted[0])
				completedDF = (seasonDF.loc[seasonDF.Round.isin(roundsCompleted[:cut])])
				tenWeekFormTemp = DataPrep.form(completedDF, roundTeamOrder)
			else:
				tenWeekFormTemp = seasonFormTemp
			#------Make-Entier-5--Week-Form----#
			if num_roundsCompleted >= 5:
				cut = (5 - roundsCompleted[0])
				completedDF = (seasonDF.loc[seasonDF.Round.isin(roundsCompleted[:cut])])
				fiveWeekFormTemp = DataPrep.form(completedDF, roundTeamOrder)
			else:
				fiveWeekFormTemp = seasonFormTemp
			#------Stack-Current-Rounds-Form-To-Output-Array-----#
			seasonForm = np.vstack((seasonForm,seasonFormTemp))
			tenWeekForm = np.vstack((tenWeekForm,tenWeekFormTemp))
			fiveWeekForm = np.vstack((fiveWeekForm,fiveWeekFormTemp))
			#---Increment-Rounds-Completed---#
			roundsCompleted.append(round_num)
		#-------------------------------------------------------------#
		#------------Insert-Forms-To-Season-DataFrame-----------------#
		seasonDF.insert(loc=5 ,column="Season_Form" , value=seasonForm)
		seasonDF.insert(loc=5 ,column="Ten_Week_Form" , value=tenWeekForm)
		seasonDF.insert(loc=5 ,column="Five_Week_Form" , value=fiveWeekForm)
		return seasonDF
#------------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------------#
	def add_home_away_form(seasonDF):
		"""This Method is used to make the form for Home recored for home teams and
		away recored for away team"""
		#-------------------------------------------------------------#
		#------------Set-Up-The-Inital-Form-Arrays--------------------#	
		teamsInLeague = seasonDF.loc[seasonDF.Round == 1].Team.shape[0]
		homeAwayForm = np.zeros((teamsInLeague, 1))
		#-------------------------------------------------------------#
		#--------------Set-For-Controll-loop--------------------------#	
		rounds = seasonDF.Round.value_counts().index.sort_values()
		rounds = np.delete(rounds,0)
		completed = [1]
		Home = True
		#-------------------------------------------------------------#
		#----------------------Controll-loop--------------------------#	
		for round_num in rounds:
			#-----Set-Up-Inner-Loop---------------------------#
			teamOrder = seasonDF.loc[seasonDF.Round==round_num].Team.values
			completedDf = seasonDF.loc[seasonDF.Round.isin(completed)]
			tempFormVector = []
    			#-----Inner-Loop-Making-Round-Vector-----#
			for teams in teamOrder:
				if Home:
					temp = completedDf[::2].loc[completedDf[::2].Team == teams]
					Home = False
				else:
					temp = completedDf[1::2].loc[completedDf[1::2].Team == teams]
					Home = True
				if temp.shape[0] == 0:
					tempFormVector.append(0)
				else:
					tempFormVector.append(temp.Points.sum())
			#------------Appending-Round-To-Season-Vector----------#
			completed.append(round_num)
			tempFormVector = np.array(tempFormVector)
			tempFormVector = tempFormVector.reshape(tempFormVector.shape[0],1)
			homeAwayForm = np.vstack((homeAwayForm,tempFormVector))
		#-----Attaching-Form-To-Season-Data-Frame-------------------------#
		seasonDF.insert(loc=5 ,column="Home_Away_Form" , value=homeAwayForm)
		return seasonDF
#------------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------------#
	def form(completedDF, roundTeamOrder):
		"""This Method is curently only inuse for the method add_form_to_data
to make a form of desiered size for deaiesed week"""
		formTemp = []
		for teams in roundTeamOrder:
			teamPoints = completedDF.loc[completedDF.Team == teams].Points.sum()
			formTemp.append(teamPoints)
		formTemp = np.array(formTemp)
		formTemp = formTemp.reshape(formTemp.shape[0],1)
		return formTemp
#------------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------------#
	def encode_team_names(seasonDF):
		"""This Method encodes team name into an interger"""
		teams = seasonDF.Team.values
		vectorHex = np.vectorize(DataPrep._getHex)
		teams = vectorHex(teams)
		seasonDF.insert(loc=5 ,column= "Encoded", value=teams)
		return seasonDF
#------------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------------#
	def _getHex(team):
		"""Privete: This metheod is for vectorizing encode_team_names method"""
		team_encode = hashlib.sha224((team.encode())).hexdigest()
		team_hex = int(team_encode, 16)
		return str(team_hex)
#------------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------------#
	def getTeamStats(team, df):
		"""This method returns a pandas DF of all of the teams stats"""
		return df.loc[df.Team==team]
#------------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------------#
#TEMP#############

	def make_Home_Head_to_Head(homeTeam, df):
		home = df.loc[(df.Team==homeTeam)&(df.Away!=homeTeam)]
		openet = df.loc[(df.Team!=homeTeam)&(df.Home==homeTeam)]
		return home, openet
#------------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------------#
#TEMP#############
	def make_Away_Head_to_Head(awayTeam, df):
		away = df.loc[(df.Team==awayTeam)&(df.Home!=awayTeam)]
		openet = df.loc[(df.Team!=awayTeam)&(df.Away==awayTeam)]
		return away, openet
#------------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------------#
