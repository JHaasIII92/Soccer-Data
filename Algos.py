from RAU_NEAF import RAU
from contractive_autoencoder import encode
from soccerData import DataPrep as dp
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
import pandas as pd
from scipy.stats import poisson
#-----------------------------------------------------------------------#
class Models:
#-----------------------------------------------------------------------#
	def nural_net(train, target, predictionSet, prediction_week=None):
		model = keras.Sequential([
keras.layers.Dense(train.shape[0],
activity_regularizer=keras.regularizers.l1(0.0001),
kernel_regularizer=keras.regularizers.l1(0.0001),
activation='relu'),
keras.layers.BatchNormalization(),
keras.layers.Dropout(0.5),
keras.layers.Dense(32,
activity_regularizer=keras.regularizers.l1(0.0001),
kernel_regularizer=keras.regularizers.l1(0.0001),
activation='sigmoid'),
keras.layers.BatchNormalization(),
keras.layers.Dropout(0.5),
keras.layers.Dense(19,activity_regularizer=keras.regularizers.l1(0.0001),
kernel_regularizer=keras.regularizers.l1(0.0001),
activation='sigmoid'),
keras.layers.Dropout(0.5),
keras.layers.BatchNormalization(),
keras.layers.Dense(8,
activity_regularizer=keras.regularizers.l1(0.0001),
kernel_regularizer=keras.regularizers.l1(0.0001),
activation='sigmoid'),
keras.layers.Dropout(0.5),
keras.layers.BatchNormalization(),
keras.layers.Dense(3, activation='softmax')
])
		model.compile(optimizer='rmsProp',loss='logcosh')
		callBack = keras.callbacks.EarlyStopping(monitor="loss",min_delta=0,patience=50)
		callBack2 = keras.callbacks.ReduceLROnPlateau(monitor="loss",patience=25,min_delta=0)
		model.fit(train,target,epochs=1000,steps_per_epoch=2,callbacks=[callBack,callBack2],verbose = False)
		predictions = (model.predict(predictionSet))
		return predictions
#-----------------------------------------------------------------------#
#-----------------------------------------------------------------------#
#-----------------------------------------------------------------------#
	def lstm(train, target, predictionSet, prediction_week):

		games_per_week = predictionSet.shape[0]
		num_input_var = predictionSet.shape[1]
		train = pd.DataFrame(train)
		target = pd.DataFrame(target).T
		predictionSet = pd.DataFrame(predictionSet)
		predictionSet = pd.DataFrame(np.concatenate((train.values, predictionSet.values), axis=0)).T
		train = train.T
		train_split, target_split = [], []
		predictionSet_split = []
		[train_split.append(train.iloc[:, i : i + games_per_week].values)for i in range(0, games_per_week * (prediction_week - 1), games_per_week)]
		[target_split.append(target.iloc[:, i : i + games_per_week].values)for i in range(0, games_per_week * (prediction_week - 1), games_per_week)]
		[predictionSet_split.append(predictionSet.iloc[:, i : i + games_per_week].values)for i in range(0, games_per_week * (prediction_week), games_per_week)]


#BEGIN: Network Architecture
		Reset_Gate_Layers = [(num_input_var,128,'swish'),(128,'BatchNorm'),(128,64,'sigmoid'),(64,num_input_var,'sigmoid')]
		Update_Gate_Layers = [(num_input_var,128,'swish'),(128,'BatchNorm'),(128,64,'sigmoid'),(64,num_input_var,'sigmoid')]
		Output_Gate_Layers = [(num_input_var,128,'swish'),(128,'BatchNorm'),(128,64,'sigmoid'),(64,num_input_var,'sigmoid')]
		Attention_Gate_Layer = [(games_per_week,games_per_week,'softmax'),(games_per_week,num_input_var,'tanh')]
		Stepdown_Network_Layers = [(num_input_var,'BatchNorm'),(num_input_var,3,'linear')]
		Architecture = [Reset_Gate_Layers,Update_Gate_Layers,Output_Gate_Layers,Attention_Gate_Layer,Stepdown_Network_Layers]

# BEGIN: Declare and run RAU NEAF
		break_pt = 5
		lr = 0.001           # Learning rate
		key_LF = "Softmax_CrossEntropy"     # Cost function
		key_Opt = "Adam"  # Optimization technique
		network = RAU(prediction_week-1, Architecture, lr, key_LF, key_Opt,break_pt)
		epochs = 10001
	
		for i in range(epochs):
			early_break = network.train(train_split, target_split, i)
			if early_break == True: break

		results = (network.query(predictionSet_split)).T
		return results
#-----------------------------------------------------------------------#
#-----------------------------------------------------------------------#
#-----------------------------------------------------------------------#
	def GRU(train, target, predictionSet, prediction_week):
		games_per_week = predictionSet.shape[0]
		num_input_var = predictionSet.shape[1]
		train = train.reshape(prediction_week - 1, games_per_week, num_input_var)
		target = target.reshape(prediction_week - 1, games_per_week, 3)
		test_x = train[:-1,:,:]
		test_y = target[:-1,:,:]
		predictionSet = predictionSet.reshape(1, games_per_week, num_input_var)
		input = keras.layers.Input(shape=(games_per_week, num_input_var))
		output = keras.layers.GRU(128,return_sequences=True,unroll=True,recurrent_dropout=0.1,kernel_regularizer=keras.regularizers.l2(0.01),activity_regularizer=keras.regularizers.l1(0.01),input_shape=(games_per_week, num_input_var),)(input)
		output = keras.layers.GaussianNoise(0.1)(output)
		X_shortcut = output
		output = keras.layers.GRU(128,return_sequences=True,unroll=True,recurrent_dropout=0.1,kernel_regularizer=keras.regularizers.l2(0.01),activity_regularizer=keras.regularizers.l1(0.01),input_shape=(games_per_week, num_input_var),)(output)
		output = keras.layers.Add()([output, X_shortcut])
		output = keras.layers.GaussianNoise(0.1)(output)
		output = keras.layers.GRU(128,return_sequences=True,unroll=True,recurrent_dropout=0.1,kernel_regularizer=keras.regularizers.l2(0.01),activity_regularizer=keras.regularizers.l1(0.01),
    	input_shape=(games_per_week, num_input_var),)(output)
		output = keras.layers.BatchNormalization()(output)
		output = keras.layers.Dense(10,activation="sigmoid",kernel_regularizer=keras.regularizers.l2(0.01),activity_regularizer=keras.regularizers.l1(0.01),)(output)
		output = keras.layers.GaussianNoise(0.1)(output)
		output = keras.layers.Dense(3,activation="softmax",kernel_regularizer=keras.regularizers.l2(0.01),activity_regularizer=keras.regularizers.l1(0.01),)(output)
		model = keras.Model(inputs=input, outputs=output)
# ----- Train model -----#
		callbacks = [EarlyStopping(monitor='val_loss', patience=10),]
		model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["acc"])
		model.fit(train, target, epochs=700,callbacks=callbacks,verbose=False,validation_data=(test_x, test_y))
# ----- Query model for prediction -----#
		prediction = model.predict(predictionSet)
		return prediction[0]

#-----------------------------------------------------------------------#
#-----------------------------------------------------------------------#
#-----------------------------------------------------------------------#
	def poissonMonte(league, prediction_week, df):
		try:
			fn = league +"/"+ league + "_Round" + str(prediction_week) + ".csv"
			opDF = pd.read_csv(fn)
		except:
			cols = ["Home","Away"]
			opDF = df.loc[df.Round==prediction_week][::2][cols]
		lastFiveWeeks = list(range(prediction_week-5,prediction_week))
		oP = []
		for i, rows in opDF.iterrows():
			homeWins = 0
			awayWins = 0
			draw = 0
			homeWins_LF = 0
			awayWins_LF = 0
			draw_LF = 0
			homeTeamName = rows.Home
			awayTeamName = rows.Away
			homeTeam = df[::2].loc[df[::2].Team==homeTeamName].GF.values
			awayTeam = df[1::2].loc[df[1::2].Team==awayTeamName].GF.values
			homeTeam_LF = df.loc[(df.Team==homeTeamName)&(df.Round.isin(lastFiveWeeks))].GF.values
			awayTeam_LF = df.loc[(df.Team==awayTeamName)&(df.Round.isin(lastFiveWeeks))].GF.values
			for i in range(1,101):
				home = np.random.poisson(homeTeam.mean(), 1)[0]
				away = np.random.poisson(awayTeam.mean(), 1)[0]
				if home > away:
					homeWins += 1
				elif home < away:
					awayWins += 1
				else:
					draw +=1
				home = np.random.poisson(homeTeam_LF.mean(), 1)[0]
				away = np.random.poisson(awayTeam_LF.mean(), 1)[0]
				if home > away:
					homeWins_LF += 1
				elif home < away:
					awayWins_LF += 1
				else:
					draw_LF +=1
				temp = [homeWins,draw,awayWins,homeWins_LF,draw_LF,awayWins_LF]

			oP.append(temp)
		oP = np.array(oP)
		return oP
#-----------------------------------------------------------------------#
#-----------------------------------------------------------------------#
#-----------------------------------------------------------------------#

	def makePointTable(league, prediction_week, df):
		try:
			fn = league +"/"+ league + "_Round" + str(prediction_week) + ".csv"
			opDF = pd.read_csv(fn)
		except:
			cols = ["Home","Away"]
			opDF = df.loc[df.Round==prediction_week][::2][cols]
		lastFiveWeeks = list(range(prediction_week-5,prediction_week))
		pointTable = []
		for i, rows in opDF.iterrows():
			tempTable = []
			totalHomeTemp = df.loc[df.Team==rows.Home].Points
			totalAwayTemp = df.loc[df.Team==rows.Away].Points
			justHomeTemp = df[::2].loc[df[::2].Team==rows.Home].Points
			justAwayTemp = df[1::2].loc[df[1::2].Team==rows.Away].Points
			last5HomeTemp = df.loc[(df.Team==rows.Home)&(df.Round.isin(lastFiveWeeks))].Points
			last5AwayTemp = df.loc[(df.Team==rows.Away)&(df.Round.isin(lastFiveWeeks))].Points
			tempTable.append(totalHomeTemp.sum()/totalHomeTemp.count())
			tempTable.append(justHomeTemp.sum()/justHomeTemp.count())
			tempTable.append(last5HomeTemp.sum()/last5HomeTemp.count())
			tempTable.append(totalAwayTemp.sum()/totalAwayTemp.count())
			tempTable.append(justAwayTemp.sum()/justAwayTemp.count())
			tempTable.append(last5AwayTemp.sum()/last5AwayTemp.count())
			pointTable.append(tempTable)
		pointTable = np.array(pointTable)
		homePointTable = pointTable[:,:3]
		awayPointTable = pointTable[:,3:]
		return homePointTable, awayPointTable
