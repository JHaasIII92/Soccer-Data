from soccer24 import scraper as sS24
from soccer24 import derived as dS24
from mySoup import mySoup

import hashlib
import pandas as pd
import time
from os import system, name
import pyfiglet

League_sheets ={
'GER1':'GER1_ID.xlsx',
'ENG2':'ENG2_ID.xlsx',
'ENG1':'ENG1_ID.xlsx',
'IT1':'IT1_ID.xlsx',
'TUR1':'TUR1_ID.xlsx',
'FRN1':'FRN1_ID.xlsx',
#'FRN2':'FRN2_ID.xlsx',
'FRN2':'FRN22018_ID.xlsx',
'SP1':'SP1_ID.xlsx',
'SP2':'SP2_ID.xlsx',
'NET1':'NET1_ID.xlsx',
'PORT1':'PORT1_ID.xlsx',
'RUS1':'RUS1_ID.xlsx',
'RUS2':'RUS2_ID.xlsx',
'IRL1':'IRL1_ID.xlsx',
'JP1':'JP1_ID.xlsx',
#'JP1':'JP12018_ID.xlsx'
}

def main():


	league = "ENG1"
	round_num = ['2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20']
	sheet_name = league +'/'+ League_sheets[league]

	header = "League,Team,Team_Code,GF,GA,Ball_Possession,Goal_Attempts,Shots_on_Goal,Shots_off_Goal,Blocked_Shots,Free_Kicks,Corner_Kicks,Offsides,Throw_in,Goalkeeper_Saves,Fouls,Red_Cards,Yellow_Cards,Total_Passes,Completed_Passes,Tackles,lineup,points,HomeAway,HW,D,HL,Target\n'"

	totalRounds = len(round_num)
	_ = system('clear')
	for i in range (totalRounds):
		result = pyfiglet.figlet_format("First Half 24!", font =  "slant")
		print(result)
		print("\t=> League: ", league)
		print("\t=> Rounds: ", round_num)
		print("\t=> Current Round: ", round_num[i],'\n')
	
		csv_name = league + '/' + league + "FirstHalfRound" + round_num[i] + ".csv"
		fo1 = open(csv_name,'+a')
		fo1.write(header)
		make_round_sheet(sheet_name,round_num[i], fo1, league)
		fo1.close()




#-----------------------------------------------------------------------#
#-----------------------------------------------------------------------#
#-----------------------------------------------------------------------#
def make_round_sheet(sheet_name,round_num, fo1, txt_name):
	wb = "Round "+round_num
	
	stat_urls, formation_urls, match_urls = get_round_urls(sheet_name, wb)
	num_matches = len(stat_urls[0])
	for i in range (num_matches):
		
		stat_soup = []
		line_up_soup = []
		match_soup = []
		stat_soup = mySoup.get_my_soup(stat_urls[0][i])
		line_up_soup = mySoup.get_my_soup(formation_urls[0][i])
		match_soup = mySoup.get_my_soup(match_urls[0][i])
		home, away = makeRows(stat_soup, line_up_soup, match_soup, round_num, txt_name)
	
		fo1.write(home)
		fo1.write(away)
	
	_ = system('clear')
#-----------------------------------------------------------------------#
#-----------------------------------------------------------------------#
#-----------------------------------------------------------------------#
def get_round_urls(file_name, work_sheet):
	stats = "#match-statistics;1"
	lineups = "#lineups;1"
	match = "#match-summary"
	df = pd.read_excel(file_name,sheet_name=work_sheet,header=None).values
	stat_urls = (df + stats).T.tolist()
	formation_urls = (df + lineups).T.tolist()
	match_urls = (df + match).T.tolist()
	return stat_urls, formation_urls, match_urls 
#-----------------------------------------------------------------------#
#-----------------------------------------------------------------------#
#-----------------------------------------------------------------------#
def makeRows(stat_soup, line_up_soup, match_soup, round_num, league):
    
	home_team, away_team = sS24.get_team_names(stat_soup)
	print('\t=>', home_team,' vs. ',away_team)
	date, time = sS24.get_date_time(stat_soup)
	home_lineup, away_lineup  = sS24.get_line_up(line_up_soup)
	home_goals, away_goals = sS24.get_goals_scored(match_soup,"1")
	fullTime_home_goals, fullTime_away_goals = sS24.get_goals_scored(line_up_soup,"0")
	home_stat_table, away_stat_table = sS24.get_stat_table(stat_soup, "1")
	
	home_team_hex ,away_team_hex = dS24.getHex(home_team, away_team)
	home_points, away_points, Htarget, Atarget, Jtarget  = dS24.get_points(fullTime_home_goals, fullTime_away_goals)
	
	home = date +' '+ time + ',' + league + ',' + home_team + ',' + home_team_hex + ',' + home_goals + ',' + away_goals + ',' + home_stat_table + ',' + home_lineup + ','+ ',' + '1' + ',' + Htarget + ',' + Jtarget + '\n'
    
	away = date +' '+ time + ',' + league + ',' + away_team + ',' + away_team_hex + ',' + away_goals + ',' + home_goals + ',' + away_stat_table + ',' + away_lineup + ','  + ',' + '-1' + ',' + Atarget + ',' + Jtarget +'\n'

	return home,away


main()
