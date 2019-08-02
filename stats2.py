#-----------------------------------------------------------------------------------------#
# Purpose:	The purpose of the of this script is to use the scrapper libary
# 	to scrape match stats from Soccer24.com for a specific league
#	and spcified rounds.
#
# Dependents:	the excel spread sheet made by Links.py titled LEAGUE_ID.xlsx 
#	this spread sheet must be located in the leagues folder exactly where 
#	Links.py places it.
#-------------------------------------------------------------------------------------#
from scrapper import s24Scraper as sS24
from scrapper import s24Derived as dS24
from mySoup import mySoup

import pandas as pd
from os import system, name
import pyfiglet

League_sheets ={
'GER1':'GER1_ID.xlsx',
'ENG2':'ENG2_ID.xlsx',
'ENG1':'ENG1_ID.xlsx',
#'ENG1':'ENG1_ID_1718.xlsx',
'IT1':'IT1_ID.xlsx',
'TUR1':'TUR1_ID.xlsx',
'FRN1':'FRN1_ID.xlsx',
'SP1':'SP1_ID.xlsx',
'SP2':'SP2_ID.xlsx',
'NET1':'NET1_ID.xlsx',
'PORT1':'PORT1_ID.xlsx',
'RUS1':'RUS1_ID.xlsx',
'RUS2':'RUS2_ID.xlsx',
'IRL1':'IRL1_ID.xlsx',
'JP1':'JP1_ID.xlsx',
#'JP1':'JP12018_ID.xlsx',
'CHI1':'CHI1_ID.xlsx',
'SK1':'SK1_ID.xlsx',
'SWD1':'SWD1_ID.xlsx',
'BRZ1':'BRZ1_ID.xlsx'
}

def main():
	league = "CHI1"
	sheet_name = league +'/'+ League_sheets[league]
	round_num = ['19']
	totalRounds = len(round_num)

	_ = system('clear')
	for i in range (totalRounds):
		result = pyfiglet.figlet_format("SOCCER 24!", font =  "slant")
		print(result)
		print("\t=> League: ", league)
		print("\t=> Rounds: ", round_num)
		print("\t=> Current Round: ", round_num[i],'\n')
		
		csv_name = league + '/' + league + "Round" + round_num[i] + ".csv"
		fo1 = open(csv_name,'+a')
		fo1.write(getHeader())
		
		make_round_sheet(sheet_name,round_num[i], fo1, league)
		fo1.close()

#-----------------------------------------------------------------------#
#-----------------------------------------------------------------------#
#-----------------------------------------------------------------------#
def make_round_sheet(sheet_name,round_num, fo1, txt_name):
    
	wb = "Round "+round_num
	
	stat_urls, formation_urls = get_round_urls(sheet_name, wb)
	num_matches = len(stat_urls[0])
	for i in range (num_matches):
	    
		stat_soup = []
		line_up_soup = []
		
		stat_soup = mySoup.get_my_soup(stat_urls[0][i])
		line_up_soup = mySoup.get_my_soup(formation_urls[0][i])

		K_home, K_away = make_K_rows(stat_soup, line_up_soup, round_num, txt_name)

		fo1.write(K_home)
		fo1.write(K_away)
	
	_ = system('clear')

#-----------------------------------------------------------------------#
#-----------------------------------------------------------------------#
#-----------------------------------------------------------------------#
def get_round_urls(file_name, work_sheet):

	stats = "#match-statistics;1"
	lineups = "#lineups;1"
	    
	df = pd.read_excel(file_name,sheet_name=work_sheet,header=None).values

	stat_urls = (df + stats).T.tolist()
	formation_urls = (df + lineups).T.tolist()
	
	return stat_urls, formation_urls

#-----------------------------------------------------------------------#
#-----------------------------------------------------------------------#
#-----------------------------------------------------------------------#
def make_K_rows(stat_soup, line_up_soup ,round_num, league):
    
    home_team, away_team = sS24.get_team_names(stat_soup)
    print('\t=>', home_team,' vs. ',away_team)
    date, time = sS24.get_date_time(stat_soup)
    home_lineup, away_lineup  = sS24.get_line_up(line_up_soup)
    home_goals, away_goals = sS24.get_goals_scored(line_up_soup, "0")
    home_points, away_points, Htarget, Atarget, Jtarget  = dS24.get_points(home_goals, away_goals)
    
    first_home_stat_table, first_away_stat_table = sS24.get_stat_table(stat_soup, "1")
    second_home_stat_table, second_away_stat_table = sS24.get_stat_table(stat_soup, "2")

    
    home = date +' '+ time + ',' + league + ',' + home_team + ',' + home_team + ',' + away_team + ',' + home_goals + ',' + away_goals + ',' + first_home_stat_table + ',' + second_home_stat_table + ',' + home_lineup + ',' + home_points + ',' + '1' + ',' + Htarget + ',' + Jtarget + '\n'
    
    away = date +' '+ time + ',' + league + ',' + away_team + ',' + home_team + ',' + away_team + ',' + home_goals + ',' + away_goals + ',' + first_away_stat_table + ',' + second_away_stat_table + ',' + away_lineup + ',' + away_points + ',' + '-1' + ',' + Atarget + ',' + Jtarget +'\n'
    
    return home,away
#-----------------------------------------------------------------------#
#-----------------------------------------------------------------------#
#-----------------------------------------------------------------------#
def getHeader():
	return 'League,Team,Home,Away,GF,GA,First_Half_Ball_Possession,First_Half_Goal_Attempts,First_Half_Shots_on_Goal,First_Half_Shots_off_Goal,First_Half_Blocked_Shots,First_Half_Free_Kicks,First_Half_Corner_Kicks,First_Half_Offsides,First_Half_Throw_in,First_Half_Goalkeeper_Saves,First_Half_Fouls,First_Half_Red_Cards,First_Half_Yellow_Cards,First_Half_Total_Passes,First_Half_Completed_Passes,First_Half_A,First_Half_DA,First_Half_Tackles,Second_Half_Ball_Possession,Second_Half_Goal_Attempts,Second_Half_Shots_on_Goal,Second_Half_Shots_off_Goal,Second_Half_Blocked_Shots,Second_Half_Free_Kicks,Second_Half_A,Second_Half_DA,Second_Half_Corner_Kicks,Second_Half_Offsides,Second_Half_Throw_in,Second_Half_Goalkeeper_Saves,Second_Half_Fouls,Second_Half_Red_Cards,Second_Half_Yellow_Cards,Second_Half_Total_Passes,Second_Half_Completed_Passes,Second_Half_Tackles,Lineup,Points,HomeAway,HW,D,HL,Target\n'
main()


            
            
            
