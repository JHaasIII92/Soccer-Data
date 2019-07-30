from bs4 import BeautifulSoup as soup
import numpy as np
import re
import hashlib

# Soccer 24 scrapper
# Soup not included

class s24Scraper:
#-----------------------------------------------------------------------#
#-----------------------------------------------------------------------#
#-----------------------------------------------------------------------#
	def get_stat_table(my_soup,half):
		"""Returns both the home and away teams stat table of a specific portion of the game my_soup: must be s24 url ending in #match-statistics;1 half: can be 0, 1, 2 0: full 1: first half 2: second half Header: 'Ball_Possession','Goal_Attempts','Shots_on_Goal','Shots_off_Goal','Blocked_Shots','Free_Kicks','Corner_Kicks','Offsides','Throwin','Goalkeeper_Saves','Fouls','Red_Cards','Yellow_Cards','Total_Passes','Completed_Passes','Tackles'"""


		bpDict ={
'Ball Possession':0,
'Goal Attempts':1,
'Shots on Goal':2,
'Shots off Goal':3,
'Blocked Shots':4,
'Free Kicks':5,
'Corner Kicks':6,
'Offsides':7,
'Throw-in':8,
'Goalkeeper Saves':9,
'Fouls':10,
'Red Cards':11,
'Yellow Cards':12,
'Total Passes':13,
'Completed Passes':14,
'Tackles':15,
'Attacks':16,
'Dangerous Attacks':17
}
	
		home = ['0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0']
		away = ['0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0']
		stat_names = []
		home_stats = []
		away_stats = []
	
		shearch = "tab-statistics-" + half + "-statistic"
		div_tag = my_soup.find("div",{"id":shearch})
		home_html_list = div_tag.find_all("div",{"style":"float: left;"})
		away_html_list = div_tag.find_all("div",{"style":"float: right;"})
		score_stat_list = my_soup.find("div",{"id":shearch}).find_all("td",{"class":"score stats"})

		stat_removes = ['<td class="score stats" style="border-top: 0px;">','</td>']
		home_removes = ['<div style="float: left;">', '</div>']
		away_removes = ['<div style="float: right;">', '</div>']
				       
				       
		for i in range(len(home_html_list)):
			stat = re.sub(home_removes[0],'',str(home_html_list[i]))
			stat = re.sub(home_removes[1],'',stat)
			stat = re.sub("%",'',stat)
			stat = stat.strip()
			home_stats.append(stat)
				       
		for i in range(len(home_html_list)):
			stat = re.sub(away_removes[0],'',str(away_html_list[i]))
			stat = re.sub(away_removes[1],'',stat)
			stat = re.sub("%",'',stat)
			stat = stat.strip()
			away_stats.append(stat)
					       
		for i in range(len(score_stat_list)):
			stat = re.sub(stat_removes[0],'',str(score_stat_list[i]))
			stat = re.sub(stat_removes[1],'',stat)
			stat = stat.strip()
			stat_names.append(stat)
				
		for i in range(len(stat_names)):
			statLoc = bpDict[stat_names[i]]
			home[statLoc] = home_stats [i]
			away[statLoc] = away_stats[i]
							       
		home = (str(home))
		away = (str(away))

		home = (((home.replace('[','')).replace(']','')).replace("'",'')).replace(" ",'')
		away = (((away.replace('[','')).replace(']','')).replace("'",'')).replace(" ",'')

		return home, away


#-----------------------------------------------------------------------#
#-----------------------------------------------------------------------#
#-----------------------------------------------------------------------#
	def get_team_names(my_soup):
		"""Returns the name of Both home and away my_soup: use s24 url ending in #match-statistics;1"""
	    
		home_team = my_soup.find("td",{"class":'tname-home logo-enable'}).get_text()
		away_team = my_soup.find("td",{"class":'tname-away logo-enable'}).get_text()
		str_home_team = (str(home_team).replace(' ','_')).strip()
		str_away_team = (str(away_team).replace(' ','_')).strip()
		return str_home_team ,str_away_team

#-----------------------------------------------------------------------#
#-----------------------------------------------------------------------#
#-----------------------------------------------------------------------#
	def get_date_time(my_soup):
		"""Returns Date and time two separate strings my_soup: use s24 url ending in #match-statistics;1"""
	    
		line = my_soup.find('td',{'class':'mstat-date'})
		strLine = str(line)
    
		s = strLine.rfind('"') + 2
		e = strLine.rfind('<')
		date_time = (strLine[s:e]).split('.')
		year_time = (date_time[2]).split(' ')
    
		time = year_time[1]
		date = date_time[1] +"/"+ date_time[0] +"/"+ year_time[0]
    
		return date, time

#-----------------------------------------------------------------------#
#-----------------------------------------------------------------------#
#-----------------------------------------------------------------------#
	def get_goals_scored(my_soup, half):
    
		if (half == "0"):
			match_score = my_soup.find("td",{"class":"current-result"}).get_text().split('-')
			home_goals = match_score[0]
			away_goals = match_score[1]

		elif (half == "1"):
			homeFirstHalf = my_soup.find("table",{"class":"parts-first vertical"}).find("span",{"class":"p1_home"})
			awayFirstHalf = my_soup.find("table",{"class":"parts-first vertical"}).find("span",{"class":"p1_away"})
			home_goals = homeFirstHalf.text
			away_goals = awayFirstHalf.text

		else:
			homeSecondHalf = my_soup.find("table",{"class":"parts-first vertical"}).find("span",{"class":"p2_home"})
			awaySecondHalf = my_soup.find("table",{"class":"parts-first vertical"}).find("span",{"class":"p2_away"})
			home_goals = homeSecondHalf.text
			away_goals = awaySecondHalf.text

		return home_goals, away_goals
#-----------------------------------------------------------------------#
#-----------------------------------------------------------------------#
#-----------------------------------------------------------------------#
	def get_line_up(my_soup):
		"""Returns Home and Away teams linup if linup not avalabile returns defualt 433 my_soup"""
    	
		try:
			formations = my_soup.find('table',{'id':'parts'}).find_all('b')
			home = ''
			away = ''
		
			raw_home = formations[0].text
			raw_away = formations[1].text
	
			for i in range(len(raw_home)):
				if (raw_home[i].isdigit()):
					home += raw_home[i]
	
			for i in range(len(raw_away)):
				if (raw_away[i].isdigit()):
					away+= raw_away[i]
			return home, away

		except:
			return '433', '433'
#-----------------------------------------------------------------------#
#-----------------------------------------------------------------------#
#-----------------------------------------------------------------------#
	def upcoming_Round(my_soup):
		"""Use this Method to scrape any leagues main page for the upcomming round lineup
make sure to do this befor any games have started or you will not get any of those matches
Returns numpy arrays of Dates, Home, Away as well as the string Round number"""
		tableTop = my_soup.find('div',{'id':'fs-summary-fixtures'}).find('tbody')
		temp = []
		topOfRound = True
		endloop = False
		for tr in tableTop.findAll(re.compile("^tr")):
			if(endloop == True):
				break
			for td in tr.findAll(re.compile("^td")):


				strTd = td.text
				if(("Round" in strTd)and(topOfRound == False)):
					endloop = True
					break
				elif("Round" in strTd and topOfRound == True):
					topOfRound = False
					roundName = strTd
					roundName = roundName.replace(' ','')
				else:
					if((strTd != '')and(u'\xa0' not in strTd)):
						temp.append(strTd)
				
				dateSlice = slice(0, len(temp), 3)
				homeSlice = slice(1, len(temp), 3)
				awaySlice = slice(2, len(temp), 3)
			
				dates = np.array(temp[dateSlice])
				homeTeams = np.array(temp[homeSlice])
				awayTeams = np.array(temp[awaySlice])

				dates = dates.reshape(dates.shape[0], 1)
				homeTeams = homeTeams.reshape(homeTeams.shape[0], 1)
				awayTeams = awayTeams.reshape(awayTeams.shape[0], 1)

		return roundName, dates, homeTeams, awayTeams
#-----------------------------------------------------------------------#
#-----------------------------------------------------------------------#
#-----------------------------------------------------------------------#
	def get_url_IDS(my_soup):
		table = my_soup.find('table',{'class':'soccer'})

		round_nums = []
		ids = []
    
		for tag in table (re.compile('^tr')):
			tr = str(tag)
	
			if(re.search('event_round',tr)):
				start_pos = tr.rfind('"6">')+4
				end_pos = tr.rfind('</td>')
				round_num = tr[start_pos:end_pos]
				round_nums.append(round_num)

			elif(re.search('id="g_1_',tr)):
				start_pos = tr.rfind('id="g_1_')+8
				end_pos = tr.rfind('"><td')
				id_num = tr[start_pos:end_pos]
				ids.append(id_num)

			else:pass

		round_num = np.array(round_num, ndmin=2).T
		ids = np.array(ids, ndmin=2).T
		combine = np.hstak((round_num,ids))

			
			    
		return combine

#-----------------------------------------------------------------------#
#-----------------------------------------------------------------------#
#-----------------------------------------------------------------------#
class s24Derived:
#-----------------------------------------------------------------------#
#-----------------------------------------------------------------------#
#-----------------------------------------------------------------------#
	def get_points(home_score, away_score):
    
		if (home_score == away_score):
			home_points = '1'
			away_points = '1'
			Htarget = '0,1,0'
			Atarget = '0,1,0'
			Jtarget = '0'
		elif (home_score > away_score):
			home_points = '3'
			away_points = '0'
			Htarget = '1,0,0'
			Atarget = '0,0,1'
			Jtarget = '1'
		else:
			home_points = '0'
			away_points = '3'
			Htarget = '0,0,1'
			Atarget ='1,0,0'
			Jtarget = '-1'

		return home_points, away_points, Htarget, Atarget, Jtarget
