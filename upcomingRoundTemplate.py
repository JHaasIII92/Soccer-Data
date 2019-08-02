from mySoup import mySoup
from scrapper import s24Scraper as s24
from bs4 import BeautifulSoup as sp
import pandas as pd
import numpy as np
import re

leagueDict = {
"CHI1":"https://www.soccer24.com/china/super-league/",
"JP1":"https://www.soccer24.com/japan/j-league/",
"SK1":"https://www.soccer24.com/south-korea/k-league-1/",
"SWD1":"https://www.soccer24.com/sweden/allsvenskan/",
"BRZ1":"https://www.soccer24.com/brazil/serie-a/"
}

def main():

	league = input("Enter League: ")
	url = leagueDict[league]
	my_soup = mySoup.get_my_soup(url)
	roundName, dates, homeTeams, awayTeams = s24.upcoming_Round(my_soup)
	fn = league + '/' + league + "_" + roundName + ".csv"
	prepareDate(dates, homeTeams, awayTeams, fn)


def prepareDate(dates, homeTeams, awayTeams, fn):
	vectorRemoveSpace = np.vectorize(removeSpace)
	vectorFixDates = np.vectorize(fixDates)
	homeTeams = vectorRemoveSpace(homeTeams)
	awayTeams = vectorRemoveSpace(awayTeams)
	dates = vectorFixDates(dates)

	table = np.hstack((dates,homeTeams,awayTeams))
	tableDF = pd.DataFrame(table, columns = ["Dates","Home","Away"])

	tableDF.to_csv(fn)


def removeSpace(a):
	strA = str(a)
	if(' ' in strA):
	    a = strA.replace(" ", '_')
	return a

def fixDates(a):
	year = "2019"
	day, month, time = str(a).split(".")
	a = month + '/' + day + '/' + year + ' ' + time
	return a


main()
