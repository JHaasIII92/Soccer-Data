from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from bs4 import BeautifulSoup as soup
import xlsxwriter as xl
from openpyxl import load_workbook
import openpyxl
import time
import re
from os import system, name


links = {
'JP1':'https://www.flashscore.com/football/japan/j-league/results/',
'BRZ1':'https://www.flashscore.com/football/brazil/serie-a/results/',
'CHI1':'https://www.flashscore.com/football/china/super-league/results/',
'SK1':'https://www.flashscore.com/football/south-korea/k-league-1/results/',
'SWD1':'https://www.flashscore.com/football/sweden/allsvenskan/results/'
}

def main():
	league = ''
	url = links[league]
	file_name = league + "/" + league +'_ID.xlsx'
	my_soup = get_soup(url)
	round_nums, ids = get_id(my_soup)
	write_to_xlsx(file_name, round_nums, ids)

#-----------------------------------------------------------------------#
#-----------------------------------------------------------------------#
#-----------------------------------------------------------------------#
def get_soup(url):
	options = Options()
	options.add_argument("--headless")
	driver = webdriver.Firefox(firefox_options=options,executable_path = '/usr/local/bin/geckodriver')
	driver.get(url)
	html = driver.page_source
	driver.quit()
	my_soup = soup(html,"html.parser")
	return my_soup

#-----------------------------------------------------------------------#
#-----------------------------------------------------------------------#
#-----------------------------------------------------------------------#
def get_id(my_soup):
    table = my_soup.find('div',{'class':'soccer'})
    round_nums = []
    ids = []
    for div in table (re.compile('^div')):
        strDiv = str(div)
        if("id=" in strDiv):
            begin = strDiv.rfind('id="') + 8
            end = strDiv.rfind('title="') - 2
            matchID = strDiv[begin:end]
            ids.append((Round, matchID))
        if("Round" in strDiv):
            begin = strDiv.rfind('c">') + 3
            end = strDiv.rfind('<')
            Round = strDiv[begin:end]
            round_nums.append(Round)
    return round_nums, ids

#-----------------------------------------------------------------------#
#-----------------------------------------------------------------------#
#-----------------------------------------------------------------------#
def write_to_xlsx(file_name, round_nums, ids):
    id_book = xl.Workbook(file_name)
    new_round = []
    for i in range(len(round_nums)):
        try:
            id_book.add_worksheet(round_nums[i])
            new_round.append(round_nums[i])
        except:
            pass
    id_book.close()
    for i in range(len(new_round)):
        wb = load_workbook(file_name)
        ws = wb[(new_round[i])]
        row = 1
        col = 1
        for r in range(len(ids)):
            if(ids[r][0] == new_round[i]):
                ws.cell(row,col).value = ('https://www.soccer24.com/match/'+ids[r][1]+'/')
                row += 1
        else:
            pass
        wb.save(file_name)


main()
