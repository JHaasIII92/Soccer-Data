from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from bs4 import BeautifulSoup as soup
import re


class mySoup:

	def get_my_soup(url):
        
		options = Options()
		options.add_argument("--headless")
		driver = webdriver.Firefox(firefox_options=options,
				       executable_path = '/usr/local/bin/geckodriver')
		driver.get(url)
		html = driver.page_source
		driver.close()
		my_soup = soup(html,"html.parser")

		return my_soup



