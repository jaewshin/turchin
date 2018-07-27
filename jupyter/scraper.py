"""
Scrape all the NGAs and their respective time periods from seshat dataset 
"""

import requests
from bs4 import BeautifulSoup
import csv 

URL = 'http://seshatdatabank.info/data/'

def ngas(URL):
	"""
	Given a url for seshat dataset, retrieve all the links 
	to each NGA as well as a list of all the NGAs 
	"""
	response = requests.get(URL)

	soup = BeautifulSoup(response.content, "html.parser")

	ngas = [nga.text for nga in soup.find_all('span', class_ = 'nga')] #list of all ngas
	ngas_htmls = [html.get('href') for html in soup.find_all('a', class_='list-group-item')] # html for each ngas 

	return ngas, ngas_htmls 

def nga_inspect(ngas, ngas_html):
	"""
	Inspect for each NGAs and return a csv file that contains NGA name, polity name, the starting time, 
	and the ending time. 
	"""
	ngas_period = list()

	for nga in range(len(ngas_html)):
		new_url = URL+ngas_html[nga]
		response = requests.get(new_url)
		soup = BeautifulSoup(response.content, "html.parser")

		polities = list()
		periods = list()

		# scrap all the polities for the iterated nga
		for html in soup.find_all('span', class_='nga-name'):
			try: 
				pol = html.find('a')['href'].split('-')[-1]
				polities.append(pol)
			except Exception as e:
				pol = html.text
				polities.append(pol)

		# scrap all the time periods for the iterated nga
		for html in soup.find_all('span', class_='variable-name'):

			times = str(html.text)
			times.replace(" ", "")
			
			try:
				start, end = times.split('-')

				try: 
					assert not int(start) 

				except AssertionError: 
					time = ''
					for i in end:
						if not i.isdigit():
							time += i
					start += time 

				except ValueError:
					pass

				periods.append([start, end])

			except Exception as e:
				pass

		assert len(polities) == len(periods) # check if we have matching number of polities and periods

		for i in range(len(polities)):
			ngas_period.append([ngas[nga], polities[i], periods[i][0], periods[i][1]])	

	return ngas_period 

def write_csv(ngas_period):
	"""
	Given a list of [NGA, polity, starting time, ending time], write it to a 
	csv file
	"""
	with open('scraped_seshat.csv', 'w', newline = '') as f:
		writer = csv.writer(f)
		writer.writerow(['NGA', 'Polity', 'Start Period', 'End Period'])
		writer.writerows(ngas_period)

if __name__ == "__main__":
	ngas, ngas_html = ngas(URL)
	ngas_period = nga_inspect(ngas, ngas_html)
	write_csv(ngas_period)

