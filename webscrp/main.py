import requests
from bs4 import BeautifulSoup
import csv


url = "https://realpython.github.io/fake-jobs/"
response = requests.get(url)

print(response.status_code)   # 200 = success
#print(response.text[:500])    

soup = BeautifulSoup(response.text, "lxml")

j3=soup.find_all("footer", class_="card-footer")
j2=soup.find_all("div", class_="content") 
j1=soup.find_all("div", class_="media-content")


titles = []
companies = []
locations = []
links = []

for j in j1:
    title_tag = j.find("h2", class_="title is-5")
    comp_tag = j.find("h3", class_="subtitle is-6 company")
    title = title_tag.get_text(strip=True) if title_tag else "N/A"
    comp = comp_tag.get_text(strip=True) if comp_tag else "N/A"
    #print(title, "@", comp)
    titles.append(title)
    companies.append(comp)

for j in j2:
    loc_tag = j.find("p", class_="location")
    loc = loc_tag.get_text(strip=True) if loc_tag else "Location not found"
    #print(loc)
    locations.append(loc)

for j in j3:
    a = j.find("a")
    link = a.get("href") if a else "No link available"
    #print(link)
    links.append(link)

with open("jobs.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["title", "company", "location"," details"])
    for t, c,l,d in zip(titles,companies,locations,links):
        writer.writerow([t,c,l,d])
