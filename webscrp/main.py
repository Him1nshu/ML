import requests
from bs4 import BeautifulSoup


url = "https://realpython.github.io/fake-jobs/"
response = requests.get(url)

print(response.status_code)   # 200 = success
#print(response.text[:500])    

soup = BeautifulSoup(response.text, "lxml")

j3=soup.find_all("footer", class_="card-footer")
j2=soup.find_all("div", class_="content") 
j1=soup.find_all("div", class_="media-content")

for j in j1:
    ttile= j.find("h2", class_="title is-5").text
    comp = j.find("h3", class_="subtitle is-6 company").text
    print(ttile,"@",comp)

for j in j2:
    loc=j.find("p",class_="location").text
    print(loc)

for j in j3:
    a=j.find("a")
    link=a.get("href")
    print(link)

