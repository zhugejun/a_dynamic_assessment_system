import requests

#Specify a URL that resolves to your workspace
URL = "http://0.0.0.0:8000/"



#Call each API endpoint and store the responses
response1 = requests.get(URL + 'prediction', params={'file_name': 'testdata.csv'}).content
response2 = requests.get(URL + 'scoring').content
response3 = requests.get(URL + 'summarystats').content
response4 = requests.get(URL + 'diagnostics').content

#combine all API responses
responses = [response1, response2, response3, response4]

#write the responses to your workspace
with open('apireturns2.txt', 'wb') as f:
    for response in responses:
        f.write(response + b'\n')

