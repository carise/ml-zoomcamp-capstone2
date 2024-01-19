import json
import requests

url = 'http://localhost:8080/2015-03-31/functions/function/invocations'

data = {'url': 'https://www.nwf.org/-/media/NEW-WEBSITE/Shared-Folder/Wildlife/Invertebrates/invertebrate_octopus_600x300.jpg'}
event = {'body': json.dumps(data)}

result = requests.post(url, json=event).json()
print(result)
