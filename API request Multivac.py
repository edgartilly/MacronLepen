import requests
import json

def write_tweets(tweets, filename):
    ''' Function that appends tweets to a file. '''
    with open(filename, 'a') as f:
        for result in tweets:
            json.dump(result, f)
            f.write('\n')

api_key = ''
total = requests.get('https://api.iscpif.fr/v2/pvt/politic/france/twitter/search?q=presidentielle2017&output=id_str,ca,tx,usr.tmz&since=2017-05-03T23:59&until=2017-05-03T23:59&count=1&from=1&api_key=' + api_key).json()['results']['total']
from_arg = 1
print('number of tweets', total)
while from_arg < total / 100:
  print('Doing tweet {}'.format(from_arg))
  results = requests.get('https://api.iscpif.fr/v2/pvt/politic/france/twitter/search?q=presidentielle2017&output=id_str,ca,tx,usr.tmz&since=2017-05-03T23:00&until=2017-05-03T23:59&count=100&from=' + str(int(from_arg)) + '&api_key=' + api_key).json()['results']['hits']
  write_tweets(results, "tweetMacronLepen.json")
  from_arg += 1
