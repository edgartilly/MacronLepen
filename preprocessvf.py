#import regex
import re
import json

#start process_tweet
def processTweet(tweet):
    # process the tweets
    #Convert to lower case
    tweet = tweet.lower()
    #Convert www.* or https?://* to URL
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','URL',tweet)
    #Remove @
    tweet = re.sub(r'@([^\s]+)', r'\1',tweet)
    #Remove additional white spaces
    tweet = re.sub('[\s]+', ' ',tweet)
    #Replace #word with word
    tweet = re.sub(r'#([^\s]+)', r'\1',tweet)
    #trim
    tweet = tweet.strip('\'"')
    #Remove RT
    tweet = re.sub('(rt([^\s]+))', r'\1',tweet)
    return tweet
#end

def clean_tweet(tweet, new_text, count):

    new_tweet = {'idx':count, 'tx': new_text, 'ca': tweet["_source"]["ca"], 'tmz': tweet["_source"]["usr"]["tmz"], 'id_str' : tweet["_source"]["id_str"], 'EMvsLP': 0, 'sent': 0}
    return new_tweet

#Read the tweets one by one and process it
fp = open('dataset_consolidevf.json','r')
cleandata = open("clean_dataset_vf","w")

count = 1
for line in fp:
    tweet = json.loads(line)
    texte = tweet["_source"]["tx"]
    ntxt = processTweet(texte)

    new_tweet = clean_tweet(tweet, ntxt, count)

    json.dump(new_tweet, cleandata)
    cleandata.write("\n")
    count = count + 1
