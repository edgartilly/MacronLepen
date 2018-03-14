import json


data1 = open('tweetMacronlepen1_consolide.json','r')
data2 = open('tweetMacronlepen2_consolide.json','r')
nodupe2 = open("nodupe2.json","w")

count = 1.0
total_tweet = 391356.0

for line2 in data2:
    tweet2 = json.loads(line2)
    test = 0
    print ("Pourcentage complete: ", round((count/total_tweet)*100,2), "%")
    count = count + 1
    for line1 in data1:
        tweet1 = json.loads(line2)
        if tweet1['id_str'] == tweet2['id_str']:
            test = 1
            break
    if test == 0:
        json.dump(tweet2, nodupe2)
        nodupe2.write("\n")
