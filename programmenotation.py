
import json
import os


def file_exists(fichier):
   try:
      file(fichier)
      return True
   except:
      return False



file_num = input("Numero du fichier : ")
print ("\n")

while file_exists("trainedset" + str(file_num) + ".json"):
    print ("Ce fichier exist deja, essaye encore!")
    file_num = input("Numero du fichier : ")
    print ("\n")

tweet_depart = input("Prochain tweet a classifier : ")
print ("\n")

trainset = open('trainingset.json','r')

trainedset = open("trainedset" + str(file_num) + ".json","w")

for line in trainset:
    tweet = json.loads(line)
    if tweet['idx'] >= int(tweet_depart):
	    print (tweet['tx'])
	    EMvsLP = input("Macron (1) vs LePen (0), exit (-1) : ")
	    if EMvsLP == str(-1):
                print ("prochain tweet a classifier: ", tweet['idx'])
                break
	    sent = input("Sentiment : ")
	    tweet['EMvsLP'] = EMvsLP
	    tweet['sent'] = sent
	    print ("\n")

	    json.dump(tweet, trainedset)
	    trainedset.write("\n")
