import random
import json

#Read the tweets one by one and process it
data1 = open('trainingset_done.json','r')
data2 = open('trainingset_done.json','r')

clean_data1 = open('trainingset_testmodele.json','w')
clean_data2 = open('trainingset_final.json','w')

#create a random list with the range of the training set

X = []
for line in data1:
    text = json.loads(line)
    X.append(text['idx'])

liste = random.sample(range(0,5000),500)
new_liste = sorted (liste)

k = 0 

#extraitre les tweets et les copier dans un autre fichier

for line in data2:
    print ("I'm here")
    tweet = json.loads(line) 
    index = tweet['idx']

    if X[new_liste[k]] == index:
        json.dump(tweet, clean_data1) 
        clean_data1.write("\n") #
        if k < 499: 
            k = k + 1
    else:
        json.dump(tweet, clean_data2) 
        clean_data2.write("\n") 
