import random
import json

#Read the tweets one by one and process it
data1 = open('trainingset_done.json','r')
data2 = open('trainingset_done.json','r')

clean_data1 = open('trainingset_testmodele.json','w')
clean_data2 = open('trainingset_final.json','w')

X = []
for line in data1:
    text = json.loads(line)
    X.append(text['idx'])

liste = random.sample(range(0,5000),500)
new_liste = sorted (liste)

k = 0  # indice initial dans ta liste

for line in data2:
    print ("I'm here")
    tweet = json.loads(line) # tu charge la ligne dans la variable tweet
    index = tweet['idx'] # tu charge l'index du tweet dans la variable index

    if X[new_liste[k]] == index: # tu verifies si l'element d'indice k de ta liste est egal a index; si oui:
        json.dump(tweet, clean_data1) # tu copie la ligne dans ton training set
        clean_data1.write("\n") # tu passe a la ligne dans ton training set
        if k < 499: # tu evite de regarder apres l'element d'indice 4999, sinon ton programme bloque
            k = k + 1 # tu passe a l'indice suivant dans ta liste
    else: #si non:
        json.dump(tweet, clean_data2) # tu copie la ligne dans ton test set
        clean_data2.write("\n") # tu passe a la ligne dans ton test set
