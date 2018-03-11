import json

content1 = open("dataset1-2-3.json","r").readlines()

content_set1 = set(content1)

cleandata = open("dataset_consolidevf.json","w")

for line in content_set1:
    cleandata.write(line)
