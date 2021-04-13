import numpy as np
import csv
import random
import os

np.random.seed(1)
random.seed (1)

def construct_train_test():
    class_1 = []
    class_2 = []
    class_3 = []
    with open('Wine.csv', newline='') as csvfile:
        rows = csv.reader(csvfile)
        for row in rows:
            if row[0] == '1':
                class_1.append(row)
            elif row[0] == '2':
                class_2.append(row)
            elif row[0] == '3':
                class_3.append(row)

    samle_indexs_1 = random.sample(range(0, len(class_1)), 18)
    samle_indexs_2 = random.sample(range(0, len(class_2)), 18)
    samle_indexs_3 = random.sample(range(0, len(class_3)), 18)

    for i in range(0, len(class_1), 1):
        if i not in samle_indexs_1:
            with open('train.csv', 'a', newline='') as csvFile:
                csvWriter = csv.writer(csvFile)
                csvWriter.writerow(class_1[i])
        else:
            with open('test.csv', 'a', newline='') as csvFile:
                csvWriter = csv.writer(csvFile)
                csvWriter.writerow(class_1[i])

    for i in range(0, len(class_2), 1):
        if i not in samle_indexs_2:
            with open('train.csv', 'a', newline='') as csvFile:
                csvWriter = csv.writer(csvFile)
                csvWriter.writerow(class_2[i])
        else:
            with open('test.csv', 'a', newline='') as csvFile:
                csvWriter = csv.writer(csvFile)
                csvWriter.writerow(class_2[i])

    for i in range(0, len(class_3), 1):
        if i not in samle_indexs_3:
            with open('train.csv', 'a', newline='') as csvFile:
                csvWriter = csv.writer(csvFile)
                csvWriter.writerow(class_3[i])
        else:
            with open('test.csv', 'a', newline='') as csvFile:
                csvWriter = csv.writer(csvFile)
                csvWriter.writerow(class_3[i])

if __name__ == "__main__":
    try:
        os.remove('train.csv')
        os.remove('test.csv')
    except:
        pass
    construct_train_test()