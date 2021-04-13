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

def estimate_prior():
    class_1_cnt = 0
    class_2_cnt = 0
    class_3_cnt = 0
    with open('train.csv', newline='') as csvfile:
        rows = csv.reader(csvfile)
        for row in rows:
            if row[0] == '1':
                class_1_cnt += 1
            elif row[0] == '2':
                class_2_cnt += 1
            elif row[0] == '3':
                class_3_cnt += 1
    total_cnt = class_1_cnt + class_2_cnt + class_3_cnt
    return class_1_cnt/total_cnt, class_2_cnt/total_cnt, class_3_cnt/total_cnt
    
def estimate_mean_std(mean_std):
    with open('train.csv', newline='') as csvfile:
        rows = csv.reader(csvfile, delimiter=',')
        datas = np.asarray(list(rows))
    datas = datas.astype(np.float32)
    
    class_2_index = -1
    class_3_index = -1
    i = 0
    for data in datas:
        if data[0] == 2.0 and class_2_index == -1:
            class_2_index = i
        elif data[0] == 3.0 and class_3_index == -1:
            class_3_index = i
        i += 1
    
    datas_1 = np.vsplit(datas, [class_2_index])
    datas_2 = np.vsplit(datas_1[1], [class_3_index- datas_1[0].shape[0]])
    
    datas_class_1 = datas_1[0]
    datas_class_2 = datas_2[0]
    datas_class_3 = datas_2[1]
    
    for j in range(1, datas_class_1.shape[1], 1):
        #print(datas_class_1[:,j])  # 讀取某直行        
        specific_feature_values = datas_class_1[:,j]
        mean = np.mean(specific_feature_values)
        std = np.std(specific_feature_values)
        mean_std[1].append((mean, std))

        specific_feature_values = datas_class_2[:,j]
        mean = np.mean(specific_feature_values)
        std = np.std(specific_feature_values)
        mean_std[2].append((mean, std))

        specific_feature_values = datas_class_3[:,j]
        mean = np.mean(specific_feature_values)
        std = np.std(specific_feature_values)
        mean_std[3].append((mean, std))
    return datas_class_1, datas_class_2, datas_class_3

if __name__ == "__main__":
    try:
        os.remove('train.csv')
        os.remove('test.csv')
    except:
        pass
    construct_train_test()

    # training
    prior_1, prior_2, prior_3 = estimate_prior()

    mean_std = {1:[], 2:[], 3:[]}
    a, b, c  = estimate_mean_std(mean_std)

    