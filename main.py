import numpy as np
import matplotlib.pyplot as plt
import csv
import random
import os
import math
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

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

def estimate_joint_prob(input_feature, prior_1, prior_2, prior_3, mean_std):
    likelihood1 = 1
    likelihood2 = 1
    likelihood3 = 1
    index = 0
    for feature in input_feature:
        mean = mean_std[1][index][0]
        std = mean_std[1][index][1]
        gaussian_prob_1 = ( 1 / (std* (2*math.pi)**(1/2))) * math.exp((-1/2) * ( (feature - mean) / std)**2)
        likelihood1 = likelihood1 * gaussian_prob_1

        mean = mean_std[2][index][0]
        std = mean_std[2][index][1]
        gaussian_prob_2 = ( 1 / (std* (2*math.pi)**(1/2))) * math.exp((-1/2) * ( (feature - mean) / std)**2)
        likelihood2 = likelihood2 * gaussian_prob_2

        mean = mean_std[3][index][0]
        std = mean_std[3][index][1]
        gaussian_prob_3 = ( 1 / (std* (2*math.pi)**(1/2))) * math.exp((-1/2) * ( (feature - mean) / std)**2)
        likelihood3 = likelihood3 * gaussian_prob_3

        index += 1
    return prior_1*likelihood1, prior_2*likelihood2, prior_3*likelihood3

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
    datas_class_1, datas_class_2, datas_class_3 = estimate_mean_std(mean_std)
    
    # testing
    correct_cnt = 0
    total_test_cnt = 0
    with open('test.csv', newline='') as csvfile:
        rows = csv.reader(csvfile)
        for row in rows:
            input_feature = np.array(row[1:]).astype(np.float32)
            joint_prob1, joint_prob2, joint_prob3 = estimate_joint_prob(input_feature, prior_1, prior_2, prior_3, mean_std)
            marginal_prob = joint_prob1 + joint_prob2 + joint_prob3
            posterior_probabilities = np.array([joint_prob1/marginal_prob, joint_prob2/marginal_prob, joint_prob3/marginal_prob])
            predict_class = np.argmax(posterior_probabilities) + 1
            if predict_class == int(row[0]):
                correct_cnt += 1
            total_test_cnt += 1
    print('Accuracy rate of MAP detector:', correct_cnt*100/total_test_cnt, '%')

    # PCA
    datas_class_1 = datas_class_1[:,1:]
    datas_class_2 = datas_class_2[:,1:]
    datas_class_3 = datas_class_3[:,1:]

    # 2-D
    fig = plt.figure()
    
    pca = PCA(n_components = 2, copy=True)

    pca.fit(datas_class_1)
    datas_class_1_pca = pca.transform(datas_class_1)
    print('Type1 variance ratio of two principal components:', pca.explained_variance_ratio_)
    print('Type1 variance of two principal components:', pca.explained_variance_)
    plt.scatter(datas_class_1_pca[:, 0], datas_class_1_pca[:, 1], label = 'type1', c = 'red')

    pca.fit(datas_class_2)
    datas_class_2_pca = pca.transform(datas_class_2)
    print('Type2 variance ratio of two principal components:', pca.explained_variance_ratio_)
    print('Type2 variance of two principal components:', pca.explained_variance_)
    plt.scatter(datas_class_2_pca[:, 0], datas_class_2_pca[:, 1], label = 'type2', c = 'green')

    pca.fit(datas_class_3)
    datas_class_3_pca = pca.transform(datas_class_3)
    print('Type3 variance ratio of two principal components:', pca.explained_variance_ratio_)
    print('Type3 variance of two principal components:', pca.explained_variance_)
    plt.scatter(datas_class_3_pca[:, 0], datas_class_3_pca[:, 1], label = 'type3', c = 'blue')

    fig.legend(loc = 'center right')
    plt.axis('equal')
    plt.show()
    print('=========================================================================')
    # 3-D
    fig = plt.figure()
    ax = Axes3D(fig)
    
    pca = PCA(n_components = 3, copy=True)

    pca.fit(datas_class_1)
    datas_class_1_pca = pca.transform(datas_class_1)
    print('Type1 variance ratio of three principal components:', pca.explained_variance_ratio_)
    print('Type1 variance of three principal components:', pca.explained_variance_)
    ax.scatter(datas_class_1_pca[:, 0], datas_class_1_pca[:, 1], datas_class_1_pca[:, 2], label = 'type1', c = 'red')

    pca.fit(datas_class_2)
    datas_class_2_pca = pca.transform(datas_class_2)
    print('Type2 variance ratio of three principal components:', pca.explained_variance_ratio_)
    print('Type2 variance of three principal components:', pca.explained_variance_)
    ax.scatter(datas_class_2_pca[:, 0], datas_class_2_pca[:, 1], datas_class_2_pca[:, 2], label = 'type2', c = 'green')

    pca.fit(datas_class_3)
    datas_class_3_pca = pca.transform(datas_class_3)
    print('Type3 variance ratio of three principal components:', pca.explained_variance_ratio_)
    print('Type3 variance of three principal components:', pca.explained_variance_)
    ax.scatter(datas_class_3_pca[:, 0], datas_class_3_pca[:, 1], datas_class_3_pca[:, 2], label = 'type3', c = 'blue')

    fig.legend(loc = 'center right')
    plt.show()
    