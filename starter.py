import math
import random
import numpy as np


def euclidean(a,b):
    res = 0
    for i in range(len(a)):
        res += abs(float(a[i]) - float(b[i]))**2
    return math.sqrt(res)

def getDot(a,b):
    res = 0
    for i in range(len(a)):
        res += float(a[i])*float(b[i])
    return res

def getVecLen(x):
    res = 0
    for i in range(len(x)):
        res += (float(x[i])-0)**2
    return math.sqrt(res)
    
# returns Cosine Similarity between examples a dn b
def cosim(a,b):
    abDot = getDot(a,b)
    distance_product = getVecLen(a)*getVecLen(b)
    res = abDot/(distance_product)
    return res

def downsample_image(image, n=2):
    downsampled_image = []

    for row in range(0, 28, n):
        for col in range(0, 28, n):
            avg_pixel = 0
            for i in range(n):
                for j in range(n):
                    avg_pixel += float(image[(row+i)*28 + col+j])
            avg_pixel //= (n*n)
            downsampled_image.append(avg_pixel)
        
    return downsampled_image

# Modify KNN function
def knn(train, query, metric):
    labels = []
    k = 5

    downsampled_train = [[label, downsample_image(data, 2)] for label, data in train]

    for q in query:
        # Downsample query data
        downsampled_q = downsample_image(q, 2)

        if metric == 'euclidean':
            distances = [(euclidean(downsampled_q, t[1]), t[0]) for t in downsampled_train]
            distances.sort()
        elif metric == 'cosim':
            distances = [(cosim(downsampled_q, t[1]), t[0]) for t in downsampled_train]
            distances.sort(reverse=True)

        labelCount = {}
        distanceCount = {}
        
        for i in range(k):
            distance = distances[i][0]
            label = distances[i][1]
            if label not in labelCount:
                distanceCount[label] = distance
                labelCount[label] = 1
            else:
                distanceCount[label] += (distanceCount[label]*labelCount[label])+distance
                labelCount[label] += 1
                distanceCount[label] = distanceCount[label]/labelCount[label]

        #this is new below
        #sorted_labels = dict(sorted(labelCount.items(), key=lambda item: item[1], reverse=True))
        #mostCommon, mostCommonCount = next(iter(sorted_labels.items()))
        #mostCommon = max(labelCount, key=labelCount.get)
        mostCommon = max(labelCount, key=lambda label: (labelCount[label], -distanceCount[label]))


        #do the mode thing here

        labels.append(mostCommon)

        #k=4 was their best, maybe play around with it
        #change the mostCommon selection to if there were more than one max, chose the one thats closest
        
    return labels


# returns a list of labels for the query dataset based upon observations in the train dataset. 
# labels should be ignored in the training set
# metric is a string specifying either "euclidean" or "cosim".  
# All hyper-parameters should be hard-coded in the algorithm.
def kmeans(train,query,metric):
    # k has to be 10 for the values 0-9
    k = 10
    # keep this in here, add convergence test if no centroids are updated if we have time
    max_iters = 400
    centroids, cluster_labels = kmeans_train(train, metric, k, max_iters)
    predicted_labels = []
    for query in queries:
        downsampled_query = downsample_image(query, 2)

        if metric == 'euclidean':
            distances = [euclidean(downsampled_query, centroid) for centroid in centroids]
        elif metric == 'cosim':
            distances = [-cosim(downsampled_query, centroid) for centroid in centroids]
            
        cluster_idx = distances.index(min(distances))
        predicted_labels.append(cluster_labels[cluster_idx])

    return predicted_labels


def kmeans_train(train, metric, k, max_iterations):

    # downsample to 14x14
    data = [downsample_image(x[1], 2) for x in train]

    centroids = random.sample(data, k)
    prev_centroids = [c[:] for c in centroids]  # Make a deep copy of the centroids

    for _ in range(max_iterations):
        # Assign each data point to the closest centroid
        clusters = [[] for _ in range(k)]
        for i, point in enumerate(data):
            if metric == 'euclidean':
                distances = [euclidean(point, centroid) for centroid in centroids]
            elif metric == 'cosim':
                distances = [-cosim(point, centroid) for centroid in centroids]
            cluster_idx = distances.index(min(distances))
            clusters[cluster_idx].append((train[i][0], point))

        # centroid update formula from the slides
        for i, cluster in enumerate(clusters):
            if cluster:
                centroids[i] = [sum(float(x[1][j]) for x in cluster) / len(cluster) for j in range(len(cluster[0][1]))]

        #check convergence
        changes = [euclidean(centroids[i], prev_centroids[i]) for i in range(k)]
        if max(changes) < 1e-4:
            break

        prev_centroids = [c[:] for c in centroids]
        # label cluster by the mode
        cluster_labels = []
        for cluster in clusters:
            labels = [x[0] for x in cluster]
            most_common = max(labels, key=labels.count)
            cluster_labels.append(most_common)

    return centroids, cluster_labels


def read_data(file_name):
    
    data_set = []
    with open(file_name,'rt') as f:
        for line in f:
            line = line.replace('\n','')
            tokens = line.split(',')
            label = tokens[0]
            attribs = []
            for i in range(784):
                attribs.append(tokens[i+1])
            data_set.append([label,attribs])
    return(data_set)
        
def show(file_name,mode):
    
    data_set = read_data(file_name)
    for obs in range(len(data_set)):
        for idx in range(784):
            if mode == 'pixels':
                if data_set[obs][1][idx] == '0':
                    print(' ',end='')
                else:
                    print('*',end='')
            else:
                print('%4s ' % data_set[obs][1][idx],end='')
            if (idx % 28) == 27:
                print(' ')
        print('LABEL: %s' % data_set[obs][0],end='')
        print(' ')
            
def main():
    show('valid.csv','pixels')
    
if __name__ == "__main__":
    main()
    training_data = read_data('train.csv')
    validation_or_test_set = read_data('test.csv')
    # should we move this into the functions?? ask in OH
    queries = [q[1] for q in validation_or_test_set]
    correct = [q[0] for q in validation_or_test_set]
    resKnn = knn(training_data, queries, 'euclidean')


    # knn test
    cor = 0
    total = len(queries)
    confusion_matrix = [[0]*10 for _ in range(10)]

    for i in range(len(queries)):
        confusion_matrix[int(resKnn[i])][int(correct[i])] += 1
        if resKnn[i] == correct[i]:
            cor += 1

    print("KNN Score Euclidean:")
    print(confusion_matrix)
    print(cor/total)
    resKmeans = kmeans(training_data, queries, 'euclidean')

    #kMeans test
    cor = 0
    total = len(queries)
    confusion_matrix = [[0]*10 for _ in range(10)]

    for i in range(len(queries)):
        predicted_label = int(resKmeans[i])
        true_label = int(correct[i])
        confusion_matrix[predicted_label][true_label] += 1
        if predicted_label == true_label:
            cor += 1

    print("KMeans Score Euclidean:")
    print(confusion_matrix)
    print(cor/total)
    
    resKnn = knn(training_data, queries, 'cosim')


    # knn test
    cor = 0
    total = len(queries)
    confusion_matrix = [[0]*10 for _ in range(10)]

    for i in range(len(queries)):
        confusion_matrix[int(resKnn[i])][int(correct[i])] += 1
        if resKnn[i] == correct[i]:
            cor += 1

    print("KNN Score Cosine Similarity:")
    print(confusion_matrix)
    print(cor/total)
    resKmeans = kmeans(training_data, queries, 'cosim')

    #kMeans test
    cor = 0
    total = len(queries)
    confusion_matrix = [[0]*10 for _ in range(10)]

    for i in range(len(queries)):
        predicted_label = int(resKmeans[i])
        true_label = int(correct[i])
        confusion_matrix[predicted_label][true_label] += 1
        if predicted_label == true_label:
            cor += 1

    print("KMeans Score Cosine Similarity:")
    print(confusion_matrix)
    print(cor/total)
