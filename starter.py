import math

def euclidean(a,b):
    
    res = 0
    for i in range(len(a)):
        res += abs(int(a[i]) - int(b[i]))**2

    return math.sqrt(res)
    

def getDot(a,b):
    res = 0
    for i in range(len(a)):
        res += int(a[i])*int(b[i])
    return res
    
def getVecLen(x):
    res = 0
    for i in range(len(x)):
        res += (int(x[i])-0)**2
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
                    avg_pixel += int(image[(row+i)*28 + col+j])
            avg_pixel //= (n*n)
            downsampled_image.append(avg_pixel)
            
    return downsampled_image

# Modify KNN function
def knn(train, query, metric):
    labels = []
    k = 2  # hyperparameter

    downsampled_train = [[label, downsample_image(data, 2)] for label, data in train]
    
    for q in query:
        # Downsample query data
        downsampled_q = downsample_image(q, 2)

        if metric == 'euclidean':
            distances = [(euclidean(downsampled_q, t[1]), t[0]) for t in downsampled_train]
        elif metric == 'cosim':
            distances = [(cosim(downsampled_q, t[1]), t[0]) for t in downsampled_train]

        distances.sort()
        labelCount = {}
        
        for i in range(k):
            label = distances[i][1]
            if label not in labelCount:
                labelCount[label] = 1
            else:
                labelCount[label] += 1
        
        mostCommon = max(labelCount, key=labelCount.get)
        labels.append(mostCommon)
        
    return labels


# returns a list of labels for the query dataset based upon observations in the train dataset. 
# labels should be ignored in the training set
# metric is a string specifying either "euclidean" or "cosim".  
# All hyper-parameters should be hard-coded in the algorithm.
def kmeans(train,query,metric):
    return(labels)

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
    training_data = read_data('test.csv')
    example1 = training_data[16][1]
    example2 = training_data[8][1]

    validation_set = read_data('valid.csv')
    queries = [q[1] for q in validation_set]
    correct = [q[0] for q in validation_set]
    res = knn(training_data, queries, 'euclidean')

    cor = 0
    total = len(queries)

    for i in range(len(queries)):
        if res[i] == correct[i]:
            cor += 1

    print(cor/total)

