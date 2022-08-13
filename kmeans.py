import random

class KMeansClusterClassifier():

    def __init__(self,n_cluster):
        self.n_cluster = n_cluster
        self.iteration_times = 100
        self.inertia = 0
        self.centroids = []

    def fit(self,X):
        self.centroids = self.centroidCreate(X)                 #random centroidler olusturuldu
        clusters = self.clusterUpdate(X)                        #clusterlar, centroidlere gore update edildi
        for _ in range(self.iteration_times):                   #iteration time kadar varyasyon hesaplandı
            self.centroids = self.centroidsUpdate(X,clusters)   #random centroidler olusturuldu
            clusters = self.clusterUpdate(X)                    #clusterlar, centroidlere gore update edildi
        return clusters

    def predict(self,X):
        return self.clusterUpdate(X)

    def centroidCreate(self,X): 
        weights = [[0]*2]*len(X[0])
        for i in range(len(X[0])):
            weights[i] = [min(list(zip(*X))[i]),max(list(zip(*X))[i])] 
        centroids = []
        for i in range(self.n_cluster): 
            centroid = []
            random.seed(i)
            for j in range(len(X[0])):
                centroid.append(weights[j][0]  + random.random()*(weights[j][1]-weights[j][0])) 
            centroids.append(centroid)
        return centroids

    def clusterUpdate(self,X):
        self.inertia = 0
        features = []
        for i in range(len(X[0])):
            features.append(list(zip(*X))[i])
        clusters = []
        for i in range(len(X)):
            minimum = float('inf')
            cluster = -100  #0, 1 ve 2 cluster'ları var (cluster sayısı = 3 oldugu icin). O yuzden bu kümede olmayan rastgele bi sayı verdim
            for j in range(self.n_cluster):
                distance = 0
                for k in range(len(X[0])):
                    distance += (features[k][i] - self.centroids[j][k])**2
                distance = distance**(0.5)
                if (distance < minimum): 
                    minimum = distance
                    cluster = j
            self.inertia += minimum
            clusters.append(cluster)
        return clusters

    def centroidsUpdate(self,X,clusters) :
        centroid_k = []
        for _ in range(self.n_cluster):
            centroid_k.append([0]*len(X[0]))
        count = [0]*self.n_cluster
        for idx,element in enumerate(X):
            for j in range(self.n_cluster):
                if clusters[idx] == j:
                    for k in range(len(X[0])):
                        centroid_k[j][k] += element[k]
                    count[j]+=1
        for i in range(self.n_cluster):
            if count[i] != 0:
                centroid_k[i] = [x / count[i] for x in centroid_k[i]]
        return centroid_k