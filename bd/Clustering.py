import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from ucimlrepo import fetch_ucirepo
import numpy as np
from sklearn.datasets import load_iris, fetch_olivetti_faces, load_wine
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn import metrics
from sklearn.impute import SimpleImputer
from sklearn.datasets import load_diabetes
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import KMeans, MeanShift, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

# fetch dataset
default_of_credit_card_clients = fetch_ucirepo(id=350)

# data (as pandas dataframes)
X = default_of_credit_card_clients.data.features
y = default_of_credit_card_clients.data.targets


def zad1():
    print(X.describe())
    #po describe widac ze nie ma brakujacych wartosci

    # Usunięcie brakujących wartości
    #data.dropna(inplace=True)

    # wszystko jest inegerem wiec kategorie juz sa zakodowane

def zad2():
    # Wybór cech do grupowania
    X_new = X[['X5', 'X12']]
    # Wybór liczby klastrów
    k = 5
    # Inicjalizacja i dopasowanie modelu K-means
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_new)
    # Przewidywanie przynależności do klastrów dla każdej próbki
    labels = kmeans.labels_
    # Wyświetlenie wyników
    plt.figure(figsize=(10, 6))
    plt.scatter(X_new['X5'], X_new['X12'], c=labels, cmap='viridis', alpha=0.5)
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='x', s=200, c='red')
    plt.xlabel('Wiek')
    plt.ylabel('Dług na stan April')
    plt.title('Grupowanie za pomocą metody K-means')
    plt.show()

#zad2()


def zad3():
    # Implementacja metody Mean Shift
    ms = MeanShift()
    dat = X[:5000]
    X_new = dat[['X5', 'X12']]
    ms.fit(X_new)
    cluster_centers = ms.cluster_centers_
    # Zwizualizowanie wyników
    plt.figure(figsize=(10, 7))
    plt.scatter(X_new['X5'], X_new['X12'], c=ms.labels_, cmap='viridis')
    plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], marker='o', s=300,
                edgecolor='k', facecolor='none')
    plt.title('Metoda przesunięcia średniej (Mean Shift)')
    plt.xlabel('Cecha 1')
    plt.ylabel('Cecha 2')
    plt.show()
#zad3()

def zad4():

    data = load_iris()
    X = data.data
    y = data.target

    # Implementacja klastrowania aglomeracyjnego
    ac = AgglomerativeClustering(n_clusters=4)
    ac.fit(X)
    # Zwizualizowanie wyników
    plt.figure(figsize=(10, 7))
    plt.scatter(X[:, 1], X[:, 2], c=ac.labels_, cmap='viridis')
    plt.title('Klastrowanie aglomeracyjne (Agglomerative Clustering)')
    plt.xlabel('Cecha 1')
    plt.ylabel('Cecha 2')
    plt.show()
#zad4()

def zad5():
    # Wczytanie danych dotyczących wina
    wine = load_wine()
    X = wine.data
    y = wine.target

    # Zdefiniowanie modelu GMM
    gmm = GaussianMixture(n_components=2, random_state=42)

    # Dopasowanie modelu do danych
    gmm.fit(X)

    # Przewidywanie przynależności do klastrów
    labels = gmm.predict(X)

    # Wizualizacja wyników za pomocą analizy składowych głównych (PCA)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    plt.figure(figsize=(10, 6))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis')
    plt.title('Grupowanie danych Wine za pomocą GMM')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.colorbar(label='Cluster')
    plt.show()

    # Ocena skuteczności modelu
    silhouette_score = metrics.silhouette_score(X, labels, metric='euclidean')
    print("Silhouette Score:", silhouette_score)

zad5()

def zad6():
    faces_data = fetch_olivetti_faces()

    # Dane obrazów twarzy
    X_faces = faces_data.data

    # Metoda DBSCAN
    dbscan = DBSCAN(eps=6, min_samples=2)
    clusters = dbscan.fit_predict(X_faces)

    # Redukcja wymiarów do wizualizacji
    pca = PCA(n_components=2).fit(X_faces)
    X_pca = pca.transform(X_faces)

    # Zwizualizowanie wyników
    plt.figure(figsize=(10, 7))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis')
    plt.title('DBSCAN na danych Olivetti Faces')
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.colorbar(label='Klastry')
    plt.show()

#zad6()

def zad7():
    # Wczytanie danych dotyczących cukrzycy
    diabetes = load_diabetes()
    X = diabetes.data
    feature_names = diabetes.feature_names

    df = pd.DataFrame(X, columns=feature_names)

    Z = linkage(df, method='ward')

    # Wyświetlenie dendrogramu
    plt.figure(figsize=(12, 6))
    dendrogram(Z, labels=np.arange(len(df)))
    plt.title('Dendrogram Hierarchicznego Klastrowania Diabetes')
    plt.xlabel('Indeks próbki')
    plt.ylabel('Odległość Euclidean')
    plt.show()
#zad7()

def zad8():
    from ucimlrepo import fetch_ucirepo

    # fetch dataset
    heart_disease = fetch_ucirepo(id=45)

    # data (as pandas dataframes)
    X = heart_disease.data.features
    y = heart_disease.data.targets

    # Zastąpienie brakujących wartości medianą
    imputer = SimpleImputer(strategy='median')
    X = imputer.fit_transform(X)

    # Metoda K-means
    kmeans = KMeans(n_clusters=2, random_state=42)
    kmeans_labels = kmeans.fit_predict(X)

    # Metoda Agglomerative Clustering
    agglomerative = AgglomerativeClustering(n_clusters=2)
    agglomerative_labels = agglomerative.fit_predict(X)

    # Metoda DBSCAN
    dbscan = DBSCAN(eps=6, min_samples=1)
    dbscan_labels = dbscan.fit_predict(X)

    # Ocena jakości grupowania
    kmeans_silhouette = silhouette_score(X, kmeans_labels)
    agglomerative_silhouette = silhouette_score(X, agglomerative_labels)
    dbscan_silhouette = silhouette_score(X, dbscan_labels)

    kmeans_calinski_harabasz = calinski_harabasz_score(X, kmeans_labels)
    agglomerative_calinski_harabasz = calinski_harabasz_score(X, agglomerative_labels)
    dbscan_calinski_harabasz = calinski_harabasz_score(X, dbscan_labels)

    kmeans_davies_bouldin = davies_bouldin_score(X, kmeans_labels)
    agglomerative_davies_bouldin = davies_bouldin_score(X, agglomerative_labels)
    dbscan_davies_bouldin = davies_bouldin_score(X, dbscan_labels)

    print("Metoda K-means:")
    print("Silhouette Score:", kmeans_silhouette)
    print("Calinski-Harabasz Score:", kmeans_calinski_harabasz)
    print("Davies-Bouldin Score:", kmeans_davies_bouldin)
    print()
    print("Metoda Agglomerative Clustering:")
    print("Silhouette Score:", agglomerative_silhouette)
    print("Calinski-Harabasz Score:", agglomerative_calinski_harabasz)
    print("Davies-Bouldin Score:", agglomerative_davies_bouldin)
    print()
    print("Metoda DBSCAN:")
    print("Silhouette Score:", dbscan_silhouette)
    print("Calinski-Harabasz Score:", dbscan_calinski_harabasz)
    print("Davies-Bouldin Score:", dbscan_davies_bouldin)

    """
Metoda K-means:
Silhouette Score: 0.38891796219674873
Calinski-Harabasz Score: 240.89476955307632
Davies-Bouldin Score: 0.9685680160166489

Metoda Agglomerative Clustering:
Silhouette Score: 0.35091628775584294
Calinski-Harabasz Score: 221.70155657251044
Davies-Bouldin Score: 1.0219125525443817

Metoda DBSCAN:
Silhouette Score: 0.015393337067620005
Calinski-Harabasz Score: 249.39780228498302
Davies-Bouldin Score: 0.10281679209634811

Process finished with exit code 0


    Dzieki temu mozna jasno stwierdzic ze K-Means ma najlepsze wyniki, tuz za nia jest Agglomerative a  potem DBSCAN ktorej pare razy podostoswalem parametry wiec jest opcja,
    gdyby jescze troche poprobowac polepszyc jej wyniki. (zostawilem 6 - 1, w ktorej dostaje najlepsze wyniki z paru prob ktore wykonalem)
    
    
    """

#zad8()