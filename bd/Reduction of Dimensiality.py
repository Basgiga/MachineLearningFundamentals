import scipy.signal as sig
from scipy.signal import find_peaks
from statsmodels.tsa.stattools import acf
from statsmodels.robust import mad
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pywt
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_digits, fetch_lfw_people, load_wine, load_iris, fetch_20newsgroups
from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, NMF
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction import text
from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import LabelEncoder



def z2():
    bc = load_breast_cancer()
    X = bc.data
    Y = bc.target

    pca = PCA(n_components=2)
    X_Pca = pca.fit_transform(X)

    plt.figure()
    for i in range(len(bc.target_names)):
        plt.scatter(X_Pca[Y == i ,0], X_Pca[Y == i , 1], label = bc.target_names[i])
    plt.xlabel('Pierwsza składowa główna')
    plt.ylabel('Druga składowa główna')
    plt.legend()
    plt.show()


#z2()

def z3():
    ld = load_digits()
    X = ld.data
    Y = ld.target

    # Redukcja wymiarowości przy użyciu t-SNE
    tsne = TSNE(n_components=2)
    X_embedded = tsne.fit_transform(X)
    target_ids = range(len(ld.target_names))


    # Wizualizacja danych w nowej przestrzeni
    colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'pink', 'orange', 'purple'
    for i, c, label in zip(target_ids, colors, ld.target_names):
        plt.scatter(X_embedded[Y == i, 0], X_embedded[Y == i, 1], c=c, label=label)
    plt.legend()
    plt.show()
#z3()


def z4():
    pf = fetch_lfw_people(min_faces_per_person=100, resize=0.035)
    X = pf.data
    Y = pf.target

    # Tworzenie obiektu NMF
    nmf_model = NMF(n_components=2)  # ustalenie liczby składowych na 3
    # Dopasowanie modelu do danych
    nmf_model.fit(X)
    # Pobranie macierzy bazowych i wagowych
    W = nmf_model.transform(X)  # macierz bazowa
    H = nmf_model.components_  # macierz wagowa
    # Wyświetlenie macierzy bazowych i wagowych
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(W, cmap='viridis', aspect='auto')
    plt.title('Macierz bazowa (W)')
    plt.xlabel('Składowe')
    plt.ylabel('Próbki')
    plt.subplot(1, 2, 2)
    plt.imshow(H, cmap='viridis', aspect='auto')
    plt.title('Macierz wagowa (H)')
    plt.xlabel('Cechy')
    plt.ylabel('Składowe')
    plt.tight_layout()
    plt.show()


def z5():
    wine_data = load_wine()
    X = wine_data.data
    y = wine_data.target

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)


    svd = TruncatedSVD(n_components=X.shape[1] - 1)
    X_svd = svd.fit_transform(X_scaled)

    wariancja = svd.explained_variance_ratio_
    cumsumwar = np.cumsum(wariancja)

    # calka warianiancji vs ilosc komponentow
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(cumsumwar) + 1), cumsumwar, marker='o',
             linestyle='-')
    plt.title('calka warianiancji vs ilosc komponentow')
    plt.xlabel('ilosc komponentow')
    plt.ylabel('Cumsum po wariancji')
    plt.grid(True)
    plt.show()

    '''
    mozna zastosowac metode lokcia dla tych danych
    
    '''

    #  ile komponentow to 95% wariancji
    n_components_95 = np.argmax(cumsumwar >= 0.95) + 1
    print("Liczba komponentów dla 95% wyjaśnionej wariancji:", n_components_95)

    # Przekształcenie danych do przestrzeni o mniejszej liczbie wymiarów
    svd_final = TruncatedSVD(n_components=n_components_95)
    X_svd_final = svd_final.fit_transform(X_scaled)

    # Wykres danych w nowej przestrzeni cech (jeśli liczba komponentów wynosi 2)
    if n_components_95 == 2:
        plt.figure(figsize=(10, 6))
        for label in np.unique(y):
            plt.scatter(X_svd_final[y == label, 0], X_svd_final[y == label, 1], label=f'Class {label + 1}')
        plt.title('Data Visualization in Reduced Dimensionality Space')
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.legend()
        plt.grid(True)
        plt.show()
    else:
        print("Liczba komponentów jest większa niż 2, więc nie możemy zwizualizować danych w dwóch wymiarach.")



def z6():
    data = fetch_20newsgroups(subset='all', shuffle=True, random_state=42)

    # Definicja listy stop words z dodatkowymi słowami
    stop_words = list(text.ENGLISH_STOP_WORDS.union(["from", "subject", "re", "edu", "use"]))
    # Definicja CountVectorizera
    vectorizer = CountVectorizer(stop_words=stop_words, max_features=1000)

    # Przetwarzanie tekstu i przekształcenie dokumentów na wektory cech
    X = vectorizer.fit_transform(data.data)

    lda_model = LatentDirichletAllocation(n_components=20, max_iter=10, learning_method='online', random_state=42)
    lda_output = lda_model.fit_transform(X) 

    # Wyświetlenie najważniejszych słów dla każdego tematu
    no_top_words = 10
    feature_names = vectorizer.get_feature_names_out()
    for topic_idx, topic in enumerate(lda_model.components_):
        print(f"Temat {topic_idx}:")
        print(" ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))
        topic_idx+=1
        print()

    #wizualizacja
    plt.figure(figsize=(12, 6))
    plt.bar(range(20), lda_output.sum(axis=0))
    plt.title('Przyporządkowanie dokumentów do tematów')
    plt.xlabel('Temat')
    plt.ylabel('Liczba dokumentów')
    plt.show()

    '''
    (Na podstawie najwazniejszych slow):
    widac jak w temacie 1 poruszane sa same religijne tematy,
    a np w temacie 5 sa same cyfry, co mogloby wskazywac na artykuly z liczeniem/danymi
    tak samo temat 10
    za to temat 13 wydaje sie polityczny 
    itd.
    
    '''


def z7():
    from ucimlrepo import fetch_ucirepo

    # fetch dataset
    differentiated_thyroid_cancer_recurrence = fetch_ucirepo(id=915)

    # data (as pandas dataframes)
    X = differentiated_thyroid_cancer_recurrence.data.features
    y = differentiated_thyroid_cancer_recurrence.data.targets


    '''
    # metadata
    print(differentiated_thyroid_cancer_recurrence.metadata)

    # variable information
    print(differentiated_thyroid_cancer_recurrence.variables)

    print(X)

    print('\n\na\n\n')

    print(y)
    '''
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)



    X = pd.get_dummies(X, columns=['Gender', 'Smoking', 'Hx Smoking', 'Hx Radiothreapy', 'Thyroid Function', 'Physical Examination', 'Adenopathy','Pathology', 'Focality', 'Risk','T','N', 'M', 'Stage', 'Response'])

    # Redukcja wymiarowości za pomocą PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    # Redukcja wymiarowości za pomocą NMF
    nmf = NMF(n_components=2)
    X_nmf = nmf.fit_transform(X)

    # Redukcja wymiarowości za pomocą t-SNE
    tsne = TSNE(n_components=2)
    X_tsne = tsne.fit_transform(X)

    # Wykres porównawczy
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_encoded, cmap='viridis', edgecolor='k')
    plt.title('PCA')

    plt.subplot(1, 3, 2)
    plt.scatter(X_nmf[:, 0], X_nmf[:, 1],c=y_encoded, cmap='viridis', edgecolor='k')
    plt.title('NMF')

    plt.subplot(1, 3, 3)
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1],c=y_encoded, cmap='viridis', edgecolor='k')
    plt.title('t-SNE')

    plt.tight_layout()
    plt.show()


    '''
    w tych danych dotyczacych Differentiated Thyroid Cancer Recurrence, czyli wsytepowania raka, nie sa to dane psychologiczne, ale z UC irvine na ktorych bardzo przyjemnie sie analizuje wyniki
    widac ze tsne jest zupelnie nie oplacalne, natomiast PCA czy NMF juz predzej nadawaly by się do pozniejszego treningu jako klasyfikatory
    PCA dodatkowo znajduje dobre skupisko fioletowych (czyli nie majacych raka) co pozwalaloby z wieksza dokladnoscia odrzucic zdrowych pacjentow.
    
    '''


#z2()
#z3()
#z4()
#z5()
#z6()
#z7()