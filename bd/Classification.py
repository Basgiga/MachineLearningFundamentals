import pandas as pd
from sklearn.datasets import load_breast_cancer, load_digits, fetch_lfw_people, load_wine, load_iris, fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.svm import SVC
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, auc
from sklearn.tree import DecisionTreeClassifier


def z2():
    dane = load_iris()
    X = dane.data
    Y = dane.target

    dane_df = pd.DataFrame(X, columns=dane.feature_names)

    print(f'\npare pierwszych linijek\n',dane_df.head())
    print(f'\n\n informacje o kolumnach\n', dane_df.describe())

    # Podział danych na zbiór treningowy i testowy
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

    #k-ns gdzie k =3
    knn3 = KNeighborsClassifier(n_neighbors=3)
    knn3.fit(X_train,Y_train)

    # k-ns gdzie k = 4
    knn4 = KNeighborsClassifier(n_neighbors=4)
    knn4.fit(X_train, Y_train)

    #k-ns gdzie k =50
    knn5 = KNeighborsClassifier(n_neighbors=50)
    knn5.fit(X_train,Y_train)

    #predykcja
    y_predykowane3 = knn3.predict(X_test)
    y_predykowane4 = knn4.predict(X_test)
    y_predykowane5 = knn5.predict(X_test)

    # ocena3
    dokladnosc3 = accuracy_score(Y_test,y_predykowane3)
    precyzja3 = precision_score(Y_test,y_predykowane3, average='weighted')
    czulosc3 = recall_score(Y_test, y_predykowane3, average='weighted')
    conf_matrix3 = confusion_matrix(Y_test, y_predykowane3)
    cm_display = ConfusionMatrixDisplay(conf_matrix3)
    cm_display.plot()
    plt.show()
    print(conf_matrix3)
    tn = conf_matrix3[1,1]+conf_matrix3[2,1]+conf_matrix3[2,2]+conf_matrix3[1,2]
    fp = conf_matrix3[0, 1]+conf_matrix3[0, 2]
    specyficznosc3 = tn/(tn+fp)


    # ocena 4
    dokladnosc4 = accuracy_score(Y_test,y_predykowane4)
    precyzja4 = precision_score(Y_test,y_predykowane4, average='weighted')
    czulosc4 = recall_score(Y_test, y_predykowane4, average='weighted')
    conf_matrix4 = confusion_matrix(Y_test, y_predykowane4)
    cm_display = ConfusionMatrixDisplay(conf_matrix4)
    cm_display.plot()
    plt.show()
    print(conf_matrix4)
    tn = conf_matrix4[1, 1] + conf_matrix4[1, 2] + conf_matrix4[2, 1] + conf_matrix4[2, 2]
    fp = conf_matrix4[0, 1] + conf_matrix4[0, 2]
    specyficznosc4 = tn / (tn + fp)


    # ocena 5
    dokladnosc5 = accuracy_score(Y_test,y_predykowane5)
    precyzja5 = precision_score(Y_test,y_predykowane5, average='weighted')
    czulosc5 = recall_score(Y_test, y_predykowane5, average='weighted')
    conf_matrix5 = confusion_matrix(Y_test, y_predykowane5)
    cm_display = ConfusionMatrixDisplay(conf_matrix5)
    cm_display.plot()
    plt.show()
    print(conf_matrix5)
    tn = conf_matrix5[1, 1] + conf_matrix5[2, 1] + conf_matrix5[2, 2] + conf_matrix5[1, 2]
    fp = conf_matrix5[0, 1] + conf_matrix5[0, 2]
    specyficznosc5 = tn / (tn + fp)

    print('\n\nBedziemy liczyc specyficznosc dla klasy "setosa"\n\n\n')
    print('ocena modelu k-nn dla k = 3')
    print('\ndokladnosc:',dokladnosc3)
    print('\nprecyzja: ', precyzja3)
    print('\nczulosc: ', czulosc3)
    print('\n specyficznosc: ', specyficznosc3)
    print('\n\n\n')
    print('ocena modelu k-nn dla k = 4')
    print('\ndokladnosc:',dokladnosc4)
    print('\nprecyzja: ', precyzja4)
    print('\nczulosc: ', czulosc4)
    print('\n specyficznosc: ', specyficznosc4)
    print('\n\n\n')
    print('ocena modelu k-nn dla k = 50')
    print('\ndokladnosc:',dokladnosc5)
    print('\nprecyzja: ', precyzja5)
    print('\nczulosc: ', czulosc5)
    print('\n specyficznosc: ', specyficznosc5)


    '''
    widac ze dla k=3 i k=4 czynniki sa identyczne, a w dodatku bardzo dobre (bliskie 1)
    ale im wiecej damy sasiadow (skomplikujemy model) tym bardziej on stanie sie przetrenowany i dla danych testowych traci wartosci tych czynnikow
    sprawdzilem jeszcze dla k = 1 i sa idenczyczne wyniki co oznacza ze jest to najlepszy model, poniewaz jest najprostszy obliczeniowo, przy najlepszych wynikach z czynnikow oceny 
    '''

def z3():
    dane = load_breast_cancer()
    X = dane.data
    Y = dane.target


    df = pd.DataFrame(X,columns=dane.feature_names)
    print('\n\n\nzadanie3\n\nprzed skalowaniem i oczysczaniem:\n')
    print(df.head())
    print('\n',df.describe())

    # Usunięcie brakujących wartości
    df.dropna(inplace=True)
    # Usunięcie duplikatów
    df.drop_duplicates(inplace=True)

    # Skalowanie cech
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(X)
    df_X = pd.DataFrame(df_scaled, columns=dane.feature_names)

    print('po skalowaniu i oczyszczaniu dla X: \n',df_X.head(),'\n',df_X.describe())

    df_Y = pd.DataFrame(Y, columns=['target'])

    print('\n i dla Y: \n', df_Y.head())
    print('\n', df_Y.describe())

    X_train, X_test, y_train, y_test = train_test_split(df_X, df_Y, test_size=0.2, random_state=0)

    # Inicjalizacja i trenowanie modelu regresji logistycznej
    logistic_regression = LogisticRegression()
    logistic_regression.fit(X_train, y_train)

    # Predykcja etykiet dla danych testowych
    y_pred = logistic_regression.predict(X_test)

    # ocena3
    dokladnosc = accuracy_score(y_test,  y_pred)
    precyzja = precision_score(y_test,  y_pred, average='weighted')
    czulosc = recall_score(y_test,  y_pred, average='weighted')
    conf_matrix = confusion_matrix(y_test,  y_pred)
    cm_display = ConfusionMatrixDisplay(conf_matrix)
    cm_display.plot()
    plt.show()
    print(conf_matrix)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    specyficznosc = tn / (tn + fp)

    print('\n\n\n')
    print('ocena modelu regresji logistycznej')
    print('\ndokladnosc:', dokladnosc)
    print('\nprecyzja: ', precyzja)
    print('\nczulosc: ', czulosc)
    print('\n specyficznosc: ', specyficznosc)

    # Inicjalizacja i trenowanie modelu regresji logistycznej tym razem z parametrem regularyzacji c
    logistic_regression = LogisticRegression(C=0.01)
    logistic_regression.fit(X_train, y_train)

    # Predykcja etykiet dla danych testowych
    y_pred = logistic_regression.predict(X_test)

    # ocena
    dokladnosc = accuracy_score(y_test, y_pred)
    precyzja = precision_score(y_test, y_pred, average='weighted')
    czulosc = recall_score(y_test, y_pred, average='weighted')
    conf_matrix = confusion_matrix(y_test, y_pred)
    cm_display = ConfusionMatrixDisplay(conf_matrix)
    cm_display.plot()
    plt.show()
    print(conf_matrix)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    specyficznosc = tn / (tn + fp)

    print('\n\n\n')
    print('ocena modelu regresji logistycznej z parametrem regularyzacji C = 0.01')
    print('\ndokladnosc:', dokladnosc)
    print('\nprecyzja: ', precyzja)
    print('\nczulosc: ', czulosc)
    print('\n specyficznosc: ', specyficznosc)


    '''
    dla silnej regularyzacji okazuje sie ze pojawia się więcej FN ale znikaja FP co może być użyteczne w niektorych scenariuasz,
    ale ogólnikowo ( w nie skomplikowanych scenariuszach lepszy jest ten prostszy model
    (porownalem jeszce reg. log. z większym C (=0.5, =0.9), ale nie roznia sie wyniki od zwyklej reg log.
    
    '''



def z4():
    digits = load_digits()
    #print(digits.DESCR)
    X = digits.data
    y = digits.target

    # Spłaszczenie obrazów
    X_flat = X.reshape((X.shape[0], -1))

    # Standaryzacja wartości pikseli
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_flat)

    # Podział danych na zbiór treningowy i testowy
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Inicjalizacja i trenowanie modelu SVM
    svm_classifier = SVC(kernel='linear')
    svm_classifier.fit(X_train, y_train)

    # Predykcja etykiet dla danych testowych
    y_pred = svm_classifier.predict(X_test)

    # ocena
    dokladnosc = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    cm_display = ConfusionMatrixDisplay(conf_matrix)
    cm_display.plot()
    plt.show()
    print(conf_matrix)

    print('\n\n\n')
    print('ocena modelu regresji logistycznej z parametrem regularyzacji C = 0.01')
    print('\ndokladnosc:', dokladnosc)


    '''
    dokladnosc jest bardzo wysoka, wiec wyglada na to ze model sam w sobie jest niezly, 
    ale najwieksze problemy ma z '9', gdzie 3 razy wskazal ze to inna cyfra
    dodatkowo cyfra 3 jest 2 razy wskazana zle, na i tak niski stan cyfr 3 w zbiorze (32) (37 bylo 9-tek)
    '''

def z5():
    dane = pd.read_csv('train.csv', sep=',')
    print(dane.head())
    print(dane.describe())
    # gdzie nulle
    print(dane.isnull().sum())

    # Usunięcie kolumny "Cabin" i "Name" i "Ticket"
    dane.drop(columns=['Cabin'], inplace=True)
    dane.drop(columns=['Name'], inplace=True)
    dane.drop(columns=['Ticket'], inplace=True)

    # Uzupełnienie brakujących wartości w kolumnie "Embarked" najczęściej występującą wartością
    most_common_embarked = dane['Embarked'].mode()[0]
    dane['Embarked'] = dane['Embarked'].fillna(most_common_embarked)

    # Uzupełnienie brakujących wartości w kolumnie "Age" medianą wieku
    median_age = dane['Age'].median()
    dane['Age'] = dane['Age'].fillna(median_age)

    # czy zostaly jakies nulle
    print(dane.isnull().sum())

    # one-hot
    dane_kat = dane.select_dtypes(include=['object'])
    dane_numer = dane.select_dtypes(include=['number'])
    dane_kod = pd.get_dummies(dane_kat)
    dane_przetworzone = pd.concat([dane_numer, dane_kod], axis=1)
    print('\n\n\n',dane_przetworzone.columns,'\n\n\n')





    # Wizualizacja rozkładu przeżycia
    sns.countplot(x='Survived', data=dane)
    plt.show()

    # Wizualizacja rozkładu przeżycia w zależności od płci
    sns.countplot(x='Survived', hue='Sex', data=dane)
    plt.show()

    # Wizualizacja rozkładu przeżycia w zależności od klasy podróżnej
    sns.countplot(x='Survived', hue='Pclass', data=dane)
    plt.show()

    # Wizualizacja rozkładu przeżycia w zależności od Parch
    sns.countplot(x='Survived', hue='Parch', data=dane)
    plt.show()

    # Wizualizacja rozkładu przeżycia w zależności od SibSp
    sns.countplot(x='Survived', hue='SibSp', data=dane)
    plt.show()




    '''
    najlepsze szanse na dobry klasyfikator ma miejsce w Sex i Pclass"
    '''


    X = dane_przetworzone[['Sex_female', 'Sex_male', 'Pclass']]
    print(X.head())
    print(X.describe())

    Y = dane_przetworzone['Survived']

    print(Y.head())
    print(Y.describe())

    # Podział danych na zbiór treningowy i testowy
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2,
                                                        random_state=42)

    # Inicjalizacja i trenowanie modelu regresji logistycznej
    logistic_regression = LogisticRegression(max_iter=1000)
    logistic_regression.fit(X_train, y_train)

    # Predykcja etykiet dla danych testowych
    y_pred = logistic_regression.predict(X_test)

    # ocena
    dokladnosc = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    cm_display = ConfusionMatrixDisplay(conf_matrix)
    cm_display.plot()
    plt.show()

    print('\n\n\n')
    print('ocena modelu regresji logistycznej z wybranymi kolumnami Sex i Pclass')
    print('\ndokladnosc:', dokladnosc)
    print("AUC:", roc_auc)
    print("Macierz pomyłek:")
    print(conf_matrix)

    '''
    Sprawdzmy teraz klasyfikator oparty na Parch i SibSp
     '''

    X = dane_przetworzone[['Parch', 'SibSp']]
    print(X.head())
    print(X.describe())

    Y = dane_przetworzone['Survived']

    print(Y.head())
    print(Y.describe())

    # Podział danych na zbiór treningowy i testowy
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2,
                                                        random_state=42)

    # Inicjalizacja i trenowanie modelu regresji logistycznej
    logistic_regression = LogisticRegression(max_iter=1000)
    logistic_regression.fit(X_train, y_train)

    # Predykcja etykiet dla danych testowych
    y_pred = logistic_regression.predict(X_test)

    # ocena
    dokladnosc = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    cm_display = ConfusionMatrixDisplay(conf_matrix)
    cm_display.plot()
    plt.show()

    print('\n\n\n')
    print('ocena modelu regresji logistycznej z wybranymi kolumnami parch i SipSp')
    print('\ndokladnosc:', dokladnosc)
    print("AUC:", roc_auc)
    print("Macierz pomyłek:")
    print(conf_matrix)


    '''
    jak widac klasyfikator oparty na plci i klasy pasazerskiej jest o wiele lepszy 
    niz klasyfikator oparty na powiazaniach rodzinnych,
    ten drugi prawie wszystko wrzuca do Positive, co generuje dużą ilość FP
    
    '''




    '''
    zrobie jeszcze model drzewa decyzyjnego
    '''


    '''
    najpierw dla naszych wybranych najlepszych kolumn (sex i pclass)
    '''

    X = dane_przetworzone[['Sex_female', 'Sex_male', 'Pclass']]
    print(X.head())
    print(X.describe())

    Y = dane_przetworzone['Survived']

    print(Y.head())
    print(Y.describe())

    # Podział danych na zbiór treningowy i testowy
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2,
                                                        random_state=42)

    # Budowa modelu drzewa decyzyjnego
    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    # ocena
    dokladnosc = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    cm_display = ConfusionMatrixDisplay(conf_matrix)
    cm_display.plot()
    plt.show()

    print('\n\n\n')
    print('ocena modelu regresji logistycznej z wybranymi kolumnami Sex i Pclass')
    print('\ndokladnosc:', dokladnosc)
    print("AUC:", roc_auc)
    print("Macierz pomyłek:")
    print(conf_matrix)



    '''
    Widac ze jest roznica, drzewo decyzyjne znakomicie radzi sobie z FN, ale za cene wiekszej ilosci FP i gorszego odrzucania(TN)
    tak jak wczesniej wspominalem w niektorych klasyfikatorach moze to miec znaczenie czy jest wiecej FN i FP
    tutaj mozemy z wieksza pewnoscia powiedziec ze jesli ktos ma nie przezyc tytanica, to raczej go nie przezyje
    mimo wszystko regresja logistyczna przewyzsza ocenami dokladnosci i roc-auc metode drzewa decyzyjnego
    
    
    sprawdzmy jeszcze wybranie slabych kolumn do predykcji (SibSp, Parch)
    '''

    X = dane_przetworzone[['Parch', 'SibSp']]
    print(X.head())
    print(X.describe())

    Y = dane_przetworzone['Survived']

    print(Y.head())
    print(Y.describe())

    # Podział danych na zbiór treningowy i testowy
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2,
                                                        random_state=42)

    # Budowa modelu drzewa decyzyjnego
    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    # ocena
    dokladnosc = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    cm_display = ConfusionMatrixDisplay(conf_matrix)
    cm_display.plot()
    plt.show()

    print('\n\n\n')
    print('ocena modelu regresji logistycznej z wybranymi kolumnami Sex i Pclass')
    print('\ndokladnosc:', dokladnosc)
    print("AUC:", roc_auc)
    print("Macierz pomyłek:")
    print(conf_matrix)


    '''
    sytuacja podobna co w reg log. ale wydaje sie odrobine lepsza, z pewnoscia nie wrzuca wszystkiego w positive, tylko choc troche wyroznia negativy
    ale jednak tylko 2/3 przypadkow negatywnych klasyfikuje poprawnie, przez te dwa czynniki jego oceny slabna (ale i tak sa lepsze od tego wariantu w wersji reg log.)
    
    '''


def z6():
    from ucimlrepo import fetch_ucirepo
    test_size = 0.2
    # fetch dataset
    heart_disease = fetch_ucirepo(id=45)

    # data (as pandas dataframes)
    X = heart_disease.data.features
    y = heart_disease.data.targets

    print(y.head())
    print(X.describe())
    # gdzie nulle
    print(X.isnull().sum())

    # Uzupełnienie brakujących wartości w kolumnie "ca" i "thal" najczęściej występującą wartością
    most_common_embarked_ca = X['ca'].mode()[0]
    X.loc[:, 'ca'] = X['ca'].fillna(most_common_embarked_ca)

    # Uzupełnienie brakujących wartości w kolumnie "ca" i "thal" najczęściej występującą wartością
    most_common_embarked_thal = X['thal'].mode()[0]
    X.loc[:, 'thal'] = X['thal'].fillna(most_common_embarked_thal)

    #czy sa jeszcze nulle
    print(X.isnull().sum())

    X['target'] = y
    # Wizualizacja korelacji między cechami
    correlation_matrix = X.corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Korelacja między cechami')
    plt.show()


    '''
    na podstawie macierzy korelacji mozna zauwazyc ze najwieksza korelacje z target (choroba serca) maja kolumhy:
    cp: 0.41
    thalach: -0.42
    oldpeak: 0.5
    ca: 0.52
    thal: 0.51
    
    wezme pod uwage teraz je w modelu klasyfikatora
    '''
    X.drop(columns=['target'], inplace=True)

    # top 3
    X_nowy = X[['oldpeak','ca','thal']]
    # Podział danych na zbiór treningowy i testowy
    X_train, X_test, y_train, y_test = train_test_split(X_nowy, y, test_size=test_size,random_state=42)

    # Inicjalizacja i trenowanie modelu SVM
    svm_classifier = SVC(kernel='linear')
    svm_classifier.fit(X_train, y_train)
    # Predykcja etykiet dla danych testowych
    y_pred = svm_classifier.predict(X_test)

    # ocena
    precyzja = precision_score(y_test, y_pred, average='weighted')
    czulosc = recall_score(y_test, y_pred, average='weighted')
    conf_matrix = confusion_matrix(y_test, y_pred)
    cm_display = ConfusionMatrixDisplay(conf_matrix)
    cm_display.plot()
    plt.show()
    print(conf_matrix)
    # specyficzniosc dla 0
    TN = conf_matrix[0, 0]
    FP = conf_matrix[0, 1:].sum()
    specyficznosc = TN / (TN + FP)

    print('\n\n\n')
    print('ocena modelu svc')
    print('\nprecyzja: ', precyzja)
    print('\nczulosc: ', czulosc)
    print('\n specyficznosc dla klasy 0: ', specyficznosc)



    '''
    teraz wezmy moze thalach, ca i oldpeak
    '''
    #X_nowy = X[['oldpeak', 'ca', 'thalach', 'thal', 'cp']]
    X_nowy = X[['oldpeak', 'ca', 'thalach']]
    # Podział danych na zbiór treningowy i testowy
    X_train, X_test, y_train, y_test = train_test_split(X_nowy, y, test_size=test_size, random_state=42)

    # Inicjalizacja i trenowanie modelu SVM
    svm_classifier = SVC(kernel='linear')
    svm_classifier.fit(X_train, y_train)
    # Predykcja etykiet dla danych testowych
    y_pred = svm_classifier.predict(X_test)

    # ocena
    precyzja = precision_score(y_test, y_pred, average='weighted')
    czulosc = recall_score(y_test, y_pred, average='weighted')
    conf_matrix = confusion_matrix(y_test, y_pred)
    cm_display = ConfusionMatrixDisplay(conf_matrix)
    cm_display.plot()
    plt.show()
    print(conf_matrix)
    # specyficzniosc dla 0
    TN = conf_matrix[0, 0]
    FP = conf_matrix[0, 1:].sum()
    specyficznosc = TN / (TN + FP)

    print('\n\n\n')
    print('ocena modelu svc')
    print('\nprecyzja: ', precyzja)
    print('\nczulosc: ', czulosc)
    print('\n specyficznosc dla klasy 0: ', specyficznosc)

    '''
        teraz wezmy te z najmniejsza korelacja
    '''
    X_nowy = X[['chol', 'fbs', 'trestbps']]
    # Podział danych na zbiór treningowy i testowy
    X_train, X_test, y_train, y_test = train_test_split(X_nowy, y, test_size=test_size, random_state=42)

    # Inicjalizacja i trenowanie modelu SVM
    svm_classifier = SVC(kernel='linear')
    svm_classifier.fit(X_train, y_train)
    # Predykcja etykiet dla danych testowych
    y_pred = svm_classifier.predict(X_test)

    # ocena
    precyzja = precision_score(y_test, y_pred, average='weighted')
    czulosc = recall_score(y_test, y_pred, average='weighted')
    conf_matrix = confusion_matrix(y_test, y_pred)
    cm_display = ConfusionMatrixDisplay(conf_matrix)
    cm_display.plot()
    plt.show()
    print(conf_matrix)
    # specyficzniosc dla 0
    TN = conf_matrix[0, 0]
    FP = conf_matrix[0, 1:].sum()
    specyficznosc = TN / (TN + FP)

    print('\n\n\n')
    print('ocena modelu svc')
    print('\nprecyzja: ', precyzja)
    print('\nczulosc: ', czulosc)
    print('\n specyficznosc dla klasy 0: ', specyficznosc)

    '''
    i jeszce wezmy model z wszystkimi z najwieksza korelacja (>0.4)
    '''


    X_nowy = X[['oldpeak', 'ca', 'thalach', 'thal', 'cp']]
    # Podział danych na zbiór treningowy i testowy
    X_train, X_test, y_train, y_test = train_test_split(X_nowy, y, test_size=test_size, random_state=42)

    # Inicjalizacja i trenowanie modelu SVM
    svm_classifier = SVC(kernel='linear')
    svm_classifier.fit(X_train, y_train)
    # Predykcja etykiet dla danych testowych
    y_pred = svm_classifier.predict(X_test)

    # ocena
    precyzja = precision_score(y_test, y_pred, average='weighted')
    czulosc = recall_score(y_test, y_pred, average='weighted')
    conf_matrix = confusion_matrix(y_test, y_pred)
    cm_display = ConfusionMatrixDisplay(conf_matrix)
    cm_display.plot()
    plt.show()
    print(conf_matrix)
    # specyficzniosc dla 0
    TN = conf_matrix[0, 0]
    FP = conf_matrix[0, 1:].sum()
    specyficznosc = TN / (TN + FP)

    print('\n\n\n')
    print('ocena modelu svc')
    print('\nprecyzja: ', precyzja)
    print('\nczulosc: ', czulosc)
    print('\n specyficznosc dla klasy 0: ', specyficznosc)


    '''
    podsumowujac bazujac na precyzji
    mozna powiedziec ze najlepiej brac te wszytstkie wartosci ktore wymienilem wczesniej z dobra korelacja
    ale z nich najlepiej wypada kombinacja najlepszych 3 czyli
    'oldpeak','ca','thal'
    za to, czego sie troche spodziewalem patrzac na poprzednie zadania, gdy bierzemy wartosci malo skorelowane
    klasyfikator wypluwa same positive co oczywiscie jest slabe.
    '''
z2()
z3()
z4()
z5()
z6()