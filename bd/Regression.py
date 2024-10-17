from sklearn.linear_model import LinearRegression, Ridge, Lasso
import numpy as np
import random
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVR

'''

Generowanie danych

'''

def generate_data(N):
     # Definiowanie listy, która będzie przechowywać dane
     data = []
     # Generowanie N losowych wierszy danych
     for _ in range(N):
         area = random.randint(50, 120)
         rooms = random.randint(1, 5)
         floor = random.randint(1, 10)
         year_of_construction = random.randint(1950, 2022)
         price = random.randint(150000, 1000000) * (0.50 * area) #domnozylem area zeby bylo widac jakakolwiek zaleznosc
         data.append([area, rooms, floor, year_of_construction, price])
     # Tworzenie obiektu DataFrame z listy danych
     df = pd.DataFrame(data, columns=['area', 'rooms', 'floor', 'year_of_construction', 'price'])
     # Zapisanie danych do pliku CSV
     df.to_csv('appartments.csv', index=False)
     print(f"Plik 'appartments.csv' został wygenerowany z {N} wierszami danych.")
# Wywołanie funkcji generate_data() z określoną ilością wierszy danych (np.100)
#generate_data(100)


'''

Przygotowanie danych

'''

# Wczytanie zbioru danych z pliku CSV
df = pd.read_csv('appartments.csv')

# Zbadanie struktury danych
print("Struktura danych:")
print(df.head())
print("\nTypy danych:")
print(df.dtypes)

# Identyfikacja brakujących wartości i obsługa brakujących danych
missing_values = df.isnull().sum()
print("\nBrakujące wartości:")
print(missing_values)

# Dla brakujących wartości można zastosować różne strategie, np. imputacja wartości średnich
imputer = SimpleImputer(strategy='mean')
df_filled = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

# Zastosowanie kodowania zmiennych kategorycznych
# Jeśli istnieją zmienne kategoryczne, możemy je zakodować za pomocą Label Encoding lub One-Hot Encoding
categorical_columns = df.select_dtypes(include=['object']).columns
if len(categorical_columns) > 0:
     label_encoder = LabelEncoder()
     for col in categorical_columns:
         df_filled[col] = label_encoder.fit_transform(df_filled[col])

# Podział zbioru danych na zbiór treningowy i testowy
X = df_filled.drop('price', axis=1) # usunięcie kolumny z wartościami celu
y = df_filled['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Wyświetlenie podsumowania zbioru danych treningowych i testowych
print("\nRozmiar zbioru treningowego:", X_train.shape)
print("Rozmiar zbioru testowego:", X_test.shape)

def zad3():

    plt.figure(figsize=(10,8))

    #pole vs cena
    Xnowy = X_train['area'].values.reshape(-1,1)
    Ynowy = y_train.values.reshape(-1, 1)

    model = LinearRegression()
    model.fit(Xnowy,Ynowy)

    y_pred = model.predict(X_test[['area']])
    mse_area = mean_squared_error(y_test, y_pred)
    r2_area = r2_score(y_test, y_pred)

    plt.subplot(3,1,1)
    plt.scatter(Xnowy,Ynowy, color = 'red', label = 'Dane')
    plt.plot(Xnowy,model.predict(Xnowy), color = 'blue', label='krzywa regresji')
    plt.title(f'Regresja Liniowa (pole powierzchni vs cena mieszkan)\n'
              f'MSE: {mse_area:.2f}, R^2: {r2_area:.2f}')
    plt.legend()

    # pokoje vs cena
    Xnowy = X_train['rooms'].values.reshape(-1,1)
    Ynowy = y_train.values.reshape(-1, 1)

    model = LinearRegression()
    model.fit(Xnowy,Ynowy)

    y_pred = model.predict(X_test[['rooms']])
    mse_area = mean_squared_error(y_test, y_pred)
    r2_area = r2_score(y_test, y_pred)

    plt.subplot(3,1,2)
    plt.scatter(Xnowy,Ynowy, color = 'red', label = 'Dane')
    plt.plot(Xnowy,model.predict(Xnowy), color = 'blue', label='krzywa regresji')
    plt.title(f'Regresja Liniowa (pokoje vs cena mieszkan)\n'
              f'MSE: {mse_area:.2f}, R^2: {r2_area:.2f}')
    plt.legend()

    # rok vs cena
    Xnowy = X_train['year_of_construction'].values.reshape(-1, 1)
    Ynowy = y_train.values.reshape(-1, 1)

    model = LinearRegression()
    model.fit(Xnowy, Ynowy)

    y_pred = model.predict(X_test[['year_of_construction']])
    mse_area = mean_squared_error(y_test, y_pred)
    r2_area = r2_score(y_test, y_pred)

    plt.subplot(3,1,3)
    plt.scatter(Xnowy,Ynowy, color = 'red', label = 'Dane')
    plt.plot(Xnowy,model.predict(Xnowy), color= 'blue', label='krzywa regresji')
    plt.title(f'Regresja Liniowa (rok vs cena mieszkan)\n'
              f'MSE: {mse_area:.2f}, R^2: {r2_area:.2f}')
    plt.legend()

    plt.tight_layout()
    plt.show()


    '''
    
    Dla mojego przebiegu:
    
    pole powierzchni vs cena:
    R^2 = 0.20
    
    liczba pokoi vs cena:
    R^2 = -0.00
    
    rok vs cena:
    R^2 = -0.03
    
    w kazdej regresji czynnik MSE jest duzy, a R^2 w dwoch (tych nie uzaleznionych przy generatorze) przypadkach sa bliskie lub rowne zero
    najlepiej wypada pole powierzchni ktore jako jedyne ma ten czynnik >0.00
    co oznacza ze istnieje niska zaleznosc meidzy powierzchnia a cena
    
    '''

def zad4(stopien):
    #ladowanie z pliku csv
    df = pd.read_csv('Miesiace.csv', sep=';')

    # Zbadanie struktury danych
    print("Struktura danych:")
    print(df.head())
    print("\nTypy danych:")
    print(df.dtypes)

    label_encoder = LabelEncoder()
    categorical_columns = df.select_dtypes(include=['object']).columns
    if len(categorical_columns) > 0:
        for col in categorical_columns:
            df[col] = label_encoder.fit_transform(df[col])

    print("Struktura danych:")
    print(df.head())
    print("\nTypy danych:")
    print(df.dtypes)

    X = df.drop('Temperatura', axis=1)  # usunięcie kolumny z wartościami celu
    y = df['Temperatura']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Wyświetlenie podsumowania zbioru danych treningowych i testowych
    print("\nRozmiar zbioru treningowego:", X_train.shape)
    print("Rozmiar zbioru testowego:", X_test.shape)

    #budowanie modelu
    Xnowy = X_train['Numer'].values.reshape(-1,1)
    Ynowy = y_train.values.reshape(-1,1)


    model = make_pipeline(PolynomialFeatures(stopien), LinearRegression())
    model.fit(Xnowy, Ynowy)
    x_wartosci = np.linspace(Xnowy.min(), Xnowy.max(), 100).reshape(-1, 1)
    y_pred = model.predict(x_wartosci)


    #Ocena modelu
    y_pred2 = model.predict(X_test['Numer'].values.reshape(-1, 1))
    mse_area = mean_squared_error(y_test, y_pred2)
    r2_area = r2_score(y_test, y_pred2)

    #wizualizacja
    plt.figure(figsize=(10,8))
    plt.scatter(Xnowy,Ynowy, color='red', label = 'Dane')
    plt.plot(x_wartosci,y_pred, color = 'blue', label ='krzywa regresji')
    plt.legend()
    plt.title(f'predykowanie temperatury w zaleznosci od miesiaca \nMSE: {mse_area:.2f}, R^2: {r2_area:.2f}')

    plt.tight_layout()
    plt.show()


'''
Dla stopnia 4tego otrzymujemu niskie MSE i R^2 = 0.98 
co dobrze pokazuje zaleznosc i dopasowanie

'''

def zad5(alfa):
    # Wczytanie danych
    df = pd.read_csv('temperatura_zuzycie_energii.csv', sep=';')

    # Podział danych na zbiór treningowy i testowy
    X = df.drop('zużycie_energii', axis=1)
    y = df['zużycie_energii']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    # Funkcja do budowania modelu i oceny wyników
    def zrob_model(model, X_train, y_train, X_test, y_test):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        return mse, r2

    #Budowa modeli regresji liniowej, grzbietowej i Lasso
    linear_model = LinearRegression()
    ridge_model = Ridge(alpha=alfa)
    lasso_model = Lasso(alpha=alfa)

    # Ocena modeli
    lin_mse, lin_r2 = zrob_model(linear_model, X_train, y_train, X_test, y_test)
    ridge_mse, ridge_r2 = zrob_model(ridge_model, X_train, y_train, X_test, y_test)
    lasso_mse, lasso_r2 = zrob_model(lasso_model, X_train, y_train, X_test, y_test)

    # Wizualizacja wyników
    plt.figure(figsize=(15, 10))
    # Regresja liniowa
    plt.subplot(3, 1, 1)
    plt.scatter(X_train, y_train, color='red', label='Dane treningowe')
    plt.plot(X_train, linear_model.predict(X_train), color='blue', label='Regresja liniowa')
    plt.legend()
    plt.title(f'Regresja Liniowa\nMSE: {lin_mse:.2f}, R^2: {lin_r2:.2f}')
    plt.xlabel('Temperatura')
    plt.ylabel('Zużycie energii')

    # Regresja grzbietowa
    plt.subplot(3, 1, 2)
    plt.scatter(X_train, y_train, color='red', label='Dane treningowe')
    plt.plot(X_train, ridge_model.predict(X_train), color='green', label='Regresja grzbietowa')
    plt.legend()
    plt.title(f'Regresja Grzbietowa\nMSE: {ridge_mse:.2f}, R^2: {ridge_r2:.2f}')
    plt.xlabel('Temperatura')
    plt.ylabel('Zużycie energii')

    # Regresja Lasso
    plt.subplot(3, 1, 3)
    plt.scatter(X_train, y_train, color='red', label='Dane treningowe')
    plt.plot(X_train, lasso_model.predict(X_train), color='orange', label='Regresja Lasso')
    plt.legend()
    plt.title(f'Regresja Lasso\nMSE: {lasso_mse:.2f}, R^2: {lasso_r2:.2f}')
    plt.xlabel('Temperatura')
    plt.ylabel('Zużycie energii')

    plt.tight_layout()
    plt.show()
'''
Dla alfy rownej 0.1 wszystkie wspolczynniki R^2 są rowne 0.50, a wspolczynniki MSE sa bardzo podobne
Dla alfy rownej 1 wszystkie wspolczynnik R^2 dalej wynosza 0.50, ale wspolczynnik MSE dla regresji grzbietowej jest odrobine mniejszy
Dla alfy rownej 100 wspolczynniki R^2 dla Liniowej i Lasso wynosza 0.50, a ich wspolczynnik MSE sa porownywalne
    natomiast dla Regresji Grzbietowej R^2 0.53 czyli odrobine wiecej
Dla alfy rownej 1000:
Liniowa R^2: 0.50
Grzbietowa R^2: 0.46
Lasso R^2: 0.52

Grzbietowa wydaje sie wyplaszczac przy ogromnych wartosciac alfy, ale przy alfa rownej 100 jest najlepszym dopsowaniem
Lasso natomiast przy ogromnej alfie wyprzedza Grzbietową i liniową

'''

def zad6():
    def generate_data2(N,wsp):
        # Definiowanie listy, która będzie przechowywać dane
        data = []
        # Generowanie N losowych wierszy danych
        for _ in range(N):
            wiek = random.randint(20, 80)
            BMI = random.randint(15, 40)
            cisnienie_krwi = random.randint(100, 200)
            poziom_glukozy = random.randint(50,100)
            cholesterol = random.randint(90, 250)
            kretynina = random.randint(50,115)
            BMI = BMI + wsp * wiek - 10
            cisnienie_krwi = cisnienie_krwi + wsp * wiek - 20
            poziom_glukozy = poziom_glukozy + wsp * wiek - 15
            cholesterol = cholesterol + wsp * wiek - 15
            kretynina = kretynina + wsp * wiek

            czas_przezycia = random.randint(50, 60) - 0.3 * wiek
            data.append([wiek, BMI, cisnienie_krwi, poziom_glukozy, cholesterol,kretynina, czas_przezycia])
        # Tworzenie obiektu DataFrame z listy danych
        df = pd.DataFrame(data, columns=['wiek', 'BMI', 'cisnienie_krwi', 'poziom_glukozy', 'cholesterol','kretynina', 'czas_przezycia'])
        # Zapisanie danych do pliku CSV
        df.to_csv('krew.csv', index=False)
        print(f"Plik 'appartments.csv' został wygenerowany z {N} wierszami danych.")
    #generate_data2(100,0.2)

    # Wczytanie zbioru danych z pliku CSV
    df = pd.read_csv('krew.csv')

    # Zbadanie struktury danych
    print("Struktura danych:")
    print(df.head())
    print("\nTypy danych:")
    print(df.dtypes)

    X = df.drop('czas_przezycia', axis=1)  # usunięcie kolumny z wartościami celu
    y = df['czas_przezycia']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Wyświetlenie podsumowania zbioru danych treningowych i testowych
    print("\nRozmiar zbioru treningowego:", X_train.shape)
    print("Rozmiar zbioru testowego:", X_test.shape)

    # budowanie modelu
    Xnowy = X_train['wiek'].values.reshape(-1, 1)
    Ynowy = y_train.values.reshape(-1, 1)

    model = SVR(kernel = 'linear')
    model.fit(Xnowy,Ynowy)

    y_pred = model.predict(X_test[['wiek']])
    mse_wiek = mean_squared_error(y_test, y_pred)
    r2_wiek = r2_score(y_test, y_pred)

    # wizualizacja
    plt.figure(figsize=(10, 8))

    plt.subplot(4,1,1)
    plt.scatter(Xnowy, Ynowy, color='red', label='Dane')
    plt.plot(Xnowy, model.predict(Xnowy), color='blue', label='krzywa regresji')
    plt.legend()
    plt.title(f'predykowany czas zycia w zaleznosci od wieku \nMSE: {mse_wiek:.2f}, R^2: {r2_wiek:.2f}')

    # budowanie modelu
    Xnowy = X_train['BMI'].values.reshape(-1, 1)
    Ynowy = y_train.values.reshape(-1, 1)

    model = SVR(kernel='linear')
    model.fit(Xnowy, Ynowy)

    y_pred = model.predict(X_test[['BMI']])
    mse_bmi = mean_squared_error(y_test, y_pred)
    r2_bmi = r2_score(y_test, y_pred)

    plt.subplot(4,1,2)
    plt.scatter(Xnowy, Ynowy, color='red', label='Dane')
    plt.plot(Xnowy, model.predict(Xnowy), color='blue', label='krzywa regresji')
    plt.legend()
    plt.title(f'(SVR) predykowany czas zycia w zaleznosci od bmi \nMSE: {mse_bmi:.2f}, R^2: {r2_bmi:.2f}')

    # budowanie modelu
    Xnowy = X_train['BMI'].values.reshape(-1, 1)
    Ynowy = y_train.values.reshape(-1, 1)

    model = SVR(kernel='linear')
    model.fit(Xnowy, Ynowy)

    y_pred = model.predict(X_test[['BMI']])
    mse_bmi = mean_squared_error(y_test, y_pred)
    r2_bmi = r2_score(y_test, y_pred)

    plt.subplot(4, 1, 2)
    plt.scatter(Xnowy, Ynowy, color='red', label='Dane')
    plt.plot(Xnowy, model.predict(Xnowy), color='blue', label='krzywa regresji')
    plt.legend()
    plt.title(f'(SVR) predykowany czas zycia w zaleznosci od bmi \nMSE: {mse_bmi:.2f}, R^2: {r2_bmi:.2f}')

    # budowanie modelu
    Xnowy = X_train['wiek'].values.reshape(-1, 1)
    Ynowy = y_train.values.reshape(-1, 1)

    model = Ridge(alpha=10)
    model.fit(Xnowy, Ynowy)

    y_pred = model.predict(X_test[['wiek']])
    mse_bmi = mean_squared_error(y_test, y_pred)
    r2_bmi = r2_score(y_test, y_pred)

    plt.subplot(4, 1, 3)
    plt.scatter(Xnowy, Ynowy, color='red', label='Dane')
    plt.plot(Xnowy, model.predict(Xnowy), color='blue', label='krzywa regresji')
    plt.legend()
    plt.title(f'(GRZBIETOWA) predykowany czas zycia w zaleznosci od wieku \nMSE: {mse_bmi:.2f}, R^2: {r2_bmi:.2f}')


    # budowanie modelu
    Xnowy = X_train['wiek'].values.reshape(-1, 1)
    Ynowy = y_train.values.reshape(-1, 1)

    model = LinearRegression()
    model.fit(Xnowy, Ynowy)

    y_pred = model.predict(X_test[['wiek']])
    mse_bmi = mean_squared_error(y_test, y_pred)
    r2_bmi = r2_score(y_test, y_pred)

    plt.subplot(4, 1, 4)
    plt.scatter(Xnowy, Ynowy, color='red', label='Dane')
    plt.plot(Xnowy, model.predict(Xnowy), color='blue', label='krzywa regresji')
    plt.legend()
    plt.title(f'(Liniowa) predykowany czas zycia w zaleznosci od wieku \nMSE: {mse_bmi:.2f}, R^2: {r2_bmi:.2f}')


    plt.tight_layout()
    plt.show()


#zad3()
#zad4(4)
#zad5(alfa = 0.1)
#zad5(alfa = 1)
#zad5(alfa = 1000)
#zad6()