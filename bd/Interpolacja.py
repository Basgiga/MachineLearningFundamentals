import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import splrep, splev, CubicHermiteSpline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
import random

def zad2():
    plik = pd.read_csv("Dane_l4z1.csv", sep=';', decimal=",")
    print(plik,'\n\n\n\n')
    print(plik.info())
    # usuwanie pustych kolumn
    df = plik.dropna(axis=1, how='all', inplace=False)
    # usuwanie pustych wierszy
    df = df.dropna(subset=df.columns, axis=0, how='any', inplace=False)

    print(df,'\n\n\n\n')
    print(df.head())
    print(df.info())

    #jak wygladaja dane:
    print(df.describe())

    #wizualizacja bez interpolacji
    x_val = df['Wartosci X']
    y_val = df['Wartosci Y']

    plt.figure()
    plt.scatter(x_val,y_val,color = 'red')
    plt.title('Dane Bez interpolacji')
    plt.show()

    #wybieram interpolacje B-sklejana Danych z 3cim stopniem
    stopien = 3
    tck = splrep(x_val, y_val,k=stopien)
    x_interpolated = np.linspace(min(x_val),max(x_val),100)
    y_interpolated = splev(x_interpolated,tck)

    #wizualizacja interpolacji
    plt.scatter(x_val, y_val, color='blue', label='Dane')
    plt.plot(x_interpolated, y_interpolated, color='red', label='Interpolacja Bsklejana')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Interpolacja B-sklejana danych')
    plt.legend()
    plt.show()



def generate_weather_data(num_stations, num_days):
     """
     Funkcja generuje przykładowe dane meteorologiczne dla wielu stacji pomiarowych i dni i zapisuje je do pliku CSV.
     Parametry:
     - num_stations: liczba stacji pomiarowych
     - num_days: liczba dni pomiarowych
     Zwraca:
     - None
     """
     # Temperatury miesięczne dla stacji 1
     temperatures1 = np.array([-2, 0, 5, 12, 18, 23, 26, 25, 21, 15, 8, 2])
     # Generowanie danych dla stacji 1
     np.random.seed(110)
     dates = pd.date_range(start='2023-01-01', periods=num_days)
     station_ids = ['Station_' + str(i) for i in range(1, num_stations + 1)]
     data = {station: [] for station in station_ids}

     for day in range(num_days):
         month = dates[day].month - 1 # Indeksowanie od zera
         temperature1 = temperatures1[month]
         # Generowanie danych dla pozostałych stacji z odchyłkami
         for station in station_ids:
             temperature = temperature1 + np.random.uniform(low=-2,
             high=2) if station == 'Station_1' else temperature1 + np.random.uniform(low=-4, high=4)
             if day > 0 and np.random.rand() < 0.05: # Rzadkie skoki temperatury
                 temperature += np.random.uniform(low=-10, high=10)
             data[station].append(temperature)
     # Utworzenie ramki danych
     df = pd.DataFrame(data)
     df['Date'] = dates
     df = df[['Date'] + station_ids]
     # Zapisanie danych do pliku CSV
     df.to_csv('weather_data.csv', index=False)

# Wygenerowanie przykładowych danych dla 5 stacji pomiarowych przez 15 dni
#generate_weather_data(num_stations=5, num_days=15)



def zad3():
    plik = pd.read_csv("weather_data.csv", sep=',')
    print(plik, '\n\n\n\n')
    print(plik.info())

    # usuwanie pustych kolumn
    df = plik.dropna(axis=1, how='all', inplace=False)
    # usuwanie pustych wierszy
    df = df.dropna(subset=df.columns, axis=0, how='any', inplace=False)

    print(df, '\n\n\n\n')
    print(df.info())

    # wizualizacja bez interpolacji
    df['Date'] = pd.to_datetime(df['Date'])
    # Pobranie tylko dnia z daty
    df['Day'] = df['Date'].apply(lambda x: x.day)
    # Wybór wartości na osi X
    x_val = df['Day']

    y_val1 = df['Station_1']
    y_val2 = df['Station_2']
    y_val3 = df['Station_3']
    y_val4 = df['Station_4']
    y_val5 = df['Station_5']



    plt.figure(figsize=(17, 5))
    plt.subplot(1,5,1)
    plt.scatter(x_val, y_val1, color='red')
    plt.title('Dane Bez interpolacji Stacja 1')

    plt.subplot(1, 5, 2)
    plt.scatter(x_val, y_val2, color='red')
    plt.title('Dane Bez interpolacji Stacja 2')

    plt.subplot(1, 5, 3)
    plt.scatter(x_val, y_val3, color='red')
    plt.title('Dane Bez interpolacji Stacja 3')

    plt.subplot(1, 5, 4)
    plt.scatter(x_val, y_val4, color='red')
    plt.title('Dane Bez interpolacji Stacja 4')

    plt.subplot(1, 5, 5)
    plt.scatter(x_val, y_val5, color='red')
    plt.title('Dane Bez interpolacji Stacja 5')

    plt.tight_layout()
    plt.show()


    #interpolacja B-sklejane
    stopien = 3

    tck = splrep(x_val, y_val1,k=stopien)
    x_interpolated = np.linspace(min(x_val),max(x_val),100)
    y_interpolated1 = splev(x_interpolated,tck)

    tck = splrep(x_val, y_val2,k=stopien)
    x_interpolated = np.linspace(min(x_val),max(x_val),100)
    y_interpolated2 = splev(x_interpolated,tck)

    tck = splrep(x_val, y_val3,k=stopien)
    x_interpolated = np.linspace(min(x_val),max(x_val),100)
    y_interpolated3 = splev(x_interpolated,tck)

    tck = splrep(x_val, y_val4,k=stopien)
    x_interpolated = np.linspace(min(x_val),max(x_val),100)
    y_interpolated4 = splev(x_interpolated,tck)

    tck = splrep(x_val, y_val5,k=stopien)
    x_interpolated = np.linspace(min(x_val),max(x_val),100)
    y_interpolated5 = splev(x_interpolated,tck)


    #wizualizacja interpolacji

    plt.figure(figsize=(17, 5))  # Rozmiar całego wykresu
    plt.subplot(1,5,1)
    plt.scatter(x_val, y_val1, color='blue', label='Dane')
    plt.plot(x_interpolated, y_interpolated1, color='red', label='Interpolacja Bsklejana')

    plt.subplot(1, 5, 2)
    plt.scatter(x_val, y_val2, color='blue', label='Dane')
    plt.plot(x_interpolated, y_interpolated2, color='red', label='Interpolacja Bsklejana')

    plt.subplot(1, 5, 3)
    plt.scatter(x_val, y_val3, color='blue', label='Dane')
    plt.plot(x_interpolated, y_interpolated3, color='red', label='Interpolacja Bsklejana')

    plt.subplot(1, 5, 4)
    plt.scatter(x_val, y_val4, color='blue', label='Dane')
    plt.plot(x_interpolated, y_interpolated4, color='red', label='Interpolacja Bsklejana')

    plt.subplot(1, 5, 5)
    plt.scatter(x_val, y_val5, color='blue', label='Dane')
    plt.plot(x_interpolated, y_interpolated5, color='red', label='Interpolacja Bsklejana')

    plt.tight_layout()
    plt.show()

def zad4():
    plik = pd.read_csv('zużycie_energii.csv', sep=',')

    print(plik.head())
    print(plik.info())

    # usuwanie pustych kolumn
    df = plik.dropna(axis=1, how='all', inplace=False)
    # usuwanie pustych wierszy
    df = df.dropna(subset=df.columns, axis=0, how='any', inplace=False)

    print(df.head())
    print(df.info())
    print(df.describe())

    #przygotowanie danych
    x_val = df['Miesiąc']
    y_val1 = df['Domowe']
    y_val2 = df['Przemysłowe']
    y_val3 = df['Komercyjne']

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.scatter(x_val, y_val1, color='red')
    plt.title('Dane Bez interpolacji Domowe')

    plt.subplot(1,3, 2)
    plt.scatter(x_val, y_val2, color='red')
    plt.title('Dane Bez interpolacji Przemyslowe')

    plt.subplot(1, 3, 3)
    plt.scatter(x_val, y_val3, color='red')
    plt.title('Dane Bez interpolacji Komercyjne')

    plt.tight_layout()
    plt.show()


    #przewidywanie metoda wielomianowa
    def polynomial_interpolation(x_values, y_values, x):
        # Wyznaczenie współczynników wielomianu interpolacyjnego
        coefficients = np.polyfit(x_values, y_values, len(x_values) - 1)

        # Wyliczenie wartości y dla podanej wartości x
        y = np.polyval(coefficients, x)
        return y

    x_interpolated = 5.5 # polowa maja :D

    y_interpolated1 = polynomial_interpolation(x_val, y_val1, x_interpolated)
    y_interpolated2 = polynomial_interpolation(x_val, y_val2, x_interpolated)
    y_interpolated3 = polynomial_interpolation(x_val, y_val3, x_interpolated)

    #wizualizacja B-splajnami
    stopien = 3

    tck = splrep(x_val, y_val1, k=stopien)
    x_interpolated1 = np.linspace(min(x_val), max(x_val), 100)
    y_interpolated11 = splev(x_interpolated1, tck)

    tck = splrep(x_val, y_val2, k=stopien)
    x_interpolated1 = np.linspace(min(x_val), max(x_val), 100)
    y_interpolated22 = splev(x_interpolated1, tck)

    tck = splrep(x_val, y_val3, k=stopien)
    x_interpolated1 = np.linspace(min(x_val), max(x_val), 100)
    y_interpolated33 = splev(x_interpolated1, tck)

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.scatter(x_val, y_val1, color='red')
    plt.scatter(x_interpolated, y_interpolated1, color='green', label=f'Interpolacja (wielomianowa) dla x = {x_interpolated}')
    plt.plot(x_interpolated1, y_interpolated11, color = 'blue', label = 'Interpolacja B-Splajny')
    plt.title('Dane interpolacji Domowe')
    plt.legend()

    plt.subplot(1,3, 2)
    plt.scatter(x_val, y_val2, color='red')
    plt.scatter(x_interpolated, y_interpolated2, color='green', label=f'Interpolacja (wielomianowa) dla x = {x_interpolated}')
    plt.plot(x_interpolated1, y_interpolated22, color = 'blue', label = 'Interpolacja B-Splajny')
    plt.title('Dane interpolacji Przemyslowe')
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.scatter(x_val, y_val3, color='red')
    plt.scatter(x_interpolated, y_interpolated3, color='green', label=f'Interpolacja (wielomianowa) dla x = {x_interpolated}')
    plt.plot(x_interpolated1, y_interpolated33, color = 'blue', label = 'Interpolacja B-Splajny')
    plt.title('Dane interpolacji Komercyjne')
    plt.legend()

    plt.tight_layout()
    plt.show()
    '''
    w domowych widac ze interpolacje maja nieco odmienne zdanie na temat srodku maja
    w komercyjnym i przemyslowym za to obydwie metody sa bardzo bliskie sobie
    
    
    '''

def zad5():
    plik = pd.read_csv('dane_gieldowe.csv', sep=',')

    print(plik.head())
    print(plik.info())

    def cubic_hermite_interpolation(x_values, y_values, y_derivatives, x):
        """
        Funkcja wykonuje interpolację kubiczną Hermite'a dla danych punktów.
        Parametry:
        - x_values: tablica NumPy zawierająca wartości x dla znanych punktów danych
        - y_values: tablica NumPy zawierająca odpowiadające wartości y dla znanych
       punktów danych
        - y_derivatives: tablica NumPy zawierająca pochodne pierwszego rzędu
       wartości y w punktach danych
        - x: wartość x, dla której ma zostać przewidziana wartość y
        Zwraca:
        - y: przewidywana wartość y dla podanej wartości x
        """
        # Utworzenie obiektu interpolacyjnego kubicznego Hermite'a
        spline = CubicHermiteSpline(x_values, y_values, y_derivatives)

        # Interpolacja wartości y dla podanej wartości x
        y = spline(x)
        return y

    # Wybór interesującego instrumentu finansowego
    instrument = 'Instrument_1'
    df_instrument = plik[plik['Instrument'] == instrument].head(100)

    #Interpolacja  Hermite
    dydx = np.gradient(df_instrument['Close'], df_instrument.index)

    x_val = np.linspace(df_instrument.index.min(), df_instrument.index.max(), 1000)
    y_val = cubic_hermite_interpolation(df_instrument.index, df_instrument['Close'],dydx,x_val)

    # Znalezienie lokalnych maksimów i minimów
    local_maxima = np.where((y_val[1:-1] > y_val[:-2]) & (y_val[1:-1] > y_val[2:]))[0] + 1
    local_minima = np.where((y_val[1:-1] < y_val[:-2]) & (y_val[1:-1] < y_val[2:]))[0] + 1



    #wizualizacja
    plt.figure(figsize=(10, 6))
    plt.plot(df_instrument.index, df_instrument['Close'], label='Cena na koniec dnia', color='blue')
    plt.plot(x_val, y_val, label = 'Interpolacja', color='green')
    plt.scatter(x_val[local_maxima], y_val[local_maxima], color='red', label='Lokalne maksima')
    plt.scatter(x_val[local_minima], y_val[local_minima], color='green', label='Lokalne minima')
    plt.title('Analiza trendow cen akcji - Interpolacja Hermite\'a')
    plt.xlabel('czas')
    plt.ylabel('Cena podczas zakonczenia dnia')
    plt.legend()
    plt.grid(True)
    plt.show()



def zad6():
    plik = pd.read_csv('DANE2022.csv', sep=';')
    print(plik.head())
    print(plik.info())

    # usuwanie pustych kolumn
    df = plik.dropna(axis=1, how='all', inplace=False)
    # usuwanie pustych wierszy
    df = df.dropna(subset=df.columns, axis=0, how='any', inplace=False)
    # Tworzenie nowej kolumny łączącej TYDZ i NRDNIA
    df['TYDZ_NRDNIA'] = df['TYDZ'].astype(int)*7 + df['NRDNIA'].astype(int) - 7
    # Usunięcie dwóch pierwszych wierszy
    df = df.iloc[2:]

    print(df.head(10))
    print(df.info())
    print(df.describe())

    n = 100
    df_n = df.head(n)
    x_val = df_n['TYDZ_NRDNIA']
    y_val = df_n['SDRD']

    plt.figure()
    plt.scatter(x_val,y_val,color = 'red')
    plt.title('Dane Bez interpolacji')
    plt.show()

    # B-sklejana Danych z 3cim stopniem
    stopien = 3
    tck = splrep(x_val, y_val,k=stopien)
    x_interpolated = np.linspace(min(x_val),max(x_val),100)
    y_interpolated = splev(x_interpolated,tck)

    #wizualizacja interpolacji
    plt.scatter(x_val, y_val, color='blue', label='Dane')
    plt.plot(x_interpolated, y_interpolated, color='red', label='Interpolacja Bsklejana')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Interpolacja B-sklejana danych')
    plt.legend()
    plt.show()

    #hermite
    def cubic_hermite_interpolation(x_values, y_values, y_derivatives, x):
        spline = CubicHermiteSpline(x_values, y_values, y_derivatives)
        y = spline(x)
        return y

    # Interpolacja  Hermite
    dydx = np.gradient(y_val, x_val)

    x_interpolated = np.linspace(x_val.min(), x_val.index.max(), 100)
    y_interpolated = cubic_hermite_interpolation(x_val, y_val, dydx, x_interpolated)

    # wizualizacja interpolacji
    plt.scatter(x_val, y_val, color='blue', label='Dane')
    plt.plot(x_interpolated, y_interpolated, color='red', label='Interpolacja hermita')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Interpolacja Hermite')
    plt.legend()
    plt.show()

    '''
    porownalem Interpolacje Hermite z Interpolacja B-Sklejanych, 
    B-sklejanych wydaje sie zbyt mocno dopasowana do tych danych przez co wyglada ona fajnie ale problem moze byc z faktyczna prognoza
    Hermite natomiast wydaje sie bardziej realnie prognozowac interpolacyjnie, ale ma problem na koncach przedzialow gdzie zupelnie traci zdolnoc prognozy

    '''



#zad2()
zad3()
zad4()
#zad5()
#zad6()