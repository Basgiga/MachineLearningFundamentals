import scipy.signal as sig
from scipy.signal import find_peaks
from statsmodels.tsa.stattools import acf
from statsmodels.robust import mad
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pywt
import pandas as pd
import tkinter as tk
from tkinter import ttk

np.random.seed(110)

data = pd.read_csv('CSV.csv', sep=',')

dff = pd.DataFrame(data)
dff = dff.interpolate(method='linear')
print(dff.head())

date_range = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
data = np.random.randn(len(date_range))

# sztucznie dorabiam brakujace wartosci zeby faktycznie moc je obsluzyc
ile=20
brak = np.random.choice(len(date_range), size=ile, replace=False)
data[brak] = np.nan


df = pd.DataFrame({'Date': date_range, 'Value': data})

df = df.interpolate(method='linear')  # interpolujemy brakujace wartosci

print("Pierwsze kilka wierszy danych:")
print(df.head())

print("\nTypy danych:")
print(df.dtypes)

def zad2():
    # Analiza szeregu czasowego
    plt.figure(figsize=(10, 6))
    plt.plot(dff['Date'], dff['Close'])
    plt.title('Wykres czasowy danych szeregu czasowego')
    plt.xlabel('Data')
    plt.ylabel('Wartość')
    plt.grid(True)
    plt.show()

    # Obliczamy podstawowe statystyki opisowe
    stats = dff['Close'].describe()
    print("\nPodstawowe statystyki opisowe:")
    print(stats)

def zad3():
    series = dff['Close']
    dates = pd.date_range(start = '2023-04-03', periods=249)
    ts = dff.set_index(dates)
    seasonality = ts['Close'].resample('M').mean() # Średnia wartość miesięczna
    trend = series.rolling(window=10).mean()  # Średnia ruchoma
    sequence_length = len(series)  # Długość sekwencji
    differences = series.diff()  # Różnice między kolejnymi wartościami
    variability = series.rolling(window=10).std()  # Odchylenie standardowe dla średniej ruchomej
    # Wyświetlenie wyników



    # Wyświetlenie wyników
    print("\n\nZad2\n\nŚrednia wartość miesięczna:")
    print(seasonality)
    print("\nŚrednia ruchoma:")
    print(trend)
    print("\nDługość sekwencji:", sequence_length)
    print("\nRóżnice między kolejnymi wartościami:")
    print(differences)
    print("\nOdchylenie standardowe dla średniej ruchomej:")
    print(variability)

    # Rysowanie wyników
    plt.figure(figsize=(12, 8))
    # Wykres oryginalnego szeregu czasowego
    plt.subplot(3, 2, 1)
    plt.plot(dff['Date'], dff['Close'])
    plt.title('Oryginalny szereg czasowy')
    plt.xlabel('Data')
    plt.ylabel('Wartość')

    # Wykres średniej wartości miesięcznej (sezonowość)
    plt.subplot(3, 2, 2)
    plt.plot(seasonality)
    plt.title('Średnia wartość miesięczna')
    plt.xlabel('Data')
    plt.ylabel('Średnia wartość')

    # Wykres średniej ruchomej (trend)
    plt.subplot(3, 2, 3)
    plt.plot(trend)
    plt.title('Średnia ruchoma')
    plt.xlabel('Data')
    plt.ylabel('Średnia wartość')

    # Wykres różnic między kolejnymi wartościami
    plt.subplot(3, 2, 4)
    plt.plot(differences)
    plt.title('Różnice między kolejnymi wartościami')
    plt.xlabel('Data')
    plt.ylabel('Różnica')

    # Wykres odchylenia standardowego dla średniej ruchomej (zmienność w czasie)
    plt.subplot(3, 2, 5)
    plt.plot(variability)

    plt.title('Odchylenie standardowe dla średniej ruchomej')
    plt.xlabel('Data')
    plt.ylabel('Odchylenie standardowe')
    plt.tight_layout()
    plt.show()



    '''
    # dorabiamy 2 kolumne zeby bylo z czym porownywac
    data2 = np.random.randn(len(date_range))  # Druga kolumna z losowymi wartościami

    df2 = pd.DataFrame({'Date': date_range, 'Value1': data, 'Value2': data2})

    df2 = df2.interpolate(method='linear')  # interpolujemy brakujace wartosci
    pca = PCA(n_components=2)
    transformed_data = pca.fit_transform(df2[['Value1', 'Value2']])
    # Wykres wyników PCA
    plt.figure(figsize=(8, 6))
    plt.scatter(transformed_data[:, 0], transformed_data[:, 1])
    plt.title('Wyniki analizy głównych składowych (PCA)')
    plt.xlabel('Składowa główna 1')
    plt.ylabel('Składowa główna 2')
    plt.grid(True)
    plt.show()
    '''
    pca = PCA(n_components=2)
    transformed_data = pca.fit_transform(dff[['Close', 'Open']])
    # Wykres wyników PCA
    plt.figure(figsize=(8, 6))
    plt.scatter(transformed_data[:, 0], transformed_data[:, 1])
    plt.title('Wyniki analizy głównych składowych (PCA)')
    plt.xlabel('Składowa główna 1')
    plt.ylabel('Składowa główna 2')
    plt.grid(True)
    plt.show()



    # Wyjaśniona wariancja
    explained_variance_ratio = pca.explained_variance_ratio_
    print("Wyjaśniona wariancja przez każdą z głównych składowych:", explained_variance_ratio)
    # Przykładowy szereg czasowy
    np.random.seed(0)
    ts = dff['Close']
    # Wykres oryginalnego szeregu czasowego
    plt.figure(figsize=(12, 6))
    plt.plot(ts)
    plt.title('Oryginalny szereg czasowy')
    plt.xlabel('Czas')
    plt.ylabel('Wartość')
    plt.grid(True)
    plt.show()
    # Dyskretna transformacja falkowa (DWT)
    # w przypadku bledu pywt zainstaluj pakiet PyWavelets poprzez Settings/Project/Python Interpreter/+
    def dwt_feature_extraction(data, wavelet='haar', level=1):
        coeffs = pywt.wavedec(data, wavelet, level=level)
        features = []
        for i in range(level + 1):
            features.extend(coeffs[i])
        return features

    # Ekstrakcja cech za pomocą DWT
    dwt_features = dwt_feature_extraction(ts)
    print("Cechy z dyskretnej transformacji falkowej:", dwt_features)
    # Wykres współczynników DWT
    plt.plot(dwt_features)
    plt.title('Współczynniki transformacji falkowej')
    plt.xlabel('Poziom dekompozycji')
    plt.ylabel('Współczynniki')
    plt.grid(True)
    plt.show()

    # Analiza falkowa
    def wavelet_analysis(data, wavelet='haar'):
        cA, cD = pywt.dwt(data, wavelet)
        return cA, cD

    # Ekstrakcja cech z analizy falkowej
    cA, cD = wavelet_analysis(ts)
    print("Wartości cech z analizy falkowej (cA):", cA)
    print("Wartości cech z analizy falkowej (cD):", cD)
    # Wykresy dla analizy falkowej
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(cA)
    plt.title('Wartości cech z analizy falkowej (cA)')
    plt.xlabel('Czas')
    plt.ylabel('Wartość')
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(cD)
    plt.title('Wartości cech z analizy falkowej (cD)')
    plt.xlabel('Czas')
    plt.ylabel('Wartość')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Autokorelacja
    def autocorrelation(data):
        return acf(data, fft=True)

    # Ekstrakcja cech za pomocą autokorelacji
    autocorr = autocorrelation(ts)
    print("Wartości cech z autokorelacji:", autocorr)
    # Wykres autokorelacji
    plt.plot(autocorr)
    plt.title('Autokorelacja')
    plt.xlabel('Opóźnienie')
    plt.ylabel('Wartość')
    plt.grid(True)
    plt.show()

    # Wykrywanie punktów ekstremalnych
    def find_extremes(data):
        peaks, _ = find_peaks(data)
        valleys, _ = find_peaks(-data)
        return peaks, valleys

    # Ekstrakcja cech za pomocą wykrywania punktów ekstremalnych
    peaks, valleys = find_extremes(ts)
    print("Lokalne maksima (peaks):", peaks)
    print("Lokalne minima (valleys):", valleys)
    # Wykres punktów ekstremalnych
    plt.figure(figsize=(12, 6))
    plt.plot(ts)
    plt.plot(peaks, ts[peaks], "x", label="Lokalne maksima", color='red')
    plt.plot(valleys, ts[valleys], "x", label="Lokalne minima", color='green')
    plt.title('Wykrywanie punktów ekstremalnych')
    plt.xlabel('Czas')
    plt.ylabel('Wartość')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Mediana absolutnego odchylenia (MAD)
    def median_absolute_deviation(data):
        return mad(data)

    # Ekstrakcja cech za pomocą mediany absolutnego odchylenia
    mad_value = median_absolute_deviation(ts)
    print("Wartość mediany absolutnego odchylenia (MAD):", mad_value)

def zad4(amplituda, frekwencja, czest_prob, t_trwania, szum_amp):
    def add_noise(signal, noise_amplitude):
        noise = np.random.normal(0, noise_amplitude, len(signal))
        return signal + noise

    def cfft(sygnal, fs):
        fft_result = np.fft.fft(sygnal)
        freqs = np.fft.fftfreq(len(sygnal), d=1 / fs)

        return freqs, fft_result

    def funkcja_calki(a, sygnal):
        t = 1 / czest_prob
        T = len(sygnal)

        s = np.sum([sygnal[i] * np.exp(-1j * 2 * np.pi * a * i * t) for i in range(T)])

        return t ** 2 / T * np.abs(s) ** 2

    # Przykładowe dane
    fs = czest_prob  #Częstotliwość próbkowania [Hz]
    # Przykładowy sygnał sinusoidalny
    t = np.linspace(0, t_trwania, int(czest_prob * t_trwania), endpoint=False)
    sygnal = amplituda * np.sin(2 * np.pi * frekwencja * t)
    sygnal = add_noise(sygnal,noise_amplitude=szum_amp)
    wgm = [funkcja_calki(a, sygnal) for a in range(czest_prob)]
    f = [f for f in range(czest_prob)]

    print("Długość wgm:", len(wgm))

    plt.figure()
    plt.plot(f, wgm)
    plt.xlabel('Częstotliwość [Hz]')
    plt.ylabel('Widmowa gęstość mocy')
    plt.title('Widmowa Gęstość Mocy')
    plt.grid(True)
    plt.show()

    _, fft_result = cfft(sygnal, czest_prob)
    # Wykres FFT
    plt.figure(figsize=(10, 6))
    plt.plot(t, np.abs(fft_result))
    plt.title('Transformata Fouriera')
    plt.xlabel('Częstotliwość [Hz]')
    plt.ylabel('Amplituda')
    plt.grid(True)
    plt.show()

def z4():
    root = tk.Tk()
    root.title("Manipulacja parametrami")

    label_amplituda = tk.Label(root, text='Amplituda:')
    label_amplituda.grid(row=0, column=0)
    slider_amplituda = tk.Scale(root, from_=0.1, to=2.0, orient=tk.HORIZONTAL, resolution=0.1)
    slider_amplituda.grid(row=0, column=1)

    label_frekwencja = tk.Label(root, text='Częstotliwość:')
    label_frekwencja.grid(row=1, column=0)
    slider_frekwencja = tk.Scale(root, from_=1.0, to=10.0, orient=tk.HORIZONTAL, resolution=0.5)
    slider_frekwencja.grid(row=1, column=1)

    label_czest_prob = tk.Label(root, text='Częstotliwość próbkowania:')
    label_czest_prob.grid(row=2, column=0)
    slider_czest_prob = tk.Scale(root, from_=50, to=200, orient=tk.HORIZONTAL, resolution=10)
    slider_czest_prob.grid(row=2, column=1)

    label_t_trwania = tk.Label(root, text='Czas trwania:')
    label_t_trwania.grid(row=3, column=0)
    slider_t_trwania = tk.Scale(root, from_=0.5, to=2.0, orient=tk.HORIZONTAL, resolution=0.1)
    slider_t_trwania.grid(row=3, column=1)

    label_szum = tk.Label(root, text='amplituda szumu:')
    label_szum.grid(row=4, column=0)
    slider_szum = tk.Scale(root, from_=0, to=5.0, orient=tk.HORIZONTAL, resolution=0.1)
    slider_szum.grid(row=4, column=1)

    button_generate = ttk.Button(root, text="Generuj sygnał",
                                 command=lambda: zad4(slider_amplituda.get(), slider_frekwencja.get(),
                                                      slider_czest_prob.get(), slider_t_trwania.get(),
                                                      slider_szum.get()))
    button_generate.grid(row=5, columnspan=2)
    root.mainloop()


#zad2()
#zad3()
#z4()