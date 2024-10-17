from keras.datasets import fashion_mnist
import numpy as np
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report


def zad2():
    # Wczytanie danych
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    # Sprawdzenie kształtu danych
    print("Kształt danych treningowych (obrazów):", train_images.shape)
    print("Kształt danych treningowych (etykiet):", train_labels.shape)
    print("Kształt danych testowych (obrazów):", test_images.shape)
    print("Kształt danych testowych (etykiet):", test_labels.shape)

    return(train_images,train_labels, test_images, test_labels)

zad2()

def zad3():
    tri, trl , tei, tel = zad2()
    # Normalizacja wartości pikseli do zakresu od 0 do 1
    train_images = tri / 255.0
    test_images = tei / 255.0

    # Przekształcenie etykiet kategorii na postać one-hot encoding
    num_classes = 10  # Fashion MNIST zawiera 10 kategorii
    train_labels_one_hot = to_categorical(trl, num_classes)
    test_labels_one_hot = to_categorical(tel, num_classes)

    # Wyświetlenie przykładowej etykiety po przekształceniu
    print("Oryginalna etykieta:", trl[0])
    print("Etykieta po one-hot encoding:", train_labels_one_hot[0])
    return(train_images, train_labels_one_hot, test_images, test_labels_one_hot)
def zad4():
    # Tworzenie instancji modelu sekwencyjnego
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
    ])


    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Wyświetlenie podsumowania architektury modelu
    model.summary()

    return(model)

def zad5():
    model = zad4()
    train_images, train_labels_one_hot, test_images, test_labels_one_hot = zad3()
    # Trenowanie modelu
    history = model.fit(train_images, train_labels_one_hot, epochs=10,
                        validation_data=(test_images, test_labels_one_hot))

    # Wydrukowanie historii trenowania
    print(history.history)
    return(history)
#zad5()

def zad6():
    model = zad4()
    train_images, train_labels_one_hot, test_images, test_labels_one_hot = zad3()
    history = zad5()
    # Ewaluacja modelu na danych testowych
    test_loss, test_accuracy = model.evaluate(test_images, test_labels_one_hot)
    print(f"Test Loss: {test_loss}")
    print(f"Test Accuracy: {test_accuracy}")

    # Wizualizacja historii trenowania
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
#zad6()

def zad7():
    model = zad4()
    train_images, train_labels_one_hot, test_images, test_labels_one_hot = zad3()

    history = model.fit(train_images, train_labels_one_hot, epochs=30, # 5 12 10
                        validation_data=(test_images, test_labels_one_hot))

    # Przeprowadzenie predykcji na danych testowych
    predictions = model.predict(test_images)
    predicted_labels = np.argmax(predictions, axis=1)

    # Ewaluacja modelu na danych testowych
    test_loss, test_accuracy = model.evaluate(test_images, test_labels_one_hot)
    print(f"Test Loss: {test_loss}")
    print(f"Test Accuracy: {test_accuracy}")

    # Tworzenie macierzy pomyłek
    conf_matrix = confusion_matrix(np.argmax(test_labels_one_hot, axis=1), predicted_labels)
    print("Confusion Matrix:")
    print(conf_matrix)

    # Generowanie raportu klasyfikacji
    class_report = classification_report(np.argmax(test_labels_one_hot, axis=1), predicted_labels)
    print("Classification Report:")
    print(class_report)

zad7()


"""
zad8:
przez dlugi czas predykowalem a potem trenowalem model co sprawialo ze macierz pomylek wypluwala dokladnosc na poziomie okolo 0.1 co mnie bardzo dziwilo
ale gdy juz udalo sie zaradzic sprobowalem rozne ilosci epok, pierw 5 gdzie dostalem dokladosc mniej wiecej na poziomie 0.88 i jedynie problemy ze znajdywaniem klasy 7,
potem sprobowalem z 12 epokami, znow problem z klasa 7 ale dokladnosc na poziomie 0.91
mozliwe ze to problem z przetrenowaniem modelu, wiec sprawdzmy 10 epok
dalej problem z klasa numer 7 a dokladnosc na identycznym poziomie 0.91

to zrobmy w druga strone, 30 epok
dokladnosc 0.92 i o wiele mniejszy problem z klasa numer 7

wiec mozliwe ze gdybysmy zostawili to na noc na jakies 1000 epok wyszedl by dobry model, albo bardzo przetrenowany model, 30 wydaje sie calkiem logiczna liczba na limitowany czas i moc obliczeniowa.
ewentualnie mozna zmienic cos w sieci neuronowej by lepiej klasyfikowala klase 7

"""



