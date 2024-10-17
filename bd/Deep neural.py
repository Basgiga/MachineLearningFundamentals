from keras.datasets import fashion_mnist
import numpy as np
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
import fiftyone.zoo as foz
import os
from PIL import Image
import cv2
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate


def calculate_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp)
    return accuracy, precision, recall, specificity



def zad2():
    x = load_breast_cancer().data
    y = load_breast_cancer().target

    x_train, x_test, y_train, y_test = train_test_split(x,y, test_size= 0.3, random_state= 42)

    """
    # Sprawdzenie kształtu danych
    print("Kształt danych treningowych (obrazów):", x.shape)
    print("Kształt danych treningowych (etykiet):", y.shape)
    print("Kształt danych testowych (obrazów):", x.shape)
    print("Kształt danych testowych (etykiet):", y.shape)
    """
    # Tworzenie instancji modelu sekwencyjnego
    model = Sequential([
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    # Kompilacja modelu
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    # Trenowanie modelu
    model.fit(x_train, y_train, epochs=10, batch_size=32, verbose = 1)
    # Ocena na zbiorze testowym
    loss, accuracy = model.evaluate(x_test, y_test)
    print(f'Loss: {loss}, Accuracy: {accuracy}')

    y_pred = np.round(model.predict(x_test))
    #print(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    print("Macierz pomyłek:")
    print(conf_matrix)



    accuracy, precision, recall, specificity = calculate_metrics(y_test, y_pred)

    print(f'Accuracy: {accuracy}')
    print(f'Precision: {precision}')
    print(f'Recall (Sensitivity): {recall}')
    print(f'Specificity: {specificity}')

    Cn = ConfusionMatrixDisplay(conf_matrix)
    Cn.plot()
    plt.show()

#zad2()

def zad3():
    x = load_iris().data
    y = load_iris().target

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
    y_train = to_categorical(y_train, 3)
    y_test = to_categorical(y_test, 3)

    #print(x_train,y_train)
    """
    # Sprawdzenie kształtu danych
    print("Kształt danych treningowych (obrazów):", x.shape)
    print("Kształt danych treningowych (etykiet):", y.shape)
    print("Kształt danych testowych (obrazów):", x.shape)
    print("Kształt danych testowych (etykiet):", y.shape)
    """
    # Tworzenie instancji modelu sekwencyjnego
    model = Sequential([
        Dense(64, activation='relu'),
        Dense(3, activation='softmax')
    ])
    # Kompilacja modelu
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Trenowanie modelu
    history = model.fit(x_train, y_train, epochs=50, batch_size=30, validation_split=0.2)

    y_pred = np.round(model.predict(x_test))
    # Wizualizacja krzywych uczenia
    plt.plot(history.history['loss'], label='Funkcja straty (trening)')
    plt.plot(history.history['val_loss'], label='Funkcja straty (walidacja)')
    plt.xlabel('Liczba epok')
    plt.ylabel('Wartość funkcji straty')
    plt.legend()
    plt.title('Krzywa funkcji straty')
    plt.show()
    plt.plot(history.history['accuracy'], label='Dokładność (trening)')
    plt.plot(history.history['val_accuracy'], label='Dokładność (walidacja)')
    plt.xlabel('Liczba epok')
    plt.ylabel('Dokładność')
    plt.legend()
    plt.title('Krzywa dokładności')
    plt.show()

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')

    print(f'Accuracy: {accuracy}')
    print(f'Precision: {precision}')
    print(f'Recall (Sensitivity): {recall}')
#zad3()


def zad4():
    # Syntetyczne dane
    num_classes = 5  # Liczba klas do klasyfikacji (dla przykładu)
    num_samples = 100  # Liczba próbek w każdej klasie
    img_size = 224  # Rozmiar obrazów

    # Generowanie syntetycznych danych (losowe obrazy)
    def generate_synthetic_data(num_samples, img_size, num_classes):
        X = np.random.rand(num_samples, img_size, img_size, 3) * 255  # Losowe obrazy
        y = np.random.randint(0, num_classes, num_samples)  # Losowe etykiety
        y = tf.keras.utils.to_categorical(y, num_classes)  # One-hot encoding etykiet
        return X, y

    X_train, y_train = generate_synthetic_data(num_samples * num_classes, img_size, num_classes)
    X_test, y_test = generate_synthetic_data(num_samples * num_classes, img_size, num_classes)
    X_train = tf.keras.applications.vgg16.preprocess_input(X_train)
    X_test = tf.keras.applications.vgg16.preprocess_input(X_test)

    # Załadowanie pretrenowanego modelu VGG16
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(img_size, img_size, 3))

    # Zamrożenie warstw konwolucyjnych
    for layer in base_model.layers:
        layer.trainable = False

    # Dodanie nowych warstw klasyfikacyjnych
    x = Flatten()(base_model.output)
    x = Dense(512, activation='relu')(x)
    x = Dense(128, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=6,
        batch_size=32
    )

    # Ocena modelu
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f'Test celnosc: {test_acc:.2f}')
    Y_pred = model.predict(X_test)
    y_pred = np.argmax(Y_pred, axis=1)
    y_true = np.argmax(y_test, axis=1)
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[f'Class {i}' for i in range(num_classes)])
    disp.plot(cmap=plt.cm.Blues)
    plt.show()

#zad4()

def zad5():
    dataset = foz.load_zoo_dataset("coco-2017")
    images_dir = dataset.default_classes_dir
    labels_dir = dataset.default_labels_dir

    # Przetwarzanie danych
    images = []
    labels = []

    # Przeglądaj obrazy i etykiety
    for filename in os.listdir(images_dir):
        if filename.endswith(".png"):
            image_path = os.path.join(images_dir, filename)
            label_path = os.path.join(labels_dir, filename.split('.')[0] + '_P.png')  # Odpowiednia etykieta pikseli

            # Wczytaj obraz i etykietę
            image = np.array(Image.open(image_path))
            label = np.array(Image.open(label_path))

            images.append(image)
            labels.append(label)

    # Konwertuj listy na tablice numpy
    images = np.array(images)
    labels = np.array(labels)

    # Podziel dane na zbiór treningowy i testowy
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

    # Definicja architektury modelu
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(10, activation='softmax')  # 10 klas dla przykładu
    ])

    # Kompilacja modelu
    model.compile(optimizer=Adam(),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Trenowanie modelu
    history = model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test))

    # Ocena modelu na zbiorze testowym
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f'celnosc testowa: {test_acc:.2f}')

    # Wizualizacja wyników za pomocą krzywej ROC
    y_pred_proba = model.predict(X_test)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(10):  # Dla każdej klasy
        fpr[i], tpr[i], _ = roc_curve(y_test == i, y_pred_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure(figsize=(8, 6))
    for i in range(10):  # Dla każdej klasy
        plt.plot(fpr[i], tpr[i], label=f'Class {i} (AUC = {roc_auc[i]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('FP')
    plt.ylabel('TP')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.show()

#zad5()

def zad6():
    def load_data(image_folder, img_height, img_width):
        images = []
        for filename in os.listdir(image_folder):
            if filename.endswith(".png") or filename.endswith(".jpg"):
                img_path = os.path.join(image_folder, filename)
                img = Image.open(img_path).resize((img_width, img_height))
                img = np.array(img)
                images.append(img)
        images = np.array(images)
        return images

    # Funkcja do załadowania etykiet z pliku tekstowego
    def load_labels_from_txt(filepath):
        labels = {}
        with open(filepath, 'r') as file:
            for line in file:
                values = line.strip().split()
                rgb = tuple(map(int, values[:3]))
                label = values[3]
                labels[rgb] = label
        return labels

    # Funkcja do konwersji maski kolorowej na maskę z wartościami etykiet
    def convert_rgb_to_labels(image, label_dict):
        label_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        for rgb, label in label_dict.items():
            mask = np.all(image == rgb, axis=-1)
            label_mask[mask] = list(label_dict.keys()).index(rgb)
        return label_mask

    # Ścieżki do danych
    image_folder = 'C:/szkola/strdanych/LabeledApproved_full'
    label_file = 'C:/szkola/strdanych/cmvidlabel.txt'

    # Rozmiar obrazów do przeskalowania
    IMG_HEIGHT, IMG_WIDTH = 256, 256

    # Załaduj etykiety
    labels = load_labels_from_txt(label_file)

    # Załaduj dane
    images = load_data(image_folder, IMG_HEIGHT, IMG_WIDTH)

    # Normalizacja danych
    images = images / 255.0
    masks = np.array([convert_rgb_to_labels(img, labels) for img in images])
    masks = np.expand_dims(masks, axis=-1)

    X_train, X_test, y_train, y_test = train_test_split(images, masks, test_size=0.2, random_state=42)

    def definicja_modelu(img_height, img_width, num_classes):
        inputs = Input((img_height, img_width, 3))

        conv1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
        conv1 = Conv2D(64, 3, activation='relu', padding='same')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = Conv2D(128, 3, activation='relu', padding='same')(pool1)
        conv2 = Conv2D(128, 3, activation='relu', padding='same')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = Conv2D(256, 3, activation='relu', padding='same')(pool2)
        conv3 = Conv2D(256, 3, activation='relu', padding='same')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = Conv2D(512, 3, activation='relu', padding='same')(pool3)
        conv4 = Conv2D(512, 3, activation='relu', padding='same')(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

        conv5 = Conv2D(1024, 3, activation='relu', padding='same')(pool4)
        conv5 = Conv2D(1024, 3, activation='relu', padding='same')(conv5)

        up6 = UpSampling2D(size=(2, 2))(conv5)
        up6 = concatenate([up6, conv4])
        conv6 = Conv2D(512, 3, activation='relu', padding='same')(up6)
        conv6 = Conv2D(512, 3, activation='relu', padding='same')(conv6)

        up7 = UpSampling2D(size=(2, 2))(conv6)
        up7 = concatenate([up7, conv3])
        conv7 = Conv2D(256, 3, activation='relu', padding='same')(up7)
        conv7 = Conv2D(256, 3, activation='relu', padding='same')(conv7)

        up8 = UpSampling2D(size=(2, 2))(conv7)
        up8 = concatenate([up8, conv2])
        conv8 = Conv2D(128, 3, activation='relu', padding='same')(up8)
        conv8 = Conv2D(128, 3, activation='relu', padding='same')(conv8)

        up9 = UpSampling2D(size=(2, 2))(conv8)
        up9 = concatenate([up9, conv1])
        conv9 = Conv2D(64, 3, activation='relu', padding='same')(up9)
        conv9 = Conv2D(64, 3, activation='relu', padding='same')(conv9)

        outputs = Conv2D(num_classes, 1, activation='softmax')(conv9)

        model = Model(inputs=[inputs], outputs=[outputs])
        return model

    num_classes = len(labels)
    model = definicja_modelu(IMG_HEIGHT, IMG_WIDTH, num_classes)
    model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    # Trenowanie modelu
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=1, batch_size=16)

    def iou_metric(y_true, y_pred):
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()
        intersection = np.sum(y_true * y_pred)
        union = np.sum(y_true) + np.sum(y_pred) - intersection
        return intersection / union

    y_pred = model.predict(X_test)
    y_pred = np.argmax(y_pred, axis=-1)

    ious = []
    for i in range(len(y_test)):
        iou = iou_metric(y_test[i], y_pred[i])
        ious.append(iou)

    mean_iou = np.mean(ious)
    print(f' IoU: {mean_iou}')

    def display_sample_predictions(X, y_true, y_pred, sample_index=0):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(X[sample_index])
        axes[0].set_title('Original Image')

        axes[1].imshow(y_true[sample_index].squeeze(), cmap='gray')
        axes[1].set_title('Ground Truth')

        axes[2].imshow(y_pred[sample_index], cmap='gray')
        axes[2].set_title('Predicted Mask')

        plt.show()

    for i in range(3):
        display_sample_predictions(X_test, y_test, y_pred, sample_index=i)


    '''
    ten model niestety ze wzgledu na wielkosc pliku od Camvid liczyl mi sie kolo godziny, ale osiagnal celnosc na poziomie 40,
    przy tylko 1 epoc, co uznaje za caliem niezly wynik,
    
    niestety ale loss przez dlugi czas uczenia tez rosl, prawdopodobnie gdyby ostawic go na dzien z 10 epokami bylby to solidny model rozpoznawania klasyfikowania etykiet
    '''

zad2()
zad3()
#zad4()
#zad5()
#zad6()