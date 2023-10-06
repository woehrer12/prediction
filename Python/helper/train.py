import pandas as pd
import numpy as np
import os
import multiprocessing
from rich import print
import random

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from tensorflow import keras
from tensorflow.keras.layers import Dense, LSTM, Dropout, Flatten, Reshape
from tensorflow.keras import Sequential
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras.regularizers import l2
from scikeras.wrappers import KerasRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score



import matplotlib.pyplot as plt


class CustomCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        # Verwende rich.print() anstelle von print() für eine schönere Ausgabe
        print(f"Epoch {epoch + 1}/{self.params['epochs']} - loss: {logs['loss']:.4f} - accuracy: {logs['accuracy']:.4f}")


def split_in_batches(df):

    # Anzahl der Datensätze im DataFrame
    total_records = len(df)

    # Größe jedes Pakets
    batch_size = 1000

    # Liste zum Speichern der aufgeteilten Pakete
    data_batches = []

    # Mische die Daten
    # df = df.sample(frac=1, random_state=42)  # Shuffle mit zufälligem Seed für Reproduzierbarkeit

    # Schleife zum Aufteilen des DataFrames in Pakete
    for i in range(0, total_records, batch_size):
        data_batch = df[i:i+batch_size]
        data_batches.append(data_batch)

    # Mische die aufgeteilten Batches zufällig
    random.shuffle(data_batches)

    
    return data_batches

def create_model1(X_train):
    
    # Definieren des Modells
    model = Sequential()

    model.add(Reshape((X_train.shape[1], 1), input_shape=(X_train.shape[1],)))
    model.add(LSTM(128, return_sequences=True))  # LSTM-Schicht mit 64 Neuronen und Rückgabe der Sequenzen
    model.add(Dropout(0.2))  # Dropout-Schicht mit 20% Dropout-Rate
    model.add(LSTM(128, return_sequences=True))  # LSTM-Schicht mit 64 Neuronen und Rückgabe der Sequenzen
    model.add(Dropout(0.2))  # Dropout-Schicht mit 20% Dropout-Rate
    model.add(LSTM(64, return_sequences=True))  # LSTM-Schicht mit 32 Neuronen
    model.add(Dropout(0.2))  # Dropout-Schicht mit 20% Dropout-Rate
    model.add(LSTM(64))  # LSTM-Schicht mit 32 Neuronen
    model.add(Dense(1, activation='linear', kernel_regularizer=l2(0.01)))


    # Kompilieren des Modells
    # optimizer = Adam(lr=0.001)
    # model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

    # Modellübersicht anzeigen
    # model.summary()

    return model

def plot(CurrencyPair, round_number, Y_train, Y_pred, Y_test, test_acc, test_loss):

    # Vorhersage in die richtige Reihenfolge bringen
    Y_test = Y_test.sort_index()
    # Y_test = Y_test.reset_index(drop=True)

    # Setze den Index von Y_train auf 0
    Y_train.index = range(len(Y_train))

    # Setze den Index von Y_test dahinter
    Y_test.index = range(len(Y_train), len(Y_train) + len(Y_test))
    

    # Berechne die Metriken
    mae = mean_absolute_error(Y_test, Y_pred)
    mse = mean_squared_error(Y_test, Y_pred)
    r2 = r2_score(Y_test, Y_pred)

    # Ausgabe der Metriken
    print("Mean Absolute Error (MAE):", mae)
    print("Mean Squared Error (MSE):", mse)
    print("R-squared (R2) Wert:", r2)

    # Plot der Vorhersagen
    plt.figure(figsize=(12, 6))
    plt.plot(Y_test, label='True')
    plt.plot(range(len(Y_train), len(Y_train) + len(Y_pred)), Y_pred, label='Predicted')
    plt.plot(Y_train, label='Train')
    plt.title('Vorhersagen des Modells')
    plt.xlabel('Index')
    plt.ylabel('Prediction')
    plt.legend()

    # Print the test accuracy on the plot
    plt.text(0.5, 0.95, f'Test accuracy: {test_acc}', transform=plt.gca().transAxes, 
            fontsize=12, va='top', ha='left')

    # Print the test loss on the plot
    plt.text(0.5, 0.9, f'Test loss: {test_loss}', transform=plt.gca().transAxes, 
            fontsize=12, va='top', ha='left')
    
    # Print MAE on the plot
    plt.text(0.5, 0.85, f'MAE: {mae}', transform=plt.gca().transAxes, 
            fontsize=12, va='top', ha='left')

    # Print MSE on the plot
    plt.text(0.5, 0.8, f'MSE: {mse}', transform=plt.gca().transAxes, 
            fontsize=12, va='top', ha='left')

    # Print R2 on the plot
    plt.text(0.5, 0.75, f'R2: {r2}', transform=plt.gca().transAxes, 
            fontsize=12, va='top', ha='left')

    # Speichern des Plots in einer Datei
    plt.savefig('./KI/Predict/CurrencyPair_{}/prediction_plot_{}.png'.format(CurrencyPair, round_number))


def train(CurrencyPair):

    if not os.path.exists('./KI/Predict/CurrencyPair_{}'.format(CurrencyPair)):
        os.makedirs('./KI/Predict/CurrencyPair_{}'.format(CurrencyPair))

    df = pd.read_csv('./KI/Data/CurrencyPair_{}/Sorted.csv'.format(CurrencyPair))

    ## Split Batches

    data_batches = split_in_batches(df)

    ## Model creation

    # Einmaliges einlesen der Daten um die Form bekannt zu machen
    X = df.drop("Prediction", axis=1)  # Features sind alle Spalten außer "Weighted_Price"
    Y = df["Prediction"]  # Ziel ist die Spalte "Weighted_Price"

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=42, train_size=0.8, test_size=0.2, shuffle=False)

    model_path = "./KI/Predict/CurrencyPair_{}/model.h5".format(CurrencyPair)

    if os.path.exists(model_path):
        model = keras.models.load_model(model_path)
        print("Das Modell wurde geladen.")
    else:
        print("Das Modell existiert nicht und muss trainiert oder erstellt werden.")
        model = create_model1(X_train)

    # model = find_best_model(X_train, Y_train)

    num_cpus = multiprocessing.cpu_count() - 2 
    print("Anzahl der zu verwendenden Prozessoren: " + str(num_cpus))

    round_number = 1

    for data in data_batches:

        print("Round number: ", round_number, "/", len(data_batches))

        # Aufteilung in Features (X) und Ziel (Y)
        X = data.drop("Prediction", axis=1)  # Features sind alle Spalten außer "Prediction"
        Y = data["Prediction"]  # Ziel ist die Spalte "Weighted_Price"

        volatilität_prozent = (Y.std() / Y.mean()) * 100

        print("Volatilität in Prozent:", volatilität_prozent)

        if volatilität_prozent > 3.0:

            # Feature Scaling - Normalisierung der Daten
            scaler = MinMaxScaler()
            X = scaler.fit_transform(X)

            # Aufteilung in Trainings- und Testdatensätze
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=0, train_size=0.8, test_size=0.2, shuffle= False)

            # EarlyStopping
            custom_early_stopping = EarlyStopping(
                monitor='loss', 
                patience=1000, 
                min_delta=1, 
                mode='min'
            )

            print("Start fit")

            # fit the model to the training data
            model.fit(  X_train, 
                        Y_train, 
                        epochs=1000, 
                        batch_size=1024, 
                        callbacks=[custom_early_stopping], 
                        shuffle=False, 
                        verbose=1,
                        workers=num_cpus,  # Verwenden Sie die Anzahl der verfügbaren CPUs
                        use_multiprocessing=True  # Aktivieren Sie die multiprocessing-Unterstützung)
                    )

            # Evaluate the model using test data
            test_loss, test_acc = model.evaluate(X_test, Y_test)

            # Print the test accuracy
            print('Test accuracy:', test_acc)
            print('Test loss:', test_loss)

            # Vorhersagen auf den Testdaten
            Y_pred = model.predict(X_test)

            plot(CurrencyPair, round_number, Y_train, Y_pred, Y_test, test_acc, test_loss)

            model.save("./KI/Predict/CurrencyPair_{}/model.h5".format(CurrencyPair))

        round_number = round_number + 1
