import pandas as pd
import numpy as np
import os

from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from tensorflow.keras.layers import Dense, LSTM, Dropout, Flatten, Reshape
from tensorflow.keras import Sequential
from statsmodels.graphics.tsaplots import plot_acf
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras.regularizers import l2

import matplotlib.pyplot as plt



def train(CurrencyPair):

    if not os.path.exists('./KI/Predict/CurrencyPair_{}'.format(CurrencyPair)):
        os.makedirs('./KI/Predict/CurrencyPair_{}'.format(CurrencyPair))

    df = pd.read_csv('./KI/Data/CurrencyPair_{}/Sorted.csv'.format(CurrencyPair))

    # Teile den df in Pakete zu je 1000 Datensätze auf

    # Anzahl der Datensätze im DataFrame
    total_records = len(df)

    # Größe jedes Pakets
    batch_size = 1000

    # Liste zum Speichern der aufgeteilten Pakete
    data_batches = []

    # Mische die Daten
    df = df.sample(frac=1, random_state=42)  # Shuffle mit zufälligem Seed für Reproduzierbarkeit

    # Schleife zum Aufteilen des DataFrames in Pakete
    for i in range(0, total_records, batch_size):
        data_batch = df[i:i+batch_size]
        data_batches.append(data_batch)

    ## Model creation

    # Einmaliges einlesen der Daten um die Form bekannt zu machen
    X = df.drop("Prediction", axis=1)  # Features sind alle Spalten außer "Weighted_Price"
    Y = df["Prediction"]  # Ziel ist die Spalte "Weighted_Price"

    # Splitten in Train und Test
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=42, train_size=0.8, test_size=0.2)

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
    model.add(Dense(1, activation='sigmoid', kernel_regularizer=l2(0.01)))


    # Kompilieren des Modells
    optimizer = Adam(lr=0.001)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    # model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

    # Modellübersicht anzeigen
    model.summary()

    round_number = 1

    for data in data_batches:

        print("Round number: ", round_number, "/", len(data_batches))

        # Aufteilung in Features (X) und Ziel (Y)
        X = data.drop("Prediction", axis=1)  # Features sind alle Spalten außer "Weighted_Price"
        Y = data["Prediction"]  # Ziel ist die Spalte "Weighted_Price"

        # Feature Scaling - Normalisierung der Daten
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)

        # Aufteilung in Trainings- und Testdatensätze
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=42, train_size=0.8, test_size=0.2)

        # EarlyStopping
        custom_early_stopping = EarlyStopping(
            monitor='accuracy', 
            patience=100, 
            min_delta=0.001, 
            mode='max'
        )

        print(X_train.shape)
        print(X_test.shape)

        # fit the model to the training data
        model.fit(X_train, Y_train, epochs=500, batch_size=64, callbacks=[custom_early_stopping], shuffle=True)

        # Evaluate the model using test data
        test_loss, test_acc = model.evaluate(X_test, Y_test)

        # Print the test accuracy
        print('Test accuracy:', test_acc)
        print('Test loss:', test_loss)

        # Vorhersagen auf den Testdaten
        Y_pred = model.predict(X_test)

        # Plot der Vorhersagen
        plt.figure(figsize=(12, 6))
        plt.plot(Y_test, label='True')
        plt.plot(Y_pred, label='Predicted')
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

        # Speichern des Plots in einer Datei
        plt.savefig('./KI/Predict/CurrencyPair_{}/prediction_plot_{}.png'.format(CurrencyPair, round_number))

        model.save("./KI/Predict/CurrencyPair_{}/model.h5".format(CurrencyPair))

        round_number = round_number + 1
