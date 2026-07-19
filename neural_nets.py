# CUSTOMER CONVERSION ENGINE - NEURAL NETWORK MODULE
# SUB-MODULE A

import os
import numpy as np
import pandas as pd

import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD, Adam, RMSprop

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# 1. LOAD DATASET + FEATURE ENGINEERING

def load_and_prepare_data(file_path):

    df = pd.read_csv(
        file_path,
        encoding="latin1"
    )


    print("Dataset Loaded Successfully")

    print(df.head())

    # DATA CLEANING

    df = df.dropna(
        subset=["CustomerID"]
    )


    df = df[
        (df["Quantity"] > 0)
        &
        (df["UnitPrice"] > 0)
    ]


    print(
        "Cleaned Shape:",
        df.shape
    )

    # FEATURE ENGINEERING


    df["TotalAmount"] = (

        df["Quantity"]

        *

        df["UnitPrice"]

    )



    customer_data = df.groupby(
        "CustomerID"
    ).agg({

        "InvoiceNo":"nunique",

        "Quantity":"sum",

        "TotalAmount":"sum"

    }).reset_index()



    customer_data.columns = [

        "CustomerID",

        "TotalOrders",

        "TotalQuantity",

        "TotalSpending"

    ]


    # TARGET CREATION

    median_spending = (

        customer_data["TotalSpending"]

        .median()

    )


    customer_data["HighValueCustomer"] = (

        customer_data["TotalSpending"]

        >

        median_spending

    ).astype(int)



    X = customer_data[

        [

        "TotalOrders",

        "TotalQuantity",

        "TotalSpending"

        ]

    ]


    y = customer_data[

        "HighValueCustomer"

    ]



    return customer_data, X, y

# 2. PREPROCESS DATA


def preprocess_data(X,y):


    X_train, X_test, y_train, y_test = train_test_split(

        X,

        y,

        test_size=0.2,

        random_state=42

    )



    scaler = StandardScaler()



    X_train = scaler.fit_transform(

        X_train

    )


    X_test = scaler.transform(

        X_test

    )



    return (

        X_train,

        X_test,

        y_train,

        y_test,

        scaler

    )

# 3. MANUAL PERCEPTRON

def train_perceptron(

        X_train,

        X_test,

        y_train,

        y_test

):


    weights = np.zeros(

        X_train.shape[1]

    )


    bias = 0



    # Forward + Weight Update


    for x,target in zip(

        X_train,

        y_train

    ):


        output = np.dot(

            weights,

            x

        ) + bias



        prediction = (

            1

            if output >=0

            else 0

        )


        error = target - prediction



        weights += error*x


        bias += error




    predictions=[]



    for x in X_test:


        output = (

            np.dot(

                weights,

                x

            )

            +

            bias

        )


        predictions.append(

            1 if output>=0 else 0

        )



    accuracy = accuracy_score(

        y_test,

        predictions

    )


    return {

        "accuracy":accuracy,

        "predictions":predictions

    }


# 4. OPTIMIZER SELECTION

def select_optimizer(name):


    if name=="SGD":


        return SGD(

            learning_rate=0.01

        )


    elif name=="Adam":


        return Adam(

            learning_rate=0.001

        )


    else:


        return RMSprop(

            learning_rate=0.001

        )



# 5. BUILD MLP MODEL


def build_mlp(

        activation,

        optimizer_name

):


    model = Sequential()


    # Forward Propagation

    model.add(

        Dense(

            64,

            activation=activation,

            input_shape=(3,)

        )

    )



    model.add(

        Dense(

            32,

            activation=activation

        )

    )



    model.add(

        Dense(

            1,

            activation="sigmoid"

        )

    )



    optimizer = select_optimizer(

        optimizer_name

    )



    model.compile(

        optimizer=optimizer,

        loss="binary_crossentropy",

        metrics=[

            "accuracy"

        ]

    )



    return model

# 6. TRAIN MLP

def train_mlp(

        model,

        X_train,

        X_test,

        y_train,

        y_test,

        epochs

):


    history = model.fit(

        X_train,

        y_train,


        validation_data=(

            X_test,

            y_test

        ),


        epochs=epochs,


        verbose=0

    )



    return history


# 7. EVALUATION

def evaluate_model(

        model,

        X_test,

        y_test

):


    probability = model.predict(

        X_test

    )



    predictions = (

        probability > 0.5

    ).astype(int)



    accuracy = accuracy_score(

        y_test,

        predictions

    )



    report = classification_report(

        y_test,

        predictions

    )


    matrix = confusion_matrix(

        y_test,

        predictions

    )



    return (

        accuracy,

        report,

        matrix,

        predictions

    )


# 8. SAVE MODEL

def save_model(model):


    os.makedirs(

        "models",

        exist_ok=True

    )


    model.save(

        "models/conversion_model.h5"

    )



    print(

        "Model saved successfully"

    )