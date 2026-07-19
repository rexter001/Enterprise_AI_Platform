# CUSTOMER CONVERSION ENGINE
# STREAMLIT DASHBOARD

import streamlit as st
import matplotlib.pyplot as plt

from neural_nets import (
    load_and_prepare_data,
    preprocess_data,
    train_perceptron,
    build_mlp,
    train_mlp,
    evaluate_model,
    save_model
)

# PAGE CONFIG

st.set_page_config(

    page_title="Customer Conversion Engine",

    layout="wide"

)


st.title(
    " Customer Conversion Engine"
)


st.write(
"""
Neural Network based customer purchasing action classifier.

Includes:
1. Perceptron
2. Multi Layer Perceptron
3. Forward Propagation
4. Backpropagation
5. Activation Selection
6. Optimizer Selection
"""
)


# LOAD DATA

customer_data, X, y = load_and_prepare_data(

    "../datasets/online_retail.csv"

)



st.subheader(
"Customer Feature Data"
)


st.dataframe(

    customer_data.head()

)

# PREPROCESS

X_train, X_test, y_train, y_test, scaler = preprocess_data(

    X,

    y

)


# SIDEBAR CONFIGURATION

st.sidebar.header(

"Neural Network Configuration"

)



activation = st.sidebar.selectbox(

    "Select Activation Function",

    [

        "relu",

        "sigmoid",

        "tanh"

    ]

)



optimizer = st.sidebar.selectbox(

    "Select Optimizer",

    [

        "SGD",

        "Adam",

        "RMSProp"

    ]

)



epochs = st.sidebar.slider(

    "Training Epochs",

    min_value=10,

    max_value=200,

    value=50

)

# PERCEPTRON

st.header(
"1. Perceptron Model"
)



perceptron_result = train_perceptron(

    X_train,

    X_test,

    y_train,

    y_test

)



st.success(

f"Perceptron Accuracy : {perceptron_result['accuracy']:.4f}"

)


# MLP MODEL

st.header(

"2. Multi Layer Perceptron"

)



model = build_mlp(

    activation,

    optimizer

)



st.info(

f"""
Selected Activation : {activation}

Selected Optimizer : {optimizer}

"""

)



# TRAIN BUTTON

if st.button(

    "Train Neural Network"

):


    with st.spinner(

        "Training Neural Network..."

    ):


        history = train_mlp(

            model,

            X_train,

            X_test,

            y_train,

            y_test,

            epochs

        )



        accuracy, report, matrix, predictions = evaluate_model(

            model,

            X_test,

            y_test

        )



        save_model(

            model

        )



        st.session_state["history"] = history

        st.session_state["accuracy"] = accuracy

        st.session_state["report"] = report

        st.session_state["matrix"] = matrix



        st.success(

            "Training Completed Successfully"

        )


# RESULTS

if "accuracy" in st.session_state:


    st.header(

        "Model Performance"

    )


    col1,col2 = st.columns(2)



    with col1:

        st.metric(

            "MLP Accuracy",

            round(

                st.session_state["accuracy"],

                4

            )

        )



    with col2:

        st.metric(

            "Perceptron Accuracy",

            round(

                perceptron_result["accuracy"],

                4

            )

        )





    st.subheader(

        "Classification Report"

    )


    st.text(

        st.session_state["report"]

    )



    st.subheader(

        "Confusion Matrix"

    )


    st.write(

        st.session_state["matrix"]

    )


# TELEMETRY

if "history" in st.session_state:


    st.header(

        " Training Convergence Telemetry"

    )


    history = st.session_state["history"]



    col1,col2 = st.columns(2)



    with col1:


        fig1,ax1 = plt.subplots()


        ax1.plot(

            history.history["accuracy"]

        )


        ax1.set_title(

            "Accuracy Curve"

        )


        ax1.set_xlabel(

            "Epoch"

        )


        ax1.set_ylabel(

            "Accuracy"

        )


        st.pyplot(fig1)




    with col2:


        fig2,ax2 = plt.subplots()


        ax2.plot(

            history.history["loss"]

        )


        ax2.set_title(

            "Loss Curve"

        )


        ax2.set_xlabel(

            "Epoch"

        )


        ax2.set_ylabel(

            "Loss"

        )


        st.pyplot(fig2)
