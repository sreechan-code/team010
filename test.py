import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import LabelEncoder
from io import StringIO
from zipfile import BadZipFile

# Internal CSS for background color and custom styles
st.markdown("""
    <style>
        /* Background color for the main app */
        body {
            background-color: #f0f2f6;
            color: #333333;
        }

        /* Sidebar background color and text styling */
        section[data-testid="stSidebar"] {
            background-color: #2e4057;
            color: white;
        }

        /* Sidebar text color */
        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3,
        section[data-testid="stSidebar"] label {
            color: white;
        }

        /* Style for main app titles and headers */
        h1, h2, h3 {
            color: #2e4057;
        }

        /* Style for buttons */
        button[kind="primary"] {
            background-color: #2e4057;
            color: white;
        }

        button[kind="primary"]:hover {
            background-color: #3a506b;
            color: white;
        }

        /* Style for dataframe headers and table text */
        .stDataFrame th {
            background-color: #2e4057;
            color: white;
        }

        .stDataFrame td {
            color: #333333;
        }    
    </style>
    """, unsafe_allow_html=True)

# Load and display the team logo
st.sidebar.image("teamlogo.png", use_column_width=True)  # Add the logo at the top of the sidebar

# Title for the web app
st.title("FINANCIAL TRANSACTION FRAUD DETECTION APP")

# Sidebar for user to upload data or input data as text
st.sidebar.header("Upload or Input Data")
data_option = st.sidebar.selectbox("Choose how you want to input data:", ["Upload CSV", "Upload Excel", "Input Data Manually"])

# Initialize an empty dataframe
df = None

# Option 1: Upload CSV File
if data_option == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.write("## Dataset Preview")
            st.dataframe(df.head())
        except Exception as e:
            st.error(f"An error occurred while reading the CSV file: {e}")

# Option 2: Upload Excel File
elif data_option == "Upload Excel":
    uploaded_file = st.sidebar.file_uploader("Upload Excel File", type=["xls", "xlsx"])

    if uploaded_file is not None:
        try:
            df = pd.read_excel(uploaded_file, engine='openpyxl')  # Specify the engine
            st.write("## Dataset Preview")
            st.dataframe(df.head())
        except ValueError as ve:
            st.error(f"ValueError: {ve}")
        except BadZipFile:
            st.error("The uploaded file is not a valid Excel file. Please upload a .xls or .xlsx file.")
        except Exception as e:
            st.error(f"An unexpected error occurred while reading the file: {e}")

# Option 3: Manually Input Data as Text
elif data_option == "Input Data Manually":
    st.write("## Enter Data as Text")
    
    input_text = st.text_area("Paste your data here (CSV format)", value="", height=200)
    
    if input_text:
        input_data = StringIO(input_text)
        df = pd.read_csv(input_data)
        
        st.write("## Dataset Preview")
        st.dataframe(df.head())

# Proceed if a dataset is provided (either uploaded or manually entered)
if df is not None:
    # Display basic dataset info
    st.write("### Basic Information")
    st.write(f"Number of rows: {df.shape[0]}")
    st.write(f"Number of columns: {df.shape[1]}")

    # Checking for missing values
    st.write("### Missing Values")
    st.write(df.isnull().sum())

    # Handling missing values
    if st.button("Impute Missing Values"):
        impute = KNNImputer()
        for i in df.select_dtypes(include="number").columns:
            df[i] = impute.fit_transform(df[[i]])
        st.write("Missing values filled using KNN Imputation")
        st.write(df.isnull().sum())

    # Encode categorical variables
    st.write("### Categorical Encoding")
    label_encoder = LabelEncoder()
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = label_encoder.fit_transform(df[col])

    st.write("### Data after Label Encoding")
    st.dataframe(df.head())

    # Split data into features and target
    Y = df['is_fraud']
    X = df.drop(columns=["is_fraud"])

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Feature selection using Variance Threshold
    var_thres = VarianceThreshold(threshold=2.0)
    var_thres.fit(X_train)
    constant_columns = [column for column in X_train.columns if column not in X_train.columns[var_thres.get_support()]]
    X_train.drop(constant_columns, axis=1, inplace=True)
    X_test.drop(constant_columns, axis=1, inplace=True)

    # Streamlit model section
    st.subheader("Model Training and Evaluation")

    # Select the noise level for the target variable
    noise_level = st.slider("Select Noise Level (%)", 0.0, 100.0, 20.0) / 100.0  # Convert to fraction

    # Introduce noise to the target variable (Y)
    noisy_Y = Y.copy()  # Create a copy of the original target variable
    for i in range(len(noisy_Y)):
        if np.random.random() < noise_level:
            noisy_Y.iloc[i] = 1 - noisy_Y.iloc[i]  # Assuming binary classification with labels 0 and 1

    # Split the data into training and testing sets using the noisy target variable
    X_train, X_test, y_train, y_test = train_test_split(X, noisy_Y, test_size=0.25, random_state=0)

    # Naive Bayes model
    nb_model = GaussianNB()

    # Handle NaN values in the feature sets
    X_train.fillna(X_train.mean(), inplace=True)
    X_test.fillna(X_test.mean(), inplace=True)

    # Convert y_train to discrete values
    y_train = y_train.astype(int)

    # Train the model
    nb_model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = nb_model.predict(X_test)

    # Evaluate the model's accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Display results
    st.write(f"Naive Bayes Model Accuracy: {accuracy * 100:.2f}%")
    st.text("Classification Report:")
    st.text(classification_report(y_test, y_pred))

else:
    st.write("Please upload a CSV or Excel file or input data to proceed.")
