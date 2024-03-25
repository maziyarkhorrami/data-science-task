# data-science-task

Mit dieser Aufgabe wollen wir gerne sehen, wie Du Probleme mit Code löst. Es geht weder
darum das beste Modell zu finden noch darum eine fertige App zu entwickeln.

## Aufgaben

1. Trainiere ein Modell für die Daten aus https://www.kaggle.com/c/rossmann-store-sales/overview. Für diese Aufgabe kannst du die test.csv und die sample_submission.csv ignorieren und dich nur mit den Daten aus der train.csv und store.csv beschäftigen. Wähle aus den ca. 3000 Stores die Teilmenge aus, die die obersten 10% der Gesamt-Sales-Menge in den Trainings-Daten ausmachen (die 'Top-Stores'). Trainiere ein Modell für die Vorhersage der täglichen Sales pro Store für diese Teilmenge.

2. Überprüfe die Vorhersageleistung deines Modells auf den Testdaten mit dem root mean square percentage error (RSMPE) und einer weiteren Metrik Deiner Wahl. Führe dazu ein Splitting deiner Wahl auf der train.csv Datei aus. Der RSMPE sollte <= 0.4 sein, ansonsten ist die Performance aber nicht weiter wichtig.

3. Binde dein Modell in eine REST API über die Route `/sales/store/<store-id>` ein. Die Route 
   sollte einen POST-Request mit dem Body der Form

   ```js
   {
       DayOfWeek: 4,
       Date: "2015-09-17",
       Open: true,
       Promo: true,
       StateHoliday: false, 
       SchoolHoliday: false
   }
   ```

   entgegenehmen und eine Antwort der Form

    ```js
    {
       Store: 1,
       Date: "2015-09-17",
       PredictedSales: 5263
    }
    ```

    zurückgeben.

    Für Stores, die das Modell nicht kennt, sollte 404 zurückgegeben werden.

4. Dokumentiere deine Lösung so, dass Andere in der Lage sind, das Modell neu zu trainieren
   und die REST API lokal zu starten. Das kannst Du hier direkt im README machen (Englisch oder
   Deutsch). Checke alles ins repository ein, was notwendig ist, um Reproduzierbarkeit
   sicherzustellen.

Die Aufgabe sollte vollständig in Python gelöst werden. Du kannst aber alle Libraries und Tools Deiner Wahl verwenden.
Deine vollständige Lösung sollte auf einem separaten Branch eingecheckt sein.
Stelle einen pull request gegen den 'main' Branch, sobald Du fertig bist.

## Dokumentation

Maziyar Khorrami
## Predict the Sales value of a Rosmann shop on a time series

The FastAPI Sales Prediction API is designed to assist users in making accurate sales predictions for different stores. By leveraging a pre-trained regression model and dynamic feature calculation, it provides a reliable and user-friendly interface for predicting sales based on various input parameters.

## Prerequisites

- Python 3.11.6
- Dependencies: look at the requirements.txt

## Installation

it is highly recommended to use python venv for ease of use. 
In case of using IDEs like PyCharm one can create the venv directly in IDE
```bash
pip install -r requirements.txt
```

## Run the FastAPI Application:
run it using the following command in your terminal:

```bash
uvicorn src.app:app --reload
```


Access the Swagger UI:
Open a web browser and go to http://127.0.0.1:8000/docs. This will open the Swagger UI, where you can test your API by providing input values and checking the responses.

Test the API Endpoints:
You can use the Swagger UI to test the /sales/store/{store_id} endpoint. Provide the required input values and click the "Execute" button. You should see the response, and if everything is set up correctly, it will show the predicted sales.



## Data Preparation and Model Training Documentation

### Data Preparation

1. **Loading DataFrames:**
   - Load the training data from the CSV file `train.csv`.
   - Load the store data from the CSV file `store.csv`.


2. **Merging DataFrames:**
   - Merge the `store` and `train` DataFrames on the 'Store' column.
     ```python
     mos_df = pd.merge(df_store, df_train, on='Store')
     ```

3. **Calculate Total Sales for Each Store:**
   - Group the merged DataFrame by 'Store' and aggregate the sum of 'Sales'.
     ```python
     tsfes_df = mos_df.groupby('Store')['Sales'].sum()
     ```

4. **Identify Top 10% Stores Based on Sales:**
   - Calculate the 90th percentile of total sales.
   - Identify stores with sales greater than or equal to the 90th percentile.
     ```python
     ttpsos = tsfes_df.quantile(0.9)
     top_stores_list = tsfes_df[tsfes_df >= ttpsos].index.tolist()
     top_stores_df = mos_df[mos_df['Store'].isin(top_stores_list)]
     ```

### Data Exploration on Sales and Date

1. **Convert 'Date' to DateTime:**
   - Convert the 'Date' column to a DateTime object.


2. **Plot Sales Over Time:**
   - Plot sales over time for each year, highlighting the variation in sales.
     ```python
     # Assuming 'Date' is not in datetime format, convert it
     df['Year'] = df['Date'].dt.year
     df['Month'] = df['Date'].dt.month

     # Create subplots for each year
     # ...

     # Enable interactive zooming
     mplcursors.cursor(hover=True)
     plt.tight_layout()
     plt.show()
     ```

3. **Decompose Seasonality and Trend:**
   - Use methods like Seasonal-Trend decomposition using LOESS (STL) to decompose seasonality, trend, and residuals.
     - Not implemented here, but the intention is explained.

### Feature Engineering

1. **Normalize Sales and Save Range:**
   - Normalize 'Sales' between 0 and 1.
   - Plot the normalized distribution.
   - Save the minimum and maximum sales values in ```notebooks/sales_range.yml``` for later use in FastAPI.
     ```python
     min_sales = df['Sales'].min()
     max_sales = df['Sales'].max()

     # Normalize 'Sales'
     df['Normalized_Sales'] = (df['Sales'] - min_sales) / (max_sales - min_sales)

     # Save the sales range to a YAML file
     sales_dict = {'min_sales': min_sales, 'max_sales': max_sales}
     # ...

     # Plot normalized distribution
     # ...
     ```

2. **Profile the DataFrame:**
   - Use Data Profiling to gain insights into the DataFrame features.
   - Identify highly correlated features for model consideration.

### Feature Selection and Model Training

1. **Select Relevant Features:**
   - Identify relevant features for model training, considering store information, promotion information, and date information.
     ```python
     X = df[['Promo2','Assortment', 'StoreType', 'DayOfWeek', 'SchoolHoliday','WeekNumber']]
     y = df['Normalized_Sales']
     ```

2. **Remove Zero Values from Sales:**
   - Remove rows where normalized sales are zero.
     ```python
     df = df[df['Normalized_Sales'] != 0]
     ```

3. **Train-Test Split and Model Training:**
   - Split the dataset into training and testing sets.
   - Initialize and train the Linear Regression model.
   - Make predictions on the test data.
   - Calculate RMSPE and MSE.
   - Save the trained model to a file.
     ```python
     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

     # Regression Model initialization and train
     model_reg = LinearRegression()
     model_reg.fit(X_train, y_train)

     # Prediction on the test data
     y_pred = model_reg.predict(X_test)

     # Calculate RMSPE and MSE
     # ...

     # Save the trained model to a file
     file_path = '../models/maziyar_regression_model.pkl'
     with
     ```