from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, constr
import pandas as pd
import joblib
from datetime import datetime
import yaml

app = FastAPI()

# Load your trained model
model = joblib.load("models/maziyar_regression_model.pkl")
# Load the known store IDs from the CSV file
known_store_ids_df = pd.read_csv('data/store.csv')
known_store_ids = known_store_ids_df['Store'].tolist()


# Define the date range constraints

class PredictionInput(BaseModel):
    Date: str
    Assortment: int = Field(..., ge=0, le=2)
    StoreType: int = Field(..., ge=0, le=3)
    DayOfWeek: int = Field(3, ge=1, le=7)
    Promo2: int = Field(..., ge=0, le=1)
    SchoolHoliday: int = Field(..., ge=0, le=1)
    WeekNumber: int = Field(4, ge=1, le=52)

    @property
    def week_number(self):
        date_obj = datetime.strptime(self.Date, "%Y-%m-%d")
        return date_obj.strftime("%W")


class PredictionOutput(BaseModel):
    Store: int
    Date: str
    PredictedSales: float


@app.post("/sales/store/{store_id}")
async def predict_sales(store_id: int, input_data: PredictionInput):
    """
        Predict sales for a specific store based on input features.

        Parameters:
            - store_id (int): The ID of the store.
            - input_data (PredictionInput): Input data for prediction.

        Returns:
            - PredictionOutput: Predicted sales for the given store and date.
    """
    # Check if the store_id is known
    if store_id not in known_store_ids:
        raise HTTPException(status_code=404, detail="Store not found")

    input_data.WeekNumber = input_data.week_number

    # Create a DataFrame with the input data
    data = [[
        input_data.Promo2,
        input_data.Assortment,
        input_data.StoreType,
        input_data.DayOfWeek,
        input_data.SchoolHoliday,
        input_data.WeekNumber,
    ]]
    columns = ['Promo2', 'Assortment', 'StoreType', 'DayOfWeek', 'SchoolHoliday', 'WeekNumber']

    input_df = pd.DataFrame(data, columns=columns)

    # Potential TODOS:
    # we can check if the distribution of input data is desired

    # Make predictions using the loaded model
    predictions = model.predict(input_df)

    # unmap the prediction value
    min_sales, max_sales = load_sales_range()
    predicted_sales_unmapped = predictions[0] * (max_sales - min_sales) + min_sales
    # Prepare the response , round the predicted sale to an int
    response = PredictionOutput(Store=store_id, Date=input_data.Date, PredictedSales=int(predicted_sales_unmapped))

    return response


def load_sales_range(file_path='config/sales_range.yaml'):
    """
        Load sales range from a YAML file.

        Parameters:
            - file_path (str): Path to the YAML file.

        Returns:
            - tuple: Minimum and maximum sales values.
    """
    with open(file_path, 'r') as yaml_file:
        sales_range = yaml.safe_load(yaml_file)

    min_sales = sales_range['min_sales']
    max_sales = sales_range['max_sales']

    return min_sales, max_sales
