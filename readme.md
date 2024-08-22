# NeSPReSO API and Client

## Overview

This project consists of a FastAPI-based web service (`nespreso_host.py`) that serves predictions of temperature and salinity profiles based on latitude, longitude, and date inputs. It also includes a client (`nespreso_client.py`) that interacts with this API to retrieve the predictions either in JSON format or as a NetCDF file. The project also includes utility functions (`utils.py`) to preprocess the inputs to ensure they are in the correct format.

## Files and Structure

### 1. `utils.py`

This module contains helper functions to preprocess the input data (`lat`, `lon`, `date`). It supports various input formats, including numpy arrays, pandas Series, xarray DataArray, Python datetime objects, and MATLAB datenum.

#### Functions:

- **`convert_to_numpy_array(data)`**: Converts the input data to a numpy array if it is not already in that format.
- **`convert_to_list_of_floats(data)`**: Ensures the data is a list of floats. If the data is already in the correct format, it is returned unchanged.
- **`convert_date_to_iso_strings(date)`**: Converts date inputs to a list of ISO 8601 strings (`'YYYY-MM-DD'`). Handles various date formats.
- **`preprocess_inputs(lat, lon, date)`**: Orchestrates the conversion of `lat`, `lon`, and `date` to ensure they are in the correct format (list of floats for lat/lon and list of ISO strings for date).

### 2. `nespreso_host.py`

This script defines the FastAPI web service that provides predictions based on input parameters. It loads a pre-trained model and dataset to generate synthetic temperature and salinity profiles.

#### Key Components:

- **Global Namespace Addition**: Adds `TemperatureSalinityDataset` and `PredictionModel` to the global namespace to ensure compatibility when running from bash.
- **`load_model_and_dataset()`**: Loads the pre-trained model and dataset, preparing them for inference.
- **`save_to_netcdf()`**: Saves the prediction results to a NetCDF file.
- **`datetime_to_datenum()`**: Converts Python datetime objects to MATLAB datenum format.
- **`startup_event()`**: Initializes the model and dataset when the API starts.
- **`predict()`**: The main endpoint (`/predict`) that accepts POST requests and returns the predictions in either JSON or NetCDF format.

### 3. `nespreso_client.py`

This script serves as a client to interact with the FastAPI service. It preprocesses inputs using functions from `utils.py` and sends a request to the API to fetch predictions.

#### Key Functions:

- **`fetch_predictions(lat, lon, date, filename="output.nc", format="netcdf")`**: Asynchronously fetches predictions from the API and either saves the result to a file or returns the JSON data.
- **`get_predictions(lat, lon, date, filename="output.nc", format="netcdf")`**: A synchronous wrapper for `fetch_predictions`, making it easy to use in non-async contexts.

## Usage

### Running the FastAPI Server

To start the FastAPI server, run the following command:

```bash
uvicorn nespreso_host:app --host 0.0.0.0 --port 8000 --reload
```

### Interacting with the API using the Client

You can interact with the FastAPI service using the provided client. Here’s an example of how to use the client:

```python
from nespreso_client import get_predictions

# Define your inputs
latitudes = [45.0, 46.0, 47.0]
longitudes = [-30.0, -29.0, -28.0]
dates = ["2020-08-20", "2020-08-21", "2020-08-22"]
output_file = "my_output.nc"

# Fetch predictions and save to a NetCDF file
result = get_predictions(latitudes, longitudes, dates, filename=output_file, format="netcdf")
print("Result:", result)  # Should print the path to the saved NetCDF file
```

### Example Test Case (using different formats):

```python
import numpy as np
import pandas as pd
import xarray as xr
from nespreso_client import get_predictions

# Example input data in different formats
lat_np = np.array([45.0, 46.0, 47.0])
lon_pd = pd.Series([-30.0, -29.0, -28.0])
date_xr = xr.DataArray(pd.to_datetime(["2020-08-20", "2020-08-21", "2020-08-22"]))

# Fetch predictions
result = get_predictions(lat_np, lon_pd, date_xr, filename="output.nc", format="netcdf")
print("NetCDF file saved as:", result)
```

## Dependencies

Install all dependencies via `pip`:

```bash
pip install fastapi uvicorn httpx numpy pandas xarray torch scipy
```

## Notes

* Ensure that the paths to the model and dataset files are correctly set in `nespreso_host.py` before running the server.
* The client and server are designed to work together seamlessly, but you can adapt the client to work with other similar services if needed.

## License

This project is licensed under the MIT License.
