import httpx
import asyncio
from utils import preprocess_inputs
import datetime

async def fetch_predictions(lat, lon, date, filename="output.nc", format="netcdf"):
    """
    Fetch predictions from the FastAPI service.

    Parameters:
    - lat: list of float, list of latitudes
    - lon: list of float, list of longitudes
    - date: list of str, list of dates in 'YYYY-MM-DD' format
    - filename: str, path where the output NetCDF file will be saved (default is 'output.nc')
    - format: str, either 'netcdf' or 'json' (default is 'netcdf')

    Returns:
    - Saves the NetCDF file to `filename` and returns the file path.
    """

    # Define the API URL
    API_URL = "http://127.0.0.1:5000/predict" # working remote (IP)
    # API_URL = "http://ozavala.coaps.fsu.edu:8000/predict" # working remote (IP)
    # API_URL = "http://144.174.11.107:8000/predict" # working remote (IP)

    # Prepare the data payload
    data = {
        "lat": lat,
        "lon": lon,
        "date": date
    }

    # Set a custom timeout (e.g., 60 seconds)
    timeout = httpx.Timeout(5000, connect=10.0)

    # Make an async request to the API
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(API_URL, json=data)
        
        if response.status_code == 200:
            content_type = response.headers.get('Content-Type')
            
            if content_type == 'application/x-netcdf':
                # Save the NetCDF file
                with open(filename, "wb") as f:
                    f.write(response.content)
                print(f"NetCDF file saved as {filename}")
                
                # Get missing data information from headers
                missing_data = response.headers.get('X-Missing-Data')
                successful_data = response.headers.get('X-Successful-Data')
                print(f"{successful_data} profiles were successfully generated.")
                if missing_data:
                    print(f"{missing_data} profiles are missing satellite data.")
                return filename
            else:
                # Assume it's JSON and return the data
                result = response.json()
                print(f"{result['success']} profiles successfully generated.")
                if result['missing_data'] > 0:
                    print(f"{result['missing_data']} points missing satellite data (not generated).")
                return result
        else:
            print(f"Request failed with status code {response.status_code}")
            print("Response content:", response.content)
            return None

def get_predictions(lat, lon, date, filename="output.nc"):
    """
    Wrapper function to run the async function in a synchronous context.

    Parameters:
    - lat: list of float, list of latitudes
    - lon: list of float, list of longitudes
    - date: list of str, list of dates in 'YYYY-MM-DD' format
    - filename: str, path where the output NetCDF file will be saved (default is 'output.nc')

    Returns:
    - The result from fetch_predictions.
    """
    lat, lon, date = preprocess_inputs(lat, lon, date)
    print(f"Fetching predictions for {len(lat)} points...")
    if asyncio.get_event_loop().is_running():
        return asyncio.ensure_future(fetch_predictions(lat, lon, date, filename))
    else:
        return asyncio.run(fetch_predictions(lat, lon, date, filename))

# Example of how to use the function from another script
if __name__ == "__main__":
    # Example usage
    latitudes = [25.0, 26.0, 27.0]
    longitudes = [-83.0, -84.0, -85.0]
    dates = ["2015-08-20", "2018-08-21", "2018-08-22"]
    #let's use 300 consecutive datetime dates:
    # start = datetime.datetime(2015, 1, 1)
    # dates = [start + datetime.timedelta(days=i) for i in range(300)]
    
    # # make lat lon and dates 100 times longer (just repeat)
    # latitudes = latitudes * 100
    # longitudes = longitudes * 100
    # dates = dates * 100
    output_file = "my_output.nc"

    result = get_predictions(latitudes, longitudes, dates, filename=output_file)
    print("Result:", result)