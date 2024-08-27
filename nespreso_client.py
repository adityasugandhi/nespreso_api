import httpx
import asyncio
from utils import preprocess_inputs

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
    - If format is 'netcdf', saves the file to `filename` and returns the file path.
    - If format is 'json', returns the JSON response.
    """

    # Define the API URL
    # API_URL = "http://0.0.0.0:8000/predict" # old, local for testing
    API_URL = "http://poseidon.sc.fsu.edu:8000/predict" # working remote
    # API_URL = "http://144.174.11.107:8000/predict" # working remote (IP)

    # Prepare the data payload
    data = {
        "lat": lat,
        "lon": lon,
        "date": date,
        "format": format
    }

    # Make an async request to the API
    async with httpx.AsyncClient() as client:
        response = await client.post(API_URL, json=data)
        
        if response.status_code == 200:
            content_type = response.headers.get('Content-Type')
            
            if content_type == 'application/x-netcdf' and format == "netcdf":
                # Save the NetCDF file
                with open(filename, "wb") as f:
                    f.write(response.content)
                print(f"NetCDF file saved as {filename}")
                return filename
            else:
                # Assume it's JSON and return the data
                return response.json()
        else:
            print(f"Request failed with status code {response.status_code}")
            print("Response content:", response.content)
            return None

def get_predictions(lat, lon, date, filename="output.nc", format="netcdf"):
    """
    Wrapper function to run the async function in a synchronous context.

    Parameters:
    - lat: list of float, list of latitudes
    - lon: list of float, list of longitudes
    - date: list of str, list of dates in 'YYYY-MM-DD' format
    - filename: str, path where the output NetCDF file will be saved (default is 'output.nc')
    - format: str, either 'netcdf' or 'json' (default is 'netcdf')

    Returns:
    - The result from fetch_predictions.
    """
    lat, lon, date = preprocess_inputs(lat, lon, date)
    print(f"Fetching predictions for {len(lat)} points...")
    if asyncio.get_event_loop().is_running():
        return asyncio.ensure_future(fetch_predictions(lat, lon, date, filename, format))
    else:
        return asyncio.run(fetch_predictions(lat, lon, date, filename, format))

# Example of how to use the function from another script
if __name__ == "__main__":
    # Example usage
    latitudes = [45.0, 46.0, 47.0]
    longitudes = [-30.0, -29.0, -28.0]
    dates = ["2020-08-20", "2020-08-21", "2020-08-22"]
    output_file = "my_output.nc"

    result = get_predictions(latitudes, longitudes, dates, filename=output_file, format="netcdf")
    print("Result:", result)
