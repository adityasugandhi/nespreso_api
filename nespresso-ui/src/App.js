import React, { useState, useRef } from 'react';
import MyMap from './components/Map';
import { DatePicker } from '@mui/x-date-pickers';
import { AdapterDayjs } from '@mui/x-date-pickers/AdapterDayjs';
import { LocalizationProvider } from '@mui/x-date-pickers/LocalizationProvider';
import { TextField, Slider } from '@mui/material';
import { PickersSectionListSectionSeparator } from '@mui/x-date-pickers/PickersSectionList/PickersSectionList';

function App() {
  const [selectedFeatures, setSelectedFeatures] = useState({
    points: false,
    lines: false,
    areas: false,
  });
  const [mapData, setMapData] = useState({ latitudes: [], longitudes: [] });
  const [selectedCoords, setSelectedCoords] = useState(null);
  const [selectedDate, setSelectedDate] = useState(null);
  const [PickersSectionListSectionSeparator, setPickersSectionListSectionSeparator] = useState(false);
  const [sliderValue, setSliderValue] = useState(0.5);
  const mapRef = useRef();
  const [visibility, setVisibility] = useState(false);

  const handleCheckboxChange = (feature) => {
    if (feature === 'areas') {
    setVisibility(!visibility);}

    if (feature === 'lines') {
      console.log(feature, PickersSectionListSectionSeparator);
      setPickersSectionListSectionSeparator(!PickersSectionListSectionSeparator);
    }
    setSelectedFeatures((prev) => ({
      ...prev,
      [feature]: !prev[feature],
    }));
  };

  const handleSubmit = async () => {
    if (mapData.latitudes.length === 0 || mapData.longitudes.length === 0) {
      alert('Please draw features on the map before submitting.');
      return;
    }

    if (selectedDate === null) {
      alert('Please select a date before submitting.');
      return;
    }

    try {
      const response = await fetch('/api/submit-map-data', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          selectedFeatures,
          mapData,
          selectedDate: selectedDate.toISOString().split('T')[0],
          sliderValue: sliderValue, // Added slider value to the request body
        }),
      });

      if (response.ok) {
        const result = await response.json();
        alert('Data submitted successfully!');
        console.log(result);
      } else {
        throw new Error('Failed to submit data');
      }
    } catch (error) {
      alert('Error submitting data: ' + error.message);
    }
  };

  const handleMapClick = (coords) => {
    setSelectedCoords(coords);
  };

  const handleReset = () => {
    setSelectedFeatures({
      points: false,
      lines: false,
      areas: false,
    });
    setMapData({ latitudes: [], longitudes: [] });
    setSelectedCoords(null);
    setSelectedDate(null);
    if (mapRef.current) {
      mapRef.current.reset();
    }
  };

  const handleFinishPolygon = () => {
    if (mapRef.current) {
      mapRef.current.finishPolygon();
    }
  };

  const handleSliderChange = (event, newValue) => {
    setSliderValue(newValue); // Update the slider value when it changes
  };

  return (
    <div className="flex flex-col min-h-screen bg-gray-100">
      <header className="bg-blue-700 text-white p-6 shadow-md">
        <h1 className="text-3xl font-bold text-center">NetCDF Generation</h1>
      </header>
      <main className="flex flex-col md:flex-row flex-grow">
        {/* Map Section */}
        <div className="w-full md:w-1/2 p-6">
          <MyMap
            ref={mapRef}
            onDataChange={setMapData}
            selectedFeatures={selectedFeatures}
            onMapClick={handleMapClick}
          />
        </div>
        {/* Form Section */}
        <div className="w-full md:w-1/2 p-6 bg-white shadow-lg rounded-lg">
          <h2 className="text-xl font-semibold mb-4 text-gray-800">Select Features</h2>
          <div className="space-y-4">
            {/* Points Feature */}
            <label className="flex items-center space-x-2">
              <input
                type="checkbox"
                checked={selectedFeatures.points}
                onChange={() => handleCheckboxChange('points')}
                className="form-checkbox"
              />
              <span className="text-gray-700">Points</span>
            </label>

            {/* Lines Feature */}
            <label className="flex items-center space-x-2">
              <input
                type="checkbox"
                checked={selectedFeatures.lines}
                onChange={() => handleCheckboxChange('lines')}
                className="form-checkbox"
              />
              <span className="text-gray-700">Lines</span>
            </label>
            <div className={`pl-3  w-full ${!PickersSectionListSectionSeparator ? 'hidden' : ''}`}>
              <label className="text-gray-700">Number of Points :</label>
              <input type="number" className="w-22 border-2 border-gray-300 rounded-md p-2 m-1" />
            </div>

            {/* Areas Feature */}
            <div className="flex flex-col space-y-4">
              <label className="flex items-center space-x-2">
                <input
                  type="checkbox"
                  checked={selectedFeatures.areas}
                  onChange={() => handleCheckboxChange('areas')}
                  className="form-checkbox"
                />
                <span className="text-gray-700">Areas</span>
              </label>
              {selectedFeatures.areas && (
                <button
                  onClick={handleFinishPolygon}
                  className="ml-2 bg-green-600 hover:bg-green-700 text-white font-bold py-1 px-3 rounded-lg text-sm"
                >
                  Finish Polygon
                </button>
              )}

              {/* Slider for Points */}
              <div className={`flex items-center space-x-3 ${!visibility ? 'hidden' : ''}`}>
                <span className="text-gray-700">Points:</span>
                <Slider
                  className="w-48"
                  value={sliderValue}
                  valueLabelDisplay="auto"
                  step={0.1}
                  marks
                  min={0}
                  max={1}
                  onChange={handleSliderChange}
                />
                <div className="font-bold text-gray-700">:{sliderValue}</div>
              </div>
            </div>

            {/* Date Picker */}
            <div className="w-full">
              <LocalizationProvider dateAdapter={AdapterDayjs}>
                <DatePicker
                  label="Select Date"
                  value={selectedDate}
                  onChange={(newValue) => setSelectedDate(newValue)}
                  renderInput={(params) => (
                    <TextField {...params} className="w-full border-2 border-gray-300 rounded-md" />
                  )}
                />
              </LocalizationProvider>
            </div>

            {/* Submit and Reset Buttons */}
            <div className="mt-4 space-x-2">
              <button
                onClick={handleSubmit}
                className="bg-blue-600 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded-lg"
              >
                Submit
              </button>
              <button
                onClick={handleReset}
                className="bg-red-500 hover:bg-red-600 text-white font-bold py-2 px-4 rounded-lg"
              >
                Reset
              </button>
            </div>
          </div>

          {/* Selected Coordinates Display */}
          {selectedCoords && (
            <div className="mt-6 p-4 bg-white border rounded-md shadow-md">
              <h3 className="font-semibold mb-2 text-gray-800">Selected Coordinates:</h3>
              <p className="text-gray-600">Latitude: {selectedCoords.lat.toFixed(4)}</p>
              <p className="text-gray-600">Longitude: {selectedCoords.lng.toFixed(4)}</p>
            </div>
          )}

          {/* Display All Selected Points */}
          <div className="mt-6 p-4 bg-white border rounded-md shadow-md">
            <h3 className="font-semibold mb-2 text-gray-800">All Selected Points:</h3>
            <div className="max-h-60 overflow-y-auto">
              {mapData.latitudes.map((lat, index) => (
                <div key={index} className="mb-2">
                  <span className="font-medium text-gray-800">Point {index + 1}:</span> Lat: {lat.toFixed(4)}, Lng: {mapData.longitudes[index].toFixed(4)}
                </div>
              ))}
            </div>
          </div>

          {/* Display Selected Date */}
          {selectedDate && (
            <div className="mt-6 p-4 bg-white border rounded-md shadow-md">
              <h3 className="font-semibold mb-2 text-gray-800">Selected Date:</h3>
              <p className="text-gray-600">{selectedDate.toISOString().split('T')[0]}</p>
              </div>
          )}
        </div>
      </main>

      {/* Footer Section */}
      <footer className="bg-gray-800 text-white p-4 text-center">
        <p>&copy; {new Date().getFullYear()} Your Company. All rights reserved.</p>
      </footer>
    </div>
  );
}

export default App;

