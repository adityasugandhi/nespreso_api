// src/App.js
import React, { useState, useRef } from 'react';
import MyMap from './components/Map';
import { DatePicker } from '@mui/x-date-pickers';
import { AdapterDayjs } from '@mui/x-date-pickers/AdapterDayjs';
import { LocalizationProvider } from '@mui/x-date-pickers/LocalizationProvider';
import { TextField, Slider} from '@mui/material';
// import { BrowserRouter, Route, Routes } from 'react-router-dom';
// import MapWithGeoTIFF from './newmap';

// const Router = () => {
//   return (
//     <BrowserRouter>
//       <Routes>
//       </Routes>
//     </BrowserRouter>
//   );
// };


function App() {
  const [selectedFeatures, setSelectedFeatures] = useState({
    points: false,
    lines: false,
    areas: false,
  });
  const [mapData, setMapData] = useState({ latitudes: [], longitudes: [] });
  const [selectedCoords, setSelectedCoords] = useState(null);
  const [selectedDate, setSelectedDate] = useState(null);
  const [sliderValue, setSliderValue] = useState(0.5);
  const mapRef = useRef();

  const handleCheckboxChange = (feature) => {
    setSelectedFeatures(prev => ({
      ...prev,
      [feature]: !prev[feature]
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
          sliderValue : sliderValue // Added  slider value to the request body
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
    <div className="flex flex-col h-screen">
      <header className="bg-blue-600 text-white p-4 shadow-md">
        <h1 className="text-2xl font-bold">NetCDF generation </h1>
      </header>
      <main className="flex flex-col md:flex-row flex-grow overflow-hidden">
        <div className="w-full md:w-1/2 p-4">
          <MyMap 
            ref={mapRef}
            onDataChange={setMapData} 
            selectedFeatures={selectedFeatures} 
            onMapClick={handleMapClick}
          />
        </div>
        <div className="w-full md:w-1/2 p-4 bg-gray-100 overflow-y-auto">
          <h2 className="text-xl font-semibold mb-4">Select Features</h2>
          <div className="space-y-2">
            <label className="flex items-center space-x-2">
              <input
                type="checkbox"
                checked={selectedFeatures.points}
                onChange={() => handleCheckboxChange('points')}
                className="form-checkbox"
              />
              <span>Points</span>
            </label>
            <label className="flex items-center space-x-2">
              <input
                type="checkbox"
                checked={selectedFeatures.lines}
                onChange={() => handleCheckboxChange('lines')}
                className="form-checkbox"
              />
              <span>Lines</span>
            </label>
            <label className="flex items-center space-x-2">
              <input
                type="checkbox"
                checked={selectedFeatures.areas}
                onChange={() => handleCheckboxChange('areas')}
                className="form-checkbox"
              />
              <span>Areas</span>
              {selectedFeatures.areas && (
                <button
                  onClick={handleFinishPolygon}
                  className="ml-2 bg-green-500 hover:bg-green-600 text-white font-bold py-1 px-2 rounded text-sm"
                >
                  Finish Polygon
                </button>
              )}
            </label>
            <LocalizationProvider dateAdapter={AdapterDayjs}>
               <DatePicker
                label="Select Date"
                value={selectedDate}
                onChange={(newValue) => setSelectedDate(newValue)}
                renderInput={(params) => <TextField {...params} className="w-full" />}
              />
            </LocalizationProvider>

            <div className="flex items-center space-x-2 w-2/6">
              <span>Points:</span>
              <Slider
                value={sliderValue}
                valueLabelDisplay="auto"
                valueLabelFormat={(value) => `${value}`}
                defaultValue={0.5}
                step={0.1}
                marks
                min={0}
                max={1}
                onChange={handleSliderChange}
              />
              <div className='font-bold'>:{sliderValue}</div>
              </div>
          </div>
          

          <div className="mt-4 space-x-2">
            <button
              onClick={handleSubmit}
              className="bg-blue-500 hover:bg-blue-600 text-white font-bold py-2 px-4 rounded"
            >
              Submit
            </button>
            <button
              onClick={handleReset}
              className="bg-red-500 hover:bg-red-600 text-white font-bold py-2 px-4 rounded"
            >
              Reset
            </button>
          </div>
          {selectedCoords && (
            <div className="mt-4 p-3 bg-white rounded shadow">
              <h3 className="font-semibold mb-2">Selected Coordinates:</h3>
              <p>Latitude: {selectedCoords.lat.toFixed(4)}</p>
              <p>Longitude: {selectedCoords.lng.toFixed(4)}</p>
            </div>
          )}
          <div className="mt-4 p-3 bg-white rounded shadow">
            <h3 className="font-semibold mb-2">All Selected Points:</h3>
            <div className="max-h-60 overflow-y-auto">
              {mapData.latitudes.map((lat, index) => (
                <div key={index} className="mb-1">
                  <span className="font-medium">Point {index + 1}:</span> Lat: {lat.toFixed(4)}, Lng: {mapData.longitudes[index].toFixed(4)}
                </div>
              ))}
            </div>
          </div>
          <div className="mt-4 p-3 bg-white rounded shadow">
            {/* <h3 className="font-semibold mb-2">Selected Dates:</h3> */}
            <div className="max-h-60 overflow-y-auto">
              {selectedDate && (
                <div className="mb-1">
                  {selectedDate.toISOString().split('T')[0]}
                </div>
              )}
            </div>
          </div>
        </div>
      </main>
      <footer className="bg-gray-800 text-white p-4 text-center">
        <p>&copy; {new Date().getFullYear()} Your Company</p>
      </footer>
    </div>
  );
}

export default App;
