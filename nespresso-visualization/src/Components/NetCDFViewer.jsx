// src/components/NetCDFViewer.jsx

import React, { useState } from 'react';
import { NetCDFReader } from 'netcdfjs';
import Plot from 'react-plotly.js';

const NetCDFViewer = ({ fileUrl }) => {
  const [variables, setVariables] = useState({});
  const [selectedVariable, setSelectedVariable] = useState(null);
  
  // Function to read and process the NetCDF file
  const readNetCDFFile = async (file) => {
    const reader = new FileReader();
    reader.onload = async (event) => {
      try {
        const arrayBuffer = event.target.result;
        const dataset = new NetCDFReader(arrayBuffer);

        // Extract all variables
        const varNames = dataset.getVariableNames();
        const varData = {};

        for (const name of varNames) {
          varData[name] = dataset.getDataVariable(name);
        }

        setVariables(varData);
      } catch (error) {
        console.error('Error reading NetCDF file:', error);
      }
    };
    reader.readAsArrayBuffer(file);
  };

  // Handle variable selection
  const handleVariableSelect = (event) => {
    const variableName = event.target.value;
    if (variableName) {
      setSelectedVariable({
        name: variableName,
        data: variables[variableName],
      });
    }
  };

  // Render the selected variable data visualization
  const renderVisualization = () => {
    if (!selectedVariable) return null;

    const { name, data } = selectedVariable;

    return (
      <div className="mt-6 p-4 bg-white shadow-lg rounded-lg">
        <h2 className="text-xl font-semibold mb-4">Selected Variable: {name}</h2>
        <pre className="whitespace-pre-wrap break-words">{JSON.stringify(data, null, 2)}</pre>
        <Plot
          data={[
            {
              x: data.longitude || [],
              y: data.latitude || [],
              z: data.values || [],
              type: 'scatter3d',
              mode: 'markers',
              marker: { size: 6 },
              colorscale: 'Viridis',
            },
          ]}
          layout={{
            width: '100%',
            height: 400,
            title: `Visualization of ${name}`,
            scene: {
              xaxis: { title: 'Longitude' },
              yaxis: { title: 'Latitude' },
              zaxis: { title: 'Values' },
            },
          }}
        />
      </div>
    );
  };

  return (
    <div className="container mx-auto px-4 py-6">
      <div className="max-w-md mx-auto bg-gray-100 p-4 rounded-lg shadow-md">
        <h1 className="text-2xl font-bold mb-4 text-center">NetCDF Viewer</h1>
        <input
          type="file"
          accept=".nc"
          onChange={(e) => {
            if (e.target.files.length > 0) {
              readNetCDFFile(e.target.files[0]);
            }
          }}
          className="block w-full text-sm text-gray-600 file:py-2 file:px-4 file:border file:rounded-md file:border-gray-300 file:bg-gray-50 file:text-gray-700 file:cursor-pointer"
        />
        <div className="mt-4">
          <h2 className="text-lg font-semibold mb-2">Available Variables</h2>
          <select
            onChange={handleVariableSelect}
            className="block w-full bg-white border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-2 focus:ring-indigo-500"
          >
            <option value="">Select a variable</option>
            {Object.keys(variables).map((name) => (
              <option key={name} value={name}>
                {name}
              </option>
            ))}
          </select>
        </div>
      </div>
      {renderVisualization()}
    </div>
  );
};

export default NetCDFViewer;
