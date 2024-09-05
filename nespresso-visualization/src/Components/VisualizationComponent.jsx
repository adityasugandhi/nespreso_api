// src/components/VisualizationComponent.jsx

import React from 'react';
import NetCDFViewer from './NetCDFViewer';
const VisualizationComponent = ({ data,fileUrl }) => {
  // Replace with actual visualization logic
  return (
    <div className="max-w-lg mx-auto p-6 bg-white shadow-md rounded-lg mt-6">
      <h2 className="text-xl font-bold mb-4">Visualization Result</h2>
      <p>Latitude: {data.latitude}</p>
      <p>Longitude: {data.longitude}</p>
      <p>Date: {data.date}</p>
      <div className="mt-4">
<NetCDFViewer fileUrl={fileUrl}/>
        {/* Placeholder for visualization */}
        <div className="h-64 bg-gray-100 border border-gray-300 rounded-md flex items-center justify-center">
          Visualization Area
        </div>
      </div>
    </div>
  );
};

export default VisualizationComponent;
