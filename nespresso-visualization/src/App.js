// src/App.jsx

import React, { useState } from 'react';
import VisualizationForm from './Components/VisualizationForm';
import VisualizationComponent from './Components/VisualizationComponent';
import './index.css';

const App = () => {
  const [visualizationData, setVisualizationData] = useState(null);

  const handleFormSubmit = (data) => {
    // Handle form submission and trigger visualization
    setVisualizationData(data);
  };

  return (
    <div className="min-h-screen bg-gray-50 flex flex-col">
      <header className="bg-gradient-to-r from-indigo-500 to-purple-600 text-white p-4 shadow-md">
        <div className="container mx-auto text-center">
          <h1 className="text-4xl font-extrabold"> NetCdf Data Visualization Tool</h1>
        </div>
      </header>
      <main className="flex-1 flex items-center justify-center px-4 py-8">
        <div className="bg-white p-6 rounded-lg shadow-lg border border-gray-200 w-full max-w-3xl">
          <h2 className="text-2xl font-semibold mb-4 text-indigo-600 text-center">Submit Your Data</h2>
          <VisualizationForm onSubmit={handleFormSubmit} />
        </div>
        {visualizationData && (
          <div className="mt-8 bg-white p-6 rounded-lg shadow-lg border border-gray-200 w-full max-w-3xl">
            <h2 className="text-2xl font-semibold mb-4 text-purple-600 text-center">Visualization Results</h2>
            <VisualizationComponent
              data={visualizationData.data}
              fileUrl={visualizationData.result.file_url}
            />
          </div>
        )}
      </main>
      <footer className="bg-gray-800 text-white text-center py-4 mt-auto">
        <p className="text-sm">&copy; {new Date().getFullYear()} Data Visualization Tool. All rights reserved.</p>
      </footer>
    </div>
  );
};

export default App;
