// GeoTIFFLayer.js
import React, { useEffect, useState } from 'react';
import { useMap } from 'react-leaflet';
import parseGeoraster from 'georaster';
import GeoRasterLayer from 'georaster-layer-for-leaflet';
import L from '../utils/leafleticon'; // Ensure this utility is properly configured

const GeoTIFFLayer = ({ urls, interval = 2000 }) => {
  const map = useMap();
  const offset = { lat: 0.5, lng: 0.5 }; // Adjust lat/lng offset if needed
  const [currentIndex, setCurrentIndex] = useState(0); // State to track the current URL index
  const [loading, setLoading] = useState(true); // State to manage loading status

  useEffect(() => {
    let georasterLayer;
    let timer;
    let filenameControl;

    // Function to create and add a Leaflet control to display the filename
    const createFilenameControl = () => {
      const FileNameControl = L.Control.extend({
        onAdd: function (map) {
          this._div = L.DomUtil.create('div', 'filename-control'); // Create a div with a class
          this.update(); // Initialize with blank or first filename
          return this._div;
        },
        update: function (fileName) {
          this._div.innerHTML = `<strong>File:</strong> ${fileName || 'Loading...'}`;
          this._div.style.backgroundColor = 'white';
  
  // Optionally, add some padding for better appearance
  this._div.style.padding = '10px';
  
  // You can also add more styles if needed
  this._div.style.borderRadius = '5px'; // For rounded corners
  this._div.style.boxShadow = '0 4px 8px rgba(0, 0, 0, 0.1)'; // Set the file name
        }
      });
      return new FileNameControl({ position: 'topright' }); // Position the control on the map
    };

    // Add the filename control to the map
    filenameControl = createFilenameControl();
    filenameControl.addTo(map);

    const fetchAndRenderGeoTIFF = async (url) => {
      setLoading(true); // Show loading spinner or progress indicator

      try {
        const response = await fetch(url);
        const arrayBuffer = await response.arrayBuffer();
        const georaster = await parseGeoraster(arrayBuffer);

        // Create GeoRasterLayer with parsed GeoTIFF data
        georasterLayer = new GeoRasterLayer({
          georaster,
          opacity: 0, // Start with 0 opacity for smooth fade-in
          resolution: 1024, // High resolution, adjust if necessary for performance
          noDataValue: 0, // Ensures NoData is treated as transparent
        });

        // Add the new GeoRasterLayer to the map
        georasterLayer.addTo(map);

        // Update the filename control with the current file name
        const fileName = url.split('/').pop(); // Extract the file name from the URL
        filenameControl.update(fileName);

        // Fit the map to the bounds of the GeoTIFF (with an offset if needed)
        const originalBounds = georasterLayer.getBounds();
        const shiftedBounds = L.latLngBounds(
          [
            originalBounds.getSouthWest().lat + offset.lat,
            originalBounds.getSouthWest().lng + offset.lng,
          ],
          [
            originalBounds.getNorthEast().lat + offset.lat,
            originalBounds.getNorthEast().lng + offset.lng,
          ]
        );

        map.fitBounds(shiftedBounds);

        // Smoothly fade in the layer after it’s added
        let opacity = 0;
        const fadeIn = setInterval(() => {
          opacity += 0.1;
          if (opacity >= 1) {
            clearInterval(fadeIn);
          } else {
            georasterLayer.setOpacity(opacity);
          }
        }, 100); // Adjust the interval time as needed

      } catch (error) {
        console.error('Error loading GeoTIFF:', error);
      } finally {
        setLoading(false); // Hide loading spinner once the GeoTIFF is loaded
      }
    };

    // Initial render of the first GeoTIFF
    fetchAndRenderGeoTIFF(urls[currentIndex]);

    // Set up the interval to update the layer every `interval` milliseconds
    timer = setInterval(() => {
      // Remove the previous layer if it exists
      if (georasterLayer) {
        map.removeLayer(georasterLayer);
      }

      // Update the index to the next URL
      const nextIndex = (currentIndex + 1) % urls.length; // Loop back to start
      setCurrentIndex(nextIndex); // Update state to the next index

      // Fetch and render the next GeoTIFF
      fetchAndRenderGeoTIFF(urls[nextIndex]);
    }, interval);

    // Cleanup to remove the layer and clear the timer when the component unmounts
    return () => {
      if (georasterLayer) {
        map.removeLayer(georasterLayer);
      }
      if (filenameControl) {
        map.removeControl(filenameControl); // Remove the filename control when unmounting
      }
      clearInterval(timer);
    };
  }, [map, urls, currentIndex, interval, offset.lat, offset.lng]);

  return loading ? <div className="loading-spinner">Loading...</div> : null; // Replace with your spinner component
};

export default GeoTIFFLayer;
