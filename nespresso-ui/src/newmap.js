import React, { useEffect } from 'react';
import { MapContainer, TileLayer, useMap } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import GeoRasterLayer from 'georaster-layer-for-leaflet';
import parseGeoraster from 'georaster';
const GeoTIFFLayer = ({ url }) => {

  const map = useMap();

  useEffect(() => {
    const fetchAndRenderGeoTIFF = async () => {
      try {
        const response = await fetch(url);
        const arrayBuffer = await response.arrayBuffer();
        const georaster = await parseGeoraster(arrayBuffer);


        if (georaster.geoTransform) {
          // Clone the geotransform to avoid mutating the original object
          const geoTransform = [...georaster.geoTransform];

          // Apply offsets
          geoTransform[0] += 0.1; // Shift in x (longitude)
          geoTransform[3] += 0.1; // Shift in y (latitude)

          // Update the georaster's geotransform
          georaster.geoTransform = geoTransform;
        } else {
          console.warn('GeoRaster does not have a geoTransform property.');
        }
        const georasterLayer = new GeoRasterLayer({
          georaster,
          opacity: 0.7,
          resolution: 1024,
          nearest: true,
          tileSize: 256,
          pixelPerfect: true
        });

        georasterLayer.addTo(map);
        map.fitBounds(georasterLayer.getBounds());
      } catch (error) {
        console.error('Error loading GeoTIFF:', error);
      }
    };

    fetchAndRenderGeoTIFF();
  }, [map, url]);

  return null;
};

const MapWithGeoTIFF = () => {
  const tiffUrl = '/2021-05_2021-05-01.tiff'; // Update with your local TIFF path or external URL

  return (
    <MapContainer center={[0, 0]} zoom={5} style={{ height: '100vh', width: '100%' }}>
      <TileLayer
        url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        attribution='&copy; <a href="http://osm.org/copyright">OpenStreetMap</a> contributors'
      />
      <GeoTIFFLayer url={tiffUrl} />
    </MapContainer>
  );
};

export default MapWithGeoTIFF;
