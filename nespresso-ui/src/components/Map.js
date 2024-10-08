import React, { useState, useEffect, useImperativeHandle, forwardRef } from 'react';
import {
  MapContainer,
  TileLayer,
  Marker,
  Popup,
  useMapEvents,
  Polyline,
  Polygon,
} from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import L from '../utils/leafleticon';
import GeoTIFFLayer from './GeoTIFFLayer';
const GULF_OF_MEXICO_CENTER = [25.5, -91.0];
const INITIAL_ZOOM = 6;
const GEO_TIFF_URL = ['./output_highres.tif','./2021-05_2021-05-01.tiff']; // Replace with your GeoTIFF URL

const  tiffFiles = [
  "/tiff_test/2021-05_2021-05-02.tiff",
  "/tiff_test/2021-05_2021-05-04.tiff",
  "/tiff_test/2021-05_2021-05-05.tiff",
  "/tiff_test/2021-05_2021-05-06.tiff",
  "/tiff_test/2021-05_2021-05-07.tiff",
  "/tiff_test/2021-05_2021-05-08.tiff",
  "/tiff_test/2021-05_2021-05-09.tiff",
  "/tiff_test/2021-05_2021-05-10.tiff",
  "/tiff_test/2021-05_2021-05-11.tiff",
  "/tiff_test/2021-05_2021-05-12.tiff",
  "/tiff_test/2021-05_2021-05-13.tiff",
  "/tiff_test/2021-05_2021-05-14.tiff",
  "/tiff_test/2021-05_2021-05-15.tiff",
  "/tiff_test/2021-05_2021-05-16.tiff",
  "/tiff_test/2021-05_2021-05-17.tiff",
  "/tiff_test/2021-05_2021-05-18.tiff",
  "/tiff_test/2021-05_2021-05-19.tiff",
  "/tiff_test/2021-05_2021-05-20.tiff",
  "/tiff_test/2021-05_2021-05-21.tiff",
  "/tiff_test/2021-05_2021-05-22.tiff",
  "/tiff_test/2021-05_2021-05-23.tiff",
  "/tiff_test/2021-05_2021-05-24.tiff",
  "/tiff_test/2021-05_2021-05-25.tiff",
  "/tiff_test/2021-05_2021-05-26.tiff",
  "/tiff_test/2021-05_2021-05-27.tiff",
  "/tiff_test/2021-05_2021-05-28.tiff",
  "/tiff_test/2021-05_2021-05-29.tiff",
  "/tiff_test/2021-05_2021-05-30.tiff",
  "/tiff_test/2021-05_2021-05-31.tiff"
];






const MapEventHandler = ({ addMarker, addPolyline, addPolygon, onMapClick, selectedFeatures }) => {
  const [polylinePoints, setPolylinePoints] = useState([]);
  const [polygonPoints, setPolygonPoints] = useState([]);

  useMapEvents({
    click(e) {
      const { lat, lng } = e.latlng;
      onMapClick({ lat, lng });

      if (selectedFeatures.points) {
        addMarker({ lat, lng });
      }
      if (selectedFeatures.lines) {
        addMarker({ lat, lng });
        setPolylinePoints([...polylinePoints, [lat, lng]]);
        if (polylinePoints.length === 1) {
          addPolyline([...polylinePoints, [lat, lng]]);
          setPolylinePoints([]);
        }
      }
      if (selectedFeatures.areas) {
        addMarker({ lat, lng });
        setPolygonPoints([...polygonPoints, [lat, lng]]);
      }
    },
  });

  return null;
};

const MyMap = forwardRef(({ onDataChange, selectedFeatures, onMapClick }, ref) => {
  const [markers, setMarkers] = useState([]);
  const [polylines, setPolylines] = useState([]);
  const [polygons, setPolygons] = useState([]);
  const [boundaryCoordinates, setBoundaryCoordinates] = useState([]);
  const [currentPolygon, setCurrentPolygon] = useState([]);

  useImperativeHandle(ref, () => ({
    reset: () => {
      setMarkers([]);
      setPolylines([]);
      setPolygons([]);
      setCurrentPolygon([]);
    },
    finishPolygon: () => {
      if (currentPolygon.length >= 3) {
        setPolygons((prevPolygons) => [...prevPolygons, currentPolygon]);
        setCurrentPolygon([]);
      } else {
        alert('A polygon must have at least 3 points.');
      }
    },
  }));

  useEffect(() => {
    const fetchBoundaryCoordinates = async () => {
      try {
        const response = await fetch('/coord.csv');
        const csvString = await response.text();
        const parsedCoordinates = csvString
          .split('\n')
          .slice(1)
          .map((row) => {
            const [longitude, latitude] = row.split(',');
            return [parseFloat(latitude), parseFloat(longitude)];
          })
          .filter((coord) => !isNaN(coord[0]) && !isNaN(coord[1]));

        setBoundaryCoordinates(parsedCoordinates);
      } catch (error) {
        console.error('Error fetching the CSV file:', error);
      }
    };

    fetchBoundaryCoordinates();
  }, []);

  const addMarker = (marker) => {
    setMarkers((prevMarkers) => [...prevMarkers, marker]);
    if (selectedFeatures.areas) {
      setCurrentPolygon((prevPolygon) => [...prevPolygon, [marker.lat, marker.lng]]);
    }
  };

  const addPolyline = (polyline) => {
    setPolylines((prevPolylines) => [...prevPolylines, polyline]);
  };

  const addPolygon = (polygon) => {
    setPolygons((prevPolygons) => [...prevPolygons, polygon]);
  };

  useEffect(() => {
    const latitudes = markers.map((marker) => marker.lat);
    const longitudes = markers.map((marker) => marker.lng);

    polylines.forEach((polyline) => {
      polyline.forEach((point) => {
        latitudes.push(point[0]);
        longitudes.push(point[1]);
      });
    });

    polygons.forEach((polygon) => {
      polygon.forEach((point) => {
        latitudes.push(point[0]);
        longitudes.push(point[1]);
      });
    });

    onDataChange({ latitudes, longitudes });
  }, [markers, polylines, polygons, onDataChange]);
console.log(GEO_TIFF_URL);
  return (
    <MapContainer center={GULF_OF_MEXICO_CENTER} zoom={INITIAL_ZOOM} className="h-full w-full">
      <TileLayer
        url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        attribution="&copy; OpenStreetMap contributors"
      />
      
      <GeoTIFFLayer urls={tiffFiles} interval={3000} /> 

      <MapEventHandler
        addMarker={addMarker}
        addPolyline={addPolyline}
        addPolygon={addPolygon}
        onMapClick={onMapClick}
        selectedFeatures={selectedFeatures}
      />
      {markers.map((marker, idx) => (
        <Marker key={idx} position={[marker.lat, marker.lng]} icon={new L.Icon.Default()}>
          <Popup>
            Coordinates: {marker.lat.toFixed(4)}, {marker.lng.toFixed(4)}
          </Popup>
        </Marker>
      ))}
      {polylines.map((polyline, idx) => (
        <Polyline key={idx} positions={polyline} color="blue" />
      ))}
      {polygons.map((polygon, idx) => (
        <Polygon key={idx} positions={polygon} color="green" />
      ))}
      {currentPolygon.length > 0 && <Polygon positions={currentPolygon} color="red" />}
      <Polygon
        positions={boundaryCoordinates}
        pathOptions={{ fillColor: 'white', fillOpacity: 0.1, color: 'grey', weight: 1 }}
      />
    </MapContainer>
  );
});

export default MyMap;
