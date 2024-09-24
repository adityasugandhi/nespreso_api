import React, { useState, useEffect, useImperativeHandle, forwardRef } from 'react';
import { MapContainer, TileLayer, Marker, Popup, useMapEvents, Polyline, Polygon, useMap } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import L from '../utils/leafleticon';
import GeoTIFF, { fromUrl, fromUrls, fromArrayBuffer, fromBlob } from 'geotiff';

const GULF_OF_MEXICO_CENTER = [25.5, -91.0];
const INITIAL_ZOOM = 6;

const MapEventHandler = ({ addMarker, addPolyline, addPolygon, onMapClick, selectedFeatures }) => {
  const [polylinePoints, setPolylinePoints] = useState([]);
  const [polygonPoints, setPolygonPoints] = useState([]);
  const maps = useMap();

  useEffect(() => {
    const loadGeoTiff = async () => {
      try {
        const response = await fetch('/2021-05_2021-05-01.tiff'); // Update with the correct path
        // console.log(response)
        // const tiff = await fromArrayBuffer(...);
        // console.log(response.arrayBuffer())
        const arrayBuffer = await response.arrayBuffer();
        // const tiff = await fromArrayBuffer(arrayBuffer);
       
        const tiff = await fromArrayBuffer(arrayBuffer);
        
        const image = await tiff.getImage();
        // console.log(image)
        // console.log(tiff)
        const [minX, minY, maxX, maxY] = image.getBoundingBox();
        
        const bounds = [[minY, minX], [maxY, maxX]];
        console.log(bounds)
        // Create and add GeoTIFF image overlay to the map
        const rasterData = await image.readRasters();
        console.log("RasterData",rasterData)
        const cleanedRasterData = rasterData[0].map(value => (value === -9999 ? 0 : value)); // Replace -9999 with 0 or another no-data value

        const imageOverlay = L.imageOverlay(cleanedRasterData, bounds);
        console.log(imageOverlay)
        imageOverlay.addTo(maps);
        console.log("Maps",maps)
      } catch (error) {
        console.error('Error loading GeoTIFF:', error);
      }
    };

    loadGeoTiff();
  }, [maps]);

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
  const [geoTiffLayer, setGeoTiffLayer] = useState(null); // Store GeoTIFF layer

  useImperativeHandle(ref, () => ({
    reset: () => {
      setMarkers([]);
      setPolylines([]);
      setPolygons([]);
      setCurrentPolygon([]);
      if (geoTiffLayer) {
        geoTiffLayer.remove(); // Remove GeoTIFF layer
        setGeoTiffLayer(null); // Reset GeoTIFF layer
      }
    },
    finishPolygon: () => {
      if (currentPolygon.length >= 3) {
        setPolygons(prevPolygons => [...prevPolygons, currentPolygon]);
        setCurrentPolygon([]);
      } else {
        alert("A polygon must have at least 3 points.");
      }
    }
  }));

  useEffect(() => {
    const fetchBoundaryCoordinates = async () => {
      try {
        const response = await fetch('/coord.csv');
        const csvString = await response.text();
        const parsedCoordinates = csvString.split('\n')
          .slice(1)
          .map(row => {
            const [longitude, latitude] = row.split(',');
            return [parseFloat(latitude), parseFloat(longitude)];
          })
          .filter(coord => !isNaN(coord[0]) && !isNaN(coord[1]));

        setBoundaryCoordinates(parsedCoordinates);
      } catch (error) {
        console.error('Error fetching the CSV file:', error);
      }
    };

    fetchBoundaryCoordinates();
  }, []);

  const addMarker = (marker) => {
    setMarkers(prevMarkers => [...prevMarkers, marker]);
    if (selectedFeatures.areas) {
      setCurrentPolygon(prevPolygon => [...prevPolygon, [marker.lat, marker.lng]]);
    }
  };

  const addPolyline = (polyline) => {
    setPolylines(prevPolylines => [...prevPolylines, polyline]);
  };

  const addPolygon = (polygon) => {
    setPolygons(prevPolygons => [...prevPolygons, polygon]);
  };

  useEffect(() => {
    const latitudes = markers.map(marker => marker.lat);
    const longitudes = markers.map(marker => marker.lng);

    polylines.forEach(polyline => {
      polyline.forEach(point => {
        latitudes.push(point[0]);
        longitudes.push(point[1]);
      });
    });

    polygons.forEach(polygon => {
      polygon.forEach(point => {
        latitudes.push(point[0]);
        longitudes.push(point[1]);
      });
    });

    onDataChange({ latitudes, longitudes });
  }, [markers, polylines, polygons, onDataChange]);

  return (
    <MapContainer center={GULF_OF_MEXICO_CENTER} zoom={INITIAL_ZOOM} className="h-full w-full">
      <TileLayer
        url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        attribution='&copy; <a href="http://osm.org/copyright">OpenStreetMap</a> contributors'
      />
      <MapEventHandler 
        addMarker={addMarker} 
        addPolyline={addPolyline} 
        addPolygon={addPolygon} 
        onMapClick={onMapClick}
        selectedFeatures={selectedFeatures}
      />
      {markers.map((marker, idx) => (
        <Marker key={idx} position={[marker.lat, marker.lng]} icon={new L.Icon.Default()}>
          <Popup>Coordinates: {marker.lat.toFixed(4)}, {marker.lng.toFixed(4)}</Popup>
        </Marker>
      ))}
      {polylines.map((polyline, idx) => (
        <Polyline key={idx} positions={polyline} color="blue" />
      ))}
      {polygons.map((polygon, idx) => (
        <Polygon key={idx} positions={polygon} color="green" />
      ))}
      {currentPolygon.length > 0 && (
        <Polygon positions={currentPolygon} color="red" />
      )}
      <Polygon
        positions={boundaryCoordinates}
        pathOptions={{ fillColor: 'purple', fillOpacity: 0.3, color: 'purple', weight: 2 }}
      />
    </MapContainer>
  );
});

export default MyMap;
