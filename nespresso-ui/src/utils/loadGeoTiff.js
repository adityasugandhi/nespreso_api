import GeoTIFF from 'geotiff';

export const loadGeoTiff = async (url) => {
  const response = await fetch(url);
  const arrayBuffer = await response.arrayBuffer();
  const tiff = await GeoTIFF.fromArrayBuffer(arrayBuffer);
  const image = await tiff.getImage(0); // Get the first image
  const data = await image.readRasters(); // Read raster data
  const boundingBox = image.getBoundingBox(); // Get bounding box for the image
  return { image, data, boundingBox };
};