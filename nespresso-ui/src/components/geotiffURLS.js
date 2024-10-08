const fs = require('fs');
const path = require('path');

// Define the folder path you want to iterate over
// const folderPath = './your-folder-path'; // Replace with your folder path

// Function to iterate through the folder
const iterateFolder = (folderPath, ArrayUrls) => {
  fs.readdir(folderPath, (err, files) => {
    if (err) {
      return console.error('Unable to scan folder:', err);
    }

    // Loop through the files/folders in the directory
    files.forEach((file) => {
      const fullPath = path.join(folderPath, file);

      // Check if it's a file or directory
      fs.stat(fullPath, (err, stats) => {
        if (err) {
          console.error('Error getting file stats:', err);
          return;
        }

        if (stats.isFile()) {
          console.log(`File: ${fullPath}`);
          ArrayUrls.push(fullPath);
        } else if (stats.isDirectory()) {
          console.log(`Directory: ${fullPath}`);
          // Recursively iterate through sub-directories
          iterateFolder(fullPath);
        }
      });
    });
  });
};

// Export the function so it can be used in other files
module.exports = iterateFolder;
