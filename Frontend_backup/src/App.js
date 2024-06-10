import React, { useState } from 'react';

function App() {
  const [emlFile, setEmlFile] = useState(null);
  const [isSaved, setIsSaved] = useState(false);
  const [result, setResult] = useState(null);

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file && file.name.endsWith('.eml')) {
      setEmlFile(file);
    } else {
      alert('Please select a valid .eml file');
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (emlFile) {
      const formData = new FormData();
      formData.append('file', emlFile);
      try {
        const response = await fetch('http://127.0.0.1:5000', {
          method: 'POST',
          body: formData,
          headers: {
            'Accept': 'application/json'
          }
        });

        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        setIsSaved(true);
        setResult(data);
      } catch (error) {
        console.error('Error uploading file:', error);
        alert('Failed to upload file');
      }
    } else {
      alert('No file selected');
    }
  };

  const handleRestart = () => {
    setIsSaved(false);
    setResult(null);
    setEmlFile(null);
  };

  const formatResult = (result) => {
    let formattedData = '';
  
    if (result) {
      if (result.duplicates && result.duplicates.length > 0) {
        formattedData += 'Duplicate Files:\n';
        result.duplicates.forEach((duplicate) => {
          formattedData += `File 1: ${duplicate.file1}\nFile 2: ${duplicate.file2}\nDuplicate: ${duplicate.duplicate}\nScore: ${duplicate.score}\n\n`;
        });
      } else {
        formattedData += 'No duplicates found\n';
      }
  
      if (result.files && result.files.length > 0) {
        formattedData += 'All Files:\n';
        result.files.forEach((file) => {
          formattedData += `${file}\n`;
        });
      }
    } else {
      formattedData = 'No result available';
    }
  
    return formattedData;
  };
  
  

  return (
    <div className="relative flex flex-col h-screen bg-gray-100">
      <header className="flex items-center justify-between p-4 bg-white shadow-lg">
        <div className="flex items-center space-x-6">
          <img
            src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcT0RoUK9oM2JlrDsIGdBp4NIyktSbqtoMA6ReLPYdNTXw&s"
            alt="SAP logo"
            className="w-16 h-16"
          />
        </div>
      </header>
      <div className="flex items-center justify-center flex-grow">
        <div className="p-20 bg-white rounded-lg shadow-lg">
          <h1 className="mb-4 text-2xl font-bold text-gray-800">Duplicate Email Predictor</h1>
          {isSaved ? (
            <>
              <h1 className="text-xl font-semibold text-gray-700">Upload Result:</h1>
              {result && (
                <div>
                  <h2 className="mt-4 text-lg font-semibold text-gray-700">Result:</h2>
                  <pre className="mt-2 p-2 bg-gray-200 rounded">
                    {formatResult(result)}
                  </pre>
                  <button
                    onClick={handleRestart}
                    className="mt-4 px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 focus:outline-none"
                  >
                    Start Again
                  </button>
                </div>
              )}
            </>
          ) : (
            <>
              <h1 className="mb-6 text-xl font-semibold text-gray-700">Upload a .eml file</h1>
              <form onSubmit={handleSubmit}>
                <div className="mb-6">
                  <input
                    type="file"
                    accept=".eml"
                    onChange={handleFileChange}
                    className="w-full px-6 py-3 border border-gray-300 rounded-lg focus:outline-none focus:border-blue-500"
                  />
                </div>
                {emlFile && (
                  <div className="mb-6">
                    <p className="text-gray-700">Selected .eml file: {emlFile.name}</p>
                  </div>
                )}
                <button
                  type="submit"
                  className="w-full px-7 py-3.5 font-semibold text-white bg-blue-500 rounded-lg hover:bg-blue-600 focus:outline-none"
                >
                  Upload File
                </button>
              </form>
            </>
          )}
        </div>
      </div>
    </div>
  );
}

export default App;
