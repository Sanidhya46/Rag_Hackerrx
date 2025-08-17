import React, { useState } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import UploadPage from './pages/UploadPage';
import ChatPage from './pages/ChatPage';
import UploadForm from './components/UploadForm'; 
import ChatBox from './components/ChatBox'; 
import './index.css';

function App() {
  const [uploadedFile, setUploadedFile] = useState(null);
  return (
    <Router>
      <div className="app-shell">
        <Routes>
          <Route path="/" element={<UploadPage />} />
          <Route path="/chat" element={<ChatPage />} />
          <Route path="*" element={<Navigate to="/" />} />
        </Routes>
      </div>
    </Router>
  );
}

export default App;