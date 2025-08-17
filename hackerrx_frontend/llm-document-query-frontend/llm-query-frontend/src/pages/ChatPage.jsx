import React from 'react';
import ChatBox from '../components/ChatBox';
import { ArrowLeft, FileText } from 'lucide-react';
import { Link } from 'react-router-dom';

function ChatPage({ file }) {
  return (
    <div className="chat-page-container">
       <header className="chat-header">
         <Link to="/" className="back-link">
            <ArrowLeft size={20} />
            <span>Upload New</span>
         </Link>
         <div className="file-info">
            <FileText size={20} />
            <span className="file-name">{file?.name || 'Document'}</span>
         </div>
         <div className="header-placeholder"></div>
       </header>
      <ChatBox />
    </div>
  )
}

export default ChatPage;