import React, { useState, useRef, useEffect } from 'react';
import Message from './Message';
import Spinner from './Spinner';
import { Send } from 'lucide-react';

function ChatBox() {
  const BACKEND_URL = process.env.REACT_APP_BACKEND_URL || 'http://localhost:8000';

  const [messages, setMessages] = useState([
    { sender: 'bot', text: 'Upload a document, then ask your question.', isStructure: false }
  ]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [documentId, setDocumentId] = useState(() => localStorage.getItem('document_id') || '');
  const [lastUploadInfo, setLastUploadInfo] = useState(null);
  const messagesEndRef = useRef(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleUpload = async (e) => {
    const file = e.target.files?.[0];
    if (!file || isUploading) return;
    setIsUploading(true);

    try {
      const formData = new FormData();
      formData.append('file', file);

      const res = await fetch(`${BACKEND_URL}/upload`, {
        method: 'POST',
        body: formData
      });
      if (!res.ok) throw new Error(`Upload failed: ${res.status}`);
      const data = await res.json();

      localStorage.setItem('document_id', data.document_id);
      setDocumentId(data.document_id);
      setLastUploadInfo(data);

      setMessages(prev => [
        ...prev,
        { sender: 'bot', text: `Uploaded ${data.filename}. Chunks: ${data.chunks_created}.`, isStructure: false }
      ]);
    } catch (err) {
      console.error('Upload error:', err);
      setMessages(prev => [...prev, { sender: 'bot', text: `Upload error: ${String(err)}`, isStructure: false }]);
    } finally {
      setIsUploading(false);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    const userMessage = { sender: 'user', text: input, isStructure: false };
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    try {
      const payload = {
        query: userMessage.text,
        document_id: documentId || localStorage.getItem('document_id') || null,
        use_context: true,
        max_tokens: 500
      };

      const res = await fetch(`${BACKEND_URL}/query`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      if (!res.ok) throw new Error(`Query failed: ${res.status}`);
      const data = await res.json();

      // Debug trace in console for step-by-step visibility
      if (data.debug) console.log('RAG debug:', data.debug);

      // Show answer
      setMessages(prev => [...prev, { sender: 'bot', text: data.answer, isStructure: false }]);

      // Show sources (top 3) as a compact follow-up message
      if (Array.isArray(data.sources) && data.sources.length) {
        const srcLines = data.sources.map((s, i) => `#${i + 1} (${(s.relevance_score ?? 0).toFixed(2)}): ${s.content}`).join('\n');
        setMessages(prev => [...prev, { sender: 'bot', text: `Sources:\n${srcLines}`, isStructure: false }]);
      }
    } catch (error) {
      console.error('Query error:', error);
      setMessages(prev => [...prev, { sender: 'bot', text: 'Sorry, I encountered an error. Please try again.', isStructure: false }]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="chatbox-container">
      <div className="messages-area">
        <div className="upload-bar">
          <input type="file" accept=".pdf,.txt,.md,.csv" onChange={handleUpload} disabled={isUploading} />
          {documentId && (
            <span className="doc-badge">Doc: {documentId}</span>
          )}
        </div>
        {messages.map((msg, idx) => (
          <Message 
            key={idx} 
            sender={msg.sender} 
            content={msg.isStructure ? msg.data : msg.text}
            isStructure={msg.isStructure}
          />
        ))}
        {(isLoading || isUploading) && <div className="typing-indicator"><Spinner simple={true} /></div>}
        <div ref={messagesEndRef} />
      </div>
      <footer className="chat-footer">
        <form onSubmit={handleSubmit} className="chat-form">
          <input
            type="text"
            placeholder="e.g., Is a 46-year-old male covered for knee surgery in Pune?"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            disabled={isLoading}
          />
          <button type="submit" disabled={isLoading || !input.trim()}>
            <Send size={20} />
          </button>
        </form>
        <p className="footer-note">Powered by Large Language Models</p>
      </footer>
    </div>
  );
}

export default ChatBox;