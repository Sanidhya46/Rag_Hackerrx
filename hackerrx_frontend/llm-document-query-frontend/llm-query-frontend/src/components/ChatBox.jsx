import React, { useState, useRef, useEffect } from 'react';
import Message from './Message';
import Spinner from './Spinner';
import { Send } from 'lucide-react';

function ChatBox() {
  const [messages, setMessages] = useState([
    {
      sender: 'bot',
      text: "Your document is ready. Ask a question.",
      isStructure: false
    }
  ]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    const userMessage = { sender: 'user', text: input, isStructure: false };
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    try {
      const documentId = localStorage.getItem('document_id') || null;

      const resp = await fetch('http://localhost:8000/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: userMessage.text,
          document_id: documentId,
          use_context: true,
          max_tokens: 500
        })
      });

      const data = await resp.json().catch(() => ({}));

      if (resp.ok && data.answer) {
        // Simple text response from backend
        const botMessage = { sender: 'bot', text: data.answer, isStructure: false };
        setMessages(prev => [...prev, botMessage]);

        // Optionally append sources as a second message
        if (Array.isArray(data.sources) && data.sources.length) {
          const sourcesText = data.sources
            .map((s, i) => `Source ${i + 1}: ${s.content || ''}`)
            .join('\n');
          setMessages(prev => [...prev, { sender: 'bot', text: sourcesText, isStructure: false }]);
        }
      } else {
        const errText = data.detail || data.message || 'Query failed.';
        setMessages(prev => [...prev, { sender: 'bot', text: errText, isStructure: false }]);
      }
    } catch (error) {
      setMessages(prev => [...prev, { sender: 'bot', text: 'Network error. Try again.', isStructure: false }]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="chatbox-container">
      <div className="messages-area">
        {messages.map((msg, idx) => (
          <Message
            key={idx}
            sender={msg.sender}
            content={msg.isStructure ? msg.data : msg.text}
            isStructure={msg.isStructure}
          />
        ))}
        {isLoading && <div className="typing-indicator"><Spinner simple={true} /></div>}
        <div ref={messagesEndRef} />
      </div>
      <footer className="chat-footer">
        <form onSubmit={handleSubmit} className="chat-form">
          <input
            type="text"
            placeholder="Ask about your uploaded document…"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            disabled={isLoading}
          />
          <button type="submit" disabled={isLoading || !input.trim()}>
            <Send size={20} />
          </button>
        </form>
        <p className="footer-note">Powered by your document + RAG</p>
      </footer>
    </div>
  );
}

export default ChatBox;