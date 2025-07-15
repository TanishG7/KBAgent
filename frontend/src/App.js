import { useState, useEffect, useRef } from 'react';
import './App.css';
import { FiSend } from 'react-icons/fi';
import ReactMarkdown from 'react-markdown';

function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [suggestions, setSuggestions] = useState([]);
  const [context, setContext] = useState(null);
  const [messageHistory, setMessageHistory] = useState([]);
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  const sendMessageToBackend = async (question, isSuggestion = false) => {
    const userMessage = { text: question, isUser: true };
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);
    setSuggestions([]);
    
    try {    
      const updatedHistory = [
        ...messageHistory,
        { role: 'user', content: question }
      ];
      setMessageHistory(updatedHistory);

      const requestBody = {
        question,
        is_follow_up: isSuggestion,
        previous_context: context,
        message_history: updatedHistory,
      };

      const response = await fetch('/ai/api/search-chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json'
        },
        body: JSON.stringify(requestBody),
      });

      if (!response.ok) throw new Error(`HTTP error! Status: ${response.status}`);

      const data = await response.json();
      setIsLoading(false)
      
      // Create initial empty bot message
      const botMessage = { text: '', isUser: false };
      setMessages(prev => [...prev, botMessage]);
      
      const answer = data.answer;

      for (let i = 0; i < answer.length; i++) {
        await new Promise(resolve => setTimeout(resolve, 15));
        setMessages(prev => {
          const lastMessage = prev[prev.length - 1];
          const newText = lastMessage.text + answer[i];
          return [
            ...prev.slice(0, -1),
            {
              ...lastMessage,
              text: newText
            }
          ];
        });
      }
      
      if (data.was_context_valid_old_key === false) {
        const newHistory = [
          { role: 'user', content: question },
          { role: 'model', content: data.answer }
        ];
        
        setMessageHistory(newHistory);
      } else {
        setMessageHistory(prev => [
          ...prev,
          { role: 'model', content: data.answer }
        ]);
      }

      if (data.context) {
        setContext(data.context);
      }

      if (data.suggestions?.length > 0) {
        setSuggestions(data.suggestions);
      }
      
    } catch (error) {
      console.error('Fetch error:', error);
      setMessages(prev => [
        ...prev,
        { text: '⚠️ Error: Service unavailable', isUser: false }
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    const trimmed = input.trim();
    if (trimmed) {
      setContext(null);
      sendMessageToBackend(trimmed, false);
    }
  };

  const handleSuggestionClick = (suggestion) => {
    sendMessageToBackend(suggestion, true);
  };

  return (
    <div className="app">
      <header className="header">
      </header>

      <main className="main-content">
        {messages.length === 0 ? (
          <div className="welcome-screen">
            <h1>Ready when you are</h1>
            <form onSubmit={handleSubmit} className="input-form">
              <input
                ref={inputRef}
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder="Ask your AI assistant..."
                disabled={isLoading}
              />
              <button type="submit" disabled={isLoading || !input.trim()}>
                <FiSend size={20} />
              </button>
            </form>
          </div>
        ) : (
          <>
            <div className="messages-container">
              {messages.map((msg, idx) => (
                <div key={idx} className="message-wrapper">
                  <div className={`message ${msg.isUser ? 'user-message' : 'bot-message'}`}>
                    {msg.isUser ? (
                      msg.text
                    ) : (
                      <ReactMarkdown
                        components={{
                          a: ({node, ...props}) => (
                            <a {...props} 
                              target="_blank" 
                              rel="noopener noreferrer" 
                              style={{color: '#000'}} />
                          ),
                          // strong: ({node, ...props}) => (
                          //   <strong style={{
                          //     fontWeight: 'light', 
                          //     color: '#000',
                          //     backgroundColor: '#f0f8ff',
                          //     padding: '2px 4px',
                          //     borderRadius: '3px'
                          //   }} {...props} />
                          // ),
                          li: ({node, ...props}) => (
                            <li style={{
                              marginBottom: '8px',
                              paddingLeft: '4px',
                              lineHeight: '1'
                            }} {...props} />
                          ),
                          p: ({node, ...props}) => (
                            <p style={{
                              marginBottom: '12px',
                              lineHeight: '1'
                            }} {...props} />
                          ),
                          ul: ({node, ...props}) => (
                            <ul style={{
                              paddingLeft: '20px',
                              margin: '12px 0'
                            }} {...props} />
                          ),
                          ol: ({node, ...props}) => (
                            <ol style={{
                              paddingLeft: '20px',
                              margin: '12px 0'
                            }} {...props} />
                          )
                        }}
                      >
                        {msg.text}
                      </ReactMarkdown>
                    )}
                  </div>
                  {!msg.isUser && idx === messages.length - 1 && suggestions.length > 0 && (
                    <div className="suggestions">
                      <div className="suggestions-content">
                        <span className="suggestions-title">Related questions:</span>
                        <div className="suggestions-list">
                          {suggestions.map((suggestion, index) => (
                            <button
                              key={index}
                              className="suggestion"
                              onClick={() => handleSuggestionClick(suggestion)}
                            >
                              {suggestion}
                            </button>
                          ))}
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              ))}
              {isLoading && (
                <div className="bot-message typing">
                  AI is thinking<span className="dot">.</span><span className="dot">.</span><span className="dot">.</span>
                </div>
              )}
              <div ref={messagesEndRef} />
            </div>

            <form onSubmit={handleSubmit} className="input-form">
              <input
                ref={inputRef}
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder="Ask your AI assistant..."
                disabled={isLoading}
              />
              <button type="submit" disabled={isLoading || !input.trim()}>
                <FiSend size={20} />
              </button>
            </form>
          </>
        )}
      </main>
    </div>
  );
}
export default App;