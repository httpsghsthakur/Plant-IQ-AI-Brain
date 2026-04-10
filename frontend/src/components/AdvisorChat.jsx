import React, { useState, useRef, useEffect } from 'react';
import { MessageCircle, X, Send, Leaf, Loader2 } from 'lucide-react';
import axios from 'axios';

const SMARTFARM_CHATBOT_URL = 'https://smartfarm-chatbot.onrender.com';

const AdvisorChat = () => {
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState([
    { role: 'ai', text: 'Hello! I am your PlantIQ Advisor. How can I help you with your nursery today?' }
  ]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const scrollRef = useRef(null);
  const inputRef = useRef(null);
  const [sessionId] = useState(() => `session_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`);

  useEffect(() => {
    if (scrollRef.current) {
        scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages, isLoading]);

  useEffect(() => {
    if (isOpen && inputRef.current) {
      setTimeout(() => inputRef.current?.focus(), 150);
    }
  }, [isOpen]);

  const handleSend = async () => {
    if (!input.trim() || isLoading) return;

    const userMessage = input.trim();
    setMessages(prev => [...prev, { role: 'user', text: userMessage }]);
    setInput('');
    setIsLoading(true);

    try {
      const response = await axios.post(
        `${SMARTFARM_CHATBOT_URL}/chat`,
        { message: userMessage, session_id: sessionId },
        { timeout: 30000 }
      );

      let aiText = '';
      if (typeof response.data === 'string') {
        aiText = response.data;
      } else if (response.data?.response) {
        aiText = typeof response.data.response === 'string'
          ? response.data.response
          : response.data.response?.answer || response.data.response?.message || JSON.stringify(response.data.response);
      } else if (response.data?.answer) {
        aiText = response.data.answer;
      } else if (response.data?.message) {
        aiText = response.data.message;
      } else {
        aiText = JSON.stringify(response.data);
      }
      
      setMessages(prev => [...prev, { role: 'ai', text: aiText }]);
    } catch (error) {
      console.error("Chat error:", error);
      const errorMsg = error?.response?.data?.message || "I'm having trouble connecting to the brain. Please check your connection.";
      setMessages(prev => [...prev, { role: 'ai', text: `⚠️ ${errorMsg}` }]);
    } finally {
      setIsLoading(false);
      setTimeout(() => inputRef.current?.focus(), 100);
    }
  };

  return (
    <div className="fixed bottom-6 right-6 z-50">
      {/* Chat Bubble Icon */}
      {!isOpen && (
        <button
          onClick={() => setIsOpen(true)}
          className="bg-green-600 hover:bg-green-700 text-white p-4 rounded-full shadow-2xl transition-all scale-100 hover:scale-110 active:scale-95 flex items-center gap-2 group"
        >
          <MessageCircle size={28} />
          <span className="max-w-0 overflow-hidden group-hover:max-w-xs transition-all duration-300 font-medium whitespace-nowrap px-1">Ask Advisor</span>
        </button>
      )}

      {/* Chat Window */}
      {isOpen && (
        <div className="bg-white rounded-2xl shadow-2xl w-96 h-[550px] flex flex-col border border-gray-100 animate-in slide-in-from-bottom-5 duration-300">
          {/* Header */}
          <div className="bg-green-600 p-4 rounded-t-2xl flex justify-between items-center text-white">
            <div className="flex items-center gap-3">
              <div className="bg-green-500/30 p-2 rounded-lg">
                <Leaf size={18} />
              </div>
              <div>
                <h3 className="font-bold text-sm leading-tight">PlantIQ Advisor</h3>
                <div className="flex items-center gap-1.5 mt-0.5">
                  <div className="w-1.5 h-1.5 bg-green-300 rounded-full animate-pulse"></div>
                  <p className="text-[10px] uppercase font-bold text-green-100 tracking-wider">SmartFarm API Live</p>
                </div>
              </div>
            </div>
            <button onClick={() => setIsOpen(false)} className="hover:bg-green-500 rounded-full p-1.5 transition-colors">
              <X size={18} />
            </button>
          </div>

          {/* Messages Area */}
          <div ref={scrollRef} className="flex-1 overflow-y-auto p-4 space-y-4 bg-gray-50/50">
            {messages.map((msg, i) => (
              <div key={i} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                <div className={`max-w-[85%] p-3.5 rounded-2xl text-sm shadow-sm leading-relaxed ${
                  msg.role === 'user' 
                    ? 'bg-green-600 text-white rounded-tr-none shadow-green-600/20' 
                    : 'bg-white text-gray-800 rounded-tl-none border border-gray-200'
                }`}>
                  {msg.text}
                </div>
              </div>
            ))}
            {isLoading && (
              <div className="flex justify-start">
                <div className="bg-white border border-gray-200 px-4 py-3 rounded-2xl rounded-tl-none shadow-sm flex items-center gap-2">
                  <Loader2 size={16} className="text-green-600 animate-spin" />
                  <span className="text-xs font-semibold text-gray-400">AI is thinking...</span>
                </div>
              </div>
            )}
          </div>

          {/* Input Area */}
          <div className="p-4 border-t border-gray-100 bg-white rounded-b-2xl">
            <div className="flex gap-2 relative">
              <input
                ref={inputRef}
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && handleSend()}
                placeholder="Ask about your plants..."
                autoFocus
                className="w-full bg-gray-50 border border-gray-200 rounded-xl px-4 py-3 pr-12 text-sm focus:outline-none focus:ring-2 focus:ring-green-500/20 focus:border-green-500 transition-all font-medium placeholder:text-gray-400"
              />
              <button
                onClick={handleSend}
                disabled={isLoading || !input.trim()}
                className="absolute right-1 top-1 bottom-1 bg-green-600 text-white px-3 rounded-lg hover:bg-green-700 disabled:opacity-40 transition-all flex items-center justify-center disabled:cursor-not-allowed"
              >
                <Send size={16} />
              </button>
            </div>
            <p className="text-[10px] text-gray-400 mt-2 text-center uppercase tracking-widest font-bold">
              Powered by SmartFarm AI
            </p>
          </div>
        </div>
      )}
    </div>
  );
};

export default AdvisorChat;

