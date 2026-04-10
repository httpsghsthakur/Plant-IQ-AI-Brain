import React, { useState, useRef, useEffect } from 'react';
import { MessageCircle, X, Send, Leaf } from 'lucide-react';
import { sendChatQuery } from '../services/aiBrainApi';

const AdvisorChat = () => {
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState([
    { role: 'ai', text: 'Hello! I am your PlantIQ Advisor. How can I help you with your nursery today?' }
  ]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const scrollRef = useRef(null);

  useEffect(() => {
    if (scrollRef.current) {
        scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages]);

  const handleSend = async () => {
    if (!input.trim()) return;

    const userMessage = input.trim();
    setMessages(prev => [...prev, { role: 'user', text: userMessage }]);
    setInput('');
    setIsLoading(true);

    try {
      const result = await sendChatQuery(userMessage);
      const aiResponse = result.response?.answer || "I'm sorry, I couldn't process that. Can you try again?";
      
      setMessages(prev => [...prev, { role: 'ai', text: aiResponse }]);
    } catch (error) {
      setMessages(prev => [...prev, { role: 'ai', text: "I'm having trouble connecting to the brain. Please check your connection." }]);
    } finally {
      setIsLoading(false);
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
          <span className="max-w-0 overflow-hidden group-hover:max-w-xs transition-all duration-300 font-medium">Ask Advisor</span>
        </button>
      )}

      {/* Chat Window */}
      {isOpen && (
        <div className="bg-white rounded-2xl shadow-2xl w-96 h-[550px] flex flex-col border border-gray-100 animate-in slide-in-from-bottom-5 duration-300">
          {/* Header */}
          <div className="bg-green-600 p-4 rounded-t-2xl flex justify-between items-center text-white">
            <div className="flex items-center gap-2">
              <Leaf size={20} />
              <div>
                <h3 className="font-bold text-sm">PlantIQ Advisor</h3>
                <p className="text-[10px] text-green-100">Intelligent Nursery Expert</p>
              </div>
            </div>
            <button onClick={() => setIsOpen(false)} className="hover:bg-green-500 rounded-full p-1">
              <X size={20} />
            </button>
          </div>

          {/* Messages Area */}
          <div ref={scrollRef} className="flex-1 overflow-y-auto p-4 space-y-4 bg-gray-50">
            {messages.map((msg, i) => (
              <div key={i} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                <div className={`max-w-[85%] p-3 rounded-2xl text-sm shadow-sm ${
                  msg.role === 'user' 
                    ? 'bg-green-600 text-white rounded-tr-none' 
                    : 'bg-white text-gray-800 rounded-tl-none border border-gray-200'
                }`}>
                  {msg.text}
                </div>
              </div>
            ))}
            {isLoading && (
              <div className="flex justify-start">
                <div className="bg-white border border-gray-200 p-3 rounded-2xl rounded-tl-none animate-pulse flex gap-1">
                  <div className="w-1.5 h-1.5 bg-gray-300 rounded-full animate-bounce"></div>
                  <div className="w-1.5 h-1.5 bg-gray-300 rounded-full animate-bounce delay-100"></div>
                  <div className="w-1.5 h-1.5 bg-gray-300 rounded-full animate-bounce delay-200"></div>
                </div>
              </div>
            )}
          </div>

          {/* Input Area */}
          <div className="p-4 border-t border-gray-100">
            <div className="flex gap-2">
              <input
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyPress={(e) => e.key === 'Enter' && handleSend()}
                placeholder="Ask about your plants..."
                className="flex-1 bg-gray-100 rounded-xl px-4 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-green-500 transition-all"
              />
              <button
                onClick={handleSend}
                disabled={isLoading}
                className="bg-green-600 text-white p-2 rounded-xl hover:bg-green-700 disabled:opacity-50 transition-all"
              >
                <Send size={18} />
              </button>
            </div>
            <p className="text-[10px] text-gray-400 mt-2 text-center italic">
              Advisor is trained on your nursery live data.
            </p>
          </div>
        </div>
      )}
    </div>
  );
};

export default AdvisorChat;
