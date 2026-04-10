import axios from 'axios';

// The AI Brain is the hub for all Machine Learning insights
const AI_BRAIN_URL = 'https://plant-iq-ai-brain.onrender.com/api';

const aiBrainApi = axios.create({
  baseURL: AI_BRAIN_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

/**
 * STRICT INTEGRATION RULES:
 * 1. Multi-Tenancy: Fetch nursery_id from session and append to query.
 * 2. Security: Append Supabase JWT to Authorization header.
 */
aiBrainApi.interceptors.request.use(
  (config) => {
    // 1. Get Nursery ID (Assuming it's stored in localStorage or from a global state)
    const nursery_id = localStorage.getItem('plantiq_nursery_id');
    
    if (nursery_id) {
      config.params = {
        ...config.params,
        nursery_id: nursery_id,
      };
    }

    // 2. Get Supabase Auth Token
    const supabaseToken = localStorage.getItem('sb-token'); // Update based on your Supabase client config
    if (supabaseToken) {
      config.headers.Authorization = `Bearer ${supabaseToken}`;
    }

    return config;
  },
  (error) => Promise.reject(error)
);

export default aiBrainApi;

// --- API Methods ---

export const getDashboardSummary = async () => {
    const response = await aiBrainApi.get('/ai/dashboard');
    return response.data;
};

export const getEnvironmentalAnalytics = async () => {
    const response = await aiBrainApi.get('/ai/analyze/environment');
    return response.data;
};

export const getFinancialReport = async () => {
    const response = await aiBrainApi.get('/ai/analyze/financials');
    return response.data;
};

export const sendChatQuery = async (query) => {
    const nursery_id = localStorage.getItem('plantiq_nursery_id');
    const response = await aiBrainApi.post('/ai/chat', { 
        query, 
        nursery_id 
    });
    return response.data;
};
