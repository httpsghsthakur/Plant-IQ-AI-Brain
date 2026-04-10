import React from 'react';
import DashboardHome from './pages/DashboardHome';
import AdvisorChat from './components/AdvisorChat';

function App() {
  // Initialize nursery_id in localStorage for testing if not present
  if (!localStorage.getItem('plantiq_nursery_id')) {
    localStorage.setItem('plantiq_nursery_id', 'Green-Valley-Walnut-Nursery');
  }

  return (
    <div className="App">
      <DashboardHome />
      <AdvisorChat />
      
      {/* Toast notifications or other global elements could go here */}
    </div>
  );
}

export default App;
