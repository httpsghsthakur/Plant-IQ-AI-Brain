import React, { useState, useEffect } from 'react';
import { Activity, AlertCircle, Droplets, Thermometer, Box, Users } from 'lucide-react';
import { getDashboardSummary } from '../services/aiBrainApi';

const DashboardHome = () => {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const result = await getDashboardSummary();
        setData(result);
      } catch (error) {
        console.error("Failed to fetch dashboard data:", error);
      } finally {
        setLoading(false);
      }
    };
    fetchData();
  }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen bg-gray-50">
        <div className="animate-spin text-green-600">
           <Activity size={48} />
        </div>
      </div>
    );
  }

  // --- RENDERING WITH LAYMAN TERMINOLOGY ---
  // Backend returns: nursery_wellness_score, urgent_care_alerts, summary: { growing_areas, pending_daily_tasks, sensor_system_health }

  return (
    <div className="p-8 bg-gray-50 min-h-screen">
      <div className="max-w-7xl mx-auto space-y-8">
        {/* Header */}
        <div className="flex justify-between items-end">
          <div>
            <h1 className="text-3xl font-extrabold text-gray-900 tracking-tight">AI Insights Dashboard</h1>
            <p className="text-gray-500 mt-2">Real-time nursery health and expert recommendations.</p>
          </div>
          <div className="text-right">
            <span className="text-xs font-bold uppercase text-gray-400">Current Health</span>
            <div className={`text-2xl font-black ${data?.nursery_wellness_score > 80 ? 'text-green-600' : 'text-orange-500'}`}>
              {data?.nursery_wellness_score || 0}% Wellness
            </div>
          </div>
        </div>

        {/* Top Cards */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
          <StatCard 
            icon={<Activity className="text-green-600" />} 
            label="Nursery Wellness" 
            value={`${data?.nursery_wellness_score}%`} 
            subValue="System-wide health"
          />
          <StatCard 
            icon={<Users className="text-blue-600" />} 
            label="Pending Team Tasks" 
            value={data?.summary?.pending_daily_tasks || 0} 
            subValue="Needs completion today"
          />
          <StatCard 
            icon={<Box className="text-purple-600" />} 
            label="Growing Areas" 
            value={data?.summary?.growing_areas || 0} 
            subValue="Active Zones"
          />
          <StatCard 
            icon={<Droplets className="text-cyan-600" />} 
            label="Equipment Status" 
            value="Active" 
            subValue={data?.summary?.sensor_system_health || "Healthy"}
          />
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Urgent Care Alerts */}
          <div className="lg:col-span-2 bg-white rounded-3xl p-6 shadow-sm border border-gray-100">
            <div className="flex justify-between items-center mb-6">
              <h2 className="text-xl font-bold flex items-center gap-2">
                <AlertCircle className="text-orange-500" />
                Urgent Care Alerts
              </h2>
              <span className="bg-orange-100 text-orange-600 px-3 py-1 rounded-full text-xs font-bold">
                {data?.urgent_care_alerts?.length || 0} Required Actions
              </span>
            </div>

            <div className="space-y-4">
              {data?.urgent_care_alerts?.map((alert, idx) => (
                <div key={idx} className="flex gap-4 p-4 rounded-2xl bg-orange-50 border border-orange-100 animate-pulse-slow">
                  <div className="bg-white p-2 rounded-xl shadow-sm h-fit">
                    <Activity className="text-orange-500" size={20} />
                  </div>
                  <div>
                    <h4 className="font-bold text-gray-900">{alert.message || "Attention required"}</h4>
                    <p className="text-sm text-gray-600 mt-1">Suggested action needed in Zone {alert.zone_id || 'A'}.</p>
                  </div>
                </div>
              ))}
              {(!data?.urgent_care_alerts || data.urgent_care_alerts.length === 0) && (
                <div className="text-center py-12 text-gray-400 italic">
                  No urgent care needed at this moment. Everything is running smoothly.
                </div>
              )}
            </div>
          </div>

          {/* Quick Stats Panel */}
          <div className="bg-white rounded-3xl p-6 shadow-sm border border-gray-100">
             <h2 className="text-xl font-bold mb-6">Plant Wellness Tips</h2>
             <div className="space-y-6">
                <TipCard icon={<Thermometer />} title="Adjust Misting" text="Greenhouse B is getting slightly warm. High heat stress detected." color="text-red-500" bg="bg-red-50" />
                <TipCard icon={<Droplets />} title="Check Watering" text="Nursery Bed E soil moisture is perfect. No change needed." color="text-green-500" bg="bg-green-50" />
                <TipCard icon={<Users />} title="Team Efficiency" text="Worker attendance is at 94% today. Excellent performance." color="text-blue-500" bg="bg-blue-50" />
             </div>
          </div>
        </div>
      </div>
    </div>
  );
};

const StatCard = ({ icon, label, value, subValue }) => (
  <div className="bg-white p-6 rounded-3xl shadow-sm border border-gray-100 hover:shadow-md transition-shadow">
    <div className="w-10 h-10 rounded-2xl bg-gray-50 flex items-center justify-center mb-4">
      {icon}
    </div>
    <div className="text-2xl font-black text-gray-900">{value}</div>
    <div className="text-sm font-bold text-gray-400 mt-1 uppercase tracking-wider">{label}</div>
    <div className="text-xs text-gray-400 mt-1">{subValue}</div>
  </div>
);

const TipCard = ({ icon, title, text, color, bg }) => (
    <div className="flex gap-4">
        <div className={`${bg} ${color} p-3 rounded-2xl h-fit`}>
            {icon}
        </div>
        <div>
            <h4 className="font-bold text-gray-900">{title}</h4>
            <p className="text-sm text-gray-500 mt-1 leading-relaxed">{text}</p>
        </div>
    </div>
);

export default DashboardHome;
