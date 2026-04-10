import React, { useState, useEffect } from 'react';
import { supabase } from '../services/supabase'; // Assuming typical supabase client wrapper
import { 
  AlertCircle, CheckCircle, Activity, Info, 
  MapPin, Calendar, Users, Leaf, ArrowLeft,
  Clock, Plus, Edit, PlusSquare, QrCode
} from 'lucide-react';

const getStatusColor = (status) => {
  switch(status?.toLowerCase()) {
    case 'healthy': return 'bg-green-100 text-green-800 border-green-200';
    case 'needs attention': return 'bg-yellow-100 text-yellow-800 border-yellow-200';
    case 'diseased': return 'bg-red-100 text-red-800 border-red-200';
    case 'grafted': return 'bg-blue-100 text-blue-800 border-blue-200';
    default: return 'bg-gray-100 text-gray-800 border-gray-200';
  }
};

const getActionIcon = (type) => {
  switch(type?.toLowerCase()) {
    case 'watering': return <div className="p-2 bg-blue-100 rounded-full text-blue-600"><Activity size={16} /></div>;
    case 'grafting': return <div className="p-2 bg-purple-100 rounded-full text-purple-600"><Activity size={16} /></div>;
    case 'inspection': return <div className="p-2 bg-yellow-100 rounded-full text-yellow-600"><Info size={16} /></div>;
    default: return <div className="p-2 bg-gray-100 rounded-full text-gray-600"><CheckCircle size={16} /></div>;
  }
};

export default function PlantDetails({ plantId = "sample-id", onBack }) { // Support props if not using router yet
  const [plant, setPlant] = useState(null);
  const [actions, setActions] = useState([]);
  const [workers, setWorkers] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [userRole, setUserRole] = useState('manager'); // Mock role

  useEffect(() => {
    fetchPlantData();
    
    // Subscribe to realtime changes in actions table for this plant
    const channel = supabase.channel('schema-db-changes')
      .on(
        'postgres_changes',
        {
          event: '*',
          schema: 'public',
          table: 'actions',
          filter: `plant_id=eq.${plantId}`
        },
        (payload) => {
          console.log('Realtime update received!', payload);
          // Auto-refresh data silently
          fetchPlantData();
        }
      )
      .subscribe();

    return () => {
      supabase.removeChannel(channel);
    };
  }, [plantId]);

  const fetchPlantData = async () => {
    try {
      if (!plantId) throw new Error("Invalid ID");
      
      // 1. Fetch Plant Details (Join varieties and zones)
      const { data: plantData, error: plantError } = await supabase
        .from('plants')
        .select(`
          *,
          plant_varieties (name, category, average_growth_days),
          zones (name, climate_zone)
        `)
        .eq('id', plantId)
        .single();
        
      if (plantError) throw plantError;
      if (!plantData) throw new Error("Invalid QR / Plant not found");

      // Calculate Age
      const plantedDate = new Date(plantData.planted_date);
      const ageDays = Math.floor((new Date() - plantedDate) / (1000 * 60 * 60 * 24));
      
      setPlant({ ...plantData, calculated_age: ageDays });

      // 2. Fetch Actions (Timeline)
      const { data: actionsData } = await supabase
        .from('actions')
        .select(`*, workers (full_name, role)`)
        .eq('plant_id', plantId)
        .order('timestamp', { ascending: false });
        
      setActions(actionsData || []);

      // 3. Fetch Assigned Workers (Extract from recent actions or tasks table)
      // For this demo, extracting unique workers from actions
      const uniqueWorkers = [];
      const workerIds = new Set();
      (actionsData || []).forEach(a => {
        if (a.worker_id && !workerIds.has(a.worker_id)) {
          workerIds.add(a.worker_id);
          uniqueWorkers.push({
            id: a.worker_id,
            name: a.workers?.full_name || 'Unknown',
            role: a.workers?.role || 'Caretaker',
            date: a.timestamp
          });
        }
      });
      setWorkers(uniqueWorkers);

    } catch (err) {
      console.error(err);
      setError(err.message || "Invalid QR / Plant not found");
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return <div className="flex h-screen items-center justify-center bg-gray-50"><div className="animate-pulse flex flex-col items-center"><Leaf className="h-12 w-12 text-green-500 mb-4 animate-bounce" /><p className="text-gray-500 font-medium">Scanning Digital Leaf Signature...</p></div></div>;
  }

  if (error) {
    return (
      <div className="min-h-screen bg-gray-50 p-6 flex flex-col items-center justify-center">
        <AlertCircle className="h-16 w-16 text-red-500 mb-4" />
        <h2 className="text-2xl font-bold text-gray-900 mb-2">Plant Not Found</h2>
        <p className="text-gray-500 mb-6">{error}</p>
        <button onClick={onBack} className="px-4 py-2 bg-green-600 text-white rounded-lg font-medium">Go Back to Scanner</button>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50 pb-20 md:pb-6">
      {/* Top Navbar */}
      <div className="bg-white px-4 py-3 border-b border-gray-200 sticky top-0 z-50 flex items-center justify-between shadow-sm">
         <button onClick={onBack} className="p-2 -ml-2 rounded-full hover:bg-gray-100"><ArrowLeft size={20} className="text-gray-600" /></button>
         <h1 className="font-semibold text-gray-800 truncate px-2">Plant {plant?.plant_code || plant?.qr_code?.substring(0,8)}</h1>
         <QrCode size={20} className="text-gray-500" />
      </div>

      <div className="max-w-3xl mx-auto p-4 space-y-6">
        {/* 1. Header Section */}
        <section className="bg-white rounded-2xl shadow-sm border border-gray-100 p-5 relative overflow-hidden">
          <div className="absolute top-0 right-0 p-4 opacity-5 pointer-events-none">
            <Leaf size={120} />
          </div>
          
          <div className="flex justify-between items-start mb-4 relative z-10">
            <div>
              <p className="text-sm font-medium text-gray-400 mb-1">ID: {plant.plant_code || plant.id.split('-')[0]}</p>
              <h2 className="text-2xl font-bold text-gray-900">{plant.plant_varieties?.name || "Unknown Variety"}</h2>
              <p className="text-gray-500 mt-1">{plant.plant_varieties?.category || "Standard Plant"} Type</p>
            </div>
            <span className={`px-3 py-1 text-xs font-semibold rounded-full border ${getStatusColor(plant.current_health)}`}>
              {plant.current_health?.toUpperCase()}
            </span>
          </div>

          <div className="grid grid-cols-2 gap-4 mt-6 text-sm">
            <div className="flex items-center text-gray-600">
              <MapPin size={16} className="mr-2 text-green-600" />
              <span className="truncate">{plant.zones?.name || "Zone unassigned"}</span>
            </div>
            <div className="flex items-center text-gray-600">
              <Calendar size={16} className="mr-2 text-green-600" />
              <span>Planted: {new Date(plant.planted_date).toLocaleDateString()}</span>
            </div>
          </div>
        </section>

        {/* 2. Live Status Panel */}
        <section className="grid grid-cols-2 gap-3">
          <div className="bg-white p-4 rounded-xl border border-gray-100 shadow-sm flex flex-col justify-center">
            <span className="text-xs text-gray-500 uppercase tracking-wider mb-1 font-semibold">Latest Action</span>
            <span className="font-medium text-gray-900 capitalize truncate">{actions[0] ? actions[0].action_type : 'None yet'}</span>
            <span className="text-xs text-gray-400 mt-1">{actions[0] ? new Date(actions[0].timestamp).toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'}) : '--:--'}</span>
          </div>
          <div className="bg-white p-4 rounded-xl border border-gray-100 shadow-sm flex flex-col justify-center">
             <span className="text-xs text-gray-500 uppercase tracking-wider mb-1 font-semibold">Growth Stage</span>
             <span className="font-medium text-gray-900 capitalize">{plant.stage || 'Seedling'}</span>
             <span className="text-xs text-gray-400 mt-1">Age: {plant.calculated_age} Days</span>
          </div>
        </section>

        {/* 4. Attributes Table */}
        <section className="bg-white rounded-xl shadow-sm border border-gray-100 overflow-hidden">
          <div className="p-4 border-b border-gray-50 bg-gray-50/50">
            <h3 className="font-semibold text-gray-800 flex items-center"><Info size={18} className="mr-2 text-green-600" /> Plant Attributes</h3>
          </div>
          <div className="divide-y divide-gray-100 text-sm">
            <div className="flex justify-between p-3 px-4"><span className="text-gray-500">Source</span><span className="font-medium">{plant.source || 'Nursery Grown'}</span></div>
            <div className="flex justify-between p-3 px-4"><span className="text-gray-500">Tree Size</span><span className="font-medium capitalize">{plant.tree_size || 'Unknown'}</span></div>
            <div className="flex justify-between p-3 px-4"><span className="text-gray-500">Lot/NFC ID</span><span className="font-medium text-gray-600">{plant.nfc_uid || '--'}</span></div>
            <div className="flex justify-between p-3 px-4"><span className="text-gray-500">Graft Status</span><span className="font-medium">{plant.is_grafted || plant.parent_plant_id ? 'Grafted Tree' : 'Original Root'}</span></div>
          </div>
        </section>

        {/* 3. Assigned Workers */}
        <section className="bg-white rounded-xl shadow-sm border border-gray-100 p-4">
          <h3 className="font-semibold text-gray-800 mb-4 flex items-center"><Users size={18} className="mr-2 text-green-600" /> Dedicated Caretakers</h3>
          {workers.length === 0 ? (
            <p className="text-sm text-gray-500 italic text-center py-4">No specific workers tracked yet.</p>
          ) : (
            <div className="flex -space-x-2 overflow-hidden mb-2">
              {workers.map((worker, i) => (
                <div key={worker.id} className="inline-flex flex-col items-center group">
                  <div className="h-10 w-10 rounded-full bg-green-100 border-2 border-white flex items-center justify-center text-green-800 font-bold shrink-0 z={10-i} shadow-sm">
                     {worker.name.charAt(0)}
                  </div>
                  <span className="text-[10px] mt-1 text-gray-500 truncate w-14 text-center">{worker.name.split(' ')[0]}</span>
                </div>
              ))}
            </div>
          )}
        </section>

        {/* 5. Activity Timeline */}
        <section className="bg-white rounded-xl shadow-sm border border-gray-100 p-4">
          <div className="flex justify-between items-center mb-6">
            <h3 className="font-semibold text-gray-800 flex items-center"><Clock size={18} className="mr-2 text-green-600" /> Action History</h3>
            <span className="text-xs bg-gray-100 text-gray-600 px-2 py-1 rounded-full">{actions.length} Total</span>
          </div>
          
          {actions.length === 0 ? (
            <div className="text-center py-8 bg-gray-50 rounded-lg border border-dashed border-gray-200">
               <p className="text-sm text-gray-500">No activity yet. Scan NFC to log the first action.</p>
            </div>
          ) : (
            <div className="relative border-l-2 border-gray-100 ml-3 md:ml-4 pb-2 space-y-8">
              {actions.map((action, idx) => (
                <div key={action.id} className="relative pl-6">
                  {/* Timeline Dot */}
                  <div className="absolute -left-[17px] top-0 bg-white p-1 rounded-full border border-gray-200 shadow-sm">
                    {getActionIcon(action.action_type)}
                  </div>
                  
                  {/* Content */}
                  <div className="bg-gray-50 border border-gray-100 rounded-xl p-3 hover:bg-gray-100/50 transition-colors">
                    <div className="flex justify-between items-start mb-2">
                      <h4 className="font-semibold text-gray-900 capitalize text-sm">{action.action_type}</h4>
                      <span className="text-xs text-gray-400 font-medium whitespace-nowrap">
                        {new Date(action.timestamp).toLocaleDateString(undefined, {month: 'short', day:'numeric'})}{' '}
                        {new Date(action.timestamp).toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'})}
                      </span>
                    </div>
                    {/* Notes logic based on metadata */}
                    <p className="text-sm text-gray-600 mb-2 leading-snug">
                      {action.metadata?.notes || `${action.action_type} performed per protocol.`}
                    </p>
                    <div className="flex items-center text-xs text-gray-500 border-t border-gray-200 pt-2 mt-2">
                      <span className="font-medium mr-1 text-gray-700">By</span> {action.workers?.full_name || action.worker_id?.split('-')[0]}
                      {action.metadata?.graftType && <span className="ml-auto bg-green-50 text-green-700 px-2 py-0.5 rounded border border-green-100 text-[10px]">Type: {action.metadata.graftType}</span>}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </section>

      </div>

      {/* 8. Floating Action Buttons (Sticky at bottom mapping to user role) */}
      {(userRole === 'manager' || userRole === 'worker') && (
        <div className="fixed bottom-0 left-0 right-0 p-4 bg-gradient-to-t from-white via-white to-transparent pb-6 md:pb-4 flex justify-center gap-3">
          <button className="flex-1 max-w-[200px] flex items-center justify-center bg-gray-900 text-white rounded-xl py-3.5 shadow-lg shadow-gray-200 font-medium hover:bg-black transition-transform active:scale-95">
            <Edit size={18} className="mr-2" /> Update Status
          </button>
          <button className="flex-1 max-w-[200px] flex items-center justify-center bg-green-600 text-white rounded-xl py-3.5 shadow-lg shadow-green-200 font-medium hover:bg-green-700 transition-transform active:scale-95">
            <PlusSquare size={18} className="mr-2" /> Quick Log
          </button>
        </div>
      )}
    </div>
  );
}
