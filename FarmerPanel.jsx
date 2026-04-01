import React, { useState } from 'react';

const FarmerPanel = ({ diseaseData }) => {
  const [explanation, setExplanation] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleExplain = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch('/explain', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(diseaseData),
      });
      if (!response.ok) throw new Error('Failed to fetch explanation');
      const data = await response.json();
      setExplanation(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const getUrgencyColor = (urgency) => {
    switch (urgency) {
      case 'Safe': return 'bg-green-500';
      case 'Needs Attention': return 'bg-yellow-500';
      case 'Act Now': return 'bg-red-500';
      default: return 'bg-gray-500';
    }
  };

  return (
    <div className="max-w-md mx-auto bg-white rounded-xl shadow-md overflow-hidden md:max-w-2xl m-4 border border-gray-100 p-6">
      <div className="flex justify-between items-center mb-6">
        <h2 className="text-2xl font-bold text-gray-800">Farmer Diagnostics</h2>
        <button 
          onClick={handleExplain}
          disabled={loading || !diseaseData}
          className="px-4 py-2 bg-green-600 text-white font-semibold rounded-lg shadow-md hover:bg-green-700 focus:outline-none focus:ring-2 focus:ring-green-400 focus:ring-opacity-75 disabled:opacity-50 transition-colors"
        >
          {loading ? 'Translating...' : 'Explain Simply'}
        </button>
      </div>

      {error && <p className="text-red-500 text-sm mb-4">{error}</p>}

      {explanation && (
        <div className="space-y-6 animate-fade-in">
          {/* Header Card */}
          <div className="p-5 bg-gray-50 rounded-xl border border-gray-200">
            <div className="flex justify-between items-start mb-2">
              <h3 className="text-xl font-black text-gray-900">{explanation.title}</h3>
              <span className={`px-3 py-1 text-xs font-bold text-white rounded-full uppercase tracking-wider ${getUrgencyColor(explanation.urgency)}`}>
                {explanation.urgency}
              </span>
            </div>
            <p className="text-sm font-semibold text-green-600 mb-3">{explanation.confidence_label}</p>
            <p className="text-gray-600 text-sm leading-relaxed">{explanation.what_is_this}</p>
          </div>

          {/* Action List */}
          <div>
            <h4 className="flex items-center text-sm font-bold text-gray-800 uppercase tracking-wide mb-3">
              <span className="w-6 h-6 rounded-full bg-blue-100 text-blue-600 flex items-center justify-center mr-2">✓</span>
              What to do
            </h4>
            <ul className="space-y-2">
              {explanation.actions.map((act, i) => (
                <li key={i} className="flex items-start bg-blue-50/50 p-3 rounded-lg text-sm text-gray-700">
                  <span className="mr-2 text-blue-500">•</span>
                  {act}
                </li>
              ))}
            </ul>
          </div>

          {/* Grid Layout for Impact & Cost */}
          <div className="grid grid-cols-2 gap-4">
            <div className="p-4 bg-orange-50 rounded-xl">
              <h4 className="text-xs font-bold text-orange-800 uppercase mb-1">Impact</h4>
              <p className="text-sm text-orange-900">{explanation.impact}</p>
            </div>
            <div className="p-4 bg-green-50 rounded-xl">
              <h4 className="text-xs font-bold text-green-800 uppercase mb-1">Estimated Cost</h4>
              <p className="text-sm text-green-900 font-medium">{explanation.cost}</p>
            </div>
          </div>

          {/* Safety */}
          {explanation.safety && explanation.safety.length > 0 && (
            <div className="pt-4 border-t border-gray-100">
              <h4 className="text-xs font-bold text-gray-500 uppercase mb-2">Safety Reminders</h4>
              <ul className="text-sm text-gray-500 space-y-1">
                {explanation.safety.map((safe, i) => (
                  <li key={i} className="flex items-center"><span className="mr-2">⚠️</span>{safe}</li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default FarmerPanel;
