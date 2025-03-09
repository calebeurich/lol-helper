import React, { useState, useEffect } from 'react';
import Plot from 'react-plotly.js';

const ChampionVisualizations: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'clusters' | 'comparison'>('clusters');
  const [clusterData, setClusterData] = useState<any>(null);
  const [comparisonData, setComparisonData] = useState<any>(null);

  useEffect(() => {
    // Fetch cluster visualization data
    fetch('/api/cluster_visualization')
      .then(response => response.json())
      .then(data => setClusterData(data));

    // Fetch champion comparison data
    fetch('/api/champion_comparison')
      .then(response => response.json())
      .then(data => setComparisonData(data));
  }, []);

  return (
    <div className="container">
      <h1>League of Legends Champion Analysis</h1>
      
      <ul className="nav nav-tabs" role="tablist">
        <li className="nav-item" role="presentation">
          <button 
            className={`nav-link ${activeTab === 'clusters' ? 'active' : ''}`}
            onClick={() => setActiveTab('clusters')}
          >
            Champion Clusters
          </button>
        </li>
        <li className="nav-item" role="presentation">
          <button 
            className={`nav-link ${activeTab === 'comparison' ? 'active' : ''}`}
            onClick={() => setActiveTab('comparison')}
          >
            Champion Comparison
          </button>
        </li>
      </ul>

      <div className="tab-content">
        <div className={`tab-pane fade ${activeTab === 'clusters' ? 'show active' : ''}`}>
          <div className="card">
            <div className="chart-container">
              {clusterData && (
                <Plot
                  data={clusterData.data}
                  layout={clusterData.layout}
                  style={{ width: '100%', height: '600px' }}
                />
              )}
            </div>
          </div>
        </div>
        <div className={`tab-pane fade ${activeTab === 'comparison' ? 'show active' : ''}`}>
          <div className="card">
            <div className="chart-container">
              {comparisonData && (
                <Plot
                  data={comparisonData.data}
                  layout={comparisonData.layout}
                  style={{ width: '100%', height: '600px' }}
                />
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ChampionVisualizations; 