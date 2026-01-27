import React, { useState, useEffect } from 'react';
import {
  BookOpen,
  Search,
  ChevronDown,
  ChevronRight,
  Info,
  Tag
} from 'lucide-react';
import { getConcepts, queryConcept } from '../services/api';

// ICT Concept Categories with descriptions
const CATEGORY_INFO = {
  market_structure: {
    name: 'Market Structure',
    description: 'Understanding trend direction through swing points and structural breaks',
    color: 'blue',
  },
  key_levels: {
    name: 'Key Levels',
    description: 'Institutional reference points including Order Blocks and Fair Value Gaps',
    color: 'purple',
  },
  liquidity: {
    name: 'Liquidity',
    description: 'Areas where stop losses cluster and smart money targets',
    color: 'green',
  },
  entry_models: {
    name: 'Entry Models',
    description: 'Specific setups for high-probability trade entries',
    color: 'orange',
  },
  time_based: {
    name: 'Time & Sessions',
    description: 'Kill zones and optimal trading times',
    color: 'red',
  },
  institutional: {
    name: 'Institutional Concepts',
    description: 'Smart money behavior and market maker models',
    color: 'yellow',
  },
};

// Concept Card
function ConceptCard({ concept, onLearnMore }) {
  return (
    <div className="bg-gray-800 rounded-lg p-4 border border-gray-700 hover:border-gray-600 transition-colors">
      <div className="flex justify-between items-start mb-2">
        <h3 className="font-medium">{concept.name}</h3>
        {concept.short_name && (
          <span className="text-xs px-2 py-0.5 bg-gray-700 rounded text-gray-400">
            {concept.short_name}
          </span>
        )}
      </div>
      <p className="text-sm text-gray-400 mb-3">{concept.description}</p>

      {concept.keywords?.length > 0 && (
        <div className="flex flex-wrap gap-1 mb-3">
          {concept.keywords.slice(0, 5).map((kw, i) => (
            <span key={i} className="text-xs px-2 py-0.5 bg-blue-500/20 text-blue-300 rounded">
              {kw}
            </span>
          ))}
        </div>
      )}

      <button
        onClick={() => onLearnMore(concept.name)}
        className="text-sm text-blue-400 hover:text-blue-300 flex items-center"
      >
        <Info size={14} className="mr-1" />
        Learn from transcripts
      </button>
    </div>
  );
}

// Category Section
function CategorySection({ category, concepts, expanded, onToggle, onLearnMore }) {
  const info = CATEGORY_INFO[category] || { name: category, color: 'gray' };
  const colorClasses = {
    blue: 'border-blue-500/50 bg-blue-500/10',
    purple: 'border-purple-500/50 bg-purple-500/10',
    green: 'border-green-500/50 bg-green-500/10',
    orange: 'border-orange-500/50 bg-orange-500/10',
    red: 'border-red-500/50 bg-red-500/10',
    yellow: 'border-yellow-500/50 bg-yellow-500/10',
    gray: 'border-gray-500/50 bg-gray-500/10',
  };

  return (
    <div className="border border-gray-700 rounded-xl overflow-hidden">
      <button
        onClick={onToggle}
        className={`w-full p-4 flex items-center justify-between ${colorClasses[info.color]} border-b border-gray-700`}
      >
        <div className="flex items-center space-x-3">
          <BookOpen size={20} />
          <div className="text-left">
            <h2 className="font-bold">{info.name}</h2>
            <p className="text-sm text-gray-400">{info.description}</p>
          </div>
        </div>
        <div className="flex items-center space-x-2">
          <span className="text-sm text-gray-400">{concepts.length} concepts</span>
          {expanded ? <ChevronDown size={20} /> : <ChevronRight size={20} />}
        </div>
      </button>

      {expanded && (
        <div className="p-4 grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {concepts.map((concept, i) => (
            <ConceptCard key={i} concept={concept} onLearnMore={onLearnMore} />
          ))}
        </div>
      )}
    </div>
  );
}

// Concept Detail Modal
function ConceptDetailModal({ concept, data, onClose }) {
  if (!concept) return null;

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
      <div className="bg-gray-800 rounded-xl max-w-2xl w-full max-h-[80vh] overflow-y-auto">
        <div className="p-6 border-b border-gray-700">
          <div className="flex justify-between items-start">
            <h2 className="text-xl font-bold">{concept.replace('_', ' ')}</h2>
            <button onClick={onClose} className="text-gray-400 hover:text-white">
              &times;
            </button>
          </div>
        </div>

        <div className="p-6 space-y-6">
          {data ? (
            <>
              {/* Terms */}
              <div>
                <h3 className="font-medium mb-2">Search Terms</h3>
                <div className="flex flex-wrap gap-2">
                  {data.terms?.map((term, i) => (
                    <span key={i} className="px-2 py-1 bg-gray-700 rounded text-sm">
                      {term}
                    </span>
                  ))}
                </div>
              </div>

              {/* Related Concepts */}
              {data.related_concepts?.length > 0 && (
                <div>
                  <h3 className="font-medium mb-2">Related Concepts</h3>
                  <div className="space-y-2">
                    {data.related_concepts.map(([name, score], i) => (
                      <div key={i} className="flex items-center justify-between bg-gray-700/50 rounded p-2">
                        <span>{name.replace('_', ' ')}</span>
                        <span className="text-sm text-gray-400">
                          {(score * 100).toFixed(0)}% similar
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Trading Rules */}
              {data.trading_rules?.length > 0 && (
                <div>
                  <h3 className="font-medium mb-2">Trading Rules from ICT</h3>
                  <ul className="space-y-2">
                    {data.trading_rules.map((rule, i) => (
                      <li key={i} className="bg-gray-700/50 rounded p-3 text-sm">
                        <p className="text-gray-300">"{rule.text}"</p>
                        <p className="text-xs text-gray-500 mt-1">
                          Type: {rule.type} | Source: {rule.source_video}
                        </p>
                      </li>
                    ))}
                  </ul>
                </div>
              )}

              {/* Definition Examples */}
              {data.definition?.examples?.length > 0 && (
                <div>
                  <h3 className="font-medium mb-2">Examples from Transcripts</h3>
                  <ul className="space-y-2 text-sm text-gray-400">
                    {data.definition.examples.slice(0, 3).map((ex, i) => (
                      <li key={i} className="italic">"{ex}"</li>
                    ))}
                  </ul>
                </div>
              )}
            </>
          ) : (
            <div className="text-center py-8 text-gray-500">
              <BookOpen size={48} className="mx-auto mb-4 opacity-50" />
              <p>Loading concept data...</p>
              <p className="text-sm mt-2">
                Make sure the ML model is trained to get detailed concept information.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

// Main Concepts Page
export default function Concepts() {
  const [concepts, setConcepts] = useState([]);
  const [expandedCategories, setExpandedCategories] = useState(['market_structure', 'key_levels']);
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedConcept, setSelectedConcept] = useState(null);
  const [conceptData, setConceptData] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadConcepts();
  }, []);

  const loadConcepts = async () => {
    try {
      const data = await getConcepts();
      setConcepts(data.concepts || []);
    } catch (err) {
      console.error('Failed to load concepts:', err);
    } finally {
      setLoading(false);
    }
  };

  const toggleCategory = (category) => {
    setExpandedCategories(prev =>
      prev.includes(category)
        ? prev.filter(c => c !== category)
        : [...prev, category]
    );
  };

  const handleLearnMore = async (conceptName) => {
    setSelectedConcept(conceptName);
    try {
      const data = await queryConcept(conceptName);
      setConceptData(data);
    } catch (err) {
      console.error('Failed to load concept:', err);
      setConceptData(null);
    }
  };

  // Group concepts by category
  const groupedConcepts = concepts.reduce((acc, concept) => {
    const category = concept.category || 'other';
    if (!acc[category]) acc[category] = [];
    acc[category].push(concept);
    return acc;
  }, {});

  // Filter by search
  const filteredGroups = Object.entries(groupedConcepts).reduce((acc, [category, items]) => {
    if (searchQuery) {
      const filtered = items.filter(c =>
        c.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
        c.keywords?.some(k => k.toLowerCase().includes(searchQuery.toLowerCase()))
      );
      if (filtered.length > 0) acc[category] = filtered;
    } else {
      acc[category] = items;
    }
    return acc;
  }, {});

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-2xl font-bold">ICT Concepts</h1>
          <p className="text-gray-400">Learn Inner Circle Trader methodology from AI-analyzed transcripts</p>
        </div>
      </div>

      {/* Search */}
      <div className="relative">
        <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-400" size={20} />
        <input
          type="text"
          placeholder="Search concepts..."
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          className="w-full pl-10 pr-4 py-3 bg-gray-800 border border-gray-700 rounded-xl focus:outline-none focus:border-blue-500"
        />
      </div>

      {/* Concept Categories */}
      {loading ? (
        <div className="space-y-4">
          {[1, 2, 3].map(i => (
            <div key={i} className="h-20 bg-gray-800 rounded-xl animate-pulse"></div>
          ))}
        </div>
      ) : (
        <div className="space-y-4">
          {Object.entries(filteredGroups).map(([category, items]) => (
            <CategorySection
              key={category}
              category={category}
              concepts={items}
              expanded={expandedCategories.includes(category)}
              onToggle={() => toggleCategory(category)}
              onLearnMore={handleLearnMore}
            />
          ))}
        </div>
      )}

      {/* Concept Detail Modal */}
      {selectedConcept && (
        <ConceptDetailModal
          concept={selectedConcept}
          data={conceptData}
          onClose={() => {
            setSelectedConcept(null);
            setConceptData(null);
          }}
        />
      )}
    </div>
  );
}
