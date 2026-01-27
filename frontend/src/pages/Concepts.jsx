import React, { useState, useEffect } from 'react';
import {
  BookOpen,
  Search,
  ChevronDown,
  ChevronRight,
  Info,
  Tag,
  X,
  Sparkles,
  Layers,
  Zap,
  Brain,
  Target,
  Clock,
  TrendingUp,
  Shield
} from 'lucide-react';
import { getConcepts, queryConcept } from '../services/api';

// Background Orb Component
function BackgroundOrb({ className }) {
  return (
    <div className={`absolute rounded-full blur-3xl opacity-20 animate-pulse ${className}`} />
  );
}

// Smart Money Concept Categories with descriptions
const CATEGORY_INFO = {
  market_structure: {
    name: 'Market Structure',
    description: 'Understanding trend direction through swing points and structural breaks',
    icon: TrendingUp,
    gradient: 'from-blue-500 to-cyan-500',
    bgColor: 'bg-blue-500/10',
    borderColor: 'border-blue-500/20',
    textColor: 'text-blue-400',
  },
  key_levels: {
    name: 'Key Levels',
    description: 'Institutional reference points including Order Blocks and Fair Value Gaps',
    icon: Layers,
    gradient: 'from-purple-500 to-pink-500',
    bgColor: 'bg-purple-500/10',
    borderColor: 'border-purple-500/20',
    textColor: 'text-purple-400',
  },
  liquidity: {
    name: 'Liquidity',
    description: 'Areas where stop losses cluster and smart money targets',
    icon: Target,
    gradient: 'from-emerald-500 to-teal-500',
    bgColor: 'bg-emerald-500/10',
    borderColor: 'border-emerald-500/20',
    textColor: 'text-emerald-400',
  },
  entry_models: {
    name: 'Entry Models',
    description: 'Specific setups for high-probability trade entries',
    icon: Zap,
    gradient: 'from-orange-500 to-amber-500',
    bgColor: 'bg-orange-500/10',
    borderColor: 'border-orange-500/20',
    textColor: 'text-orange-400',
  },
  time_based: {
    name: 'Time & Sessions',
    description: 'Kill zones and optimal trading times',
    icon: Clock,
    gradient: 'from-red-500 to-rose-500',
    bgColor: 'bg-red-500/10',
    borderColor: 'border-red-500/20',
    textColor: 'text-red-400',
  },
  institutional: {
    name: 'Institutional Concepts',
    description: 'Smart money behavior and market maker models',
    icon: Shield,
    gradient: 'from-yellow-500 to-orange-500',
    bgColor: 'bg-yellow-500/10',
    borderColor: 'border-yellow-500/20',
    textColor: 'text-yellow-400',
  },
};

// Concept Card
function ConceptCard({ concept, onLearnMore, delay = 0 }) {
  return (
    <div
      className="glass-card-static p-5 group hover:bg-white/[0.08] transition-all duration-300 animate-slide-up"
      style={{ animationDelay: `${delay}ms` }}
    >
      <div className="flex justify-between items-start mb-3">
        <h3 className="font-semibold text-white group-hover:text-indigo-300 transition-colors">
          {concept.name}
        </h3>
        {concept.short_name && (
          <span className="text-xs px-2.5 py-1 bg-indigo-500/10 text-indigo-300 rounded-lg border border-indigo-500/20">
            {concept.short_name}
          </span>
        )}
      </div>
      <p className="text-sm text-slate-400 mb-4 line-clamp-2">{concept.description}</p>

      {concept.keywords?.length > 0 && (
        <div className="flex flex-wrap gap-1.5 mb-4">
          {concept.keywords.slice(0, 4).map((kw, i) => (
            <span
              key={i}
              className="text-xs px-2 py-1 bg-white/5 text-slate-400 rounded-lg border border-white/5"
            >
              {kw}
            </span>
          ))}
          {concept.keywords.length > 4 && (
            <span className="text-xs px-2 py-1 text-slate-500">
              +{concept.keywords.length - 4}
            </span>
          )}
        </div>
      )}

      <button
        onClick={() => onLearnMore(concept.name)}
        className="flex items-center gap-2 text-sm text-indigo-400 hover:text-indigo-300 transition-colors group/btn"
      >
        <div className="w-6 h-6 rounded-lg bg-indigo-500/10 flex items-center justify-center group-hover/btn:bg-indigo-500/20 transition-colors">
          <Info size={12} />
        </div>
        <span>Learn from transcripts</span>
        <ChevronRight size={14} className="group-hover/btn:translate-x-1 transition-transform" />
      </button>
    </div>
  );
}

// Category Section
function CategorySection({ category, concepts, expanded, onToggle, onLearnMore, index = 0 }) {
  const info = CATEGORY_INFO[category] || {
    name: category,
    icon: BookOpen,
    gradient: 'from-slate-500 to-slate-600',
    bgColor: 'bg-slate-500/10',
    borderColor: 'border-slate-500/20',
    textColor: 'text-slate-400',
  };
  const Icon = info.icon;

  return (
    <div
      className="glass-card overflow-hidden animate-slide-up"
      style={{ animationDelay: `${index * 100}ms` }}
    >
      <button
        onClick={onToggle}
        className={`w-full p-5 flex items-center justify-between ${info.bgColor} hover:bg-opacity-20 transition-all duration-300`}
      >
        <div className="flex items-center gap-4">
          <div className={`w-12 h-12 rounded-xl bg-gradient-to-br ${info.gradient} flex items-center justify-center shadow-lg`}>
            <Icon className="w-5 h-5 text-white" />
          </div>
          <div className="text-left">
            <h2 className="font-bold text-white text-lg">{info.name}</h2>
            <p className="text-sm text-slate-400 mt-0.5">{info.description}</p>
          </div>
        </div>
        <div className="flex items-center gap-4">
          <span className={`text-sm font-medium px-3 py-1.5 rounded-lg ${info.bgColor} ${info.textColor} border ${info.borderColor}`}>
            {concepts.length} concepts
          </span>
          <div className={`w-8 h-8 rounded-lg ${expanded ? 'bg-white/10' : 'bg-white/5'} flex items-center justify-center transition-colors`}>
            {expanded
              ? <ChevronDown className="w-5 h-5 text-slate-400" />
              : <ChevronRight className="w-5 h-5 text-slate-400" />
            }
          </div>
        </div>
      </button>

      {expanded && (
        <div className="p-5 pt-0 border-t border-white/5">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 pt-5">
            {concepts.map((concept, i) => (
              <ConceptCard
                key={i}
                concept={concept}
                onLearnMore={onLearnMore}
                delay={i * 50}
              />
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

// Concept Detail Modal
function ConceptDetailModal({ concept, data, onClose }) {
  if (!concept) return null;

  return (
    <div className="fixed inset-0 bg-black/70 backdrop-blur-sm flex items-center justify-center z-50 p-4 animate-fade-in">
      <div
        className="glass-card max-w-2xl w-full max-h-[85vh] overflow-hidden animate-slide-up"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="p-6 border-b border-white/5 relative">
          <div className="absolute top-0 left-0 right-0 h-1 bg-gradient-to-r from-indigo-500 via-purple-500 to-pink-500" />
          <div className="flex justify-between items-start">
            <div className="flex items-center gap-3">
              <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-indigo-500 to-purple-500 flex items-center justify-center">
                <Brain className="w-6 h-6 text-white" />
              </div>
              <div>
                <h2 className="text-xl font-bold text-white">
                  {concept.replace(/_/g, ' ')}
                </h2>
                <p className="text-sm text-slate-400">Smart Money Concept</p>
              </div>
            </div>
            <button
              onClick={onClose}
              className="w-10 h-10 rounded-xl bg-white/5 hover:bg-white/10 flex items-center justify-center transition-colors"
            >
              <X className="w-5 h-5 text-slate-400" />
            </button>
          </div>
        </div>

        {/* Content */}
        <div className="p-6 space-y-6 overflow-y-auto max-h-[calc(85vh-120px)]">
          {data ? (
            <>
              {/* Terms */}
              {data.terms?.length > 0 && (
                <div className="animate-slide-up" style={{ animationDelay: '0ms' }}>
                  <h3 className="font-semibold text-white mb-3 flex items-center gap-2">
                    <Tag className="w-4 h-4 text-indigo-400" />
                    Search Terms
                  </h3>
                  <div className="flex flex-wrap gap-2">
                    {data.terms?.map((term, i) => (
                      <span
                        key={i}
                        className="px-3 py-1.5 bg-white/5 text-slate-300 rounded-lg border border-white/5 text-sm"
                      >
                        {term}
                      </span>
                    ))}
                  </div>
                </div>
              )}

              {/* Related Concepts */}
              {data.related_concepts?.length > 0 && (
                <div className="animate-slide-up" style={{ animationDelay: '100ms' }}>
                  <h3 className="font-semibold text-white mb-3 flex items-center gap-2">
                    <Layers className="w-4 h-4 text-indigo-400" />
                    Related Concepts
                  </h3>
                  <div className="space-y-2">
                    {data.related_concepts.map(([name, score], i) => (
                      <div
                        key={i}
                        className="glass-card-static p-3 flex items-center justify-between"
                      >
                        <span className="text-slate-300">{name.replace(/_/g, ' ')}</span>
                        <div className="flex items-center gap-2">
                          <div className="w-24 h-2 bg-white/10 rounded-full overflow-hidden">
                            <div
                              className="h-full bg-gradient-to-r from-indigo-500 to-purple-500 rounded-full"
                              style={{ width: `${score * 100}%` }}
                            />
                          </div>
                          <span className="text-sm text-slate-400 w-12 text-right">
                            {(score * 100).toFixed(0)}%
                          </span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Trading Rules */}
              {data.trading_rules?.length > 0 && (
                <div className="animate-slide-up" style={{ animationDelay: '200ms' }}>
                  <h3 className="font-semibold text-white mb-3 flex items-center gap-2">
                    <Zap className="w-4 h-4 text-indigo-400" />
                    Trading Rules
                  </h3>
                  <div className="space-y-3">
                    {data.trading_rules.map((rule, i) => (
                      <div key={i} className="glass-card-static p-4">
                        <p className="text-slate-300 text-sm italic mb-2">"{rule.text}"</p>
                        <div className="flex items-center gap-3 text-xs">
                          <span className="px-2 py-1 bg-indigo-500/10 text-indigo-300 rounded-lg border border-indigo-500/20">
                            {rule.type}
                          </span>
                          <span className="text-slate-500">
                            Source: {rule.source_video}
                          </span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Definition Examples */}
              {data.definition?.examples?.length > 0 && (
                <div className="animate-slide-up" style={{ animationDelay: '300ms' }}>
                  <h3 className="font-semibold text-white mb-3 flex items-center gap-2">
                    <Sparkles className="w-4 h-4 text-indigo-400" />
                    Examples from Transcripts
                  </h3>
                  <div className="space-y-2">
                    {data.definition.examples.slice(0, 3).map((ex, i) => (
                      <div key={i} className="glass-card-static p-4 text-sm text-slate-400 italic">
                        "{ex}"
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </>
          ) : (
            <div className="text-center py-12">
              <div className="w-20 h-20 rounded-2xl bg-slate-500/10 flex items-center justify-center mx-auto mb-6">
                <BookOpen className="w-10 h-10 text-slate-500 animate-pulse" />
              </div>
              <h3 className="text-lg font-semibold text-slate-400 mb-2">Loading concept data...</h3>
              <p className="text-sm text-slate-500">
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
    <div className="space-y-8 relative">
      {/* Background decorations */}
      <BackgroundOrb className="w-96 h-96 bg-blue-500 -top-48 -left-48" />
      <BackgroundOrb className="w-72 h-72 bg-purple-500 top-1/3 -right-36" />

      {/* Header */}
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
        <div>
          <h1 className="text-3xl font-bold text-white tracking-tight flex items-center gap-3">
            <BookOpen className="w-8 h-8 text-indigo-400" />
            Smart Money Concepts
          </h1>
          <p className="text-slate-400 mt-1">
            Learn Smart Money trading methodology from AI-analyzed transcripts
          </p>
        </div>
        <div className="flex items-center gap-2 px-4 py-2 rounded-xl bg-indigo-500/10 border border-indigo-500/20">
          <Sparkles className="w-4 h-4 text-indigo-400" />
          <span className="text-sm text-indigo-300">
            {concepts.length} concepts loaded
          </span>
        </div>
      </div>

      {/* Search */}
      <div className="glass-card p-2 relative">
        <Search className="absolute left-5 top-1/2 -translate-y-1/2 text-slate-400" size={20} />
        <input
          type="text"
          placeholder="Search concepts by name or keyword..."
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          className="w-full pl-12 pr-4 py-3 bg-transparent border-0 text-white placeholder-slate-500 focus:outline-none focus:ring-0"
        />
      </div>

      {/* Concept Categories */}
      {loading ? (
        <div className="space-y-4">
          {[1, 2, 3].map(i => (
            <div key={i} className="glass-card p-5 animate-pulse">
              <div className="flex items-center gap-4">
                <div className="w-12 h-12 skeleton rounded-xl" />
                <div className="flex-1">
                  <div className="h-5 skeleton rounded w-1/3 mb-2" />
                  <div className="h-4 skeleton rounded w-1/2" />
                </div>
              </div>
            </div>
          ))}
        </div>
      ) : (
        <div className="space-y-4">
          {Object.entries(filteredGroups).map(([category, items], index) => (
            <CategorySection
              key={category}
              category={category}
              concepts={items}
              expanded={expandedCategories.includes(category)}
              onToggle={() => toggleCategory(category)}
              onLearnMore={handleLearnMore}
              index={index}
            />
          ))}

          {Object.keys(filteredGroups).length === 0 && (
            <div className="glass-card p-12 text-center">
              <div className="w-16 h-16 rounded-2xl bg-slate-500/10 flex items-center justify-center mx-auto mb-4">
                <Search className="w-8 h-8 text-slate-500" />
              </div>
              <h3 className="text-lg font-semibold text-slate-400 mb-2">No concepts found</h3>
              <p className="text-sm text-slate-500">
                Try adjusting your search query
              </p>
            </div>
          )}
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
