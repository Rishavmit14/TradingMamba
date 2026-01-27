import React, { useState, useEffect } from 'react';
import { BrowserRouter, Routes, Route, Link, useLocation } from 'react-router-dom';
import {
  LayoutDashboard,
  TrendingUp,
  BookOpen,
  Settings,
  Activity,
  Brain,
  RefreshCw,
  BarChart3
} from 'lucide-react';

import Dashboard from './pages/Dashboard';
import Signals from './pages/Signals';
import Concepts from './pages/Concepts';
import Learning from './pages/Learning';
import Performance from './pages/Performance';

// Navigation Component
function Navigation() {
  const location = useLocation();

  const navItems = [
    { path: '/', icon: LayoutDashboard, label: 'Dashboard' },
    { path: '/signals', icon: TrendingUp, label: 'Signals' },
    { path: '/concepts', icon: BookOpen, label: 'ICT Concepts' },
    { path: '/learning', icon: Brain, label: 'ML Learning' },
    { path: '/performance', icon: BarChart3, label: 'Performance' },
  ];

  return (
    <nav className="bg-gray-800 border-b border-gray-700">
      <div className="max-w-7xl mx-auto px-4">
        <div className="flex items-center justify-between h-16">
          {/* Logo */}
          <div className="flex items-center space-x-3">
            <div className="w-10 h-10 bg-gradient-to-br from-green-400 to-blue-500 rounded-lg flex items-center justify-center">
              <span className="text-xl font-bold">TM</span>
            </div>
            <div>
              <h1 className="text-lg font-bold text-white">TradingMamba</h1>
              <p className="text-xs text-gray-400">ICT AI Signals</p>
            </div>
          </div>

          {/* Nav Links */}
          <div className="flex space-x-1">
            {navItems.map(({ path, icon: Icon, label }) => (
              <Link
                key={path}
                to={path}
                className={`flex items-center space-x-2 px-4 py-2 rounded-lg transition-colors ${
                  location.pathname === path
                    ? 'bg-gray-700 text-white'
                    : 'text-gray-400 hover:text-white hover:bg-gray-700/50'
                }`}
              >
                <Icon size={18} />
                <span className="hidden md:inline">{label}</span>
              </Link>
            ))}
          </div>

          {/* Status Indicator */}
          <div className="flex items-center space-x-2">
            <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
            <span className="text-sm text-gray-400">Live</span>
          </div>
        </div>
      </div>
    </nav>
  );
}

// Main App
function App() {
  return (
    <BrowserRouter>
      <div className="min-h-screen bg-gray-900">
        <Navigation />
        <main className="max-w-7xl mx-auto px-4 py-6">
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/signals" element={<Signals />} />
            <Route path="/concepts" element={<Concepts />} />
            <Route path="/learning" element={<Learning />} />
            <Route path="/performance" element={<Performance />} />
          </Routes>
        </main>

        {/* Footer */}
        <footer className="border-t border-gray-800 py-4 mt-8">
          <div className="max-w-7xl mx-auto px-4 text-center text-gray-500 text-sm">
            TradingMamba - Free ICT AI Trading Signals | Not Financial Advice
          </div>
        </footer>
      </div>
    </BrowserRouter>
  );
}

export default App;
