#!/usr/bin/env python3
"""
Run the TradingMamba API server.

Usage:
    python run_api.py

Or with uvicorn directly:
    uvicorn backend.app.main:app --reload --port 8000
"""

import sys
sys.path.insert(0, '/Users/kumarrishav/Library/Python/3.9/lib/python/site-packages')

import uvicorn

if __name__ == "__main__":
    print("🚀 Starting TradingMamba API...")
    print("📚 API Docs: http://localhost:8000/docs")
    print("Press Ctrl+C to stop\n")

    uvicorn.run(
        "backend.app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )
