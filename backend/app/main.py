"""
ICT AI Trading System - FastAPI Application

Main entry point for the backend API.
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List, Optional
from datetime import datetime
import json
from pathlib import Path

from .config import settings

# Create FastAPI app
app = FastAPI(
    title="TradingMamba - ICT AI Trading System",
    description="AI-powered trading signal system based on ICT methodology",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Data paths
DATA_DIR = Path(__file__).parent.parent.parent / "data"
PLAYLISTS_DIR = DATA_DIR / "playlists"
TRANSCRIPTS_DIR = DATA_DIR / "transcripts"


# ============================================================================
# Health Check
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint - API info"""
    return {
        "name": "TradingMamba API",
        "version": "0.1.0",
        "status": "running",
        "docs": "/docs"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat()
    }


# ============================================================================
# Playlists & Videos
# ============================================================================

@app.get("/api/playlists")
async def get_playlists():
    """Get all ICT playlists"""
    playlists = []

    if not PLAYLISTS_DIR.exists():
        return {"playlists": [], "total": 0}

    for playlist_file in sorted(PLAYLISTS_DIR.glob("*.json")):
        with open(playlist_file) as f:
            data = json.load(f)
            playlists.append({
                "id": data.get("playlist_id"),
                "title": data.get("title"),
                "video_count": data.get("video_count"),
                "tier": data.get("tier", 1),
            })

    return {
        "playlists": playlists,
        "total": len(playlists)
    }


@app.get("/api/playlists/{playlist_id}")
async def get_playlist(playlist_id: str):
    """Get a specific playlist with videos"""
    playlist_file = PLAYLISTS_DIR / f"{playlist_id}.json"

    if not playlist_file.exists():
        raise HTTPException(status_code=404, detail="Playlist not found")

    with open(playlist_file) as f:
        return json.load(f)


@app.get("/api/playlists/{playlist_id}/videos")
async def get_playlist_videos(playlist_id: str):
    """Get videos in a playlist"""
    playlist_file = PLAYLISTS_DIR / f"{playlist_id}.json"

    if not playlist_file.exists():
        raise HTTPException(status_code=404, detail="Playlist not found")

    with open(playlist_file) as f:
        data = json.load(f)
        return {
            "playlist_id": playlist_id,
            "videos": data.get("videos", [])
        }


# ============================================================================
# Transcripts
# ============================================================================

@app.get("/api/transcripts")
async def get_transcripts():
    """Get all available transcripts"""
    transcripts = []

    if not TRANSCRIPTS_DIR.exists():
        return {"transcripts": [], "total": 0}

    for transcript_file in sorted(TRANSCRIPTS_DIR.glob("*.json")):
        with open(transcript_file) as f:
            data = json.load(f)
            transcripts.append({
                "video_id": data.get("video_id"),
                "title": data.get("title"),
                "word_count": data.get("word_count", 0),
                "method": data.get("method"),
                "transcribed_at": data.get("transcribed_at"),
            })

    return {
        "transcripts": transcripts,
        "total": len(transcripts)
    }


@app.get("/api/transcripts/{video_id}")
async def get_transcript(video_id: str):
    """Get transcript for a specific video"""
    transcript_file = TRANSCRIPTS_DIR / f"{video_id}.json"

    if not transcript_file.exists():
        raise HTTPException(status_code=404, detail="Transcript not found")

    with open(transcript_file) as f:
        return json.load(f)


@app.get("/api/transcripts/{video_id}/search")
async def search_transcript(
    video_id: str,
    q: str = Query(..., description="Search query"),
    context: int = Query(50, description="Characters of context around match")
):
    """Search within a transcript"""
    transcript_file = TRANSCRIPTS_DIR / f"{video_id}.json"

    if not transcript_file.exists():
        raise HTTPException(status_code=404, detail="Transcript not found")

    with open(transcript_file) as f:
        data = json.load(f)

    full_text = data.get("full_text", "").lower()
    query = q.lower()

    matches = []
    start = 0

    while True:
        pos = full_text.find(query, start)
        if pos == -1:
            break

        # Get context
        context_start = max(0, pos - context)
        context_end = min(len(full_text), pos + len(query) + context)
        context_text = full_text[context_start:context_end]

        # Find timestamp
        timestamp = None
        for seg in data.get("segments", []):
            seg_text = seg.get("text", "").lower()
            if query in seg_text:
                timestamp = seg.get("start_time")
                break

        matches.append({
            "position": pos,
            "context": f"...{context_text}...",
            "timestamp": timestamp
        })

        start = pos + 1

    return {
        "video_id": video_id,
        "query": q,
        "matches": matches[:20],  # Limit to 20 matches
        "total_matches": len(matches)
    }


# ============================================================================
# ICT Concepts
# ============================================================================

@app.get("/api/concepts")
async def get_concepts():
    """Get all ICT concepts from taxonomy"""
    from .models.concept import ICT_CONCEPT_TAXONOMY

    concepts = []
    for category, data in ICT_CONCEPT_TAXONOMY.items():
        for concept in data.get("concepts", []):
            concepts.append({
                "category": category,
                "name": concept["name"],
                "short_name": concept.get("short_name", ""),
                "description": concept.get("description", ""),
                "keywords": concept.get("keywords", []),
            })

    return {
        "concepts": concepts,
        "total": len(concepts),
        "categories": list(ICT_CONCEPT_TAXONOMY.keys())
    }


@app.get("/api/concepts/{category}")
async def get_concepts_by_category(category: str):
    """Get concepts for a specific category"""
    from .models.concept import ICT_CONCEPT_TAXONOMY

    if category not in ICT_CONCEPT_TAXONOMY:
        raise HTTPException(status_code=404, detail="Category not found")

    data = ICT_CONCEPT_TAXONOMY[category]
    return {
        "category": category,
        "name": data.get("name"),
        "concepts": data.get("concepts", [])
    }


@app.get("/api/concepts/search")
async def search_concepts(q: str = Query(..., description="Search query")):
    """Search for concepts by name or keyword"""
    from .models.concept import ICT_CONCEPT_TAXONOMY

    query = q.lower()
    matches = []

    for category, data in ICT_CONCEPT_TAXONOMY.items():
        for concept in data.get("concepts", []):
            # Check name
            if query in concept["name"].lower():
                matches.append({**concept, "category": category, "match_type": "name"})
                continue

            # Check short name
            if query in concept.get("short_name", "").lower():
                matches.append({**concept, "category": category, "match_type": "short_name"})
                continue

            # Check keywords
            for keyword in concept.get("keywords", []):
                if query in keyword.lower():
                    matches.append({**concept, "category": category, "match_type": "keyword"})
                    break

    return {
        "query": q,
        "matches": matches,
        "total": len(matches)
    }


# ============================================================================
# Analysis (placeholder for future)
# ============================================================================

@app.get("/api/analysis/status")
async def get_analysis_status():
    """Get system analysis status"""
    # Count transcripts
    transcript_count = len(list(TRANSCRIPTS_DIR.glob("*.json"))) if TRANSCRIPTS_DIR.exists() else 0

    # Count videos
    total_videos = 0
    if PLAYLISTS_DIR.exists():
        for pf in PLAYLISTS_DIR.glob("*.json"):
            with open(pf) as f:
                data = json.load(f)
                total_videos += data.get("video_count", 0)

    return {
        "status": "operational",
        "playlists_loaded": len(list(PLAYLISTS_DIR.glob("*.json"))) if PLAYLISTS_DIR.exists() else 0,
        "total_videos": total_videos,
        "transcripts_ready": transcript_count,
        "transcription_progress": f"{transcript_count}/{total_videos}" if total_videos else "0/0",
        "concepts_loaded": 40,  # From taxonomy
        "signals_generated": 0,  # TODO: implement
        "last_updated": datetime.utcnow().isoformat()
    }


# ============================================================================
# Error Handlers
# ============================================================================

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc)
        }
    )


# ============================================================================
# Startup/Shutdown
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Run on application startup"""
    print("🚀 TradingMamba API starting...")
    print(f"📁 Data directory: {DATA_DIR}")

    # Ensure directories exist
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    PLAYLISTS_DIR.mkdir(parents=True, exist_ok=True)
    TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)


@app.on_event("shutdown")
async def shutdown_event():
    """Run on application shutdown"""
    print("👋 TradingMamba API shutting down...")
