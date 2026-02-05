#!/usr/bin/env python3
"""
Train remaining Forex Minions videos ONE AT A TIME in separate subprocesses.
Each video runs in its own Python process so memory is fully released between videos.
This prevents the MLX-VLM memory leak from crashing the pipeline.
"""
import subprocess
import sys
import os
import json
import time

VIDEOS = [
    ('DabKey96qmE', '01-Structure Mapping'),
    ('BtCIrLqNKe4', '02-Liquidity & Inducement'),
    ('f9rg4BDaaXE', '03-Pullback & Valid Inducement'),
    ('E1AgOEk-lfM', '04-Inducement Shift & Traps'),
    ('GunkTVpUccM', '05-Break of Structure'),
    ('Yq-Tw3PEU5U', '06-BOS vs Liquidity Sweep'),
    ('NbhVSLd18YM', '07-CHoCH & Structure Mapping'),
    ('evng_upluR0', '08-High Prob Inducement'),
    ('eoL_y_6ODLk', '09-Fake CHoCH'),
    ('HEq0YzT19kI', '10-CHoCH Confirmation'),
    ('G-pD_Ts4UEE', '11-Price Cycle Theory'),
    ('gSyIFHd3HeE', '12-Premium & Discount Zones'),
    ('hMb-cEAVKcQ', '13-Fair Value Gap'),
    ('-zPGWtuuWdU', '14-Valid Order Blocks'),
    ('hdnldU2yQMw', '15-Millions Dollar Setup'),
    ('hRuUCLE7i6U', '16-Candlestick & Sessions'),
]

KB_DIR = 'data/audio_first_training'

def is_done(video_id):
    return os.path.exists(f'{KB_DIR}/{video_id}_knowledge_base.json')

def train_single_video(video_id):
    """Run training in a SEPARATE subprocess so all memory is freed on exit."""
    code = f"""
import sys, os
sys.stdout.reconfigure(line_buffering=True)
os.chdir('{os.getcwd()}')

from backend.app.ml.audio_first_learning import AudioFirstTrainer
import json

trainer = AudioFirstTrainer(data_dir='data')
result = trainer.train_from_url('{video_id}')

concepts = len(result.get('knowledge', {{}}).get('concepts', {{}})) if isinstance(result.get('knowledge'), dict) else 0
print(f'CONCEPTS:{{concepts}}', flush=True)
"""
    proc = subprocess.Popen(
        [sys.executable, '-u', '-c', code],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=os.getcwd(),
    )
    stdout, stderr = proc.communicate()
    return proc.returncode, stdout, stderr

def main():
    print("=" * 60, flush=True)
    print("  FOREX MINIONS - SUBPROCESS-ISOLATED TRAINING", flush=True)
    print("  Each video in its own process (no memory leaks)", flush=True)
    print("=" * 60, flush=True)

    total_start = time.time()
    completed = 0
    skipped = 0
    failed = 0

    for video_id, title in VIDEOS:
        num = VIDEOS.index((video_id, title)) + 1

        if is_done(video_id):
            kb = json.load(open(f'{KB_DIR}/{video_id}_knowledge_base.json'))
            concepts = len(kb.get('concepts', {}))
            print(f"  ✅ V{num:02d} {title}: {concepts} concepts (cached)", flush=True)
            skipped += 1
            completed += 1
            continue

        print(f"\n  🔄 V{num:02d} {title} ({video_id})...", flush=True)
        v_start = time.time()

        returncode, stdout, stderr = train_single_video(video_id)
        v_time = (time.time() - v_start) / 60

        if returncode == 0 and is_done(video_id):
            # Extract concept count from stdout
            concepts = 0
            for line in stdout.split('\n'):
                if line.startswith('CONCEPTS:'):
                    concepts = int(line.split(':')[1])
            print(f"  ✅ V{num:02d} {title}: {concepts} concepts in {v_time:.1f}min", flush=True)
            completed += 1
        else:
            print(f"  ❌ V{num:02d} {title}: FAILED (exit={returncode}, {v_time:.1f}min)", flush=True)
            # Print last few lines of stderr for debugging
            err_lines = stderr.strip().split('\n')[-5:]
            for line in err_lines:
                print(f"       {line}", flush=True)
            failed += 1

    total_time = (time.time() - total_start) / 60
    print(f"\n{'=' * 60}", flush=True)
    print(f"  COMPLETE: {completed}/16 done | {skipped} skipped | {failed} failed", flush=True)
    print(f"  Total time: {total_time:.1f} minutes", flush=True)
    print(f"{'=' * 60}", flush=True)

if __name__ == '__main__':
    main()
