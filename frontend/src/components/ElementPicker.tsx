"use client";

import { useEffect, useRef } from "react";
import { ClickCandidate, SelectedElementType } from "@/lib/types";

interface ElementPickerProps {
  candidates: ClickCandidate[];
  x: number;
  y: number;
  onPick: (candidate: ClickCandidate) => void;
  onDismiss: () => void;
}

const TYPE_COLORS: Record<SelectedElementType, string> = {
  swing: "#f59e0b",
  bos: "#22d3ee",
  choch: "#a855f7",
  idm: "#60a5fa",
  fvg: "#10b981",
  ob: "#f97316",
};

const TYPE_ICONS: Record<SelectedElementType, string> = {
  swing: "\u2B24",  // filled circle
  bos: "\u2192",   // arrow right
  choch: "\u21C4", // left-right arrows
  idm: "\u2014",   // em dash (horizontal line)
  fvg: "\u2588",   // full block
  ob: "\u25A0",    // filled square
};

export default function ElementPicker({ candidates, x, y, onPick, onDismiss }: ElementPickerProps) {
  const containerRef = useRef<HTMLDivElement>(null);

  // Click-outside dismissal
  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (containerRef.current && !containerRef.current.contains(e.target as Node)) {
        onDismiss();
      }
    };
    // Delay listener to avoid catching the click that opened the picker
    const id = setTimeout(() => document.addEventListener("mousedown", handler), 50);
    return () => {
      clearTimeout(id);
      document.removeEventListener("mousedown", handler);
    };
  }, [onDismiss]);

  // Escape key dismissal
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === "Escape") onDismiss();
    };
    document.addEventListener("keydown", handler);
    return () => document.removeEventListener("keydown", handler);
  }, [onDismiss]);

  const count = candidates.length;
  const RADIUS = count <= 3 ? 52 : count <= 5 ? 60 : 68;
  const ITEM_SIZE = 40;

  // Position items in a circle around the center
  const items = candidates.map((c, i) => {
    const angle = (i / count) * 2 * Math.PI - Math.PI / 2; // start from top
    const ix = Math.cos(angle) * RADIUS;
    const iy = Math.sin(angle) * RADIUS;
    return { ...c, ix, iy };
  });

  // Ensure the picker doesn't go off-screen
  const totalSize = (RADIUS + ITEM_SIZE) * 2 + 20;
  const halfSize = totalSize / 2;
  const clampedX = Math.max(halfSize, Math.min(window.innerWidth - halfSize, x));
  const clampedY = Math.max(halfSize, Math.min(window.innerHeight - halfSize, y));

  return (
    <div
      className="fixed inset-0 z-50"
      style={{ pointerEvents: "auto" }}
    >
      {/* Subtle backdrop */}
      <div
        className="absolute inset-0"
        style={{ background: "rgba(0,0,0,0.15)" }}
        onClick={onDismiss}
      />

      {/* Radial picker */}
      <div
        ref={containerRef}
        className="absolute"
        style={{
          left: clampedX,
          top: clampedY,
          transform: "translate(-50%, -50%)",
        }}
      >
        {/* Center dot */}
        <div
          className="absolute w-3 h-3 rounded-full bg-white/20 border border-white/30"
          style={{
            left: "50%",
            top: "50%",
            transform: "translate(-50%, -50%)",
          }}
        />

        {/* Connecting lines from center to each item */}
        <svg
          className="absolute"
          style={{
            left: "50%",
            top: "50%",
            transform: "translate(-50%, -50%)",
            width: totalSize,
            height: totalSize,
            pointerEvents: "none",
          }}
        >
          {items.map((item, i) => (
            <line
              key={i}
              x1={totalSize / 2}
              y1={totalSize / 2}
              x2={totalSize / 2 + item.ix}
              y2={totalSize / 2 + item.iy}
              stroke={TYPE_COLORS[item.type]}
              strokeWidth={1.5}
              strokeOpacity={0.3}
            />
          ))}
        </svg>

        {/* Candidate buttons arranged radially */}
        {items.map((item, i) => {
          const color = TYPE_COLORS[item.type];
          return (
            <button
              key={i}
              onClick={() => onPick(item)}
              className="absolute flex items-center gap-1.5 rounded-lg text-xs font-semibold transition-all duration-150 hover:scale-110 active:scale-95 whitespace-nowrap"
              style={{
                left: `calc(50% + ${item.ix}px)`,
                top: `calc(50% + ${item.iy}px)`,
                transform: "translate(-50%, -50%)",
                color,
                background: `rgba(0,0,0,0.85)`,
                border: `1.5px solid ${color}60`,
                boxShadow: `0 0 12px ${color}25, 0 4px 12px rgba(0,0,0,0.5)`,
                padding: "6px 12px",
                backdropFilter: "blur(12px)",
                animation: `pickerItemIn 200ms ease-out ${i * 40}ms both`,
              }}
              title={item.label}
            >
              <span style={{ fontSize: "11px" }}>{TYPE_ICONS[item.type]}</span>
              <span>{item.label}</span>
            </button>
          );
        })}
      </div>

      {/* Global keyframes for picker animation */}
      <style dangerouslySetInnerHTML={{ __html: `
        @keyframes pickerItemIn {
          from { opacity: 0; transform: translate(-50%, -50%) scale(0.3); }
          to { opacity: 1; transform: translate(-50%, -50%) scale(1); }
        }
      `}} />
    </div>
  );
}
