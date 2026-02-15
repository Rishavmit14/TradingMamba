"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import { RotateCcw } from "lucide-react";
import {
  createChart,
  CrosshairMode,
  IChartApi,
  ISeriesApi,
  CandlestickData,
  ColorType,
  LineStyle,
  MouseEventParams,
} from "lightweight-charts";
import { AnalysisResult, DetectorVisibility, SelectedElement, ChartClickResult, ClickCandidate, SwingPoint, Inducement, BOS, CHoCH, FVG, OrderBlock, DemoTrade } from "@/lib/types";

interface SignalMarker {
  timestamp: number;     // ms — find matching candle by timestamp
  direction: string;     // "bullish" | "bearish"
  entryPrice: number;    // for price line
  grade: string;         // "A" | "B" | "C" | "D"
}

interface ChartProps {
  data: AnalysisResult | null;
  visibility: DetectorVisibility;
  livePrice?: number | null;
  priceChangePct?: number | null;
  onElementClick?: (result: ChartClickResult | null) => void;
  openTrades?: DemoTrade[];
  signalMarker?: SignalMarker | null;
}

/** Convert unix ms timestamp to unix seconds for TradingView. */
function toTV(ts: number) {
  return Math.floor(ts / 1000) as any;
}

/** Format a unix-seconds timestamp in New York time. */
function formatNY(timeSec: number, opts?: Intl.DateTimeFormatOptions): string {
  return new Date(timeSec * 1000).toLocaleString("en-US", {
    timeZone: "America/New_York",
    ...opts,
  });
}

/** Convert a NY-local hour on a given date to UTC milliseconds (DST-aware). */
function nyHourToUtcMs(year: number, month: number, day: number, hour: number): number {
  // Create a UTC date as if the NY time were UTC
  const approxUtc = Date.UTC(year, month - 1, day, hour, 0, 0);
  // Find what NY hour this UTC timestamp actually represents
  const nyHour = parseInt(
    new Intl.DateTimeFormat("en-US", {
      timeZone: "America/New_York",
      hour: "numeric",
      hour12: false,
    }).format(new Date(approxUtc))
  );
  // Difference = NY offset; adjust to get the real UTC for the desired NY hour
  let diff = nyHour - hour;
  if (diff > 12) diff -= 24;
  if (diff < -12) diff += 24;
  return approxUtc - diff * 3600_000;
}

/** ICT Kill Zones defined in New York local time.
 *  Opacity varies by timeframe: higher on H4 (narrow bands) vs M15/M5 (wide bands). */
const KILL_ZONES_NY = [
  { name: "LDN", startHour: 2, endHour: 5, rgb: "96, 165, 250" },
  { name: "NY",  startHour: 7, endHour: 10, rgb: "251, 191, 36" },
];
const KZ_OPACITY: Record<string, { fill: number; border: number; label: number }> = {
  H4:  { fill: 0.14, border: 0.40, label: 0.70 },
  H1:  { fill: 0.10, border: 0.28, label: 0.60 },
  M15: { fill: 0.06, border: 0.18, label: 0.50 },
  M5:  { fill: 0.06, border: 0.18, label: 0.50 },
};

// --- Box Primitive: draws filled rectangles on the chart canvas ---

/** Shared box primitive for drawing filled rectangles on the chart canvas. */
interface BoxData {
  startTime: any;
  endTime: any;
  upperPrice: number;
  lowerPrice: number;
  fillColor: string;
  borderColor: string;
  label: string;
  labelColor: string;
  rightExtendPx?: number; // extra logical pixels to add past endTime
  labelTop?: boolean; // render label near top of box instead of centered
}

class BoxPrimitive {
  _boxes: BoxData[] = [];
  _chart: IChartApi | null = null;
  _series: any = null;
  _requestUpdate: (() => void) | null = null;

  setBoxes(boxes: BoxData[]) {
    this._boxes = boxes;
    this._requestUpdate?.();
  }

  attached(param: any) {
    this._chart = param.chart;
    this._series = param.series;
    this._requestUpdate = param.requestUpdate;
  }

  detached() {
    this._chart = null;
    this._series = null;
    this._requestUpdate = null;
  }

  paneViews() {
    return [{
      zOrder: () => "bottom" as const,
      renderer: () => ({
        draw: (target: any) => {
          target.useBitmapCoordinateSpace((scope: any) => {
            const ctx = scope.context as CanvasRenderingContext2D;
            if (!this._chart || !this._series) return;
            const hpr = scope.horizontalPixelRatio;
            const vpr = scope.verticalPixelRatio;

            for (const box of this._boxes) {
              const x1 = this._chart.timeScale().timeToCoordinate(box.startTime);
              const x2 = this._chart.timeScale().timeToCoordinate(box.endTime);
              const y1 = this._series.priceToCoordinate(box.upperPrice);
              const y2 = this._series.priceToCoordinate(box.lowerPrice);

              if (x1 === null || x2 === null || y1 === null || y2 === null) continue;

              const px1 = Math.round(x1 * hpr);
              let px2 = Math.round(x2 * hpr);
              // Apply optional right extension (e.g. position boxes extending past last candle)
              if (box.rightExtendPx) px2 += Math.round(box.rightExtendPx * hpr);
              const py1 = Math.round(y1 * vpr);
              const py2 = Math.round(y2 * vpr);

              // Filled rectangle
              ctx.fillStyle = box.fillColor;
              ctx.fillRect(px1, py1, px2 - px1, py2 - py1);

              // Border
              ctx.strokeStyle = box.borderColor;
              ctx.lineWidth = Math.max(1, Math.round(1.5 * hpr));
              ctx.strokeRect(px1, py1, px2 - px1, py2 - py1);

              // Label centered in the box
              if (box.label) {
                const fontSize = Math.round(11 * vpr);
                ctx.font = `bold ${fontSize}px sans-serif`;
                ctx.fillStyle = box.labelColor;
                ctx.textAlign = "center";
                ctx.textBaseline = "middle";
                const cx = (px1 + px2) / 2;
                // labelTop: fixed offset from canvas top (ignores py1 which may be off-screen)
                const cy = box.labelTop ? fontSize + Math.round(6 * vpr) : (py1 + py2) / 2;
                ctx.fillText(box.label, cx, cy);
                ctx.textAlign = "start"; // reset
              }
            }
          });
        },
      }),
    }];
  }
}

// Keep old name as alias for readability
type OBBoxData = BoxData;

export default function Chart({ data, visibility, livePrice, priceChangePct, onElementClick, openTrades = [], signalMarker }: ChartProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const candleSeriesRef = useRef<ISeriesApi<"Candlestick"> | null>(null);
  const volumeSeriesRef = useRef<ISeriesApi<"Histogram"> | null>(null);
  const priceLinesRef = useRef<any[]>([]);
  const idmSeriesRef = useRef<ISeriesApi<"Line">[]>([]);
  const bosSeriesRef = useRef<ISeriesApi<"Line">[]>([]);
  const chochSeriesRef = useRef<ISeriesApi<"Line">[]>([]);
  const obPrimitiveRef = useRef<BoxPrimitive | null>(null);
  const fvgPrimitiveRef = useRef<BoxPrimitive | null>(null);
  const positionPrimitiveRef = useRef<BoxPrimitive | null>(null);
  const killZonePrimitiveRef = useRef<BoxPrimitive | null>(null);
  const fundingPrimitiveRef = useRef<BoxPrimitive | null>(null);
  const oiDeltaSeriesRef = useRef<ISeriesApi<"Histogram"> | null>(null);
  const positionLinesRef = useRef<any[]>([]);
  const prevCandleCountRef = useRef<number>(0);

  // Crosshair OHLCV display
  const [hoverOHLCV, setHoverOHLCV] = useState<{ o: number; h: number; l: number; c: number; v: number } | null>(null);

  // Swing ↔ IDM click interaction
  const [selectedSwingIdx, setSelectedSwingIdx] = useState<number | null>(null);
  const dataRef = useRef<AnalysisResult | null>(null);
  dataRef.current = data;
  const onElementClickRef = useRef(onElementClick);
  onElementClickRef.current = onElementClick;

  // Create chart once
  useEffect(() => {
    if (!containerRef.current) return;

    const chart = createChart(containerRef.current, {
      layout: {
        background: { type: ColorType.Solid, color: "#ffffff" },
        textColor: "#5c6370",
      },
      grid: {
        vertLines: { visible: false },
        horzLines: { visible: false },
      },
      crosshair: {
        mode: CrosshairMode.Normal,
        vertLine: { color: "#3b82f640", width: 1, style: LineStyle.Dashed },
        horzLine: { color: "#3b82f640", width: 1, style: LineStyle.Dashed },
      },
      rightPriceScale: {
        borderColor: "#1e2230",
      },
      localization: {
        timeFormatter: (time: number) =>
          formatNY(time as number, {
            month: "short", day: "numeric",
            hour: "2-digit", minute: "2-digit", hour12: false,
          }),
      },
      timeScale: {
        borderColor: "#1e2230",
        timeVisible: true,
        secondsVisible: false,
        tickMarkFormatter: (time: number) => {
          const d = new Date((time as number) * 1000);
          const ny = new Intl.DateTimeFormat("en-US", {
            timeZone: "America/New_York",
            year: "numeric", month: "2-digit", day: "2-digit",
            hour: "2-digit", minute: "2-digit", hour12: false,
          }).formatToParts(d);
          const p = (t: string) => ny.find(x => x.type === t)?.value || "";
          const hh = p("hour"), mm = p("minute");
          // Show date when at midnight, otherwise show time
          if (hh === "00" && mm === "00") {
            return `${p("month")}/${p("day")}`;
          }
          return `${hh}:${mm}`;
        },
      },
      width: containerRef.current.clientWidth,
      height: containerRef.current.clientHeight,
    });

    const candleSeries = chart.addCandlestickSeries({
      upColor: "#10b981",
      downColor: "#ef4444",
      borderUpColor: "#10b981",
      borderDownColor: "#ef4444",
      wickUpColor: "#10b98180",
      wickDownColor: "#ef444480",
    });

    // Volume histogram on a separate price scale at the bottom
    const volumeSeries = chart.addHistogramSeries({
      priceFormat: { type: "volume" },
      priceScaleId: "volume",
    });
    chart.priceScale("volume").applyOptions({
      scaleMargins: { top: 0.85, bottom: 0 },
    });
    volumeSeriesRef.current = volumeSeries;

    // Attach box primitives to candle series (kill zones behind FVG behind OB)
    const killZonePrimitive = new BoxPrimitive();
    (candleSeries as any).attachPrimitive(killZonePrimitive);
    killZonePrimitiveRef.current = killZonePrimitive;

    const fvgPrimitive = new BoxPrimitive();
    (candleSeries as any).attachPrimitive(fvgPrimitive);
    fvgPrimitiveRef.current = fvgPrimitive;

    const obPrimitive = new BoxPrimitive();
    (candleSeries as any).attachPrimitive(obPrimitive);
    obPrimitiveRef.current = obPrimitive;

    const positionPrimitive = new BoxPrimitive();
    (candleSeries as any).attachPrimitive(positionPrimitive);
    positionPrimitiveRef.current = positionPrimitive;

    // Funding rate background tinting (behind everything)
    const fundingPrimitive = new BoxPrimitive();
    (candleSeries as any).attachPrimitive(fundingPrimitive);
    fundingPrimitiveRef.current = fundingPrimitive;

    // OI delta histogram on separate price scale at the very bottom
    const oiDeltaSeries = chart.addHistogramSeries({
      priceFormat: { type: "price", precision: 2, minMove: 0.01 },
      priceScaleId: "oi_delta",
    });
    chart.priceScale("oi_delta").applyOptions({
      scaleMargins: { top: 0.92, bottom: 0 },
    });
    oiDeltaSeriesRef.current = oiDeltaSeries;

    chartRef.current = chart;
    candleSeriesRef.current = candleSeries;

    // Click handler: find nearest chart element to the click point
    chart.subscribeClick((param: MouseEventParams) => {
      const d = dataRef.current;
      const cb = onElementClickRef.current;
      if (!param.time || !d || !d.candles.length) {
        setSelectedSwingIdx(null);
        cb?.(null);
        return;
      }

      const clickedTimeSec = param.time as number;
      const clickedCandle = d.candles.find(
        (c) => Math.floor(c.timestamp / 1000) === clickedTimeSec
      );
      if (!clickedCandle) {
        setSelectedSwingIdx(null);
        cb?.(null);
        return;
      }

      const clickedPrice =
        param.point && candleSeriesRef.current
          ? candleSeriesRef.current.coordinateToPrice(param.point.y)
          : null;

      if (clickedPrice === null) {
        setSelectedSwingIdx(null);
        cb?.(null);
        return;
      }

      const ci = clickedCandle.index;

      // Calculate adaptive tolerance from visible price range (~2% of visible range)
      const allPrices = d.candles.map(c => [c.high, c.low]).flat();
      const visibleHigh = Math.max(...allPrices);
      const visibleLow = Math.min(...allPrices);
      const priceRange = visibleHigh - visibleLow;
      const tol = priceRange * 0.025; // 2.5% of visible range

      const inRange = (a: number, b: number) => ci >= Math.min(a, b) - 1 && ci <= Math.max(a, b) + 1;

      // Collect ALL candidate matches with labels for the picker
      const candidates: ClickCandidate[] = [];

      // Helper to build a label for each candidate type
      const makeLabel = (type: string, detail: string) => {
        const prefix: Record<string, string> = { bos: "BOS", choch: "CHoCH", idm: "IDM", fvg: "FVG", ob: "OB", swing: "Swing" };
        return `${prefix[type] ?? type} ${detail}`;
      };

      // 1. BOS lines
      for (let i = 0; i < d.bos_events.length; i++) {
        const b = d.bos_events[i];
        if (inRange(b.broken_swing_index, b.candle_index)) {
          const dist = Math.abs(clickedPrice - b.broken_price);
          if (dist < tol) {
            const dir = b.direction === "bullish" ? "\u25B2" : "\u25BC";
            candidates.push({ type: "bos", index: i, candle_index: b.candle_index, dist, label: makeLabel("bos", dir) });
          }
        }
      }

      // 2. CHoCH lines
      for (let i = 0; i < d.choch_events.length; i++) {
        const ch = d.choch_events[i];
        if (inRange(ch.broken_swing_index, ch.candle_index)) {
          const dist = Math.abs(clickedPrice - ch.broken_price);
          if (dist < tol) {
            const tag = ch.is_fake ? "Fake" : ch.confirmed ? "Confirmed" : "";
            candidates.push({ type: "choch", index: i, candle_index: ch.candle_index, dist, label: makeLabel("choch", tag) });
          }
        }
      }

      // 3. IDM rays
      for (let i = 0; i < d.inducements.length; i++) {
        const idm = d.inducements[i];
        const endIdx = idm.taken_at_candle ?? d.candles.length - 1;
        if (inRange(idm.candle_index, endIdx)) {
          const dist = Math.abs(clickedPrice - idm.price);
          if (dist < tol) {
            const tag = idm.is_major ? "Major" : "Minor";
            candidates.push({ type: "idm", index: i, candle_index: idm.candle_index, dist, label: makeLabel("idm", tag) });
          }
        }
      }

      // 4. OB boxes (click inside the box = distance 0)
      for (let i = 0; i < d.order_blocks.length; i++) {
        const ob = d.order_blocks[i];
        if (!ob.valid) continue;
        const endIdx = ob.mitigated && ob.mitigated_at_candle != null ? ob.mitigated_at_candle : d.candles.length - 1;
        if (inRange(ob.candle_index_start, endIdx)) {
          const dir = ob.direction === "bullish" ? "\u25B2" : "\u25BC";
          if (clickedPrice >= ob.lower_price && clickedPrice <= ob.upper_price) {
            candidates.push({ type: "ob", index: i, candle_index: ob.candle_index_start, dist: 0, label: makeLabel("ob", dir) });
          } else {
            const dist = Math.min(Math.abs(clickedPrice - ob.upper_price), Math.abs(clickedPrice - ob.lower_price));
            if (dist < tol * 0.5) {
              candidates.push({ type: "ob", index: i, candle_index: ob.candle_index_start, dist, label: makeLabel("ob", dir) });
            }
          }
        }
      }

      // 5. FVG zones
      for (let i = 0; i < d.fvgs.length; i++) {
        const f = d.fvgs[i];
        if (!f.valid) continue;
        if (Math.abs(ci - f.candle_index) <= 5) {
          if (clickedPrice >= f.lower_price && clickedPrice <= f.upper_price) {
            const dir = f.direction === "bullish" ? "\u25B2" : "\u25BC";
            candidates.push({ type: "fvg", index: i, candle_index: f.candle_index, dist: 0, label: makeLabel("fvg", dir) });
          }
        }
      }

      // 6. Swings (within ±2 candle tolerance)
      const nearbySwings = d.swings.filter(
        (s) => Math.abs(s.candle_index - ci) <= 2 && s.classification !== "unclassified"
      );
      for (const s of nearbySwings) {
        const idx = d.swings.indexOf(s);
        const dist = Math.abs(clickedPrice - s.price);
        if (dist < tol) {
          candidates.push({ type: "swing", index: idx, candle_index: s.candle_index, dist, label: makeLabel("swing", s.classification) });
        }
      }

      // Sort by distance
      candidates.sort((a, b) => a.dist - b.dist);

      if (candidates.length > 0) {
        // Highlight swing if the closest is a swing
        const best = candidates[0];
        setSelectedSwingIdx(best.type === "swing" ? best.candle_index : null);

        // Get click pixel coordinates relative to the page
        const clickX = param.point?.x ?? 0;
        const clickY = param.point?.y ?? 0;
        // Offset by chart container position
        const rect = containerRef.current?.getBoundingClientRect();
        const pageX = (rect?.left ?? 0) + clickX;
        const pageY = (rect?.top ?? 0) + clickY;

        cb?.({ candidates, clickX: pageX, clickY: pageY });
        return;
      }

      // Nothing matched
      setSelectedSwingIdx(null);
      cb?.(null);
    });

    // Crosshair move — show OHLCV of hovered candle
    chart.subscribeCrosshairMove((param: MouseEventParams) => {
      if (!param.time || !param.seriesData) {
        setHoverOHLCV(null);
        return;
      }
      const candleData = param.seriesData.get(candleSeries) as any;
      if (candleData && candleData.open !== undefined) {
        setHoverOHLCV({
          o: candleData.open,
          h: candleData.high,
          l: candleData.low,
          c: candleData.close,
          v: 0, // will be filled from volume series
        });
        // Try to get volume from volume series
        const volData = param.seriesData.get(volumeSeries) as any;
        if (volData && volData.value !== undefined) {
          setHoverOHLCV(prev => prev ? { ...prev, v: volData.value } : null);
        }
      } else {
        setHoverOHLCV(null);
      }
    });

    // ResizeObserver tracks both width and height of the container
    const ro = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const { width, height } = entry.contentRect;
        chart.applyOptions({ width, height });
      }
    });
    ro.observe(containerRef.current);

    return () => {
      ro.disconnect();
      chart.remove();
      chartRef.current = null;
      candleSeriesRef.current = null;
      volumeSeriesRef.current = null;
      priceLinesRef.current = [];
      idmSeriesRef.current = [];
      bosSeriesRef.current = [];
      chochSeriesRef.current = [];
      obPrimitiveRef.current = null;
      killZonePrimitiveRef.current = null;
      positionPrimitiveRef.current = null;
      fundingPrimitiveRef.current = null;
      oiDeltaSeriesRef.current = null;
      positionLinesRef.current = [];
    };
  }, []);

  // Update data when analysis result changes
  useEffect(() => {
    if (!data || !chartRef.current || !candleSeriesRef.current) return;
    const chart = chartRef.current;
    const candleSeries = candleSeriesRef.current;

    // Set candle data
    const candles = data.candles;
    if (!candles || candles.length === 0) return;

    const candleData: CandlestickData[] = candles.map((c) => ({
      time: toTV(c.timestamp),
      open: c.open,
      high: c.high,
      low: c.low,
      close: c.close,
    }));
    candleSeries.setData(candleData);

    // Volume bars — green for up candles, red for down
    if (volumeSeriesRef.current) {
      volumeSeriesRef.current.setData(candles.map((c) => ({
        time: toTV(c.timestamp),
        value: c.volume,
        color: c.close >= c.open ? "#10b98130" : "#ef444430",
      })));
    }

    // --- KILL ZONE SHADING: London (2-5AM NY) and NY (7-10AM NY) ---
    if (killZonePrimitiveRef.current) {
      const subDaily = ["H4", "H1", "M15", "M5"].includes(data.timeframe);
      if (subDaily) {
        const kzBoxes: BoxData[] = [];
        // Get the full price range for full-height bands
        const allHighs = candles.map(c => c.high);
        const allLows = candles.map(c => c.low);
        const priceHigh = Math.max(...allHighs);
        const priceLow = Math.min(...allLows);
        const pricePad = (priceHigh - priceLow) * 0.5;
        const boxTop = priceHigh + pricePad;
        const boxBottom = priceLow - pricePad;

        // Collect unique NY dates from chart data
        const nyDateFmt = new Intl.DateTimeFormat("en-US", {
          timeZone: "America/New_York",
          year: "numeric", month: "2-digit", day: "2-digit",
        });
        const seenDates = new Set<string>();
        const nyDates: { year: number; month: number; day: number }[] = [];
        for (const c of candles) {
          const parts = nyDateFmt.formatToParts(new Date(c.timestamp));
          const key = parts.map(p => p.value).join("");
          if (!seenDates.has(key)) {
            seenDates.add(key);
            nyDates.push({
              year: parseInt(parts.find(p => p.type === "year")!.value),
              month: parseInt(parts.find(p => p.type === "month")!.value),
              day: parseInt(parts.find(p => p.type === "day")!.value),
            });
          }
        }

        // For each day, create kill zone boxes (opacity adapts to timeframe)
        const firstTs = candles[0].timestamp;
        const lastTs = candles[candles.length - 1].timestamp;
        const opac = KZ_OPACITY[data.timeframe] || KZ_OPACITY.M15;
        for (const { year, month, day } of nyDates) {
          for (const kz of KILL_ZONES_NY) {
            const startMs = nyHourToUtcMs(year, month, day, kz.startHour);
            const endMs = nyHourToUtcMs(year, month, day, kz.endHour);
            // Skip if completely outside chart data range
            if (endMs < firstTs || startMs > lastTs) continue;
            kzBoxes.push({
              startTime: toTV(Math.max(startMs, firstTs)),
              endTime: toTV(Math.min(endMs, lastTs)),
              upperPrice: boxTop,
              lowerPrice: boxBottom,
              fillColor: `rgba(${kz.rgb}, ${opac.fill})`,
              borderColor: `rgba(${kz.rgb}, ${opac.border})`,
              label: kz.name,
              labelColor: "#000000",
              labelTop: true,
            });
          }
        }
        killZonePrimitiveRef.current.setBoxes(kzBoxes);
      } else {
        killZonePrimitiveRef.current.setBoxes([]);
      }
    }

    // --- MARKERS: Swings, BOS, CHoCH ---
    const markers: any[] = [];

    // Swing point markers
    if (visibility.swings) {
      data.swings.forEach((s: SwingPoint) => {
        const candle = candles[s.candle_index];
        if (!candle) return;

        const isHigh = s.swing_type === "swing_high";
        const label = s.classification === "unclassified" ? "?" : s.classification;
        const isSelected = s.candle_index === selectedSwingIdx;
        const validColor = isSelected
          ? "#fbbf24"  // bright yellow when selected
          : s.is_valid_smc ? (isHigh ? "#f59e0b" : "#3b82f6") : "#6b7280";

        markers.push({
          time: toTV(candle.timestamp),
          position: isHigh ? "aboveBar" : "belowBar",
          color: validColor,
          shape: isHigh ? "arrowDown" : "arrowUp",
          text: isSelected ? `▶ ${label}` : label,
        });
      });
    }

    // Signal activation marker (shown when deep analysis is open)
    if (signalMarker) {
      // Determine candle interval for fuzzy matching across TFs
      const candleInterval = candles.length >= 2
        ? Math.abs(candles[1].timestamp - candles[0].timestamp)
        : 60000;
      const sigCandle = candles.find(c =>
        Math.abs(c.timestamp - signalMarker.timestamp) < candleInterval
      );
      if (sigCandle) {
        const isBull = signalMarker.direction === "bullish";
        markers.push({
          time: toTV(sigCandle.timestamp),
          position: isBull ? "belowBar" : "aboveBar",
          color: isBull ? "#22c55e" : "#ef4444",
          shape: isBull ? "arrowUp" : "arrowDown",
          size: 2,
          text: `${isBull ? "LONG" : "SHORT"} Entry`,
        });
      }
    }

    // Sort markers by time (required by lightweight-charts)
    markers.sort((a, b) => (a.time as number) - (b.time as number));
    candleSeries.setMarkers(markers);

    // --- IDM RAYS: horizontal dashed lines from origin to taken candle ---

    // Remove old IDM line series
    for (const s of idmSeriesRef.current) {
      try { chart.removeSeries(s); } catch { /* series already removed */ }
    }
    idmSeriesRef.current = [];

    if (visibility.idm) {
      const lastIdx = candles.length - 1;
      data.inducements.forEach((idm: Inducement) => {
          const startCandle = candles[idm.candle_index];
          if (!startCandle) return;

          const endIdx = idm.taken_at_candle != null
            ? idm.taken_at_candle
            : lastIdx;
          if (endIdx <= idm.candle_index) return;
          const endCandle = candles[endIdx];
          if (!endCandle) return;

          const startTime = toTV(startCandle.timestamp);
          const endTime = toTV(endCandle.timestamp);
          if (endTime <= startTime) return;

          const isTaken = idm.status === "taken";
          const isHighlighted = idm.parent_swing_index === selectedSwingIdx;
          const color = isHighlighted
            ? "#fbbf24"  // bright yellow when parent swing is selected
            : isTaken ? "#60a5fa80" : "#60a5fa";
          const lineWidth = isHighlighted ? 3 : 1;

          const lineSeries = chart.addLineSeries({
            color,
            lineWidth,
            lineStyle: LineStyle.Dashed,
            crosshairMarkerVisible: false,
            priceLineVisible: false,
            lastValueVisible: false,
          });

          lineSeries.setData([
            { time: startTime, value: idm.price },
            { time: endTime, value: idm.price },
          ]);

          // Add "IDM" label at the midpoint of the ray
          const midIdx = Math.floor((idm.candle_index + endIdx) / 2);
          const midCandle = candles[midIdx];
          if (midCandle) {
            lineSeries.setMarkers([{
              time: toTV(midCandle.timestamp),
              position: "aboveBar" as const,
              color,
              shape: "square" as const,
              size: isHighlighted ? 1 : 0.01,
              text: "IDM",
            }]);
          }

          idmSeriesRef.current.push(lineSeries);
        });
    }

    // --- BOS RAYS: horizontal lines from broken swing to break candle ---

    // Remove old BOS line series
    for (const s of bosSeriesRef.current) {
      try { chart.removeSeries(s); } catch { /* already removed */ }
    }
    bosSeriesRef.current = [];

    if (visibility.bos) {
      data.bos_events.forEach((b: BOS) => {
        const startCandle = candles[b.broken_swing_index];
        const endCandle = candles[b.candle_index];
        if (!startCandle || !endCandle) return;

        const startTime = toTV(startCandle.timestamp);
        const endTime = toTV(endCandle.timestamp);
        if (endTime <= startTime) return;

        const color = b.valid ? "#22d3ee" : "#6b728080"; // cyan valid, gray invalid

        const lineSeries = chart.addLineSeries({
          color,
          lineWidth: 2,
          lineStyle: LineStyle.Dashed,
          crosshairMarkerVisible: false,
          priceLineVisible: false,
          lastValueVisible: false,
        });

        // Add midpoint data point so the marker can sit in the center of the ray
        const midIdx = Math.floor((b.broken_swing_index + b.candle_index) / 2);
        const midCandle = candles[midIdx];
        const lineData: { time: any; value: number }[] = [
          { time: startTime, value: b.broken_price },
        ];
        if (midCandle) {
          const midTime = toTV(midCandle.timestamp);
          if (midTime > startTime && midTime < endTime) {
            lineData.push({ time: midTime, value: b.broken_price });
          }
        }
        lineData.push({ time: endTime, value: b.broken_price });
        lineSeries.setData(lineData);

        // Add "BOS" label with directional arrow at the midpoint of the ray
        if (midCandle) {
          const midTime = toTV(midCandle.timestamp);
          if (midTime > startTime && midTime < endTime) {
            const isBull = b.direction === "bullish";
            const label = b.valid ? "BOS" : "xBOS";
            lineSeries.setMarkers([{
              time: midTime,
              position: "inBar" as const,
              color,
              shape: isBull ? "arrowUp" as const : "arrowDown" as const,
              size: 0.5,
              text: label,
            }]);
          }
        }

        bosSeriesRef.current.push(lineSeries);
      });
    }

    // --- CHoCH RAYS: horizontal lines from broken swing to break candle ---

    // Remove old CHoCH line series
    for (const s of chochSeriesRef.current) {
      try { chart.removeSeries(s); } catch { /* already removed */ }
    }
    chochSeriesRef.current = [];

    if (visibility.choch) {
      data.choch_events.forEach((ch: CHoCH) => {
        const startCandle = candles[ch.broken_swing_index];
        const endCandle = candles[ch.candle_index];
        if (!startCandle || !endCandle) return;

        const startTime = toTV(startCandle.timestamp);
        const endTime = toTV(endCandle.timestamp);
        if (endTime <= startTime) return;

        // MSS = gold/solid, confirmed CHoCH = pink/dashed, fake = gray, default = purple
        let color = "#a855f7"; // purple default
        let lineWidth = 2;
        let style = LineStyle.Dashed;
        if (ch.is_fake) {
          color = "#6b728080";
        } else if (ch.is_mss) {
          color = "#f59e0b"; // amber/gold for MSS
          lineWidth = 3;
          style = LineStyle.Solid;
        } else if (ch.confirmed) {
          color = "#ec4899"; // pink confirmed CHoCH
        }

        const lineSeries = chart.addLineSeries({
          color,
          lineWidth: lineWidth as any,
          lineStyle: style,
          crosshairMarkerVisible: false,
          priceLineVisible: false,
          lastValueVisible: false,
        });

        // Add midpoint data point so the marker can sit in the center of the ray
        const midIdx = Math.floor((ch.broken_swing_index + ch.candle_index) / 2);
        const midCandle = candles[midIdx];
        const lineData: { time: any; value: number }[] = [
          { time: startTime, value: ch.broken_price },
        ];
        if (midCandle) {
          const midTime = toTV(midCandle.timestamp);
          if (midTime > startTime && midTime < endTime) {
            lineData.push({ time: midTime, value: ch.broken_price });
          }
        }
        lineData.push({ time: endTime, value: ch.broken_price });
        lineSeries.setData(lineData);

        // MSS gets "MSS" label, CHoCH gets "CHoCH" or "xCHoCH"
        if (midCandle) {
          const midTime = toTV(midCandle.timestamp);
          if (midTime > startTime && midTime < endTime) {
            const isBull = ch.direction === "bullish";
            const label = ch.is_mss ? "MSS" : ch.is_fake ? "xCHoCH" : "CHoCH";
            lineSeries.setMarkers([{
              time: midTime,
              position: "inBar" as const,
              color,
              shape: isBull ? "arrowUp" as const : "arrowDown" as const,
              size: ch.is_mss ? 0.8 : 0.5,
              text: label,
            }]);
          }
        }

        chochSeriesRef.current.push(lineSeries);
      });
    }

    // --- PRICE LINES: Premium/Discount, FVG ---

    // Remove all previous price lines before creating new ones
    for (const line of priceLinesRef.current) {
      candleSeries.removePriceLine(line);
    }
    priceLinesRef.current = [];

    // Signal activation entry price line (shown when deep analysis is open)
    if (signalMarker) {
      const isBull = signalMarker.direction === "bullish";
      priceLinesRef.current.push(candleSeries.createPriceLine({
        price: signalMarker.entryPrice,
        color: isBull ? "#22c55e" : "#ef4444",
        lineWidth: 1 as 1,
        lineStyle: LineStyle.Dashed,
        axisLabelVisible: true,
        title: `Entry ${signalMarker.grade}`,
      }));
    }

    // Premium/Discount equilibrium line
    if (visibility.pd && data.premium_discount) {
      priceLinesRef.current.push(candleSeries.createPriceLine({
        price: data.premium_discount.equilibrium,
        color: "#eab308",
        lineWidth: 1,
        lineStyle: LineStyle.Dashed,
        axisLabelVisible: true,
        title: "EQ 50%",
      }));
      priceLinesRef.current.push(candleSeries.createPriceLine({
        price: data.premium_discount.swing_high,
        color: "#ef444480",
        lineWidth: 1,
        lineStyle: LineStyle.Dotted,
        axisLabelVisible: false,
        title: "Premium",
      }));
      priceLinesRef.current.push(candleSeries.createPriceLine({
        price: data.premium_discount.swing_low,
        color: "#22c55e80",
        lineWidth: 1,
        lineStyle: LineStyle.Dotted,
        axisLabelVisible: false,
        title: "Discount",
      }));
    }

    // --- QUANT OVERLAYS: ATR bands + Max Pain line (when quant_context available) ---
    const qc = data.quant_context;
    if (qc && qc.atr_m15 > 0) {
      const lastCandle = candles[candles.length - 1];
      if (lastCandle) {
        const currentPrice = lastCandle.close;
        const atr = qc.atr_m15;
        // ATR bands: ±2×ATR from current price
        priceLinesRef.current.push(candleSeries.createPriceLine({
          price: currentPrice + atr * 2,
          color: "rgba(139, 92, 246, 0.3)",
          lineWidth: 1,
          lineStyle: LineStyle.Dotted,
          axisLabelVisible: true,
          title: "ATR+2",
        }));
        priceLinesRef.current.push(candleSeries.createPriceLine({
          price: currentPrice - atr * 2,
          color: "rgba(139, 92, 246, 0.3)",
          lineWidth: 1,
          lineStyle: LineStyle.Dotted,
          axisLabelVisible: true,
          title: "ATR-2",
        }));
      }
    }

    // Max pain line from options data
    if (qc?.options_data?.max_pain && qc.options_data.max_pain > 0) {
      priceLinesRef.current.push(candleSeries.createPriceLine({
        price: qc.options_data.max_pain,
        color: "rgba(6, 182, 212, 0.5)",
        lineWidth: 1,
        lineStyle: LineStyle.LargeDashed,
        axisLabelVisible: true,
        title: "Max Pain",
      }));
    }

    // FVG zones as filled boxes (rendered via canvas primitive)
    if (visibility.fvg && fvgPrimitiveRef.current) {
      const lastIdx = candles.length - 1;
      // Show both active AND recently mitigated FVGs (mitigated = faded)
      // Take last N of each direction to ensure both buy/sell zones are visible
      const validFvgs = data.fvgs.filter((f: FVG) => f.valid);
      const bullFvgs = validFvgs.filter((f: FVG) => f.direction === "bullish").slice(-15);
      const bearFvgs = validFvgs.filter((f: FVG) => f.direction === "bearish").slice(-15);
      const balancedFvgs = [...bullFvgs, ...bearFvgs];

      const fvgBoxes: BoxData[] = balancedFvgs
        .map((f: FVG) => {
          const startCandle = candles[f.candle_index];
          if (!startCandle) return null;

          // Mitigated FVGs end at mitigated candle; active extend to chart end
          const endIdx = f.mitigated && f.mitigated_at_candle != null
            ? f.mitigated_at_candle
            : lastIdx;
          const endCandle = candles[endIdx];
          if (!endCandle) return null;

          const startTime = toTV(startCandle.timestamp);
          const endTime = toTV(endCandle.timestamp);
          if (endTime <= startTime) return null;

          const isBull = f.direction === "bullish";
          const rgb = isBull ? "34, 197, 94" : "239, 68, 68"; // green-500 / red-500
          const fillAlpha = f.mitigated ? 0.06 : 0.18;
          const borderAlpha = f.mitigated ? 0.2 : 0.6;
          const labelAlpha = f.mitigated ? 0.35 : 0.85;
          const arrow = isBull ? " \u25B2" : " \u25BC";
          const label = (f.mitigated ? "xFVG" : "FVG") + arrow;

          return {
            startTime,
            endTime,
            upperPrice: f.upper_price,
            lowerPrice: f.lower_price,
            fillColor: `rgba(${rgb}, ${fillAlpha})`,
            borderColor: `rgba(${rgb}, ${borderAlpha})`,
            label,
            labelColor: `rgba(${rgb}, ${labelAlpha})`,
          } as BoxData;
        })
        .filter((b): b is BoxData => b !== null);

      fvgPrimitiveRef.current.setBoxes(fvgBoxes);
    } else if (fvgPrimitiveRef.current) {
      fvgPrimitiveRef.current.setBoxes([]);
    }

    // --- OB ZONES: filled rectangular boxes via canvas primitive ---

    if (visibility.ob && obPrimitiveRef.current) {
      const lastIdx = candles.length - 1;
      // Take last N of each direction to ensure both buy/sell zones are visible
      const validObs = data.order_blocks.filter((ob: OrderBlock) => ob.valid);
      const bullObs = validObs.filter((ob: OrderBlock) => ob.direction === "bullish").slice(-8);
      const bearObs = validObs.filter((ob: OrderBlock) => ob.direction === "bearish").slice(-8);
      const balancedObs = [...bullObs, ...bearObs];

      const boxes: OBBoxData[] = balancedObs
        .map((ob: OrderBlock) => {
          const startCandle = candles[ob.candle_index_start];
          if (!startCandle) return null;

          const endIdx = ob.mitigated && ob.mitigated_at_candle != null
            ? ob.mitigated_at_candle
            : lastIdx;
          const endCandle = candles[endIdx];
          if (!endCandle) return null;

          const startTime = toTV(startCandle.timestamp);
          const endTime = toTV(endCandle.timestamp);
          if (endTime <= startTime) return null;

          const isBull = ob.direction === "bullish";
          const rgb = isBull ? "59, 130, 246" : "249, 115, 22";
          const fillAlpha = ob.mitigated ? 0.08 : 0.18;
          const borderAlpha = ob.mitigated ? 0.25 : 0.7;
          const labelAlpha = ob.mitigated ? 0.4 : 0.9;
          const arrow = isBull ? " \u25B2" : " \u25BC";
          const label = (ob.mitigated ? "xOB" : "OB") + arrow;

          return {
            startTime,
            endTime,
            upperPrice: ob.upper_price,
            lowerPrice: ob.lower_price,
            fillColor: `rgba(${rgb}, ${fillAlpha})`,
            borderColor: `rgba(${rgb}, ${borderAlpha})`,
            label,
            labelColor: `rgba(${rgb}, ${labelAlpha})`,
          } as OBBoxData;
        })
        .filter((b): b is OBBoxData => b !== null);

      obPrimitiveRef.current.setBoxes(boxes);
    } else if (obPrimitiveRef.current) {
      obPrimitiveRef.current.setBoxes([]);
    }

    // --- FUTURES OVERLAYS: OI Delta histogram + Funding rate tinting ---

    // OI Delta histogram — green bars for rising OI, red for falling
    if (visibility.futures && data.futures_context?.oi_deltas?.length && oiDeltaSeriesRef.current) {
      // Map OI delta timestamps to nearest candle timestamps
      const oiMap = new Map<number, number>(); // tvTime -> delta_pct
      for (const oi of data.futures_context.oi_deltas) {
        let bestCandle = candles[0];
        let bestDist = Math.abs(candles[0].timestamp - oi.timestamp);
        for (const c of candles) {
          const dist = Math.abs(c.timestamp - oi.timestamp);
          if (dist < bestDist) {
            bestDist = dist;
            bestCandle = c;
          }
        }
        oiMap.set(toTV(bestCandle.timestamp), oi.delta_pct);
      }

      const oiData = Array.from(oiMap.entries())
        .sort((a, b) => a[0] - b[0])
        .map(([time, delta]) => ({
          time: time as any,
          value: delta,
          color: delta >= 0 ? "rgba(34, 197, 94, 0.6)" : "rgba(239, 68, 68, 0.6)",
        }));
      oiDeltaSeriesRef.current.setData(oiData);
    } else if (oiDeltaSeriesRef.current) {
      oiDeltaSeriesRef.current.setData([]);
    }

    // Funding rate background tinting — red for crowded longs, green for crowded shorts
    if (visibility.futures && data.futures_context?.funding_history?.length && fundingPrimitiveRef.current) {
      const allHighs = candles.map(c => c.high);
      const allLows = candles.map(c => c.low);
      const priceHigh = Math.max(...allHighs);
      const priceLow = Math.min(...allLows);
      const pricePad = (priceHigh - priceLow) * 0.5;
      const boxTop = priceHigh + pricePad;
      const boxBottom = priceLow - pricePad;

      const fundingBoxes: BoxData[] = [];
      for (const fh of data.futures_context.funding_history) {
        const rate = fh.rate;
        if (Math.abs(rate) <= 0.0005) continue; // Only tint extreme funding
        const startTime = toTV(fh.timestamp);
        const endTime = toTV(fh.timestamp + 8 * 3600_000); // Funding periods are ~8h
        const alpha = Math.min(0.08, Math.abs(rate) * 50);
        const color = rate > 0
          ? `rgba(239, 68, 68, ${alpha})`   // Red = crowded longs (bearish pressure)
          : `rgba(34, 197, 94, ${alpha})`;   // Green = crowded shorts (bullish pressure)
        fundingBoxes.push({
          startTime,
          endTime,
          upperPrice: boxTop,
          lowerPrice: boxBottom,
          fillColor: color,
          borderColor: "transparent",
          label: "",
          labelColor: "transparent",
        });
      }
      fundingPrimitiveRef.current.setBoxes(fundingBoxes);
    } else if (fundingPrimitiveRef.current) {
      fundingPrimitiveRef.current.setBoxes([]);
    }

    // --- OPEN POSITION DISPLAY: SL / Entry / TP boxes ---
    // Remove old position price lines
    for (const line of positionLinesRef.current) {
      candleSeries.removePriceLine(line);
    }
    positionLinesRef.current = [];

    if (positionPrimitiveRef.current) {
      const posBoxes: BoxData[] = [];
      const lastCandle = candles[candles.length - 1];

      // Compute 5-candle pixel extension for position boxes
      const lastCandleCoord = chart.timeScale().timeToCoordinate(toTV(lastCandle.timestamp));
      const prevCandleCoord = candles.length >= 2
        ? chart.timeScale().timeToCoordinate(toTV(candles[candles.length - 2].timestamp))
        : null;
      const candlePxWidth = (lastCandleCoord !== null && prevCandleCoord !== null)
        ? Math.abs(lastCandleCoord - prevCandleCoord)
        : 15;
      const extendRightPx = 5 * candlePxWidth;

      for (const trade of openTrades) {
        if (!lastCandle) break;
        const entry = trade.entry_price;
        const sl = trade.stop_loss;
        const tp = trade.take_profit;
        if (!entry || !sl || !tp) continue;

        // Start from opened_at timestamp — snap to the candle whose period contains that time.
        // Candle timestamps mark the START of each period, so we find the last candle
        // with timestamp <= openedMs (the candle the trade was opened within).
        let startTime: any;
        if (trade.opened_at) {
          const openedMs = new Date(trade.opened_at.replace(" ", "T") + "Z").getTime();
          if (!isNaN(openedMs)) {
            let snapIdx = -1;
            for (let i = candles.length - 1; i >= 0; i--) {
              if (candles[i].timestamp <= openedMs) {
                snapIdx = i;
                break;
              }
            }
            // If opened before all chart data, start from first candle
            if (snapIdx === -1) snapIdx = 0;
            startTime = toTV(candles[snapIdx].timestamp);
          } else {
            startTime = toTV(lastCandle.timestamp);
          }
        } else {
          startTime = toTV(lastCandle.timestamp);
        }
        const endTime = toTV(lastCandle.timestamp);

        // TP zone (green) — between entry and TP
        posBoxes.push({
          startTime,
          endTime,
          upperPrice: Math.max(entry, tp),
          lowerPrice: Math.min(entry, tp),
          fillColor: "rgba(34, 197, 94, 0.12)",
          borderColor: "rgba(34, 197, 94, 0.0)",
          label: "",
          labelColor: "rgba(34, 197, 94, 0.9)",
          rightExtendPx: extendRightPx,
        });

        // SL zone (red) — between entry and SL
        posBoxes.push({
          startTime,
          endTime,
          upperPrice: Math.max(entry, sl),
          lowerPrice: Math.min(entry, sl),
          fillColor: "rgba(239, 68, 68, 0.12)",
          borderColor: "rgba(239, 68, 68, 0.0)",
          label: "",
          labelColor: "rgba(239, 68, 68, 0.9)",
          rightExtendPx: extendRightPx,
        });
      }
      positionPrimitiveRef.current.setBoxes(posBoxes);

      // Price lines for Entry / SL / TP
      for (const trade of openTrades) {
        const entry = trade.entry_price;
        const sl = trade.stop_loss;
        const tp = trade.take_profit;
        if (!entry || !sl || !tp) continue;

        const isLong = trade.direction === "bullish";
        const dir = isLong ? "LONG" : "SHORT";
        const isManual = trade.trade_source === "manual";
        const tag = isManual ? "Manual" : (trade.grade || "");
        const rr = trade.risk_reward_ratio?.toFixed(1) || "";
        const entryColor = isManual ? "#a855f7" : "#3b82f6"; // purple for manual, blue for signal

        // Entry line
        positionLinesRef.current.push(
          candleSeries.createPriceLine({
            price: entry,
            color: entryColor,
            lineWidth: 2,
            lineStyle: LineStyle.Solid,
            axisLabelVisible: true,
            title: `${dir} ${tag} Entry $${entry.toLocaleString(undefined, { maximumFractionDigits: 0 })}`,
          })
        );

        // TP line
        positionLinesRef.current.push(
          candleSeries.createPriceLine({
            price: tp,
            color: "#22c55e",
            lineWidth: 1,
            lineStyle: LineStyle.Dashed,
            axisLabelVisible: true,
            title: `TP $${tp.toLocaleString(undefined, { maximumFractionDigits: 0 })} (R:R ${rr})`,
          })
        );

        // SL line
        positionLinesRef.current.push(
          candleSeries.createPriceLine({
            price: sl,
            color: "#ef4444",
            lineWidth: 1,
            lineStyle: LineStyle.Dashed,
            axisLabelVisible: true,
            title: `SL $${sl.toLocaleString(undefined, { maximumFractionDigits: 0 })}`,
          })
        );
      }
    }

    // Fit content only when candle count changes (new timeframe), not on live refreshes
    if (candles.length !== prevCandleCountRef.current) {
      chart.timeScale().fitContent();
      prevCandleCountRef.current = candles.length;
    }
  }, [data, visibility, selectedSwingIdx, openTrades, signalMarker]);

  // Live price update — update the last candle's close in real-time
  useEffect(() => {
    if (!livePrice || !candleSeriesRef.current || !data?.candles?.length) return;

    const lastCandle = data.candles[data.candles.length - 1];
    if (!lastCandle) return;

    candleSeriesRef.current.update({
      time: toTV(lastCandle.timestamp),
      open: lastCandle.open,
      high: Math.max(lastCandle.high, livePrice),
      low: Math.min(lastCandle.low, livePrice),
      close: livePrice,
    });
  }, [livePrice, data]);

  const handleReset = useCallback(() => {
    if (!chartRef.current || !candleSeriesRef.current) return;
    const ts = chartRef.current.timeScale();
    const count = candleSeriesRef.current.data().length;
    if (count > 0) {
      const from = Math.max(0, count - 50);
      ts.setVisibleLogicalRange({ from, to: count + 3 });
    }
  }, []);

  return (
    <div className="relative w-full h-full">
      <div
        ref={containerRef}
        className="w-full h-full rounded-xl overflow-hidden border border-[var(--border-primary)] shadow-lg shadow-black/20"
      />
      {/* Price label + OHLCV overlay — top left of chart */}
      <div className="absolute top-2 left-3 pointer-events-none" style={{ zIndex: 10 }}>
        <div className="flex items-center gap-2">
          <span className="text-sm font-semibold text-black">BTCUSDT</span>
          {livePrice != null && (
            <>
              <span className="text-sm font-mono font-medium text-black">
                ${livePrice.toLocaleString(undefined, { minimumFractionDigits: 0, maximumFractionDigits: 0 })}
              </span>
              {priceChangePct != null && (
                <span className={`text-xs font-mono font-medium ${priceChangePct >= 0 ? "text-emerald-600" : "text-red-600"}`}>
                  {priceChangePct >= 0 ? "+" : ""}{priceChangePct.toFixed(2)}%
                </span>
              )}
            </>
          )}
        </div>
        {hoverOHLCV && (
          <div className="flex items-center gap-2.5 mt-0.5 text-[11px] font-mono text-gray-500">
            <span>O: <span className={hoverOHLCV.c >= hoverOHLCV.o ? "text-emerald-600" : "text-red-600"}>{hoverOHLCV.o.toLocaleString(undefined, { maximumFractionDigits: 2 })}</span></span>
            <span>H: <span className={hoverOHLCV.c >= hoverOHLCV.o ? "text-emerald-600" : "text-red-600"}>{hoverOHLCV.h.toLocaleString(undefined, { maximumFractionDigits: 2 })}</span></span>
            <span>L: <span className={hoverOHLCV.c >= hoverOHLCV.o ? "text-emerald-600" : "text-red-600"}>{hoverOHLCV.l.toLocaleString(undefined, { maximumFractionDigits: 2 })}</span></span>
            <span>C: <span className={hoverOHLCV.c >= hoverOHLCV.o ? "text-emerald-600" : "text-red-600"}>{hoverOHLCV.c.toLocaleString(undefined, { maximumFractionDigits: 2 })}</span></span>
            <span>V: <span className="text-gray-600">{hoverOHLCV.v >= 1e6 ? (hoverOHLCV.v / 1e6).toFixed(2) + "M" : hoverOHLCV.v >= 1e3 ? (hoverOHLCV.v / 1e3).toFixed(1) + "K" : hoverOHLCV.v.toFixed(0)}</span></span>
          </div>
        )}
      </div>
      {/* Reset / recenter button — bottom center, above time axis */}
      <button
        onClick={handleReset}
        style={{ zIndex: 10, background: "rgba(26,26,46,0.55)", borderColor: "rgba(255,255,255,0.08)" }}
        className="absolute bottom-7 left-1/2 -translate-x-1/2 flex items-center gap-1.5 px-3 py-1 text-[10px] font-medium text-[var(--text-muted)] backdrop-blur-sm border rounded-full shadow-sm hover:text-[var(--text-primary)] transition-all opacity-40 hover:opacity-100"
        title="Reset chart view"
      >
        <RotateCcw className="w-3 h-3" />
        Reset
      </button>
    </div>
  );
}
