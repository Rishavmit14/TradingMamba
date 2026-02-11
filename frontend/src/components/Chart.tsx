"use client";

import { useEffect, useRef } from "react";
import {
  createChart,
  CrosshairMode,
  IChartApi,
  ISeriesApi,
  CandlestickData,
  ColorType,
  LineStyle,
} from "lightweight-charts";
import { AnalysisResult, DetectorVisibility, SwingPoint, Inducement, BOS, CHoCH, FVG, OrderBlock } from "@/lib/types";

interface ChartProps {
  data: AnalysisResult | null;
  visibility: DetectorVisibility;
}

/** Convert unix ms timestamp to unix seconds for TradingView. */
function toTV(ts: number) {
  return Math.floor(ts / 1000) as any;
}

export default function Chart({ data, visibility }: ChartProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const candleSeriesRef = useRef<ISeriesApi<"Candlestick"> | null>(null);
  const priceLinesRef = useRef<any[]>([]);
  const idmSeriesRef = useRef<ISeriesApi<"Line">[]>([]);
  const chochSeriesRef = useRef<ISeriesApi<"Line">[]>([]);
  const prevDataRef = useRef<AnalysisResult | null>(null);

  // Create chart once
  useEffect(() => {
    if (!containerRef.current) return;

    const chart = createChart(containerRef.current, {
      layout: {
        background: { type: ColorType.Solid, color: "#0a0a0f" },
        textColor: "#9ca3af",
      },
      grid: {
        vertLines: { visible: false },
        horzLines: { visible: false },
      },
      crosshair: {
        mode: CrosshairMode.Normal,
        vertLine: { color: "#4b5563", width: 1, style: LineStyle.Dashed },
        horzLine: { color: "#4b5563", width: 1, style: LineStyle.Dashed },
      },
      rightPriceScale: {
        borderColor: "#1f2937",
      },
      timeScale: {
        borderColor: "#1f2937",
        timeVisible: true,
        secondsVisible: false,
      },
      width: containerRef.current.clientWidth,
      height: containerRef.current.clientHeight,
    });

    const candleSeries = chart.addCandlestickSeries({
      upColor: "#22c55e",
      downColor: "#ef4444",
      borderUpColor: "#22c55e",
      borderDownColor: "#ef4444",
      wickUpColor: "#22c55e",
      wickDownColor: "#ef4444",
    });

    chartRef.current = chart;
    candleSeriesRef.current = candleSeries;

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
      priceLinesRef.current = [];
      idmSeriesRef.current = [];
      chochSeriesRef.current = [];
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

    // --- MARKERS: Swings, BOS, CHoCH ---
    const markers: any[] = [];

    // Swing point markers
    if (visibility.swings) {
      data.swings.forEach((s: SwingPoint) => {
        const candle = candles[s.candle_index];
        if (!candle) return;

        const isHigh = s.swing_type === "swing_high";
        const label = s.classification === "unclassified" ? "?" : s.classification;
        const validColor = s.is_valid_smc ? (isHigh ? "#f59e0b" : "#3b82f6") : "#6b7280";

        markers.push({
          time: toTV(candle.timestamp),
          position: isHigh ? "aboveBar" : "belowBar",
          color: validColor,
          shape: isHigh ? "arrowDown" : "arrowUp",
          text: label,
        });
      });
    }

    // BOS markers
    if (visibility.bos) {
      data.bos_events.forEach((b: BOS) => {
        const candle = candles[b.candle_index];
        if (!candle) return;

        markers.push({
          time: toTV(candle.timestamp),
          position: b.direction === "bullish" ? "belowBar" : "aboveBar",
          color: b.valid ? "#22d3ee" : "#6b7280",
          shape: "circle",
          text: b.valid ? "BOS" : "xBOS",
        });
      });
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
          const color = isTaken ? "#60a5fa80" : "#60a5fa";

          const lineSeries = chart.addLineSeries({
            color,
            lineWidth: 1,
            lineStyle: LineStyle.Dashed,
            crosshairMarkerVisible: false,
            priceLineVisible: false,
            lastValueVisible: false,
          });

          lineSeries.setData([
            { time: startTime, value: idm.price },
            { time: endTime, value: idm.price },
          ]);

          idmSeriesRef.current.push(lineSeries);
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

        let color = "#a855f7"; // purple default
        if (ch.is_fake) color = "#6b728080";
        else if (ch.confirmed) color = "#ec4899"; // pink confirmed

        const lineSeries = chart.addLineSeries({
          color,
          lineWidth: 2,
          lineStyle: LineStyle.Dashed,
          crosshairMarkerVisible: false,
          priceLineVisible: false,
          lastValueVisible: false,
        });

        lineSeries.setData([
          { time: startTime, value: ch.broken_price },
          { time: endTime, value: ch.broken_price },
        ]);

        // Add "CHoCH" label at the midpoint of the ray
        const midIdx = Math.floor((ch.broken_swing_index + ch.candle_index) / 2);
        const midCandle = candles[midIdx];
        if (midCandle) {
          lineSeries.setMarkers([{
            time: toTV(midCandle.timestamp),
            position: "aboveBar" as const,
            color,
            shape: "square" as const,
            size: 0.01,
            text: ch.is_fake ? "xCHoCH" : "CHoCH",
          }]);
        }

        chochSeriesRef.current.push(lineSeries);
      });
    }

    // --- PRICE LINES: Premium/Discount, FVG, OB ---

    // Remove all previous price lines before creating new ones
    for (const line of priceLinesRef.current) {
      candleSeries.removePriceLine(line);
    }
    priceLinesRef.current = [];

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

    // Active FVG zones as price lines
    if (visibility.fvg) {
      data.fvgs
        .filter((f: FVG) => f.valid && !f.mitigated)
        .slice(-5) // show last 5 active FVGs
        .forEach((f: FVG) => {
          const color = f.direction === "bullish" ? "#22c55e40" : "#ef444440";
          priceLinesRef.current.push(candleSeries.createPriceLine({
            price: f.upper_price,
            color,
            lineWidth: 1,
            lineStyle: LineStyle.Dotted,
            axisLabelVisible: false,
            title: "",
          }));
          priceLinesRef.current.push(candleSeries.createPriceLine({
            price: f.lower_price,
            color,
            lineWidth: 1,
            lineStyle: LineStyle.Dotted,
            axisLabelVisible: false,
            title: `FVG ${f.direction === "bullish" ? "▲" : "▼"}`,
          }));
        });
    }

    // Active OB zones as price lines
    if (visibility.ob) {
      data.order_blocks
        .filter((ob: OrderBlock) => ob.valid && !ob.mitigated)
        .slice(-3) // show last 3 active OBs
        .forEach((ob: OrderBlock) => {
          const color = ob.direction === "bullish" ? "#3b82f680" : "#f97316b0";
          priceLinesRef.current.push(candleSeries.createPriceLine({
            price: ob.upper_price,
            color,
            lineWidth: 2,
            lineStyle: LineStyle.Solid,
            axisLabelVisible: false,
            title: "",
          }));
          priceLinesRef.current.push(candleSeries.createPriceLine({
            price: ob.lower_price,
            color,
            lineWidth: 2,
            lineStyle: LineStyle.Solid,
            axisLabelVisible: false,
            title: `OB ${ob.direction === "bullish" ? "▲" : "▼"}`,
          }));
        });
    }

    // Fit content only when data changes (new timeframe), not on visibility toggles
    if (data !== prevDataRef.current) {
      chart.timeScale().fitContent();
      prevDataRef.current = data;
    }
  }, [data, visibility]);

  return (
    <div
      ref={containerRef}
      className="w-full h-full rounded-lg overflow-hidden border border-gray-800"
    />
  );
}
