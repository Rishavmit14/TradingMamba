"use client";

import { useEffect, useRef } from "react";
import {
  createChart,
  IChartApi,
  ISeriesApi,
  CandlestickData,
  ColorType,
  LineStyle,
} from "lightweight-charts";
import { AnalysisResult, SwingPoint, BOS, CHoCH, FVG, OrderBlock } from "@/lib/types";

interface ChartProps {
  data: AnalysisResult | null;
  height?: number;
}

/** Convert unix ms timestamp to unix seconds for TradingView. */
function toTV(ts: number) {
  return Math.floor(ts / 1000) as any;
}

export default function Chart({ data, height = 600 }: ChartProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const candleSeriesRef = useRef<ISeriesApi<"Candlestick"> | null>(null);

  // Create chart once
  useEffect(() => {
    if (!containerRef.current) return;

    const chart = createChart(containerRef.current, {
      layout: {
        background: { type: ColorType.Solid, color: "#0a0a0f" },
        textColor: "#9ca3af",
      },
      grid: {
        vertLines: { color: "#1f2937" },
        horzLines: { color: "#1f2937" },
      },
      crosshair: {
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
      height,
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

    // Resize handler
    const handleResize = () => {
      if (containerRef.current) {
        chart.applyOptions({ width: containerRef.current.clientWidth });
      }
    };
    window.addEventListener("resize", handleResize);

    return () => {
      window.removeEventListener("resize", handleResize);
      chart.remove();
      chartRef.current = null;
      candleSeriesRef.current = null;
    };
  }, [height]);

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

    // BOS markers
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

    // CHoCH markers
    data.choch_events.forEach((ch: CHoCH) => {
      const candle = candles[ch.candle_index];
      if (!candle) return;

      let color = "#a855f7"; // purple
      if (ch.is_fake) color = "#6b7280";
      else if (ch.confirmed) color = "#ec4899"; // pink

      markers.push({
        time: toTV(candle.timestamp),
        position: ch.direction === "bullish" ? "belowBar" : "aboveBar",
        color,
        shape: "square",
        text: ch.is_fake ? "xCHoCH" : "CHoCH",
      });
    });

    // Sort markers by time (required by lightweight-charts)
    markers.sort((a, b) => (a.time as number) - (b.time as number));
    candleSeries.setMarkers(markers);

    // --- PRICE LINES: Premium/Discount, FVG, OB ---

    // Clear old line series by removing and re-adding
    // For zones (FVG, OB), we use price lines on the candle series

    // Premium/Discount equilibrium line
    if (data.premium_discount) {
      candleSeries.createPriceLine({
        price: data.premium_discount.equilibrium,
        color: "#eab308",
        lineWidth: 1,
        lineStyle: LineStyle.Dashed,
        axisLabelVisible: true,
        title: "EQ 50%",
      });
      candleSeries.createPriceLine({
        price: data.premium_discount.swing_high,
        color: "#ef444480",
        lineWidth: 1,
        lineStyle: LineStyle.Dotted,
        axisLabelVisible: false,
        title: "Premium",
      });
      candleSeries.createPriceLine({
        price: data.premium_discount.swing_low,
        color: "#22c55e80",
        lineWidth: 1,
        lineStyle: LineStyle.Dotted,
        axisLabelVisible: false,
        title: "Discount",
      });
    }

    // Active FVG zones as price lines
    data.fvgs
      .filter((f: FVG) => f.valid && !f.mitigated)
      .slice(-5) // show last 5 active FVGs
      .forEach((f: FVG) => {
        const color = f.direction === "bullish" ? "#22c55e40" : "#ef444440";
        candleSeries.createPriceLine({
          price: f.upper_price,
          color,
          lineWidth: 1,
          lineStyle: LineStyle.Dotted,
          axisLabelVisible: false,
          title: "",
        });
        candleSeries.createPriceLine({
          price: f.lower_price,
          color,
          lineWidth: 1,
          lineStyle: LineStyle.Dotted,
          axisLabelVisible: false,
          title: `FVG ${f.direction === "bullish" ? "▲" : "▼"}`,
        });
      });

    // Active OB zones as price lines
    data.order_blocks
      .filter((ob: OrderBlock) => ob.valid && !ob.mitigated)
      .slice(-3) // show last 3 active OBs
      .forEach((ob: OrderBlock) => {
        const color = ob.direction === "bullish" ? "#3b82f680" : "#f97316b0";
        candleSeries.createPriceLine({
          price: ob.upper_price,
          color,
          lineWidth: 2,
          lineStyle: LineStyle.Solid,
          axisLabelVisible: false,
          title: "",
        });
        candleSeries.createPriceLine({
          price: ob.lower_price,
          color,
          lineWidth: 2,
          lineStyle: LineStyle.Solid,
          axisLabelVisible: false,
          title: `OB ${ob.direction === "bullish" ? "▲" : "▼"}`,
        });
      });

    // Fit content
    chart.timeScale().fitContent();
  }, [data]);

  return (
    <div
      ref={containerRef}
      className="w-full rounded-lg overflow-hidden border border-gray-800"
    />
  );
}
