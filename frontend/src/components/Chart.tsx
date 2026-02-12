"use client";

import { useEffect, useRef, useState } from "react";
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
import { AnalysisResult, DetectorVisibility, SwingPoint, Inducement, BOS, CHoCH, FVG, OrderBlock } from "@/lib/types";

interface ChartProps {
  data: AnalysisResult | null;
  visibility: DetectorVisibility;
  livePrice?: number | null;
}

/** Convert unix ms timestamp to unix seconds for TradingView. */
function toTV(ts: number) {
  return Math.floor(ts / 1000) as any;
}

// --- OB Box Primitive: draws filled rectangles on the chart canvas ---

interface OBBoxData {
  startTime: any;
  endTime: any;
  upperPrice: number;
  lowerPrice: number;
  fillColor: string;
  borderColor: string;
  label: string;
  labelColor: string;
}

class OBBoxPrimitive {
  _boxes: OBBoxData[] = [];
  _chart: IChartApi | null = null;
  _series: any = null;
  _requestUpdate: (() => void) | null = null;

  setBoxes(boxes: OBBoxData[]) {
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
              const px2 = Math.round(x2 * hpr);
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
              const fontSize = Math.round(11 * vpr);
              ctx.font = `bold ${fontSize}px sans-serif`;
              ctx.fillStyle = box.labelColor;
              ctx.textAlign = "center";
              ctx.textBaseline = "middle";
              const cx = (px1 + px2) / 2;
              const cy = (py1 + py2) / 2;
              ctx.fillText(box.label, cx, cy);
              ctx.textAlign = "start"; // reset
            }
          });
        },
      }),
    }];
  }
}

export default function Chart({ data, visibility, livePrice }: ChartProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const candleSeriesRef = useRef<ISeriesApi<"Candlestick"> | null>(null);
  const priceLinesRef = useRef<any[]>([]);
  const idmSeriesRef = useRef<ISeriesApi<"Line">[]>([]);
  const bosSeriesRef = useRef<ISeriesApi<"Line">[]>([]);
  const chochSeriesRef = useRef<ISeriesApi<"Line">[]>([]);
  const obPrimitiveRef = useRef<OBBoxPrimitive | null>(null);
  const prevCandleCountRef = useRef<number>(0);

  // Swing ↔ IDM click interaction
  const [selectedSwingIdx, setSelectedSwingIdx] = useState<number | null>(null);
  const dataRef = useRef<AnalysisResult | null>(null);
  dataRef.current = data;

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
      timeScale: {
        borderColor: "#1e2230",
        timeVisible: true,
        secondsVisible: false,
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

    // Attach OB box primitive to candle series
    const obPrimitive = new OBBoxPrimitive();
    (candleSeries as any).attachPrimitive(obPrimitive);
    obPrimitiveRef.current = obPrimitive;

    chartRef.current = chart;
    candleSeriesRef.current = candleSeries;

    // Click handler: find nearest swing and highlight its IDM
    chart.subscribeClick((param: MouseEventParams) => {
      const d = dataRef.current;
      if (!param.time || !d || !d.candles.length) {
        setSelectedSwingIdx(null);
        return;
      }

      // Find which candle was clicked by matching timestamp
      const clickedTimeSec = param.time as number;
      const clickedCandle = d.candles.find(
        (c) => Math.floor(c.timestamp / 1000) === clickedTimeSec
      );
      if (!clickedCandle) {
        setSelectedSwingIdx(null);
        return;
      }

      // Get clicked price from y coordinate
      const clickedPrice =
        param.point && candleSeriesRef.current
          ? candleSeriesRef.current.coordinateToPrice(param.point.y)
          : null;

      // Find swings at this candle (could be both a high and low)
      const swingsAtCandle = d.swings.filter(
        (s) =>
          s.candle_index === clickedCandle.index &&
          s.classification !== "unclassified"
      );

      if (swingsAtCandle.length === 0) {
        // Also check 1 candle tolerance for easier clicking
        const nearby = d.swings.filter(
          (s) =>
            Math.abs(s.candle_index - clickedCandle.index) <= 1 &&
            s.classification !== "unclassified"
        );
        if (nearby.length === 0) {
          setSelectedSwingIdx(null);
          return;
        }
        // Pick closest by price
        if (clickedPrice !== null) {
          nearby.sort(
            (a, b) =>
              Math.abs(a.price - clickedPrice) -
              Math.abs(b.price - clickedPrice)
          );
        }
        setSelectedSwingIdx(nearby[0].candle_index);
        return;
      }

      // If multiple swings at same candle, pick closest to clicked price
      if (swingsAtCandle.length > 1 && clickedPrice !== null) {
        swingsAtCandle.sort(
          (a, b) =>
            Math.abs(a.price - clickedPrice) -
            Math.abs(b.price - clickedPrice)
        );
      }
      setSelectedSwingIdx(swingsAtCandle[0].candle_index);
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
      priceLinesRef.current = [];
      idmSeriesRef.current = [];
      bosSeriesRef.current = [];
      chochSeriesRef.current = [];
      obPrimitiveRef.current = null;
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

        // Add "CHoCH" label with directional arrow at the midpoint of the ray
        if (midCandle) {
          const midTime = toTV(midCandle.timestamp);
          if (midTime > startTime && midTime < endTime) {
            const isBull = ch.direction === "bullish";
            const label = ch.is_fake ? "xCHoCH" : "CHoCH";
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

        chochSeriesRef.current.push(lineSeries);
      });
    }

    // --- PRICE LINES: Premium/Discount, FVG ---

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

    // --- OB ZONES: filled rectangular boxes via canvas primitive ---

    if (visibility.ob && obPrimitiveRef.current) {
      const lastIdx = candles.length - 1;
      const boxes: OBBoxData[] = data.order_blocks
        .filter((ob: OrderBlock) => ob.valid)
        .slice(-10)
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

    // Fit content only when candle count changes (new timeframe), not on live refreshes
    if (candles.length !== prevCandleCountRef.current) {
      chart.timeScale().fitContent();
      prevCandleCountRef.current = candles.length;
    }
  }, [data, visibility, selectedSwingIdx]);

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

  return (
    <div
      ref={containerRef}
      className="w-full h-full rounded-xl overflow-hidden border border-[var(--border-primary)] shadow-lg shadow-black/20"
    />
  );
}
