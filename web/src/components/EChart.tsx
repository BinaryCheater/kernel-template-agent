import { useEffect, useRef } from "react";
import * as echarts from "echarts/core";
import { BarChart, CustomChart, HeatmapChart, LineChart, ScatterChart } from "echarts/charts";
import {
  DataZoomComponent, GridComponent, LegendComponent, MarkAreaComponent, MarkLineComponent, TooltipComponent, VisualMapComponent,
} from "echarts/components";
import { CanvasRenderer } from "echarts/renderers";

echarts.use([BarChart, CustomChart, HeatmapChart, LineChart, ScatterChart, DataZoomComponent, GridComponent, LegendComponent,
  MarkAreaComponent, MarkLineComponent, TooltipComponent, VisualMapComponent, CanvasRenderer]);

export { echarts };
export type EOption = echarts.EChartsCoreOption;

/** Thin ECharts wrapper: owns the instance, re-applies the option, forwards clicks. */
export function EChart({ option, height, onClick, onZoom }: {
  option: EOption;
  height: number;
  onClick?: (p: { data?: unknown; dataIndex: number; seriesIndex?: number; name?: string }) => void;
  onZoom?: (chart: echarts.ECharts) => void;
}) {
  const el = useRef<HTMLDivElement>(null);
  const chart = useRef<echarts.ECharts | null>(null);
  const handlers = useRef({ onClick, onZoom });
  handlers.current = { onClick, onZoom };

  useEffect(() => {
    if (!el.current) return;
    const c = echarts.init(el.current, undefined, { renderer: "canvas" });
    chart.current = c;
    c.on("click", (p) => handlers.current.onClick?.(p as never));
    c.on("datazoom", () => handlers.current.onZoom?.(c));
    const ro = new ResizeObserver(() => c.resize());
    ro.observe(el.current);
    return () => {
      ro.disconnect();
      c.dispose();
      chart.current = null;
    };
  }, []);
  useEffect(() => {
    chart.current?.setOption(option, { notMerge: true });
  }, [option]);
  useEffect(() => {
    chart.current?.resize();
  }, [height]);
  return <div ref={el} style={{ width: "100%", height }} />;
}

/** Shared axis / tooltip styling (recessive grid, text in ink tokens). */
export const AXIS = {
  axisLine: { lineStyle: { color: "#c3c2b7" } },
  axisTick: { lineStyle: { color: "#c3c2b7" } },
  axisLabel: { color: "#52514e", fontSize: 11 },
  splitLine: { lineStyle: { color: "#ecebe5" } },
  nameTextStyle: { color: "#898781", fontSize: 11 },
};
export const TOOLTIP = {
  backgroundColor: "#fcfcfb",
  borderColor: "#e1e0d9",
  textStyle: { color: "#0b0b0b", fontSize: 12 },
  extraCssText: "box-shadow: 0 4px 12px rgba(0,0,0,0.08); border-radius: 6px;",
};
