export const COLORS = ["#60a5fa", "#34d399", "#fbbf24", "#f472b6", "#a78bfa"];
export const pct = (v, d = 1) => `${(v * 100).toFixed(d)}%`;
export const f3 = (v) => Number(v).toFixed(3);
export const num = (v) => Number(v).toLocaleString("en-US");
export const secs = (s) => (s >= 60 ? `${Math.floor(s / 60)}m ${Math.round(s % 60)}s` : `${Number(s).toFixed(1)}s`);
