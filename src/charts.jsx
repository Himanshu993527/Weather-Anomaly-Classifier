import { COLORS } from "./util.js";

export function Legend({ items }) {
  return (
    <div className="legend">
      {items.map((n, i) => (
        <span key={n}><i style={{ background: COLORS[i % COLORS.length] }} />{n}</span>
      ))}
    </div>
  );
}

/** Grouped bar chart: categories on the x axis, one bar per series. */
export function GroupedBars({ categories, series, height = 300, yMin, yMax, labelFmt = (v) => (v * 100).toFixed(1) }) {
  const W = 720, m = { l: 48, r: 12, t: 16, b: 40 };
  const pw = W - m.l - m.r, ph = height - m.t - m.b;
  const all = series.flatMap((s) => s.values);
  const lo = yMin ?? Math.max(0, Math.floor((Math.min(...all) - 0.05) * 20) / 20);
  const hi = yMax ?? Math.min(1, Math.ceil((Math.max(...all) + 0.02) * 20) / 20);
  const y = (v) => m.t + ph * (1 - (v - lo) / (hi - lo || 1));
  const ticks = [0, 1, 2, 3, 4].map((i) => lo + ((hi - lo) * i) / 4);
  const gw = pw / categories.length, bw = (gw * 0.8) / series.length;
  const showLabels = categories.length * series.length <= 20;
  return (
    <div>
      <svg viewBox={`0 0 ${W} ${height}`} className="chart" role="img">
        {ticks.map((t, i) => (
          <g key={i}>
            <line x1={m.l} x2={W - m.r} y1={y(t)} y2={y(t)} className="grid" />
            <text x={m.l - 6} y={y(t) + 4} textAnchor="end" className="tick">{(t * 100).toFixed(0)}%</text>
          </g>
        ))}
        {categories.map((c, ci) => (
          <g key={c}>
            {series.map((s, si) => {
              const v = s.values[ci], x = m.l + ci * gw + gw * 0.1 + si * bw;
              return (
                <g key={s.name}>
                  <rect x={x} y={y(v)} width={bw - 2} height={Math.max(0, y(lo) - y(v))} rx="3" fill={COLORS[si % COLORS.length]} />
                  {showLabels && <text x={x + (bw - 2) / 2} y={y(v) - 4} textAnchor="middle" className="vlabel">{labelFmt(v)}</text>}
                </g>
              );
            })}
            <text x={m.l + ci * gw + gw / 2} y={height - 16} textAnchor="middle" className="tick">{c}</text>
          </g>
        ))}
      </svg>
      <Legend items={series.map((s) => s.name)} />
      {lo > 0 && <div className="note">Vertical axis starts at {(lo * 100).toFixed(0)}% so small differences are visible.</div>}
    </div>
  );
}

/** Horizontal bars for a single value per item (e.g. training time). */
export function HBars({ items, format }) {
  const max = Math.max(...items.map((i) => i.value), 1e-9);
  return (
    <div className="hbars">
      {items.map((it, i) => (
        <div className="hrow" key={it.label}>
          <span className="hl">{it.label}</span>
          <div className="htrack"><div className="hbar" style={{ width: `${(it.value / max) * 100}%`, background: COLORS[i % COLORS.length] }} /></div>
          <span className="hv">{format(it.value)}</span>
        </div>
      ))}
    </div>
  );
}

/** Line chart over epochs. series = [{ name, values }] */
export function LineChart({ series, height = 270, format = (v) => v.toFixed(3) }) {
  const W = 720, m = { l: 52, r: 12, t: 14, b: 34 };
  const pw = W - m.l - m.r, ph = height - m.t - m.b;
  const all = series.flatMap((s) => s.values);
  const mn = Math.min(...all), mx = Math.max(...all), pad = (mx - mn) * 0.08 || 0.01;
  const lo = mn - pad, hi = mx + pad, n = Math.max(...series.map((s) => s.values.length));
  const x = (i) => m.l + (n <= 1 ? pw / 2 : (pw * i) / (n - 1));
  const y = (v) => m.t + ph * (1 - (v - lo) / (hi - lo));
  const ticks = [0, 1, 2, 3, 4].map((i) => lo + ((hi - lo) * i) / 4);
  return (
    <div>
      <svg viewBox={`0 0 ${W} ${height}`} className="chart" role="img">
        {ticks.map((t, i) => (
          <g key={i}>
            <line x1={m.l} x2={W - m.r} y1={y(t)} y2={y(t)} className="grid" />
            <text x={m.l - 6} y={y(t) + 4} textAnchor="end" className="tick">{format(t)}</text>
          </g>
        ))}
        {Array.from({ length: n }, (_, i) => (i + 1 === 1 || (i + 1) % Math.ceil(n / 10) === 0) && (
          <text key={i} x={x(i)} y={height - 12} textAnchor="middle" className="tick">{i + 1}</text>
        ))}
        {series.map((s, si) => (
          <g key={s.name}>
            <polyline fill="none" strokeWidth="2.2" stroke={COLORS[si % COLORS.length]} points={s.values.map((v, i) => `${x(i)},${y(v)}`).join(" ")} />
            {n <= 25 && s.values.map((v, i) => <circle key={i} cx={x(i)} cy={y(v)} r="2.6" fill={COLORS[si % COLORS.length]} />)}
          </g>
        ))}
      </svg>
      <Legend items={series.map((s) => s.name)} />
      <div className="note">Horizontal axis: epoch.</div>
    </div>
  );
}
