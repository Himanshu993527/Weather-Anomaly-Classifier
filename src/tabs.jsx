import { useState } from "react";
import { GroupedBars, HBars, LineChart } from "./charts.jsx";
import { f3, num, pct, secs } from "./util.js";

const Card = ({ title, children, className = "" }) => (
  <section className={`card ${className}`}>{title && <h3>{title}</h3>}{children}</section>
);
const Stat = ({ label, value, sub }) => (
  <div className="stat"><div className="sv">{value}</div><div className="sl">{label}</div>{sub && <div className="ss">{sub}</div>}</div>
);

/* ------------------------------------------------------------------ Overview */
export function Overview({ r }) {
  const d = r.dataset, cc = d.class_counts;
  const share = (arr) => arr.map((v) => v / arr.reduce((a, b) => a + b, 0));
  const best = r.models.find((m) => m.name === r.best_model);
  return (
    <>
      <div className="stats">
        <Stat label="Raw records" value={num(d.rows_raw)} />
        <Stat label="After cleaning" value={num(d.rows_clean)} sub={`${num(d.rows_removed)} invalid rows removed`} />
        <Stat label="Weather stations" value={num(d.stations)} />
        <Stat label="Period" value={`${d.date_min.slice(0, 4)}–${d.date_max.slice(0, 4)}`} />
      </div>
      {best && (
        <Card title="Best model on the test set">
          <p className="big">{best.name}: macro F1 {f3(best.f1_macro)}, accuracy {pct(best.accuracy)}, trained in {secs(best.training_time_s)}.</p>
          <p className="muted">Best is chosen by macro F1, which treats the four classes equally, so the rare “Strong” classes are not hidden by the large “Weak Hot” class.</p>
        </Card>
      )}
      <div className="two">
        <Card title="Chronological split (no shuffling across time)">
          <table className="tbl">
            <thead><tr><th>Split</th><th>Years</th><th>Rows available</th><th>Rows used</th></tr></thead>
            <tbody>
              {["train", "val", "test"].map((k) => (
                <tr key={k}><td>{k === "val" ? "Validation" : k[0].toUpperCase() + k.slice(1)}</td><td>{d.split[k].years}</td>
                  <td>{num(d.split[k].rows_total)}</td><td>{num(d.split[k].rows_used)}</td></tr>
              ))}
            </tbody>
          </table>
          <p className="muted">Features: {d.features.join(", ")}.</p>
        </Card>
        <Card title="Class distribution (share of rows)">
          <GroupedBars categories={r.classes} yMin={0}
            series={[{ name: "Training (used)", values: share(cc.train) }, { name: "Test", values: share(cc.test) }]} height={250} />
        </Card>
      </div>
      <Card title="Method notes">
        <ul className="notes">
          <li>The label (Weak/Strong Hot/Cold) is derived from <code>degrees_from_mean</code>, so that column is <b>not</b> used as an input. Using it would leak the answer.</li>
          <li>Because the data are imbalanced, models use sqrt-balanced class weights and are compared with macro-averaged precision, recall and F1 as well as accuracy.</li>
          <li>Each station has a learned embedding, so a model can learn station-specific temperature behaviour.</li>
          <li>Training time is the wall-clock time of <code>model.fit</code> on the computer that ran <code>train.py</code>.</li>
        </ul>
        <table className="tbl small">
          <tbody>{Object.entries(r.config).map(([k, v]) => <tr key={k}><td>{k.replace(/_/g, " ")}</td><td>{String(v)}</td></tr>)}</tbody>
        </table>
      </Card>
    </>
  );
}

/* ------------------------------------------------------------------ Compare */
export function Compare({ r }) {
  const [avg, setAvg] = useState("macro");
  const ms = r.models;
  const cols = [
    ["Accuracy", (m) => m.accuracy, true, f3],
    ["Precision", (m) => m[`precision_${avg}`], true, f3],
    ["Recall", (m) => m[`recall_${avg}`], true, f3],
    ["F1-score", (m) => m[`f1_${avg}`], true, f3],
    ["Training time", (m) => m.training_time_s, false, secs],
    ["Epochs", (m) => m.epochs_run, false, (v) => v],
    ["Parameters", (m) => m.params, false, num],
    ["Inference (ms / 1000)", (m) => m.inference_ms_per_1000, false, (v) => v],
  ];
  const bestIdx = cols.map(([, get, higher]) => {
    const vals = ms.map(get), t = higher ? Math.max(...vals) : Math.min(...vals);
    return vals.indexOf(t);
  });
  const fastest = ms.reduce((a, b) => (a.training_time_s <= b.training_time_s ? a : b));
  return (
    <>
      <Card title="Model comparison (test set)">
        <div className="toggle">
          Averaging for precision / recall / F1:
          {["macro", "weighted"].map((a) => (
            <button key={a} className={avg === a ? "on" : ""} onClick={() => setAvg(a)}>{a}</button>
          ))}
        </div>
        <div className="scroll">
          <table className="tbl">
            <thead><tr><th>Model</th>{cols.map(([n]) => <th key={n}>{n}</th>)}</tr></thead>
            <tbody>
              {ms.map((m, i) => (
                <tr key={m.key}>
                  <td><b>{m.name}</b>{m.name === r.best_model && <span className="badge">best F1</span>}</td>
                  {cols.map(([n, get, , fmt], c) => <td key={n} className={bestIdx[c] === i ? "best" : ""}>{fmt(get(m))}</td>)}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="muted">Green = best in the column. Macro averaging gives each class equal weight; weighted averaging weights classes by how often they occur.</p>
      </Card>
      <div className="two">
        <Card title="Accuracy, precision, recall, F1">
          <GroupedBars categories={ms.map((m) => m.name)}
            series={[{ name: "Accuracy", values: ms.map((m) => m.accuracy) }, { name: "Precision", values: ms.map((m) => m[`precision_${avg}`]) },
              { name: "Recall", values: ms.map((m) => m[`recall_${avg}`]) }, { name: "F1-score", values: ms.map((m) => m[`f1_${avg}`]) }]} />
        </Card>
        <Card title="Training time">
          <HBars items={ms.map((m) => ({ label: m.name, value: m.training_time_s }))} format={secs} />
          <p className="muted">Fastest to train: <b>{fastest.name}</b>. A slower model is only worth it if its F1 gain justifies the extra time.</p>
        </Card>
      </div>
    </>
  );
}

/* ------------------------------------------------------------------ Confusion matrix */
export function ConfusionTab({ r }) {
  const [key, setKey] = useState(r.models[0].key);
  const [mode, setMode] = useState("norm");
  const m = r.models.find((x) => x.key === key) || r.models[0];
  const cm = m.confusion_matrix, nm = m.confusion_matrix_norm, C = r.classes;
  let worst = { v: -1, i: 0, j: 0 };
  nm.forEach((row, i) => row.forEach((v, j) => { if (i !== j && v > worst.v) worst = { v, i, j }; }));
  return (
    <>
      <Card title="Confusion matrix">
        <div className="toggle">
          Model:
          {r.models.map((x) => <button key={x.key} className={x.key === key ? "on" : ""} onClick={() => setKey(x.key)}>{x.name}</button>)}
        </div>
        <div className="toggle">
          Show:
          <button className={mode === "norm" ? "on" : ""} onClick={() => setMode("norm")}>% of actual class</button>
          <button className={mode === "count" ? "on" : ""} onClick={() => setMode("count")}>counts</button>
        </div>
        <div className="scroll">
          <table className="cm">
            <thead>
              <tr><th /><th colSpan={C.length} className="axis">Predicted</th><th /></tr>
              <tr><th className="axis">Actual</th>{C.map((c) => <th key={c}>{c}</th>)}<th>Support</th></tr>
            </thead>
            <tbody>
              {C.map((c, i) => (
                <tr key={c}>
                  <th>{c}</th>
                  {C.map((_, j) => (
                    <td key={j} className={i === j ? "diag" : ""} style={{ background: `rgba(59,130,246,${0.06 + 0.86 * nm[i][j]})`, color: nm[i][j] > 0.45 ? "#fff" : "inherit" }}>
                      <div className="cv">{mode === "norm" ? pct(nm[i][j]) : num(cm[i][j])}</div>
                      <div className="cs">{mode === "norm" ? num(cm[i][j]) : pct(nm[i][j])}</div>
                    </td>
                  ))}
                  <td className="sup">{num(cm[i].reduce((a, b) => a + b, 0))}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="muted">Rows are the true class, columns the prediction; the diagonal is correct. Most common mistake: actual <b>{C[worst.i]}</b> predicted as <b>{C[worst.j]}</b> ({pct(worst.v)} of that class).</p>
      </Card>
      <Card title={`Per-class metrics: ${m.name}`}>
        <table className="tbl">
          <thead><tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1-score</th><th>Support</th></tr></thead>
          <tbody>
            {m.per_class.map((c) => <tr key={c.class}><td>{c.class}</td><td>{f3(c.precision)}</td><td>{f3(c.recall)}</td><td>{f3(c.f1)}</td><td>{num(c.support)}</td></tr>)}
            <tr className="sum"><td>Macro average</td><td>{f3(m.precision_macro)}</td><td>{f3(m.recall_macro)}</td><td>{f3(m.f1_macro)}</td><td>{num(cm.flat().reduce((a, b) => a + b, 0))}</td></tr>
            <tr className="sum"><td>Weighted average</td><td>{f3(m.precision_weighted)}</td><td>{f3(m.recall_weighted)}</td><td>{f3(m.f1_weighted)}</td><td /></tr>
          </tbody>
        </table>
      </Card>
    </>
  );
}

/* ------------------------------------------------------------------ Training curves */
export function Curves({ r }) {
  const [metric, setMetric] = useState("loss");
  const [phase, setPhase] = useState("val_");
  const name = (metric === "loss" ? "loss" : "accuracy");
  return (
    <Card title="Training behaviour">
      <div className="toggle">
        Metric:
        {["loss", "accuracy"].map((x) => <button key={x} className={metric === x ? "on" : ""} onClick={() => setMetric(x)}>{x}</button>)}
        Data:
        <button className={phase === "val_" ? "on" : ""} onClick={() => setPhase("val_")}>validation</button>
        <button className={phase === "" ? "on" : ""} onClick={() => setPhase("")}>training</button>
      </div>
      <LineChart series={r.models.map((m) => ({ name: m.name, values: m.history[`${phase}${name}`] || [] }))} />
      <p className="muted">Early stopping keeps the weights from the best validation epoch. A validation curve that rises while the training curve falls signals overfitting.</p>
      <table className="tbl small">
        <thead><tr><th>Model</th><th>Epochs run</th><th>Best epoch</th><th>Training time</th></tr></thead>
        <tbody>{r.models.map((m) => <tr key={m.key}><td>{m.name}</td><td>{m.epochs_run}</td><td>{m.best_epoch}</td><td>{secs(m.training_time_s)}</td></tr>)}</tbody>
      </table>
    </Card>
  );
}

/* ------------------------------------------------------------------ Predict */
export function Predict({ r }) {
  const [api, setApi] = useState("http://localhost:5000");
  const [f, setF] = useState({ max_temp: "36", min_temp: "24", latitude: "40.78", longitude: "-73.97", date: "2012-07-15", station_id: "" });
  const [model, setModel] = useState(r.models.find((m) => m.name === r.best_model)?.key || r.models[0].key);
  const [out, setOut] = useState(null);
  const [err, setErr] = useState("");
  const [busy, setBusy] = useState(false);
  const set = (k) => (e) => setF({ ...f, [k]: e.target.value });
  async function go() {
    setBusy(true); setErr(""); setOut(null);
    try {
      const res = await fetch(`${api}/predict`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ...f, max_temp: +f.max_temp, min_temp: +f.min_temp, latitude: +f.latitude, longitude: +f.longitude, model }),
      });
      const j = await res.json();
      if (!res.ok) throw new Error(j.error || "Request failed");
      setOut(j);
    } catch (e) {
      setErr(e instanceof TypeError ? "Could not reach the API. Start it in a second terminal with: python api.py" : e.message);
    } finally { setBusy(false); }
  }
  return (
    <Card title="Classify a day's temperature">
      <div className="form">
        <label>Max temp (°C)<input value={f.max_temp} onChange={set("max_temp")} type="number" /></label>
        <label>Min temp (°C)<input value={f.min_temp} onChange={set("min_temp")} type="number" /></label>
        <label>Latitude<input value={f.latitude} onChange={set("latitude")} type="number" /></label>
        <label>Longitude<input value={f.longitude} onChange={set("longitude")} type="number" /></label>
        <label>Date<input value={f.date} onChange={set("date")} type="date" /></label>
        <label>Station ID (optional)<input value={f.station_id} onChange={set("station_id")} placeholder="e.g. USC00103882" /></label>
        <label>Model<select value={model} onChange={(e) => setModel(e.target.value)}>{r.models.map((m) => <option key={m.key} value={m.key}>{m.name}</option>)}</select></label>
        <label>API address<input value={api} onChange={(e) => setApi(e.target.value)} /></label>
      </div>
      <button className="primary" onClick={go} disabled={busy}>{busy ? "Predicting…" : "Predict"}</button>
      {err && <div className="error">{err}</div>}
      {out && (
        <div className="result">
          <div className="big">{out.predicted} <span className="muted">({pct(out.confidence)} confidence)</span></div>
          <HBars items={Object.entries(out.probabilities).map(([label, value]) => ({ label, value }))} format={(v) => pct(v)} />
          {!out.station_known && <p className="muted">Station not provided or unknown, so the station embedding was not used.</p>}
        </div>
      )}
      <p className="muted">Needs the API running (<code>python api.py</code>) and trained models. Example inputs only; this is a classroom prototype.</p>
    </Card>
  );
}
