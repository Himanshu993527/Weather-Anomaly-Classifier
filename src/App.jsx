import { useEffect, useState } from "react";
import { Compare, ConfusionTab, Curves, Overview, Predict } from "./tabs.jsx";

const TABS = [["overview", "Overview"], ["compare", "Model comparison"], ["confusion", "Confusion matrix"], ["curves", "Training curves"], ["predict", "Predict"]];
const valid = (j) => j && Array.isArray(j.models) && j.models.length > 0 && j.classes && j.dataset;

export default function App({ initialResults = null, initialTab = "overview" }) {
  const [r, setR] = useState(initialResults);
  const [tab, setTab] = useState(initialTab);
  const [msg, setMsg] = useState("");

  useEffect(() => {
    if (initialResults) return;
    fetch("results.json").then((x) => (x.ok ? x.json() : Promise.reject())).then((j) => valid(j) && setR(j)).catch(() => {});
  }, []);

  function upload(e) {
    const file = e.target.files[0];
    if (!file) return;
    const rd = new FileReader();
    rd.onload = () => {
      try {
        const j = JSON.parse(rd.result);
        if (!valid(j)) throw new Error("not a results file");
        setR(j); setMsg("");
      } catch { setMsg("That file is not a valid results.json from train.py."); }
    };
    rd.readAsText(file);
  }

  return (
    <div className="app">
      <header>
        <div>
          <h1>🌡️ Weather Anomaly Classifier</h1>
          <p className="muted">Deep-learning comparison on US station records, 1964–2013: Hot / Cold × Weak / Strong</p>
        </div>
        <label className="file">Load results.json<input type="file" accept=".json,application/json" onChange={upload} /></label>
      </header>
      {msg && <div className="error">{msg}</div>}
      {!r ? (
        <section className="card empty">
          <h3>No results loaded yet</h3>
          <p>Train the models first, then reload this page (or click “Load results.json”):</p>
          <pre>python train.py</pre>
          <p className="muted">The training script writes <code>frontend/public/results.json</code> and <code>backend/artifacts/results.json</code>.</p>
        </section>
      ) : (
        <>
          <nav>{TABS.map(([k, n]) => <button key={k} className={tab === k ? "on" : ""} onClick={() => setTab(k)}>{n}</button>)}</nav>
          <main>
            {tab === "overview" && <Overview r={r} />}
            {tab === "compare" && <Compare r={r} />}
            {tab === "confusion" && <ConfusionTab r={r} />}
            {tab === "curves" && <Curves r={r} />}
            {tab === "predict" && <Predict r={r} />}
          </main>
          <footer>Generated {r.generated_at}. Metrics are measured on the held-out test years ({r.dataset.split.test.years}). Research / classroom prototype.</footer>
        </>
      )}
    </div>
  );
}
