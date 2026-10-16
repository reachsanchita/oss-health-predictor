import { useEffect, useState } from "react";
import { ComposedChart, Area, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, ReferenceArea, CartesianGrid } from "recharts";
import "./App.css";

const API = "http://localhost:8000";
const DEMO_NAME = "demo/sample-project";
const TONE = { Healthy: "#1F8A5B", Declining: "#C98512", Abandoned: "#C2453D", "Already inactive": "#7B8580" };
const VERDICT = { Healthy: "is staying healthy.", Declining: "is slowing down.", Abandoned: "looks close to abandoned.", "Already inactive": "has already gone quiet." };
const EXAMPLES = ["psf/requests", "pallets/flask", "expressjs/express"];
const LINES = [
  { k: "commits_count", name: "Commits", color: "#2F5D8A" },
  { k: "active_contributors_count", name: "Contributors", color: "#7A4E9E" },
  { k: "prs_merged", name: "PRs merged", color: "#1F8A5B" },
  { k: "issues_closed", name: "Issues closed", color: "#C98512" },
  { k: "releases_count", name: "Releases", color: "#C2453D" },
];
const METRIC = { commits: "Commits", contributors: "Active contributors", prs_opened: "PRs opened", prs_merged: "PRs merged", issues_opened: "Issues opened", issues_closed: "Issues closed", releases: "Releases" };
const STAT = { last: "last month", mean3: "3-month average", mean6: "6-month average", std6: "swings over 6 months", slope6: "6-month trend", ratio3v3: "vs the 3 months before" };
const SPECIAL = {
  commit_zero_months6: "Silent months (of the last 6)", months_since_last_commit: "Months since last commit",
  months_since_last_release: "Months since last release", pr_merge_rate: "Share of PRs merged",
  issue_close_rate: "Share of issues closed", commits_per_contributor: "Commits per contributor",
  commits_cv6: "How uneven commits are", age_months_log: "Project age",
  pr_data_covered: "PR history available", issue_data_covered: "Issue history available",
};
const human = (f) => {
  if (SPECIAL[f]) return SPECIAL[f];
  const m = f.match(/^(.*)_(last|mean3|mean6|std6|slope6|ratio3v3)$/);
  return m ? `${METRIC[m[1]] || m[1]}, ${STAT[m[2]]}` : f;
};
const addMonths = (ym, n) => {
  const [y, m] = ym.split("-").map(Number);
  const d = new Date(y, m - 1 + n, 1);
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}`;
};
const mean = (a) => (a.length ? a.reduce((s, x) => s + (x || 0), 0) / a.length : 0);
const change = (now, before) => ((now - before) / Math.max(before, 1)) * 100;

const DEMO = {
  full_name: DEMO_NAME, status: "ready", prediction: "Declining", model_prediction: "Declining",
  already_inactive: false, as_of_month: "2026-09", horizon: "next 3 months", updated_at: "sample data",
  probabilities: { Healthy: 0.18, Declining: 0.62, Abandoned: 0.2 },
  history: Array.from({ length: 24 }, (_, i) => ({
    year_month: addMonths("2024-10", i),
    commits_count: Math.max(Math.round(60 - i * 1.9 + (i % 3) * 6), 4),
    active_contributors_count: Math.max(12 - Math.floor(i / 3), 2),
    prs_merged: Math.max(Math.round(14 - i * 0.5), 1),
    issues_closed: Math.max(10 - Math.floor(i / 3), 1),
    releases_count: i % 6 === 0 ? 1 : 0,
  })),
  top_factors: [
    { feature: "commits_ratio3v3", value: 0.55, shap: 0.42 }, { feature: "contributors_mean3", value: 3.3, shap: 0.31 },
    { feature: "months_since_last_release", value: 9, shap: 0.22 }, { feature: "pr_merge_rate", value: 0.48, shap: 0.15 },
    { feature: "issues_closed_mean3", value: 6.7, shap: -0.12 }, { feature: "age_months_log", value: 4.6, shap: -0.08 },
  ],
  recommendations: [
    { title: "Grow the contributor base", advice: "Few or falling active contributors. Label good-first-issues, improve CONTRIBUTING.md and thank new contributors.", because: "contributors_mean3", value: 3.3 },
    { title: "Publish a maintenance release", advice: "No release for a long time. Ship a small patch release to signal the project is alive.", because: "months_since_last_release", value: 9 },
    { title: "Clear the pull-request backlog", advice: "Few PRs are being merged. Run a review sprint and set a target response time.", because: "pr_merge_rate", value: 0.48 },
  ],
};

export default function App() {
  const [repos, setRepos] = useState([]);
  const [name, setName] = useState("");
  const [sel, setSel] = useState(null);
  const [d, setD] = useState(null);
  const [info, setInfo] = useState(null);
  const [msg, setMsg] = useState("");
  const [filter, setFilter] = useState("");
  const [range, setRange] = useState(24);
  const [shown, setShown] = useState(["commits_count", "active_contributors_count", "prs_merged"]);
  const [done, setDone] = useState([]);
  const [confirm, setConfirm] = useState(null);
  const [toast, setToast] = useState("");

  const loadRepos = () =>
    fetch(`${API}/repos`).then((r) => r.json()).then(setRepos)
      .catch(() => setMsg("Can't reach the backend. Start it with: uvicorn backend.app.main:app --port 8000"));

  useEffect(() => {
    loadRepos();
    fetch(`${API}/models/info`).then((r) => r.json()).then(setInfo).catch(() => {});
  }, []);

  useEffect(() => {
    const esc = (e) => e.key === "Escape" && setConfirm(null);
    window.addEventListener("keydown", esc);
    return () => window.removeEventListener("keydown", esc);
  }, []);

  useEffect(() => {
    setDone([]);
    if (!sel) return;
    if (sel === DEMO_NAME) { setD(DEMO); return; }
    let t, stop = false;
    const tick = async () => {
      const r = await fetch(`${API}/repos/${sel}`);
      if (!r.ok || stop) return;
      const j = await r.json();
      setD(j);
      if (j.status === "processing") t = setTimeout(tick, 4000); else loadRepos();
    };
    setD(null); tick();
    return () => { stop = true; clearTimeout(t); };
  }, [sel]);

  const flash = (t) => { setToast(t); setTimeout(() => setToast(""), 2800); };

  const check = async (full) => {
    setMsg("");
    const r = await fetch(`${API}/repos`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ full_name: full.trim() }) });
    const j = await r.json();
    if (!r.ok) return setMsg(j.detail || "Something went wrong.");
    setName(""); setSel(j.full_name); loadRepos();
  };

  const recheck = async () => {
    await fetch(`${API}/repos/${sel}/refresh`, { method: "POST" });
    setD({ ...d, status: "processing" });
    setSel(null); setTimeout(() => setSel(d.full_name), 50);
  };

  const remove = async (full) => {
    setConfirm(null);
    try {
      if (full !== DEMO_NAME) {
        const r = await fetch(`${API}/repos/${full}`, { method: "DELETE" });
        if (!r.ok) throw new Error();
      }
      if (sel === full) { setSel(null); setD(null); }
      loadRepos(); flash(`Removed ${full}`);
    } catch { flash("Could not remove it. Is the backend running?"); }
  };

  const clearAll = async () => {
    await Promise.all(repos.map((r) => fetch(`${API}/repos/${r.full_name}`, { method: "DELETE" })));
    setSel(null); setD(null); loadRepos(); flash("All projects removed");
  };

  const count = (p) => repos.filter((r) => r.prediction === p).length;
  const hist = d?.history || [];
  const col = (k) => hist.map((h) => h[k] || 0);
  const risk = d?.probabilities ? Math.round((1 - d.probabilities.Healthy) * 100) : 0;
  const conf = d?.probabilities ? Math.round(Math.max(...Object.values(d.probabilities)) * 100) : 0;
  const cards = d?.status === "ready" ? [
    { label: "Chance of not staying healthy", value: `${risk}%`, note: `Model confidence ${conf}%`, big: true },
    { label: "Commits per month", value: mean(col("commits_count").slice(-3)).toFixed(0), delta: change(mean(col("commits_count").slice(-3)), mean(col("commits_count").slice(-6, -3))) },
    { label: "Active contributors", value: mean(col("active_contributors_count").slice(-3)).toFixed(1), delta: change(mean(col("active_contributors_count").slice(-3)), mean(col("active_contributors_count").slice(-6, -3))) },
    { label: "PRs merged per month", value: mean(col("prs_merged").slice(-3)).toFixed(1), delta: change(mean(col("prs_merged").slice(-3)), mean(col("prs_merged").slice(-6, -3))) },
  ] : [];

  const tone = d && TONE[d.prediction];
  const probs = d?.probabilities ? Object.entries(d.probabilities) : [];
  const chart = d?.history ? [...hist.slice(-range), ...[1, 2, 3].map((n) => ({ year_month: addMonths(d.as_of_month, n) }))] : [];
  const factors = (d?.top_factors || []).slice(0, 8);
  const maxShap = Math.max(...factors.map((f) => Math.abs(f.shap)), 0.001);
  const toward = factors.filter((f) => f.shap > 0);
  const against = factors.filter((f) => f.shap < 0);
  const recs = d?.recommendations || [];
  const list = repos.filter((r) => r.full_name.toLowerCase().includes(filter.toLowerCase()));
  const toggle = (k) => setShown((s) => (s.includes(k) ? s.filter((x) => x !== k) : [...s, k]));

  const Driver = ({ f, c }) => (
    <div className="driver">
      <span>{human(f.feature)} <small>({Number(f.value).toFixed(2)})</small></span>
      <div className="track"><i style={{ width: `${(Math.abs(f.shap) / maxShap) * 100}%`, background: c }} /></div>
    </div>
  );

  return (
    <div className="app">
      <header className="hero">
        <div>
          <h1 className="title"><span className="mark" />Repo Pulse</h1>
          <p className="tagline">Forecast the health of any open-source project, three months ahead.</p>
        </div>
        <div className="stats">
          <div><b>{repos.length}</b><span>Tracked</span></div>
          <div><b style={{ color: TONE.Healthy }}>{count("Healthy")}</b><span>Healthy</span></div>
          <div><b style={{ color: TONE.Declining }}>{count("Declining")}</b><span>Declining</span></div>
          <div><b style={{ color: TONE.Abandoned }}>{count("Abandoned")}</b><span>Abandoned</span></div>
        </div>
      </header>

      <div className="body">
        <aside className="rail">
          <form onSubmit={(e) => { e.preventDefault(); if (name) check(name); }}>
            <label htmlFor="repo">Check a project</label>
            <input id="repo" value={name} onChange={(e) => setName(e.target.value)} placeholder="owner/repo, e.g. psf/requests" />
            <button type="submit">Check project</button>
          </form>
          {msg && <p className="note err">{msg}</p>}

          <div className="listhead">
            <h3>Your projects</h3>
            {repos.length > 0 && <button className="link" onClick={clearAll}>Clear all</button>}
          </div>
          {repos.length > 3 && <input className="search" value={filter} onChange={(e) => setFilter(e.target.value)} placeholder="Filter projects" />}
          {repos.length === 0 && <p className="note">Nothing yet. Add one above.</p>}
          {list.map((r) => (
            <div className={"row" + (sel === r.full_name ? " on" : "")} key={r.full_name}>
              <button className="item" onClick={() => setSel(r.full_name)}>
                <i className="dot" style={{ background: TONE[r.prediction] || "#6b7b73" }} />
                <span>{r.full_name}</span>
                <small>{r.prediction || r.status}</small>
              </button>
              <button className="x" aria-label={`Remove ${r.full_name}`} title="Remove" onClick={() => setConfirm(r.full_name)}>✕</button>
            </div>
          ))}
          {info && (
            <p className="note model">
              <b>{info.model_name}</b><br />
              Macro-F1 on newest months: {info.test_metrics?.macro_f1}<br />
              On unseen projects: {info.grouped_cv_macro_f1?.[0]} ± {info.grouped_cv_macro_f1?.[1]}
            </p>
          )}
        </aside>

        <main className="stage">
          {!sel && (
            <section className="empty">
              <h2>Will this project still be alive in three months?</h2>
              <p>Add any public GitHub repository. Repo Pulse reads its recent activity, forecasts the next three months and tells you what to do about it.</p>
              <div className="chips">
                <button className="demo" onClick={() => setSel(DEMO_NAME)}>See a demo</button>
                {EXAMPLES.map((e) => <button key={e} onClick={() => check(e)}>{e}</button>)}
              </div>
              <ol className="how">
                <li><b>1. Add</b> a repository</li>
                <li><b>2. Read</b> the verdict and the reasons</li>
                <li><b>3. Act</b> on the recommendations</li>
              </ol>
            </section>
          )}
          {sel && !d && <p className="status">Loading…</p>}
          {d?.status === "processing" && <p className="status"><i className="beat" />Collecting activity for {d.full_name}. This takes 1 to 2 minutes.</p>}
          {d?.status === "error" && <section className="empty"><h2>We couldn't analyse {d.full_name}</h2><p>{d.error}</p><div className="chips"><button onClick={() => check(d.full_name)}>Try again</button><button onClick={() => setConfirm(d.full_name)}>Remove it</button></div></section>}

          {d?.status === "ready" && (
            <>
              <section className="verdict">
                <div className="vtop">
                  <h2><span>{d.full_name.split("/")[1]}</span> {VERDICT[d.prediction]}</h2>
                  <div className="actions">
                    <a href={`https://github.com/${d.full_name}`} target="_blank" rel="noreferrer">Open on GitHub</a>
                    {sel !== DEMO_NAME && <button onClick={recheck}>Re-check</button>}
                    <button className="danger" onClick={() => setConfirm(d.full_name)}>Remove project</button>
                  </div>
                </div>
                <p className="meta">{d.full_name} · activity through {d.as_of_month} · outlook: {d.horizon} · updated {String(d.updated_at).slice(0, 10)}
                  {d.already_inactive && " · This project had almost no commits in the last 6 months, and the model only judges projects that were active."}</p>
                <div className="strip" role="img" aria-label="Probability of each outcome">
                  {probs.map(([k, v]) => <i key={k} title={`${k} ${(v * 100).toFixed(0)}%`} style={{ width: `${v * 100}%`, background: TONE[k] }} />)}
                </div>
                <ul className="legend">{probs.map(([k, v]) => <li key={k}><i style={{ background: TONE[k] }} />{k} {(v * 100).toFixed(0)}%</li>)}</ul>
              </section>

              <div className="kpis">
                {cards.map((c) => (
                  <div className={"kpi" + (c.big ? " big" : "")} key={c.label} style={c.big ? { borderTopColor: tone } : {}}>
                    <span>{c.label}</span>
                    <b style={c.big ? { color: tone } : {}}>{c.value}</b>
                    {c.note && <small>{c.note}</small>}
                    {c.delta !== undefined && <small className={c.delta >= 0 ? "up" : "down"}>{c.delta >= 0 ? "▲" : "▼"} {Math.abs(c.delta).toFixed(0)}% vs previous 3 months</small>}
                  </div>
                ))}
              </div>

              <section className="panel">
                <div className="phead">
                  <h3>Activity, and the next three months</h3>
                  <div className="seg">{[6, 12, 24].map((n) => <button key={n} className={range === n ? "on" : ""} onClick={() => setRange(n)}>{n} months</button>)}</div>
                </div>
                <div className="toggles">
                  {LINES.map((l) => (
                    <button key={l.k} className={shown.includes(l.k) ? "on" : ""} onClick={() => toggle(l.k)} style={shown.includes(l.k) ? { borderColor: l.color, color: l.color } : {}}>
                      <i style={{ background: l.color }} />{l.name}
                    </button>
                  ))}
                </div>
                <ResponsiveContainer width="100%" height={380}>
                  <ComposedChart data={chart} margin={{ top: 10, right: 12, left: -8, bottom: 0 }}>
                    <CartesianGrid stroke="#DDE2DA" vertical={false} />
                    <XAxis dataKey="year_month" minTickGap={34} tick={{ fontSize: 14 }} stroke="#9AA59E" />
                    <YAxis tick={{ fontSize: 14 }} stroke="#9AA59E" />
                    <Tooltip contentStyle={{ fontSize: 14, borderRadius: 10 }} />
                    <ReferenceArea x1={addMonths(d.as_of_month, 1)} x2={addMonths(d.as_of_month, 3)} fill={tone} fillOpacity={0.13}
                      label={{ value: "forecast window", position: "insideTop", fontSize: 14, fill: tone }} />
                    {LINES.filter((l) => shown.includes(l.k)).map((l) =>
                      l.k === "commits_count"
                        ? <Area key={l.k} dataKey={l.k} name={l.name} stroke={l.color} fill={l.color} fillOpacity={0.14} strokeWidth={2.5} />
                        : <Line key={l.k} dataKey={l.k} name={l.name} stroke={l.color} dot={false} strokeWidth={2.5} />)}
                  </ComposedChart>
                </ResponsiveContainer>
              </section>

              <div className="two">
                <section className="panel">
                  <h3>What is driving this</h3>
                  {toward.length > 0 && <><h4 style={{ color: tone }}>Pushing toward {d.model_prediction}</h4>{toward.map((f) => <Driver key={f.feature} f={f} c={tone} />)}</>}
                  {against.length > 0 && <><h4>Holding it back</h4>{against.map((f) => <Driver key={f.feature} f={f} c="#9AA59E" />)}</>}
                  <p className="muted">Longer bar = stronger influence. Numbers in brackets are this project's actual values.</p>
                </section>
                <section className="panel">
                  <div className="phead">
                    <h3>What to do about it</h3>
                    {recs.length > 0 && <span className="progress">{done.length} of {recs.length} done</span>}
                  </div>
                  {recs.length > 0 && <div className="bar"><i style={{ width: `${(done.length / recs.length) * 100}%` }} /></div>}
                  {recs.length ? recs.map((r) => (
                    <label className={"rec" + (done.includes(r.title) ? " done" : "")} key={r.title}>
                      <input type="checkbox" checked={done.includes(r.title)} onChange={() => setDone((x) => (x.includes(r.title) ? x.filter((t) => t !== r.title) : [...x, r.title]))} />
                      <div><b>{r.title}</b><p>{r.advice}</p><small>Because: {human(r.because)} is {r.value}</small></div>
                    </label>
                  )) : <p className="muted">Nothing urgent. Keep the current pace.</p>}
                </section>
              </div>
            </>
          )}
        </main>
      </div>

      {confirm && (
        <div className="overlay" onClick={() => setConfirm(null)}>
          <div className="modal" role="dialog" aria-modal="true" onClick={(e) => e.stopPropagation()}>
            <h3>Remove {confirm}?</h3>
            <p>This deletes its saved analysis from your list. You can always check it again later.</p>
            <div className="mbtns">
              <button onClick={() => setConfirm(null)}>Cancel</button>
              <button className="danger solid" onClick={() => remove(confirm)}>Remove</button>
            </div>
          </div>
        </div>
      )}
      {toast && <div className="toast">{toast}</div>}
    </div>
  );
}