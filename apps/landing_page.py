"""Product pages for the stateful.ai demonstration app.

Two self-contained pages, no build step:

- ``LANDING_HTML``  - marketing/product page served at ``/``
- ``DEMO_HTML``     - live operations console served at ``/demo``

Both share a glassmorphism design system: deep layered background,
frosted translucent panels, restrained accent gradient, and a minimal,
professional type scale.
"""

# ---------------------------------------------------------------------------
# Shared design system (inlined into both pages)
# ---------------------------------------------------------------------------

_BASE_CSS = r"""
  :root {
    color-scheme: dark;
    --bg0: #05060c;
    --bg1: #0a0d18;
    --ink: #eef1f8;
    --muted: #949cb0;
    --faint: #5d6478;
    --glass: rgba(255, 255, 255, 0.045);
    --glass-2: rgba(255, 255, 255, 0.075);
    --stroke: rgba(255, 255, 255, 0.10);
    --stroke-2: rgba(255, 255, 255, 0.16);
    --cyan: #38d4f5;
    --indigo: #7c8cf8;
    --violet: #a78bfa;
    --green: #4ade80;
    --red: #f87171;
    --amber: #fbbf24;
    --accent-grad: linear-gradient(135deg, var(--cyan), var(--indigo) 60%, var(--violet));
    --radius: 18px;
    --radius-sm: 12px;
    --sans: "Inter", ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
    --mono: "JetBrains Mono", "SFMono-Regular", ui-monospace, Menlo, Consolas, monospace;
    --shadow: 0 24px 60px -24px rgba(0, 0, 0, 0.65);
  }
  * { box-sizing: border-box; }
  html { scroll-behavior: smooth; }
  body {
    margin: 0;
    min-height: 100vh;
    font-family: var(--sans);
    color: var(--ink);
    background: var(--bg0);
    -webkit-font-smoothing: antialiased;
    overflow-x: hidden;
  }
  /* Layered ambient background */
  .ambient {
    position: fixed; inset: 0; z-index: -2; pointer-events: none;
    background:
      radial-gradient(42rem 30rem at 12% -8%, rgba(124, 140, 248, 0.16), transparent 60%),
      radial-gradient(38rem 26rem at 88% 4%, rgba(56, 212, 245, 0.11), transparent 60%),
      radial-gradient(50rem 36rem at 50% 110%, rgba(167, 139, 250, 0.10), transparent 65%),
      linear-gradient(180deg, var(--bg1), var(--bg0) 42%);
  }
  .ambient::after {
    content: ""; position: absolute; inset: 0;
    background-image:
      linear-gradient(rgba(255,255,255,0.025) 1px, transparent 1px),
      linear-gradient(90deg, rgba(255,255,255,0.025) 1px, transparent 1px);
    background-size: 56px 56px;
    mask-image: radial-gradient(70rem 42rem at 50% 0%, rgba(0,0,0,0.9), transparent 75%);
    -webkit-mask-image: radial-gradient(70rem 42rem at 50% 0%, rgba(0,0,0,0.9), transparent 75%);
  }
  /* Glass primitives */
  .glass {
    background: var(--glass);
    border: 1px solid var(--stroke);
    border-radius: var(--radius);
    backdrop-filter: blur(22px) saturate(160%);
    -webkit-backdrop-filter: blur(22px) saturate(160%);
    box-shadow: var(--shadow), inset 0 1px 0 rgba(255, 255, 255, 0.07);
  }
  .grad-text {
    background: var(--accent-grad);
    -webkit-background-clip: text; background-clip: text;
    -webkit-text-fill-color: transparent; color: transparent;
  }
  .btn {
    display: inline-flex; align-items: center; gap: 0.5rem;
    padding: 0.72rem 1.35rem; border-radius: 999px;
    font: 600 0.92rem var(--sans); letter-spacing: 0.01em;
    text-decoration: none; cursor: pointer; border: 1px solid transparent;
    transition: transform 0.16s ease, box-shadow 0.16s ease, background 0.16s ease;
    user-select: none;
  }
  .btn:active { transform: translateY(1px) scale(0.99); }
  .btn-primary {
    color: #061018; background: var(--accent-grad);
    box-shadow: 0 10px 28px -10px rgba(86, 140, 245, 0.55);
  }
  .btn-primary:hover { box-shadow: 0 14px 36px -10px rgba(86, 140, 245, 0.75); transform: translateY(-1px); }
  .btn-ghost {
    color: var(--ink); background: var(--glass-2); border-color: var(--stroke-2);
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
  }
  .btn-ghost:hover { background: rgba(255,255,255,0.11); }
  .pill {
    display: inline-flex; align-items: center; gap: 0.45rem;
    padding: 0.32rem 0.85rem; border-radius: 999px;
    font: 500 0.78rem var(--sans); color: var(--muted);
    background: var(--glass); border: 1px solid var(--stroke);
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
  }
  .dot { width: 7px; height: 7px; border-radius: 50%; background: var(--green); box-shadow: 0 0 10px var(--green); }
  ::selection { background: rgba(124, 140, 248, 0.35); }
  @media (prefers-reduced-motion: reduce) {
    * { animation: none !important; transition: none !important; }
  }
"""

# ---------------------------------------------------------------------------
# Landing page
# ---------------------------------------------------------------------------

LANDING_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>stateful.ai — Memory infrastructure for AI agents</title>
<meta name="description" content="stateful.ai is a production-grade persistent memory layer for long-running AI agents: hybrid retrieval, versioned lifecycle, contradiction detection, and an MCP server." />
<style>
__BASE_CSS__
  /* ---- nav ---- */
  nav {
    position: fixed; top: 14px; left: 50%; transform: translateX(-50%);
    width: min(1080px, calc(100% - 28px)); z-index: 50;
    display: flex; align-items: center; justify-content: space-between;
    padding: 0.65rem 1.1rem; border-radius: 999px;
  }
  .brand { display: flex; align-items: center; gap: 0.6rem; text-decoration: none; color: var(--ink); }
  .brand b { font-size: 1.02rem; font-weight: 700; letter-spacing: -0.01em; }
  .mark {
    width: 30px; height: 30px; border-radius: 9px; display: grid; place-items: center;
    background: var(--accent-grad); box-shadow: 0 6px 18px -6px rgba(86,140,245,0.7);
  }
  .navlinks { display: flex; gap: 1.6rem; }
  .navlinks a { color: var(--muted); text-decoration: none; font: 500 0.88rem var(--sans); transition: color .15s; }
  .navlinks a:hover { color: var(--ink); }
  @media (max-width: 760px) { .navlinks { display: none; } }

  main { width: min(1080px, calc(100% - 40px)); margin: 0 auto; }

  /* ---- hero ---- */
  .hero { padding: 9.5rem 0 4.5rem; display: grid; grid-template-columns: 1.05fr 0.95fr; gap: 3rem; align-items: center; }
  @media (max-width: 900px) { .hero { grid-template-columns: 1fr; padding-top: 8rem; } }
  .hero h1 {
    margin: 1.1rem 0 1rem; font-size: clamp(2.3rem, 4.6vw, 3.4rem);
    line-height: 1.07; letter-spacing: -0.035em; font-weight: 750;
  }
  .hero p.lead { color: var(--muted); font-size: 1.08rem; line-height: 1.65; max-width: 34rem; margin: 0 0 1.8rem; }
  .cta-row { display: flex; gap: 0.8rem; flex-wrap: wrap; align-items: center; }
  .hint { color: var(--faint); font: 400 0.8rem var(--mono); margin-top: 1.2rem; }

  /* terminal card */
  .term { padding: 0; overflow: hidden; font: 400 0.82rem/1.65 var(--mono); }
  .term-bar {
    display: flex; align-items: center; gap: 0.45rem; padding: 0.7rem 1rem;
    border-bottom: 1px solid var(--stroke); color: var(--faint); font-size: 0.75rem;
  }
  .term-bar i { width: 10px; height: 10px; border-radius: 50%; display: inline-block; opacity: 0.85; }
  .term-body { padding: 1.1rem 1.2rem 1.3rem; overflow-x: auto; white-space: pre; }
  .c-dim { color: var(--faint); } .c-cmd { color: var(--ink); }
  .c-key { color: var(--cyan); } .c-str { color: #b8f0a8; } .c-num { color: var(--amber); }

  /* metric strip */
  .metrics { display: grid; grid-template-columns: repeat(4, 1fr); gap: 0.9rem; margin: 0 0 5rem; }
  @media (max-width: 820px) { .metrics { grid-template-columns: repeat(2, 1fr); } }
  .metric { padding: 1.15rem 1.3rem; }
  .metric b { display: block; font-size: 1.35rem; letter-spacing: -0.02em; }
  .metric span { color: var(--muted); font-size: 0.82rem; }

  /* sections */
  section { padding: 3.4rem 0; }
  .sec-head { max-width: 38rem; margin-bottom: 2.4rem; }
  .eyebrow { font: 600 0.74rem var(--mono); text-transform: uppercase; letter-spacing: 0.16em; color: var(--cyan); }
  .sec-head h2 { margin: 0.7rem 0 0.7rem; font-size: clamp(1.6rem, 3vw, 2.2rem); letter-spacing: -0.025em; }
  .sec-head p { color: var(--muted); line-height: 1.65; margin: 0; }

  .grid3 { display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; }
  @media (max-width: 900px) { .grid3 { grid-template-columns: 1fr 1fr; } }
  @media (max-width: 620px) { .grid3 { grid-template-columns: 1fr; } }
  .card { padding: 1.5rem 1.45rem; transition: transform .2s ease, border-color .2s ease; }
  .card:hover { transform: translateY(-3px); border-color: var(--stroke-2); }
  .card .ico {
    width: 38px; height: 38px; border-radius: 11px; display: grid; place-items: center;
    background: var(--glass-2); border: 1px solid var(--stroke); margin-bottom: 1rem;
  }
  .card h3 { margin: 0 0 0.5rem; font-size: 1.02rem; letter-spacing: -0.01em; }
  .card p { margin: 0; color: var(--muted); font-size: 0.89rem; line-height: 1.6; }

  /* architecture */
  .arch { display: flex; flex-direction: column; gap: 0.7rem; }
  .layer { display: grid; grid-template-columns: 150px 1fr; gap: 1.2rem; align-items: center; padding: 1.05rem 1.35rem; }
  @media (max-width: 620px) { .layer { grid-template-columns: 1fr; gap: 0.3rem; } }
  .layer .tag { font: 600 0.78rem var(--mono); color: var(--cyan); text-transform: uppercase; letter-spacing: 0.1em; }
  .layer p { margin: 0; color: var(--muted); font-size: 0.88rem; line-height: 1.55; }
  .layer code { font: 500 0.8rem var(--mono); color: var(--ink); }
  .arrow { text-align: center; color: var(--faint); font-size: 0.85rem; line-height: 0.5; }

  /* api table */
  .api-table { width: 100%; border-collapse: collapse; overflow: hidden; }
  .api-wrap { padding: 0.4rem 0; overflow-x: auto; }
  .api-table th, .api-table td { text-align: left; padding: 0.78rem 1.2rem; font-size: 0.85rem; }
  .api-table th { color: var(--faint); font: 600 0.72rem var(--mono); text-transform: uppercase; letter-spacing: 0.12em; border-bottom: 1px solid var(--stroke); }
  .api-table tr + tr td { border-top: 1px solid rgba(255,255,255,0.05); }
  .api-table td:first-child { font: 600 0.78rem var(--mono); white-space: nowrap; }
  .api-table td:nth-child(2) { font: 400 0.82rem var(--mono); color: var(--ink); white-space: nowrap; }
  .api-table td:last-child { color: var(--muted); }
  .m-get { color: var(--green); } .m-post { color: var(--cyan); } .m-patch { color: var(--amber); } .m-del { color: var(--red); }

  /* CTA + footer */
  .cta-final { margin: 2rem 0 0; padding: 3.2rem 2rem; text-align: center; }
  .cta-final h2 { margin: 0 0 0.6rem; font-size: clamp(1.6rem, 3vw, 2.1rem); letter-spacing: -0.025em; }
  .cta-final p { color: var(--muted); margin: 0 0 1.6rem; }
  footer {
    margin: 4rem auto 2rem; width: min(1080px, calc(100% - 40px));
    display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 1rem;
    color: var(--faint); font-size: 0.82rem; padding-top: 1.6rem; border-top: 1px solid var(--stroke);
  }
  footer a { color: var(--muted); text-decoration: none; }
  footer a:hover { color: var(--ink); }

  /* reveal animation */
  .reveal { opacity: 0; transform: translateY(18px); transition: opacity .6s ease, transform .6s ease; }
  .reveal.in { opacity: 1; transform: none; }
</style>
</head>
<body>
<div class="ambient"></div>

<nav class="glass">
  <a class="brand" href="/">
    <span class="mark">
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none"><path d="M12 2l8 4v6c0 5-3.4 8.4-8 10-4.6-1.6-8-5-8-10V6l8-4z" fill="#061018"/><circle cx="12" cy="11" r="3" fill="#9be8ff"/></svg>
    </span>
    <b>stateful.ai</b>
  </a>
  <div class="navlinks">
    <a href="#features">Features</a>
    <a href="#architecture">Architecture</a>
    <a href="#api">API</a>
    <a href="https://github.com/Ajay-quan/stateful.ai" target="_blank" rel="noopener">GitHub</a>
  </div>
  <a class="btn btn-primary" href="/demo">Open console</a>
</nav>

<main>
  <header class="hero">
    <div>
      <span class="pill"><span class="dot"></span> v0.2 &middot; production hardened &middot; MCP ready</span>
      <h1>Memory infrastructure<br/>for <span class="grad-text">AI agents</span></h1>
      <p class="lead">A persistent memory layer for long-running agents: hybrid semantic + lexical retrieval, versioned lifecycle, contradiction detection, reflection — exposed over REST and MCP. Zero infrastructure to start, swappable adapters to scale.</p>
      <div class="cta-row">
        <a class="btn btn-primary" href="/demo">Launch live console</a>
        <a class="btn btn-ghost" href="#api">API reference</a>
      </div>
      <div class="hint">pip install -r requirements.txt &nbsp;&middot;&nbsp; uvicorn apps.api.main:app</div>
    </div>

    <div class="term glass">
      <div class="term-bar">
        <i style="background:#ff5f57"></i><i style="background:#febc2e"></i><i style="background:#28c840"></i>
        <span style="margin-left:.5rem">stateful_ai — recall</span>
      </div>
      <div class="term-body"><span class="c-dim">$</span> <span class="c-cmd">curl -s $BASE/api/v1/retrieve \
   -H <span class="c-str">"X-API-Key: $KEY"</span> \
   -d <span class="c-str">'{"user_id":"alice","query":"vector db preference"}'</span></span>

<span class="c-dim">{</span>
  <span class="c-key">"results"</span><span class="c-dim">: [{</span>
    <span class="c-key">"content"</span><span class="c-dim">:</span> <span class="c-str">"Alice prefers FAISS for local search"</span><span class="c-dim">,</span>
    <span class="c-key">"score"</span><span class="c-dim">:</span> <span class="c-num">0.914</span><span class="c-dim">,</span>
    <span class="c-key">"signals"</span><span class="c-dim">: {</span> <span class="c-key">"semantic"</span><span class="c-dim">:</span> <span class="c-num">0.88</span><span class="c-dim">,</span> <span class="c-key">"bm25"</span><span class="c-dim">:</span> <span class="c-num">0.79</span><span class="c-dim">,</span> <span class="c-key">"recency"</span><span class="c-dim">:</span> <span class="c-num">0.97</span> <span class="c-dim">}</span>
  <span class="c-dim">}],</span>
  <span class="c-key">"total_found"</span><span class="c-dim">:</span> <span class="c-num">1</span>
<span class="c-dim">}</span></div>
    </div>
  </header>

  <div class="metrics reveal">
    <div class="metric glass"><b class="grad-text">Hybrid</b><span>dense + BM25 fused with RRF</span></div>
    <div class="metric glass"><b class="grad-text">Versioned</b><span>full lifecycle &amp; audit trail</span></div>
    <div class="metric glass"><b class="grad-text">Zero-infra</b><span>boots with no external services</span></div>
    <div class="metric glass"><b class="grad-text">4 tools</b><span>remember &middot; recall &middot; forget &middot; list</span></div>
  </div>

  <section id="features">
    <div class="sec-head reveal">
      <span class="eyebrow">Capabilities</span>
      <h2>Everything an agent needs to remember</h2>
      <p>Chat history is not memory. stateful.ai stores observations, retrieves the right context, keeps facts current, and tells you when they conflict.</p>
    </div>
    <div class="grid3">
      <div class="card glass reveal">
        <div class="ico"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#38d4f5" stroke-width="1.8"><circle cx="11" cy="11" r="7"/><path d="M21 21l-4.3-4.3"/></svg></div>
        <h3>Hybrid retrieval</h3>
        <p>Dense vectors for meaning, BM25 for names and identifiers, Reciprocal Rank Fusion to merge rankings, plus recency, importance, and access signals.</p>
      </div>
      <div class="card glass reveal">
        <div class="ico"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#7c8cf8" stroke-width="1.8"><path d="M12 3v18M3 12h18"/><circle cx="12" cy="12" r="9"/></svg></div>
        <h3>Versioned lifecycle</h3>
        <p>Every update produces a new version with full history. Supersede, soft-delete, and audit any memory — nothing is silently overwritten.</p>
      </div>
      <div class="card glass reveal">
        <div class="ico"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#fbbf24" stroke-width="1.8"><path d="M12 9v4m0 4h.01M10.3 3.9L1.8 18a2 2 0 001.7 3h17a2 2 0 001.7-3L13.7 3.9a2 2 0 00-3.4 0z"/></svg></div>
        <h3>Contradiction detection</h3>
        <p>New observations are scanned against prior facts. Conflicts are reported with confidence scores and penalized during ranking until resolved.</p>
      </div>
      <div class="card glass reveal">
        <div class="ico"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#a78bfa" stroke-width="1.8"><path d="M12 2a7 7 0 017 7c0 2.4-1.2 4.4-3 5.7V17a2 2 0 01-2 2h-4a2 2 0 01-2-2v-2.3C6.2 13.4 5 11.4 5 9a7 7 0 017-7z"/><path d="M9 21h6"/></svg></div>
        <h3>Reflection</h3>
        <p>Periodically distills clusters of related observations into higher-level insights, keeping the store dense with meaning instead of noise.</p>
      </div>
      <div class="card glass reveal">
        <div class="ico"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#4ade80" stroke-width="1.8"><rect x="3" y="3" width="7" height="7" rx="1.5"/><rect x="14" y="3" width="7" height="7" rx="1.5"/><rect x="3" y="14" width="7" height="7" rx="1.5"/><rect x="14" y="14" width="7" height="7" rx="1.5"/></svg></div>
        <h3>MCP server</h3>
        <p>Exposes <code style="font-family:var(--mono);font-size:.82em">remember</code>, <code style="font-family:var(--mono);font-size:.82em">recall</code>, <code style="font-family:var(--mono);font-size:.82em">forget</code>, and <code style="font-family:var(--mono);font-size:.82em">list_memories</code> so any MCP-capable tool shares one memory.</p>
      </div>
      <div class="card glass reveal">
        <div class="ico"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#38d4f5" stroke-width="1.8"><path d="M3 17l5-5 4 4 8-8"/><path d="M14 7h7v7"/></svg></div>
        <h3>Production observability</h3>
        <p>Prometheus metrics, structured request logs with request IDs, liveness and readiness probes, API-key auth, and per-client rate limiting.</p>
      </div>
    </div>
  </section>

  <section id="architecture">
    <div class="sec-head reveal">
      <span class="eyebrow">Architecture</span>
      <h2>Clean layers, swappable adapters</h2>
      <p>Run it with zero infrastructure today; point the same code at Postgres, Qdrant, and Neo4j tomorrow. No layer reaches around another.</p>
    </div>
    <div class="arch reveal">
      <div class="layer glass">
        <span class="tag">API</span>
        <p>FastAPI service &amp; MCP server — auth, rate limits, validation, OpenAPI. <code>apps/api</code> &middot; <code>integrations/mcp_server.py</code></p>
      </div>
      <div class="arrow">&#8595;</div>
      <div class="layer glass">
        <span class="tag">Services</span>
        <p>Ingestion, retrieval, update, reflection, contradiction, consolidation. <code>services/</code></p>
      </div>
      <div class="arrow">&#8595;</div>
      <div class="layer glass">
        <span class="tag">Domain</span>
        <p>Scoring, BM25, RRF fusion, reranking, relevance — pure logic, no I/O. <code>domain/</code></p>
      </div>
      <div class="arrow">&#8595;</div>
      <div class="layer glass">
        <span class="tag">Adapters</span>
        <p>Relational (memory / JSON / Postgres) &middot; vectors (FAISS / Qdrant / Chroma) &middot; graph (Neo4j) &middot; embeddings &middot; LLMs. <code>adapters/</code></p>
      </div>
    </div>
  </section>

  <section id="api">
    <div class="sec-head reveal">
      <span class="eyebrow">API</span>
      <h2>A small, sharp surface</h2>
      <p>Everything below is live on this deployment — try it in the <a href="/demo" style="color:var(--cyan);text-decoration:none">console</a>.</p>
    </div>
    <div class="api-wrap glass reveal">
      <table class="api-table">
        <thead><tr><th>Method</th><th>Endpoint</th><th>Description</th></tr></thead>
        <tbody>
          <tr><td class="m-post">POST</td><td>/api/v1/memories</td><td>Ingest a memory (content, tags, importance, relations)</td></tr>
          <tr><td class="m-post">POST</td><td>/api/v1/retrieve</td><td>Hybrid semantic search over a user's memories</td></tr>
          <tr><td class="m-get">GET</td><td>/api/v1/memories?user_id=…</td><td>List memories for a user</td></tr>
          <tr><td class="m-get">GET</td><td>/api/v1/memories/{id}</td><td>Fetch a single memory</td></tr>
          <tr><td class="m-get">GET</td><td>/api/v1/memories/{id}/versions</td><td>Full version history</td></tr>
          <tr><td class="m-patch">PATCH</td><td>/api/v1/memories/{id}</td><td>Update content, tags, metadata, importance</td></tr>
          <tr><td class="m-del">DELETE</td><td>/api/v1/memories/{id}</td><td>Soft-delete a memory</td></tr>
          <tr><td class="m-get">GET</td><td>/api/v1/graph/{id}</td><td>Traverse related memories</td></tr>
          <tr><td class="m-get">GET</td><td>/api/v1/stats</td><td>Aggregate store statistics</td></tr>
          <tr><td class="m-get">GET</td><td>/api/v1/export</td><td>Export the full store as JSON</td></tr>
          <tr><td class="m-post">POST</td><td>/api/v1/import</td><td>Import a previously exported store</td></tr>
        </tbody>
      </table>
    </div>
  </section>

  <div class="cta-final glass reveal">
    <h2>Give your agents a <span class="grad-text">memory</span></h2>
    <p>Boot it locally in under a minute. No database, no API keys, no model downloads.</p>
    <div class="cta-row" style="justify-content:center">
      <a class="btn btn-primary" href="/demo">Open the console</a>
      <a class="btn btn-ghost" href="https://github.com/Ajay-quan/stateful.ai" target="_blank" rel="noopener">View source</a>
    </div>
  </div>
</main>

<footer>
  <span>&copy; 2026 stateful.ai &middot; persistent memory for LLM agents</span>
  <span><a href="/health">status</a> &nbsp;&middot;&nbsp; <a href="https://github.com/Ajay-quan/stateful.ai" target="_blank" rel="noopener">github</a></span>
</footer>

<script>
  const io = new IntersectionObserver(
    entries => entries.forEach(e => { if (e.isIntersecting) { e.target.classList.add('in'); io.unobserve(e.target); } }),
    { threshold: 0.12 }
  );
  document.querySelectorAll('.reveal').forEach(el => io.observe(el));
</script>
</body>
</html>
""".replace("__BASE_CSS__", _BASE_CSS)

# ---------------------------------------------------------------------------
# Live operations console
# ---------------------------------------------------------------------------

DEMO_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>stateful.ai Console</title>
<meta name="description" content="Live operations console for the stateful.ai memory service." />
<style>
__BASE_CSS__
  body { padding-bottom: 4rem; }
  .wrap { width: min(1180px, calc(100% - 36px)); margin: 0 auto; }

  /* top bar */
  .topbar {
    position: sticky; top: 14px; z-index: 40;
    display: flex; align-items: center; gap: 1rem; flex-wrap: wrap;
    padding: 0.7rem 1.1rem; margin: 14px auto 1.6rem; border-radius: 999px;
  }
  .brand { display: flex; align-items: center; gap: 0.55rem; text-decoration: none; color: var(--ink); margin-right: auto; }
  .brand b { font-weight: 700; }
  .mark { width: 28px; height: 28px; border-radius: 8px; display: grid; place-items: center; background: var(--accent-grad); }
  .keybox { display: flex; align-items: center; gap: 0.5rem; }
  .keybox input {
    width: 180px; padding: 0.5rem 0.85rem; border-radius: 999px;
    background: rgba(0,0,0,0.25); border: 1px solid var(--stroke);
    color: var(--ink); font: 400 0.8rem var(--mono); outline: none;
  }
  .keybox input:focus { border-color: var(--indigo); }

  /* KPI cards */
  .kpis { display: grid; grid-template-columns: repeat(5, 1fr); gap: 0.9rem; margin-bottom: 1.6rem; }
  @media (max-width: 980px) { .kpis { grid-template-columns: repeat(3, 1fr); } }
  @media (max-width: 620px) { .kpis { grid-template-columns: repeat(2, 1fr); } }
  .kpi { padding: 1.1rem 1.25rem; }
  .kpi label { display: block; color: var(--muted); font-size: 0.74rem; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 0.45rem; }
  .kpi b { font-size: 1.55rem; font-weight: 700; letter-spacing: -0.02em; }
  .kpi small { color: var(--faint); font-size: 0.72rem; display: block; margin-top: 0.3rem; }

  /* layout */
  .cols { display: grid; grid-template-columns: 420px 1fr; gap: 1rem; align-items: start; }
  @media (max-width: 980px) { .cols { grid-template-columns: 1fr; } }
  .panel { padding: 1.4rem 1.45rem; margin-bottom: 1rem; }
  .panel h2 { margin: 0 0 1.1rem; font-size: 1.0rem; letter-spacing: -0.01em; display: flex; align-items: center; gap: 0.55rem; }
  .panel h2 .ic { width: 26px; height: 26px; border-radius: 8px; display: grid; place-items: center; background: var(--glass-2); border: 1px solid var(--stroke); }

  label.f { display: block; color: var(--muted); font-size: 0.76rem; margin: 0.85rem 0 0.35rem; letter-spacing: 0.03em; }
  input.f, textarea.f, select.f {
    width: 100%; padding: 0.62rem 0.8rem; border-radius: var(--radius-sm);
    background: rgba(0,0,0,0.25); border: 1px solid var(--stroke);
    color: var(--ink); font: 400 0.88rem var(--sans); outline: none;
    transition: border-color .15s;
  }
  textarea.f { min-height: 86px; resize: vertical; font-family: var(--sans); }
  input.f:focus, textarea.f:focus { border-color: var(--indigo); }
  .row2 { display: grid; grid-template-columns: 1fr 1fr; gap: 0.8rem; }
  .range-row { display: flex; align-items: center; gap: 0.8rem; }
  input[type=range] { flex: 1; accent-color: #7c8cf8; }
  .range-val { font: 600 0.85rem var(--mono); color: var(--cyan); min-width: 2.6rem; text-align: right; }
  .panel .btn { margin-top: 1.15rem; width: 100%; justify-content: center; }

  /* results */
  .result { padding: 0.95rem 1.05rem; border: 1px solid var(--stroke); border-radius: var(--radius-sm); background: rgba(0,0,0,0.18); margin-bottom: 0.7rem; }
  .result .top { display: flex; justify-content: space-between; gap: 1rem; align-items: baseline; }
  .result .content { font-size: 0.92rem; line-height: 1.5; }
  .score { font: 700 0.82rem var(--mono); color: var(--cyan); white-space: nowrap; }
  .scorebar { height: 4px; border-radius: 99px; background: rgba(255,255,255,0.08); margin-top: 0.65rem; overflow: hidden; }
  .scorebar i { display: block; height: 100%; border-radius: 99px; background: var(--accent-grad); }
  .meta { color: var(--faint); font: 400 0.72rem var(--mono); margin-top: 0.5rem; display: flex; gap: 0.9rem; flex-wrap: wrap; }

  /* memory table */
  table.mem { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
  table.mem th { text-align: left; color: var(--faint); font: 600 0.7rem var(--mono); text-transform: uppercase; letter-spacing: 0.1em; padding: 0.55rem 0.7rem; border-bottom: 1px solid var(--stroke); }
  table.mem td { padding: 0.65rem 0.7rem; border-bottom: 1px solid rgba(255,255,255,0.05); vertical-align: top; }
  table.mem tr:hover td { background: rgba(255,255,255,0.025); }
  .mono { font: 400 0.75rem var(--mono); color: var(--muted); }
  .tag-chip { display: inline-block; padding: 0.1rem 0.55rem; border-radius: 99px; background: rgba(124,140,248,0.14); border: 1px solid rgba(124,140,248,0.3); color: #b9c3ff; font-size: 0.7rem; margin: 0 0.25rem 0.25rem 0; }
  .iconbtn {
    background: none; border: 1px solid var(--stroke); border-radius: 8px; color: var(--muted);
    cursor: pointer; padding: 0.28rem 0.55rem; font-size: 0.72rem; font-family: var(--mono);
    transition: color .15s, border-color .15s;
  }
  .iconbtn:hover { color: var(--ink); border-color: var(--stroke-2); }
  .iconbtn.danger:hover { color: var(--red); border-color: rgba(248,113,113,0.5); }
  .table-tools { display: flex; gap: 0.6rem; align-items: center; margin-bottom: 0.9rem; flex-wrap: wrap; }
  .table-tools input { flex: 1; min-width: 160px; }
  .empty { color: var(--faint); text-align: center; padding: 2.2rem 1rem; font-size: 0.88rem; }

  /* modal */
  .modal-bg { position: fixed; inset: 0; background: rgba(3,4,9,0.6); backdrop-filter: blur(6px); -webkit-backdrop-filter: blur(6px); display: none; align-items: center; justify-content: center; z-index: 100; }
  .modal-bg.open { display: flex; }
  .modal { width: min(640px, calc(100% - 32px)); max-height: 80vh; overflow: auto; padding: 1.5rem 1.6rem; }
  .modal h3 { margin: 0 0 1rem; }
  .ver { border-left: 2px solid var(--indigo); padding: 0.5rem 0.9rem; margin-bottom: 0.8rem; }
  .ver .when { color: var(--faint); font: 400 0.72rem var(--mono); }

  /* toasts */
  #toasts { position: fixed; bottom: 22px; right: 22px; display: flex; flex-direction: column; gap: 0.6rem; z-index: 200; }
  .toast {
    padding: 0.8rem 1.15rem; border-radius: var(--radius-sm); font-size: 0.86rem;
    background: rgba(20,24,38,0.85); border: 1px solid var(--stroke-2);
    backdrop-filter: blur(18px); -webkit-backdrop-filter: blur(18px); box-shadow: var(--shadow);
    animation: slidein .25s ease;
  }
  .toast.ok { border-left: 3px solid var(--green); }
  .toast.err { border-left: 3px solid var(--red); }
  @keyframes slidein { from { opacity: 0; transform: translateY(8px); } }
  .spin { display: inline-block; width: 14px; height: 14px; border: 2px solid rgba(255,255,255,0.25); border-top-color: var(--cyan); border-radius: 50%; animation: rot 0.8s linear infinite; vertical-align: -2px; }
  @keyframes rot { to { transform: rotate(360deg); } }
</style>
</head>
<body>
<div class="ambient"></div>

<div class="wrap">
  <div class="topbar glass">
    <a class="brand" href="/">
      <span class="mark"><svg width="14" height="14" viewBox="0 0 24 24" fill="none"><path d="M12 2l8 4v6c0 5-3.4 8.4-8 10-4.6-1.6-8-5-8-10V6l8-4z" fill="#061018"/><circle cx="12" cy="11" r="3" fill="#9be8ff"/></svg></span>
      <b>stateful.ai</b> <span style="color:var(--faint);font-size:.8rem">console</span>
    </a>
    <span class="pill" id="env-pill"><span class="dot" id="env-dot" style="background:var(--amber);box-shadow:0 0 10px var(--amber)"></span><span id="env-text">connecting…</span></span>
    <div class="keybox">
      <input id="api-key" type="password" placeholder="API key (if required)" autocomplete="off" />
    </div>
  </div>

  <div class="kpis" id="kpis">
    <div class="kpi glass"><label>Memories</label><b id="k-total">–</b><small id="k-active"></small></div>
    <div class="kpi glass"><label>Users</label><b id="k-users">–</b><small>distinct user_ids</small></div>
    <div class="kpi glass"><label>Avg importance</label><b id="k-imp">–</b><small>active memories</small></div>
    <div class="kpi glass"><label>Versions</label><b id="k-versions">–</b><small>total revisions</small></div>
    <div class="kpi glass"><label>Recalls</label><b id="k-access">–</b><small>total access count</small></div>
  </div>

  <div class="cols">
    <div>
      <div class="panel glass">
        <h2><span class="ic"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#4ade80" stroke-width="2"><path d="M12 5v14M5 12h14"/></svg></span>Ingest memory</h2>
        <label class="f" for="in-user">User ID</label>
        <input class="f" id="in-user" value="alice" />
        <label class="f" for="in-content">Content</label>
        <textarea class="f" id="in-content" placeholder="Alice prefers FAISS for local vector search…"></textarea>
        <div class="row2">
          <div>
            <label class="f" for="in-key">Key <span style="color:var(--faint)">(optional)</span></label>
            <input class="f" id="in-key" placeholder="pref:vector-db" />
          </div>
          <div>
            <label class="f" for="in-tags">Tags <span style="color:var(--faint)">(comma sep.)</span></label>
            <input class="f" id="in-tags" placeholder="preference, infra" />
          </div>
        </div>
        <label class="f">Importance</label>
        <div class="range-row">
          <input type="range" id="in-imp" min="0" max="1" step="0.05" value="0.5" />
          <span class="range-val" id="in-imp-val">0.50</span>
        </div>
        <button class="btn btn-primary" id="btn-ingest">Store memory</button>
      </div>

      <div class="panel glass">
        <h2><span class="ic"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#38d4f5" stroke-width="2"><circle cx="11" cy="11" r="7"/><path d="M21 21l-4.3-4.3"/></svg></span>Semantic recall</h2>
        <label class="f" for="q-user">User ID</label>
        <input class="f" id="q-user" value="alice" />
        <label class="f" for="q-query">Query</label>
        <input class="f" id="q-query" placeholder="what vector database does alice like?" />
        <div class="row2">
          <div>
            <label class="f" for="q-topk">Top K</label>
            <input class="f" id="q-topk" type="number" min="1" max="50" value="5" />
          </div>
          <div style="display:flex;align-items:flex-end">
            <button class="btn btn-ghost" id="btn-search" style="margin-top:0">Search</button>
          </div>
        </div>
      </div>
    </div>

    <div>
      <div class="panel glass">
        <h2><span class="ic"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#a78bfa" stroke-width="2"><path d="M3 17l5-5 4 4 8-8"/></svg></span>Results</h2>
        <div id="results"><div class="empty">Run a search to see ranked recall results.</div></div>
      </div>

      <div class="panel glass">
        <h2><span class="ic"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#7c8cf8" stroke-width="2"><path d="M4 6h16M4 12h16M4 18h16"/></svg></span>Memory store</h2>
        <div class="table-tools">
          <input class="f" id="ls-user" value="alice" placeholder="user_id" />
          <button class="btn btn-ghost" id="btn-list" style="margin:0;width:auto">Refresh</button>
          <button class="iconbtn" id="btn-export">⤓ export</button>
        </div>
        <div style="overflow-x:auto">
          <table class="mem">
            <thead><tr><th>Content</th><th>Tags</th><th>Imp.</th><th>v</th><th>Updated</th><th></th></tr></thead>
            <tbody id="mem-rows"><tr><td colspan="6" class="empty">No memories loaded — press Refresh.</td></tr></tbody>
          </table>
        </div>
      </div>
    </div>
  </div>
</div>

<div class="modal-bg" id="modal-bg">
  <div class="modal glass">
    <h3>Version history</h3>
    <div id="modal-body"></div>
    <button class="btn btn-ghost" onclick="document.getElementById('modal-bg').classList.remove('open')" style="width:auto;margin-top:1rem">Close</button>
  </div>
</div>

<div id="toasts"></div>

<script>
(function () {
  "use strict";
  const $ = (id) => document.getElementById(id);

  function headers() {
    const h = { "Content-Type": "application/json" };
    const key = $("api-key").value.trim();
    if (key) h["X-API-Key"] = key;
    return h;
  }

  async function api(path, options = {}) {
    const resp = await fetch(path, { headers: headers(), ...options });
    let body = null;
    try { body = await resp.json(); } catch (_) { /* no body */ }
    if (!resp.ok) {
      const message = body && body.error ? body.error.message : `HTTP ${resp.status}`;
      throw new Error(message);
    }
    return body;
  }

  function toast(message, kind = "ok") {
    const el = document.createElement("div");
    el.className = `toast ${kind}`;
    el.textContent = message;
    $("toasts").appendChild(el);
    setTimeout(() => el.remove(), 3800);
  }

  function esc(s) {
    return String(s).replace(/[&<>"']/g, (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  }

  function fmtDate(iso) {
    if (!iso) return "–";
    const d = new Date(iso);
    return isNaN(d) ? iso : d.toLocaleString(undefined, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
  }

  // --- health + stats ------------------------------------------------
  async function refreshHealth() {
    try {
      const h = await api("/health");
      $("env-dot").style.background = "var(--green)";
      $("env-dot").style.boxShadow = "0 0 10px var(--green)";
      $("env-text").textContent = `online · ${h.vector_store || "memory"} · auth ${h.auth_enabled ? "on" : "off"}`;
    } catch (e) {
      $("env-dot").style.background = "var(--red)";
      $("env-dot").style.boxShadow = "0 0 10px var(--red)";
      $("env-text").textContent = "offline";
    }
  }

  async function refreshStats() {
    try {
      const s = await api("/api/v1/stats");
      $("k-total").textContent = s.total_memories;
      $("k-active").textContent = `${s.active_memories} active · ${s.deleted_memories} deleted`;
      $("k-users").textContent = s.users;
      $("k-imp").textContent = s.avg_importance.toFixed(2);
      $("k-versions").textContent = s.total_versions;
      $("k-access").textContent = s.total_access_count;
    } catch (e) { /* stats need a key when auth is on; stay quiet */ }
  }

  // --- ingest ---------------------------------------------------------
  $("in-imp").addEventListener("input", () => {
    $("in-imp-val").textContent = Number($("in-imp").value).toFixed(2);
  });

  $("btn-ingest").addEventListener("click", async () => {
    const content = $("in-content").value.trim();
    const userId = $("in-user").value.trim();
    if (!content || !userId) return toast("user_id and content are required", "err");
    const tags = $("in-tags").value.split(",").map((t) => t.trim()).filter(Boolean);
    const btn = $("btn-ingest");
    btn.innerHTML = '<span class="spin"></span>&nbsp;Storing…';
    try {
      const payload = {
        user_id: userId,
        content,
        tags,
        importance_score: Number($("in-imp").value),
      };
      const key = $("in-key").value.trim();
      if (key) payload.key = key;
      await api("/api/v1/memories", { method: "POST", body: JSON.stringify(payload) });
      toast("Memory stored");
      $("in-content").value = "";
      refreshStats();
      if ($("ls-user").value.trim() === userId) listMemories();
    } catch (e) { toast(e.message, "err"); }
    btn.textContent = "Store memory";
  });

  // --- search ---------------------------------------------------------
  $("btn-search").addEventListener("click", runSearch);
  $("q-query").addEventListener("keydown", (e) => { if (e.key === "Enter") runSearch(); });

  async function runSearch() {
    const query = $("q-query").value.trim();
    const userId = $("q-user").value.trim();
    if (!query || !userId) return toast("user_id and query are required", "err");
    $("results").innerHTML = '<div class="empty"><span class="spin"></span>&nbsp;Searching…</div>';
    try {
      const data = await api("/api/v1/retrieve", {
        method: "POST",
        body: JSON.stringify({ user_id: userId, query, top_k: Number($("q-topk").value) || 5 }),
      });
      renderResults(data.results || []);
    } catch (e) {
      $("results").innerHTML = '<div class="empty">Search failed.</div>';
      toast(e.message, "err");
    }
  }

  function renderResults(results) {
    if (!results.length) {
      $("results").innerHTML = '<div class="empty">No matches for this query.</div>';
      return;
    }
    $("results").innerHTML = results.map((r) => {
      const memory = r.memory || r;
      const rawScore = typeof r.score === "number" ? r.score : (r.similarity ?? 0);
      const pct = Math.max(2, Math.min(100, Math.round(rawScore * 100)));
      const imp = typeof memory.importance_score === "number" ? memory.importance_score.toFixed(2) : "–";
      return `<div class="result">
        <div class="top">
          <div class="content">${esc(memory.content || "")}</div>
          <span class="score">${rawScore.toFixed(3)}</span>
        </div>
        <div class="scorebar"><i style="width:${pct}%"></i></div>
        <div class="meta">
          <span>id ${esc(String(memory.memory_id || "").slice(0, 8))}</span>
          <span>v${esc(memory.version ?? 1)}</span>
          <span>imp ${esc(imp)}</span>
          <span>${esc(fmtDate(memory.updated_at))}</span>
        </div>
      </div>`;
    }).join("");
  }

  // --- list / manage ---------------------------------------------------
  $("btn-list").addEventListener("click", listMemories);

  async function listMemories() {
    const userId = $("ls-user").value.trim();
    if (!userId) return toast("user_id is required", "err");
    const rows = $("mem-rows");
    rows.innerHTML = '<tr><td colspan="6" class="empty"><span class="spin"></span>&nbsp;Loading…</td></tr>';
    try {
      const data = await api(`/api/v1/memories?user_id=${encodeURIComponent(userId)}`);
      const memories = data.memories || [];
      if (!memories.length) {
        rows.innerHTML = '<tr><td colspan="6" class="empty">No memories for this user yet.</td></tr>';
        return;
      }
      rows.innerHTML = memories.map((m) => `<tr>
        <td style="max-width:340px">${esc(m.content)}<div class="mono">${esc(m.memory_id.slice(0, 12))}</div></td>
        <td>${(m.tags || []).map((t) => `<span class="tag-chip">${esc(t)}</span>`).join("") || '<span class="mono">–</span>'}</td>
        <td class="mono">${Number(m.importance_score).toFixed(2)}</td>
        <td class="mono">v${m.version}</td>
        <td class="mono">${esc(fmtDate(m.updated_at))}</td>
        <td style="white-space:nowrap">
          <button class="iconbtn" data-act="versions" data-id="${esc(m.memory_id)}">history</button>
          <button class="iconbtn danger" data-act="delete" data-id="${esc(m.memory_id)}">delete</button>
        </td>
      </tr>`).join("");
    } catch (e) {
      rows.innerHTML = '<tr><td colspan="6" class="empty">Failed to load.</td></tr>';
      toast(e.message, "err");
    }
  }

  $("mem-rows").addEventListener("click", async (e) => {
    const btn = e.target.closest("button[data-act]");
    if (!btn) return;
    const id = btn.dataset.id;
    if (btn.dataset.act === "delete") {
      if (!confirm("Soft-delete this memory?")) return;
      try {
        await api(`/api/v1/memories/${encodeURIComponent(id)}`, { method: "DELETE" });
        toast("Memory deleted");
        listMemories();
        refreshStats();
      } catch (err) { toast(err.message, "err"); }
    } else if (btn.dataset.act === "versions") {
      try {
        const data = await api(`/api/v1/memories/${encodeURIComponent(id)}/versions`);
        $("modal-body").innerHTML = (data.versions || []).map((v) => `
          <div class="ver">
            <div class="when">v${esc(v.version)} · ${esc(fmtDate(v.updated_at || v.created_at))}</div>
            <div>${esc(v.content)}</div>
          </div>`).join("") || '<div class="empty">No versions recorded.</div>';
        $("modal-bg").classList.add("open");
      } catch (err) { toast(err.message, "err"); }
    }
  });

  $("modal-bg").addEventListener("click", (e) => {
    if (e.target === $("modal-bg")) $("modal-bg").classList.remove("open");
  });

  $("btn-export").addEventListener("click", async () => {
    try {
      const data = await api("/api/v1/export");
      const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      a.download = "stateful_ai-export.json";
      a.click();
      URL.revokeObjectURL(a.href);
      toast("Export downloaded");
    } catch (e) { toast(e.message, "err"); }
  });

  // boot
  refreshHealth();
  refreshStats();
  setInterval(refreshHealth, 30000);
})();
</script>
</body>
</html>
""".replace("__BASE_CSS__", _BASE_CSS)
