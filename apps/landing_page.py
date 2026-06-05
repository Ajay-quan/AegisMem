"""Static product pages for the Flask demonstration app."""

LANDING_HTML = r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>AegisMem | Memory infrastructure for AI agents</title>
  <meta name="description" content="AegisMem is a persistent memory layer for long-running AI agents." />
  <style>
    :root {
      color-scheme: dark;
      --bg: #050605;
      --panel: #0d0f0c;
      --panel-2: #12150f;
      --ink: #f5f1e8;
      --muted: #a5a095;
      --line: rgba(245, 241, 232, 0.16);
      --green: #a6ff8f;
      --amber: #d8ac62;
      --blue: #9ab7c9;
      --mono: "SFMono-Regular", "JetBrains Mono", Consolas, monospace;
      --sans: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    [data-theme="light"] {
      color-scheme: light;
      --bg: #f1ecdf;
      --panel: #fffaf0;
      --panel-2: #f7f0e4;
      --ink: #11130f;
      --muted: #69645a;
      --line: rgba(17, 19, 15, 0.15);
      --green: #4e7f35;
      --amber: #9a6d2c;
      --blue: #587287;
    }
    * { box-sizing: border-box; }
    html { scroll-behavior: smooth; }
    body {
      margin: 0;
      color: var(--ink);
      font-family: var(--sans);
      background:
        radial-gradient(circle at 20% 0%, color-mix(in srgb, var(--green) 12%, transparent), transparent 28rem),
        radial-gradient(circle at 82% 12%, color-mix(in srgb, var(--blue) 13%, transparent), transparent 28rem),
        linear-gradient(var(--line) 1px, transparent 1px),
        linear-gradient(90deg, var(--line) 1px, transparent 1px),
        var(--bg);
      background-size: auto, auto, 64px 64px, 64px 64px, auto;
      letter-spacing: 0;
    }
    body::after {
      content: "";
      position: fixed;
      inset: 0;
      pointer-events: none;
      opacity: 0.12;
      background-image: radial-gradient(currentColor 0.5px, transparent 0.5px);
      background-size: 4px 4px;
      z-index: 1;
    }
    a { color: inherit; text-decoration: none; }
    button { font: inherit; }
    .page { position: relative; z-index: 2; overflow: clip; }
    .shell { width: min(1220px, calc(100% - 36px)); margin: 0 auto; }
    .nav {
      position: sticky;
      top: 0;
      z-index: 20;
      border-bottom: 1px solid var(--line);
      background: color-mix(in srgb, var(--bg) 84%, transparent);
      backdrop-filter: blur(18px);
    }
    .nav-inner {
      min-height: 74px;
      display: grid;
      grid-template-columns: 1fr auto 1fr;
      align-items: center;
      gap: 22px;
    }
    .brand { display: inline-flex; align-items: center; gap: 10px; font-weight: 760; font-size: 18px; }
    .logo { width: 36px; height: 36px; object-fit: contain; filter: invert(1); }
    [data-theme="light"] .logo { filter: none; }
    .nav-links { display: flex; gap: 20px; align-items: center; color: var(--muted); font-size: 14px; }
    .nav-links a:hover, .nav-links a.active { color: var(--ink); }
    .nav-actions { display: flex; justify-content: flex-end; gap: 10px; align-items: center; }
    .btn {
      min-height: 40px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: color-mix(in srgb, var(--panel) 80%, transparent);
      color: var(--ink);
      padding: 0 14px;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      gap: 8px;
      cursor: pointer;
      transition: transform 180ms ease, border-color 180ms ease, background 180ms ease;
    }
    .btn:hover { transform: translateY(-2px); border-color: color-mix(in srgb, var(--green) 54%, var(--line)); }
    .btn.primary { background: var(--green); color: #061006; border-color: var(--green); font-weight: 760; }
    .icon-btn { width: 40px; padding: 0; }
    section { padding: 96px 0; scroll-margin-top: 90px; }
    .hero { min-height: calc(100vh - 74px); display: grid; align-items: center; padding: 78px 0 96px; }
    .hero-grid { display: grid; grid-template-columns: minmax(0, 1fr) minmax(390px, 0.9fr); gap: 56px; align-items: center; }
    .eyebrow { margin: 0 0 18px; color: var(--green); font: 760 12px/1.2 var(--mono); letter-spacing: 0.08em; text-transform: uppercase; }
    h1, h2, h3, p { margin-top: 0; }
    h1 { font-size: clamp(62px, 8.8vw, 126px); line-height: 0.88; letter-spacing: 0; margin-bottom: 26px; max-width: 920px; }
    h2 { font-size: clamp(38px, 5vw, 72px); line-height: 0.95; letter-spacing: 0; margin-bottom: 22px; max-width: 900px; }
    h3 { font-size: 22px; margin-bottom: 10px; }
    .lede { color: var(--muted); font-size: clamp(18px, 2vw, 22px); line-height: 1.55; max-width: 780px; }
    .hero-actions { margin-top: 34px; display: flex; flex-wrap: wrap; gap: 10px; }
    .manifest {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: rgba(7, 9, 7, 0.86);
      box-shadow: 0 28px 90px rgba(0, 0, 0, 0.38);
      overflow: hidden;
    }
    [data-theme="light"] .manifest { background: #11130f; color: #f5f1e8; }
    .manifest-head {
      min-height: 48px;
      border-bottom: 1px solid rgba(245,241,232,0.14);
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0 14px;
      color: #aaa;
      font: 700 11px/1 var(--mono);
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }
    .manifest-body { padding: 18px; font: 13px/1.65 var(--mono); color: #d9d4c9; min-height: 470px; }
    .manifest .ok { color: var(--green); }
    .manifest .key { color: var(--blue); }
    .manifest .num { color: var(--amber); }
    .manifest-band { padding-top: 0; }
    .manifest-band .manifest-body { min-height: 360px; }
    .scanline {
      height: 1px;
      background: linear-gradient(90deg, transparent, var(--green), transparent);
      animation: scan 3.4s ease-in-out infinite;
    }
    @keyframes scan { 50% { transform: translateY(428px); opacity: 1; } 0%,100% { opacity: 0.35; } }
    .value-grid { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); border-top: 1px solid var(--line); border-left: 1px solid var(--line); }
    .value {
      min-height: 260px;
      padding: 24px;
      border-right: 1px solid var(--line);
      border-bottom: 1px solid var(--line);
      background: color-mix(in srgb, var(--panel) 76%, transparent);
      transition: background 180ms ease, transform 180ms ease;
    }
    .value:hover { background: var(--panel); transform: translateY(-3px); }
    .value .num { color: var(--green); font: 800 13px/1 var(--mono); margin-bottom: 70px; display: block; }
    .value p, .card p, .trust p { color: var(--muted); line-height: 1.55; }
    .product-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 16px; }
    .card, .trust, .cta-panel {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: color-mix(in srgb, var(--panel) 82%, transparent);
      padding: 26px;
      box-shadow: 0 20px 70px rgba(0,0,0,0.22);
    }
    .card {
      min-height: 300px;
      display: flex;
      flex-direction: column;
      justify-content: space-between;
      transition: transform 180ms ease, border-color 180ms ease;
    }
    .card:hover { transform: translateY(-3px); border-color: color-mix(in srgb, var(--green) 48%, var(--line)); }
    .tag { color: var(--green); font: 800 12px/1 var(--mono); letter-spacing: 0.08em; text-transform: uppercase; }
    .spec { border-top: 1px solid var(--line); padding-top: 18px; color: var(--muted); font: 700 12px/1.55 var(--mono); }
    .trust-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; align-items: stretch; }
    .trust { min-height: 340px; }
    .trust-list { display: grid; gap: 12px; margin-top: 22px; }
    .trust-list div { border: 1px solid var(--line); border-radius: 6px; padding: 13px; color: var(--muted); background: color-mix(in srgb, var(--panel-2) 72%, transparent); }
    .architecture-showcase { display: grid; grid-template-columns: 0.82fr 1.18fr; gap: 28px; align-items: center; }
    .stack-stage {
      min-height: 640px;
      position: relative;
      display: grid;
      place-items: center;
      overflow: hidden;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #070807;
      box-shadow: 0 28px 90px rgba(0,0,0,0.34);
    }
    .iso-stack {
      position: relative;
      width: min(680px, 96%);
      height: 560px;
      transform: translateY(10px);
    }
    .iso-layer {
      position: absolute;
      left: 10%;
      width: 66%;
      height: 118px;
      border: 1px solid rgba(245, 241, 232, 0.18);
      transform: skewY(29deg) rotate(-29deg);
      transform-origin: center;
      background: rgba(255,255,255,0.015);
      box-shadow: inset 0 0 0 1px rgba(255,255,255,0.02);
    }
    .iso-top {
      top: 28px;
      height: 210px;
      border-color: var(--green);
      background:
        radial-gradient(circle, rgba(245,241,232,0.82) 1.4px, transparent 1.8px),
        linear-gradient(135deg, rgba(166,255,143,0.08), rgba(154,183,201,0.04));
      background-size: 22px 22px, auto;
      animation: floatTop 4s ease-in-out infinite;
      filter: drop-shadow(2px 0 0 #ff2434) drop-shadow(-2px 0 0 #00b7ff);
    }
    .iso-mid-1 { top: 218px; }
    .iso-mid-2 { top: 318px; }
    .iso-mid-3 { top: 418px; }
    @keyframes floatTop { 50% { transform: translateY(-8px) skewY(29deg) rotate(-29deg); } }
    .iso-label {
      position: absolute;
      left: 8%;
      color: var(--ink);
      font: 900 clamp(20px, 3vw, 34px)/1 var(--mono);
      letter-spacing: 0.02em;
      transform: rotate(24deg);
      text-shadow: 2px 0 #ff2434, -2px 0 #00b7ff;
      white-space: nowrap;
    }
    .iso-label.ltop { top: 215px; }
    .iso-label.l1 { top: 330px; color: rgba(245,241,232,0.9); }
    .iso-label.l2 { top: 432px; color: rgba(245,241,232,0.84); }
    .iso-label.l3 { top: 532px; color: rgba(245,241,232,0.78); }
    .node-dot {
      position: absolute;
      width: 14px;
      height: 14px;
      border-radius: 50%;
      background: var(--green);
      filter: drop-shadow(2px 0 0 #ff2434) drop-shadow(-2px 0 0 #00b7ff);
      animation: nodePulse 2.4s ease-in-out infinite;
    }
    @keyframes nodePulse { 50% { transform: scale(1.32); } }
    .d1 { left: 31%; top: 112px; } .d2 { left: 45%; top: 86px; } .d3 { left: 58%; top: 118px; }
    .d4 { left: 36%; top: 174px; } .d5 { left: 50%; top: 150px; } .d6 { left: 64%; top: 180px; }
    .d7 { left: 42%; top: 234px; } .d8 { left: 56%; top: 258px; }
    .audience {
      position: absolute;
      right: 6%;
      color: rgba(245,241,232,0.82);
      font: 800 clamp(14px, 2vw, 24px)/1 var(--mono);
      letter-spacing: 0.12em;
      text-transform: uppercase;
    }
    .audience::before {
      content: "";
      position: absolute;
      left: -58px;
      top: 50%;
      width: 40px;
      border-top: 1px dotted rgba(245,241,232,0.55);
    }
    .a1 { top: 245px; } .a2 { top: 350px; } .a3 { top: 458px; }
    .hero .stack-stage { min-height: 570px; }
    .hero .iso-stack { transform: translateY(-10px) scale(0.9); }
    .hero .iso-label.ltop { top: 216px; }
    .engine-grid { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 16px; }
    .engine { border: 1px solid var(--line); border-radius: 8px; min-height: 230px; padding: 18px; background: linear-gradient(145deg, color-mix(in srgb, var(--panel) 86%, transparent), color-mix(in srgb, var(--green) 8%, transparent)); display: flex; flex-direction: column; justify-content: space-between; }
    .chip-row { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 18px; }
    .chip { border: 1px solid var(--line); border-radius: 999px; padding: 8px 11px; color: var(--muted); font: 700 12px/1 var(--mono); }
    .cta-panel { text-align: center; padding: 72px 26px; }
    .cta-panel .lede { margin-left: auto; margin-right: auto; }
    .footer { border-top: 1px solid var(--line); padding: 48px 0; }
    .footer-grid { display: grid; grid-template-columns: 1.4fr repeat(4, 1fr); gap: 24px; color: var(--muted); }
    .footer a { display: block; color: var(--muted); margin: 8px 0; font-size: 14px; }
    .reveal { opacity: 0; transform: translateY(18px); transition: opacity 680ms ease, transform 680ms ease; }
    .reveal.visible { opacity: 1; transform: translateY(0); }
    @media (max-width: 980px) {
      .nav-inner { grid-template-columns: 1fr auto; padding: 14px 0; }
      .nav-links { order: 3; grid-column: 1 / -1; overflow-x: auto; padding-bottom: 4px; }
      .hero-grid, .trust-grid, .architecture-showcase { grid-template-columns: 1fr; }
      .value-grid, .engine-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .product-grid { grid-template-columns: 1fr; }
      .footer-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .stack-stage { min-height: 560px; }
    }
    @media (max-width: 620px) {
      .shell { width: min(100% - 28px, 1220px); }
      section { padding: 72px 0; }
      h1 { font-size: 58px; }
      .value-grid, .engine-grid, .footer-grid { grid-template-columns: 1fr; }
      .nav-actions .btn:not(.icon-btn) { display: none; }
      .manifest-body { min-height: 380px; }
      .stack-stage { min-height: 500px; }
      .iso-stack { width: 760px; transform: translateX(-132px) scale(0.68); transform-origin: left center; }
      .audience { display: none; }
    }
  </style>
</head>
<body>
<div class="page">
  <nav class="nav">
    <div class="shell nav-inner">
      <a class="brand" href="#top"><img class="logo" src="/static/brain.png" alt="" />AegisMem</a>
      <div class="nav-links">
        <a href="#platform">Platform</a><a href="#products">Products</a><a href="#architecture">Architecture</a><a href="#trust">Trust</a><a href="#use-cases">Use cases</a>
      </div>
      <div class="nav-actions">
        <button class="btn icon-btn" id="theme-toggle" aria-label="Toggle theme">+-</button>
        <a class="btn" href="/demo">Play demo</a>
        <a class="btn primary" href="https://github.com/Ajay-quan/AegisMem">GitHub</a>
      </div>
    </div>
  </nav>
  <main id="top">
    <section class="hero shell">
      <div class="hero-grid">
        <div class="reveal">
          <p class="eyebrow">Persistent context infrastructure</p>
          <h1>Memory for agents that work over time.</h1>
          <p class="lede">AegisMem is the durable memory layer between your application and your AI agent. It captures useful context, retrieves the right records later, and keeps long-running work consistent across sessions.</p>
          <div class="hero-actions"><a class="btn primary" href="/demo">Launch memory demo</a><a class="btn" href="#architecture">View architecture</a></div>
        </div>
        <div class="stack-stage reveal" aria-label="Isometric AegisMem architecture stack">
          <div class="iso-stack">
            <div class="iso-layer iso-mid-3"></div>
            <div class="iso-layer iso-mid-2"></div>
            <div class="iso-layer iso-mid-1"></div>
            <div class="iso-layer iso-top"></div>
            <span class="iso-label ltop">Persistent memory layer</span>
            <span class="iso-label l1">Agent context</span>
            <span class="iso-label l2">Retrieval engine</span>
            <span class="iso-label l3">Memory control</span>
            <span class="node-dot d1"></span><span class="node-dot d2"></span><span class="node-dot d3"></span><span class="node-dot d4"></span><span class="node-dot d5"></span><span class="node-dot d6"></span><span class="node-dot d7"></span><span class="node-dot d8"></span>
            <span class="audience a1">Agent builders</span>
            <span class="audience a2">Product teams</span>
            <span class="audience a3">Long workflows</span>
          </div>
        </div>
      </div>
    </section>
    <section class="shell manifest-band">
      <div class="manifest reveal" aria-label="AegisMem agent manifest">
        <div class="manifest-head"><span>// AegisMem agent manifest //</span><span>ready</span></div>
        <div class="scanline"></div>
        <div class="manifest-body">
&gt; <span class="ok">[ok]</span> Agent handshake initialized<br />
&gt; <span class="ok">[ok]</span> Human session detected<br />
&gt; <span class="ok">[ok]</span> Memory layer available<br />
&gt; <span class="ok">[ok]</span> Context graph indexed<br /><br />
{<br />
&nbsp;&nbsp;<span class="key">"page_type"</span>: "agent_memory_platform",<br />
&nbsp;&nbsp;<span class="key">"primary_cta"</span>: "Play memory demo",<br />
&nbsp;&nbsp;<span class="key">"capabilities"</span>: [<br />
&nbsp;&nbsp;&nbsp;&nbsp;"capture_context",<br />
&nbsp;&nbsp;&nbsp;&nbsp;"retrieve_relevant_memory",<br />
&nbsp;&nbsp;&nbsp;&nbsp;"inspect_and_delete_records",<br />
&nbsp;&nbsp;&nbsp;&nbsp;"maintain_project_continuity"<br />
&nbsp;&nbsp;],<br />
&nbsp;&nbsp;<span class="key">"retrieval_profile"</span>: {<br />
&nbsp;&nbsp;&nbsp;&nbsp;<span class="key">"precision_at_1"</span>: <span class="num">1.0</span>,<br />
&nbsp;&nbsp;&nbsp;&nbsp;<span class="key">"avg_latency_ms"</span>: <span class="num">10.3</span><br />
&nbsp;&nbsp;}<br />
}<br /><br />
&gt; <span class="ok">[ok]</span> Agent context generated
        </div>
      </div>
    </section>
    <section class="shell" id="platform">
      <p class="eyebrow reveal">Built for agents. Ready for continuity.</p>
      <div class="value-grid">
        <article class="value reveal"><span class="num">01</span><h3>You bring agents. We bring memory.</h3><p>Keep useful user preferences, project facts, and decisions available beyond a single chat session.</p></article>
        <article class="value reveal"><span class="num">02</span><h3>Every session can build on the last.</h3><p>Agents retrieve what matters before they respond, so answers stay consistent over time.</p></article>
        <article class="value reveal"><span class="num">03</span><h3>Memory is managed, not hidden.</h3><p>Records can be inspected, exported, updated, and deleted instead of becoming an invisible black box.</p></article>
        <article class="value reveal"><span class="num">04</span><h3>Designed for compact infrastructure.</h3><p>Run a serious memory workflow with a lightweight API, local persistence, and retrieval that stays fast.</p></article>
      </div>
    </section>
    <section class="shell" id="products">
      <div class="reveal"><p class="eyebrow">Memory products</p><h2>Memory that scales with agent ambition.</h2></div>
      <div class="product-grid">
        <article class="card reveal"><span class="tag">01 Capture</span><div><h3>Session memory</h3><p>Turn important interactions into durable records with user, key, metadata, and importance signals.</p></div><div class="spec">POST /api/v1/memories<br />Capture context</div></article>
        <article class="card reveal"><span class="tag">02 Recall</span><div><h3>Relevant retrieval</h3><p>Find context for the current task without flooding the agent with unrelated history.</p></div><div class="spec">POST /api/v1/retrieve<br />Return ranked memories</div></article>
        <article class="card reveal"><span class="tag">03 Control</span><div><h3>Inspectable memory</h3><p>Look up exact memories, traverse related records, export snapshots, and remove stale context.</p></div><div class="spec">GET / DELETE / EXPORT<br />Manage memory lifecycle</div></article>
      </div>
    </section>
    <section class="shell" id="architecture">
      <div class="architecture-showcase">
        <div class="trust reveal"><p class="eyebrow">Architecture</p><h2>The memory stack for agent continuity.</h2><p>AegisMem sits between your application and your agent. It captures useful context, stores it as manageable memory, retrieves the right records, and returns a focused context pack before the agent responds.</p><div class="chip-row"><span class="chip">Applications</span><span class="chip">AI agents</span><span class="chip">Teams</span></div></div>
        <div class="stack-stage reveal" aria-label="Isometric architecture stack illustration">
          <div class="iso-stack">
            <div class="iso-layer iso-mid-3"></div>
            <div class="iso-layer iso-mid-2"></div>
            <div class="iso-layer iso-mid-1"></div>
            <div class="iso-layer iso-top"></div>
            <span class="iso-label ltop">Persistent memory layer</span>
            <span class="iso-label l1">Agent context</span>
            <span class="iso-label l2">Retrieval engine</span>
            <span class="iso-label l3">Memory control</span>
            <span class="node-dot d1"></span><span class="node-dot d2"></span><span class="node-dot d3"></span><span class="node-dot d4"></span><span class="node-dot d5"></span><span class="node-dot d6"></span><span class="node-dot d7"></span><span class="node-dot d8"></span>
            <span class="audience a1">AI developers</span>
            <span class="audience a2">Product teams</span>
            <span class="audience a3">Long-running agents</span>
          </div>
        </div>
      </div>
    </section>
    <section class="shell" id="trust">
      <div class="reveal"><p class="eyebrow">Secure by design. Manageable by default.</p><h2>Persistent memory should earn user trust.</h2></div>
      <div class="trust-grid"><div class="trust reveal"><h3>Control surfaces</h3><div class="trust-list"><div>Inspectable records</div><div>Deletable memories</div><div>Portable import/export snapshots</div><div>Optional API-key protection</div></div></div><div class="trust reveal"><h3>Quality surfaces</h3><div class="trust-list"><div>Relevant recall instead of raw history dumps</div><div>Versioned updates for memory lifecycle</div><div>Graph traversal for related context</div><div>Fast local retrieval for demo environments</div></div></div></div>
    </section>
    <section class="shell" id="use-cases">
      <div class="reveal"><p class="eyebrow">Use cases</p><h2>The engines of agent continuity.</h2></div>
      <div class="engine-grid"><article class="engine reveal"><span class="tag">Interview</span><h3>Prep agents</h3><p>Remember feedback, weak areas, and preferred answer style.</p></article><article class="engine reveal"><span class="tag">Coding</span><h3>Dev assistants</h3><p>Preserve architecture, constraints, errors, and fixes.</p></article><article class="engine reveal"><span class="tag">Research</span><h3>Research agents</h3><p>Track notes, papers, hypotheses, and conclusions.</p></article><article class="engine reveal"><span class="tag">Personal</span><h3>Productivity agents</h3><p>Remember routines, recurring tasks, preferences, and goals.</p></article></div>
    </section>
    <section class="shell"><div class="cta-panel reveal"><p class="eyebrow">AegisMem</p><h2>Build agents that remember what matters.</h2><p class="lede">Give your agent a memory layer for persistent context, relevant recall, and better continuity across sessions.</p><div class="hero-actions" style="justify-content:center"><a class="btn primary" href="/demo">Play the demo</a><a class="btn" href="https://github.com/Ajay-quan/AegisMem">View GitHub</a></div></div></section>
  </main>
  <footer class="footer"><div class="shell footer-grid"><div><div class="brand"><img class="logo" src="/static/brain.png" alt="" />AegisMem</div><p>Persistent memory for long-running AI agents.</p></div><div><strong>Platform</strong><a href="#platform">Overview</a><a href="#products">Products</a></div><div><strong>System</strong><a href="#architecture">Architecture</a><a href="#trust">Trust</a></div><div><strong>Project</strong><a href="/demo">Demo</a><a href="https://github.com/Ajay-quan/AegisMem">GitHub</a></div><div><strong>Docs</strong><a href="/docs">API docs</a><a href="/health">Health</a></div></div></footer>
</div>
<script>
const root = document.documentElement;
const savedTheme = localStorage.getItem("aegismem_theme");
if (savedTheme) root.dataset.theme = savedTheme;
document.getElementById("theme-toggle").addEventListener("click", () => {
  const next = root.dataset.theme === "light" ? "dark" : "light";
  root.dataset.theme = next;
  localStorage.setItem("aegismem_theme", next);
});
const reveals = document.querySelectorAll(".reveal");
const revealObserver = new IntersectionObserver((entries) => {
  entries.forEach((entry) => {
    if (entry.isIntersecting) {
      entry.target.classList.add("visible");
      revealObserver.unobserve(entry.target);
    }
  });
}, { threshold: 0.12 });
reveals.forEach((el) => revealObserver.observe(el));
const links = [...document.querySelectorAll(".nav-links a")];
const sections = links.map((link) => document.querySelector(link.getAttribute("href"))).filter(Boolean);
const navObserver = new IntersectionObserver((entries) => {
  entries.forEach((entry) => {
    if (!entry.isIntersecting) return;
    links.forEach((link) => link.classList.toggle("active", link.getAttribute("href") === "#" + entry.target.id));
  });
}, { rootMargin: "-35% 0px -55% 0px" });
sections.forEach((section) => navObserver.observe(section));
</script>
</body>
</html>
"""


DEMO_HTML = r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>AegisMem Demo | Guided Terminal</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #070807;
      --panel: #10130e;
      --panel-2: #151912;
      --ink: #f5f1e8;
      --muted: #aaa59a;
      --line: rgba(245, 241, 232, 0.16);
      --green: #a6ff8f;
      --amber: #d8ac62;
      --blue: #9ab7c9;
      --red: #ff7a7a;
      --mono: "SFMono-Regular", "JetBrains Mono", Consolas, monospace;
      --sans: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      color: var(--ink);
      font-family: var(--sans);
      background:
        radial-gradient(circle at 18% 8%, rgba(166,255,143,0.12), transparent 30rem),
        radial-gradient(circle at 84% 16%, rgba(154,183,201,0.12), transparent 28rem),
        linear-gradient(rgba(245,241,232,0.06) 1px, transparent 1px),
        linear-gradient(90deg, rgba(245,241,232,0.06) 1px, transparent 1px),
        var(--bg);
      background-size: auto, auto, 48px 48px, 48px 48px, auto;
      letter-spacing: 0;
    }
    a { color: inherit; text-decoration: none; }
    button, input { font: inherit; }
    .page { width: min(1180px, calc(100% - 32px)); margin: 0 auto; padding: 22px 0 34px; }
    .topbar {
      border: 1px solid var(--line);
      border-radius: 10px;
      background: rgba(12,14,11,0.82);
      min-height: 64px;
      padding: 12px 16px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      backdrop-filter: blur(18px);
    }
    .brand { display: flex; align-items: center; gap: 10px; font-weight: 780; font-size: 18px; }
    .logo { width: 36px; height: 36px; filter: invert(1); }
    .btn {
      min-height: 42px;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 0 14px;
      color: var(--ink);
      background: rgba(255,255,255,0.04);
      display: inline-flex;
      align-items: center;
      justify-content: center;
      gap: 8px;
      cursor: pointer;
      transition: transform 160ms ease, border-color 160ms ease, background 160ms ease;
    }
    .btn:hover { transform: translateY(-2px); border-color: rgba(166,255,143,0.44); }
    .btn.primary { background: var(--green); border-color: var(--green); color: #061006; font-weight: 800; }
    .hero { padding: 30px 0 18px; display: grid; grid-template-columns: 1fr auto; gap: 18px; align-items: end; }
    h1 { margin: 0 0 12px; font-size: clamp(44px, 7vw, 88px); line-height: 0.9; letter-spacing: 0; }
    .lede { margin: 0; color: var(--muted); font-size: 18px; line-height: 1.5; max-width: 780px; }
    .scoreboard { display: grid; grid-template-columns: repeat(3, 120px); gap: 10px; }
    .score { border: 1px solid var(--line); border-radius: 10px; background: var(--panel); padding: 13px; }
    .score strong { display: block; font-size: 26px; }
    .score span { color: var(--muted); font: 800 11px/1 var(--mono); text-transform: uppercase; letter-spacing: 0.08em; }
    .demo-grid { display: grid; grid-template-columns: 1fr 340px; gap: 18px; align-items: stretch; }
    .terminal, .side {
      border: 1px solid var(--line);
      border-radius: 12px;
      background: rgba(16,19,14,0.92);
      box-shadow: 0 24px 90px rgba(0,0,0,0.36);
      overflow: hidden;
    }
    .head {
      min-height: 54px;
      border-bottom: 1px solid var(--line);
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0 16px;
      color: var(--muted);
      font: 800 12px/1 var(--mono);
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }
    .traffic { display: flex; gap: 7px; }
    .traffic span { width: 10px; height: 10px; border-radius: 50%; }
    .traffic span:nth-child(1) { background: #ff6b62; } .traffic span:nth-child(2) { background: #e0b35a; } .traffic span:nth-child(3) { background: #68d783; }
    .terminal-screen {
      min-height: 540px;
      max-height: 640px;
      overflow: auto;
      padding: 18px;
      background:
        linear-gradient(rgba(166,255,143,0.035) 1px, transparent 1px),
        #080a07;
      background-size: 100% 28px;
      font: 14px/1.62 var(--mono);
    }
    .line { display: grid; grid-template-columns: 108px 1fr; gap: 12px; margin-bottom: 9px; animation: rise 220ms ease both; }
    @keyframes rise { from { opacity: 0; transform: translateY(4px); } to { opacity: 1; transform: translateY(0); } }
    .time { color: rgba(245,241,232,0.38); }
    .sys { color: var(--blue); }
    .ok { color: var(--green); }
    .warn { color: var(--amber); }
    .err { color: var(--red); }
    .user { color: var(--ink); }
    .input-row {
      border-top: 1px solid var(--line);
      display: flex;
      align-items: center;
      gap: 10px;
      padding: 16px;
      background: rgba(255,255,255,0.035);
      font-family: var(--mono);
    }
    .prompt { color: var(--green); font-weight: 900; }
    #terminal-input {
      width: 100%;
      border: 0;
      outline: 0;
      background: transparent;
      color: var(--ink);
      font-family: var(--mono);
      font-size: 15px;
    }
    .side-body { padding: 16px; display: grid; gap: 14px; }
    .guide-card {
      border: 1px solid var(--line);
      border-radius: 10px;
      background: rgba(255,255,255,0.035);
      padding: 16px;
    }
    .guide-card h3 { margin: 0 0 8px; font-size: 20px; }
    .guide-card p { color: var(--muted); line-height: 1.5; margin: 0; }
    .step-list { display: grid; gap: 10px; }
    .step {
      border: 1px solid var(--line);
      border-radius: 9px;
      padding: 13px;
      color: var(--muted);
      background: rgba(0,0,0,0.18);
      transition: transform 180ms ease, border-color 180ms ease, color 180ms ease;
    }
    .step.active { color: var(--ink); border-color: rgba(166,255,143,0.55); transform: translateX(4px); }
    .step.done { color: var(--green); border-color: rgba(154,183,201,0.42); }
    .quick { display: flex; flex-wrap: wrap; gap: 8px; }
    .quick button { min-height: 34px; border-radius: 999px; border: 1px solid var(--line); color: var(--muted); background: rgba(255,255,255,0.035); padding: 0 10px; cursor: pointer; }
    .quick button:hover { color: var(--ink); border-color: rgba(166,255,143,0.45); }
    .lifecycle {
      border: 1px solid var(--line);
      border-radius: 10px;
      background: #080a07;
      padding: 14px;
      display: grid;
      gap: 10px;
    }
    .life-node {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      color: var(--muted);
      background: rgba(255,255,255,0.03);
      position: relative;
      transition: border-color 180ms ease, color 180ms ease, transform 180ms ease;
    }
    .life-node:not(:last-child)::after {
      content: "";
      position: absolute;
      left: 22px;
      bottom: -11px;
      height: 10px;
      border-left: 1px dotted rgba(166,255,143,0.42);
    }
    .life-node.active { color: var(--ink); border-color: rgba(166,255,143,0.56); transform: translateX(4px); }
    .life-node.done { color: var(--green); border-color: rgba(154,183,201,0.42); }
    .life-node b { display: block; margin-bottom: 4px; }
    .life-node span { display: block; font: 12px/1.4 var(--mono); }
    @media (max-width: 920px) {
      .hero, .demo-grid { grid-template-columns: 1fr; }
      .scoreboard { grid-template-columns: repeat(3, minmax(0, 1fr)); }
      .terminal { order: -1; }
    }
    @media (max-width: 620px) {
      .page { width: min(100% - 24px, 1180px); }
      .topbar { align-items: flex-start; flex-direction: column; }
      .topbar .btn { width: 100%; }
      .scoreboard { grid-template-columns: 1fr; }
      .line { grid-template-columns: 1fr; gap: 2px; }
      .terminal-screen { min-height: 480px; }
    }
  </style>
</head>
<body>
<div class="page">
  <header class="topbar">
    <a class="brand" href="/"><img class="logo" src="/static/brain.png" alt="" />AegisMem Guided Terminal</a>
    <div style="display:flex; gap:10px; flex-wrap:wrap">
      <button class="btn primary" id="sample-btn">Run sample flow</button>
      <a class="btn" href="/">Back to landing</a>
    </div>
  </header>
  <section class="hero">
    <div>
      <h1>Type once. Watch the agent remember.</h1>
      <p class="lede">This demo shows the actual AegisMem lifecycle. Run the sample flow or type into the terminal: user name, memory, then a later question.</p>
    </div>
    <div class="scoreboard">
      <div class="score"><strong id="score">0</strong><span>memory score</span></div>
      <div class="score"><strong id="latency">--</strong><span>last call</span></div>
      <div class="score"><strong id="phase-label">1/4</strong><span>step</span></div>
    </div>
  </section>
  <main class="demo-grid">
    <section class="terminal">
      <div class="head"><div class="traffic"><span></span><span></span><span></span></div><span id="status">ready</span></div>
      <div class="terminal-screen" id="terminal"></div>
      <form class="input-row" id="terminal-form"><span class="prompt">&gt;</span><input id="terminal-input" autocomplete="off" /></form>
    </section>
    <aside class="side">
      <div class="head"><span>What is happening?</span><span>simple mode</span></div>
      <div class="side-body">
        <div class="guide-card"><h3 id="guide-title">Start with a user.</h3><p id="guide-copy">The terminal will ask one question at a time. Press Enter after each answer.</p></div>
        <div class="lifecycle" aria-label="AegisMem lifecycle">
          <div class="life-node active" id="l-capture"><b>Capture</b><span>Collect user/project context</span></div>
          <div class="life-node" id="l-store"><b>Store</b><span>POST /api/v1/memories</span></div>
          <div class="life-node" id="l-retrieve"><b>Retrieve</b><span>POST /api/v1/retrieve</span></div>
          <div class="life-node" id="l-respond"><b>Respond</b><span>Return context to the agent</span></div>
        </div>
        <div class="step-list">
          <div class="step active" id="s-user">1. Choose a user</div>
          <div class="step" id="s-memory">2. Store useful memory</div>
          <div class="step" id="s-question">3. Ask a later question</div>
          <div class="step" id="s-result">4. See recalled context</div>
        </div>
        <div class="guide-card"><h3>Fast examples</h3><div class="quick"><button data-fill="Alice">Alice</button><button data-fill="Alice prefers concise STAR interview answers.">STAR answers</button><button data-fill="How should Alice answer interview questions?">Interview question</button></div></div>
      </div>
    </aside>
  </main>
</div>
<script>
const apiKey = localStorage.getItem("aegismem_api_key") || "";
const terminal = document.getElementById("terminal");
const input = document.getElementById("terminal-input");
const state = { phase: "user", user: "", memory: "", memoryId: "", query: "", score: 0 };
function headers() { const h = {"Content-Type": "application/json"}; if (apiKey) h["X-API-Key"] = apiKey; return h; }
function stamp() { return new Date().toLocaleTimeString([], { hour12: false }); }
function escapeHtml(text) { return String(text).replace(/[&<>"']/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch])); }
function line(kind, text) {
  const row = document.createElement("div");
  row.className = "line";
  row.innerHTML = `<span class="time">${stamp()}</span><span class="${kind}">${escapeHtml(text)}</span>`;
  terminal.appendChild(row);
  terminal.scrollTop = terminal.scrollHeight;
}
function setStatus(text) { document.getElementById("status").textContent = text; }
function setGuide(title, copy, placeholder) {
  document.getElementById("guide-title").textContent = title;
  document.getElementById("guide-copy").textContent = copy;
  input.placeholder = placeholder;
  input.focus();
}
function setStep(active) {
  const order = ["user", "memory", "question", "result"];
  order.forEach((name, index) => {
    const el = document.getElementById(`s-${name}`);
    el.className = `step ${name === active ? "active" : order.indexOf(active) > index ? "done" : ""}`;
  });
  const lifecycle = {
    user: "capture",
    memory: "store",
    question: "retrieve",
    result: "respond"
  };
  const lifeOrder = ["capture", "store", "retrieve", "respond"];
  const activeLife = lifecycle[active];
  lifeOrder.forEach((name, index) => {
    const el = document.getElementById(`l-${name}`);
    el.className = `life-node ${name === activeLife ? "active" : lifeOrder.indexOf(activeLife) > index ? "done" : ""}`;
  });
  document.getElementById("phase-label").textContent = `${order.indexOf(active) + 1}/4`;
}
async function request(path, options = {}) {
  const started = performance.now();
  setStatus("calling api");
  const res = await fetch(path, options);
  const data = await res.json();
  document.getElementById("latency").textContent = `${Math.round(performance.now() - started)}ms`;
  setStatus(res.ok ? "ready" : "error");
  if (!res.ok) line("err", `API error: ${data.error?.message || res.statusText}`);
  return data;
}
function boot() {
  terminal.innerHTML = "";
  line("sys", "AegisMem terminal started.");
  line("sys", "We will store one memory, then ask a later question.");
  line("ok", "What user should the agent remember? Try: Alice");
  line("warn", "Shortcut: type sample or click Run sample flow.");
  setGuide("Start with a user.", "Give the agent a user name. This keeps memory scoped and easy to understand.", "Type a user name, e.g. Alice");
  setStep("user");
}
async function handle(value) {
  if (!value) return;
  line("user", `> ${value}`);
  input.value = "";
  if (value.toLowerCase() === "sample") { runSample(); return; }
  if (value.toLowerCase() === "restart") { state.phase = "user"; state.score = 0; document.getElementById("score").textContent = "0"; boot(); return; }
  if (state.phase === "user") {
    state.user = value;
    state.phase = "memory";
    line("ok", `User set to ${state.user}.`);
    line("sys", "Now type one useful thing the agent should remember.");
    setGuide("Store a useful memory.", "Write one practical preference, project fact, or decision. AegisMem will persist it through the API.", "Example: Alice prefers concise STAR interview answers.");
    setStep("memory");
    return;
  }
  if (state.phase === "memory") {
    state.memory = value;
    line("sys", "Saving memory through POST /api/v1/memories...");
    const data = await request("/api/v1/memories", { method: "POST", headers: headers(), body: JSON.stringify({ user_id: state.user, key: "guided-demo", content: state.memory, importance_score: 0.9 }) });
    if (data.memory) {
      state.memoryId = data.memory.memory_id;
      state.phase = "question";
      state.score = 50;
      document.getElementById("score").textContent = state.score;
      line("ok", "Memory saved. The agent can retrieve it later instead of asking again.");
      line("sys", "Ask a later question where that memory would help.");
      setGuide("Ask a later question.", "Pretend this is a new session. Ask something where the saved memory should help the agent respond.", "Example: How should Alice answer interview questions?");
      setStep("question");
    }
    return;
  }
  if (state.phase === "question") {
    state.query = value;
    line("sys", "Retrieving relevant memory through POST /api/v1/retrieve...");
    const data = await request("/api/v1/retrieve", { method: "POST", headers: headers(), body: JSON.stringify({ user_id: state.user, query: state.query, top_k: 3 }) });
    const top = data.results?.[0];
    state.phase = "result";
    state.score = top ? 100 : 60;
    document.getElementById("score").textContent = state.score;
    line("ok", top ? `AegisMem recalled: ${top.content}` : "No exact match returned, but the flow completed.");
    line("sys", top ? "Agent response: I used the saved memory instead of asking the user to repeat it." : "Try a question closer to the memory you stored.");
    line("ok", "Type restart to run the demo again.");
    setGuide("That is persistent memory.", "The important detail was saved once, then retrieved later for a better response. Type restart to try another example.", "Type restart to run again");
    setStep("result");
  }
}
document.getElementById("terminal-form").addEventListener("submit", (event) => {
  event.preventDefault();
  handle(input.value.trim());
});
function wait(ms) { return new Promise(resolve => setTimeout(resolve, ms)); }
async function runSample() {
  if (state.phase !== "user") {
    state.phase = "user";
    state.score = 0;
    document.getElementById("score").textContent = "0";
    boot();
    await wait(250);
  }
  await handle("Alice");
  await wait(650);
  await handle("Alice prefers concise STAR interview answers with measurable impact.");
  await wait(850);
  await handle("How should Alice answer interview practice questions?");
}
document.getElementById("sample-btn").addEventListener("click", runSample);
document.querySelectorAll("[data-fill]").forEach((button) => button.addEventListener("click", () => {
  input.value = button.dataset.fill;
  input.focus();
}));
boot();
</script>
</body>
</html>
"""
