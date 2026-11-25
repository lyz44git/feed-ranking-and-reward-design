// Interactive figures for the feed-ranking repository. Data: data.js, built by build_data.py from the result files.
(() => {
"use strict";
const D = window.DATA, NS = "http://www.w3.org/2000/svg";
const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
const $ = (s, r = document) => r.querySelector(s);
const fmt = (v, d = 0) => v.toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d });
const pct = (v, d = 1) => fmt(100 * v, d) + "%";

/* ---------- helpers (as in the ad-measurement explorer) ---------- */
function el(tag, attrs = {}, parent) {
  const e = document.createElementNS(NS, tag);
  for (const k in attrs) if (attrs[k] !== undefined && attrs[k] !== null) e.setAttribute(k, attrs[k]);
  if (parent) parent.appendChild(e);
  return e;
}
function txt(parent, x, y, s, attrs = {}) { const t = el("text", Object.assign({ x, y }, attrs), parent); t.textContent = s; return t; }
function svgIn(host, w, h) { host.innerHTML = ""; return el("svg", { width: w, height: h, viewBox: `0 0 ${w} ${h}`, role: "img" }, host); }
const lin = (d0, d1, r0, r1) => v => r0 + (v - d0) / (d1 - d0) * (r1 - r0);
const lg = (d0, d1, r0, r1) => { const a = Math.log10(d0), b = Math.log10(d1); return v => r0 + (Math.log10(v) - a) / (b - a) * (r1 - r0); };
const ease = t => t < .5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
function tween(dur, step, done) {
  if (reduce || dur <= 0 || document.hidden) { step(1); if (done) done(); return; }
  const t0 = performance.now();
  (function f(now) { const t = Math.max(0, Math.min(1, (now - t0) / dur)); step(ease(t)); if (t < 1) requestAnimationFrame(f); else if (done) done(); })(t0);
}
function animate(items, dur = 650, done) {
  const from = items.map(([e, a]) => Object.fromEntries(Object.keys(a).map(k => [k, parseFloat(e.getAttribute(k)) || 0])));
  tween(dur, t => items.forEach(([e, a], i) => { for (const k in a) { let v = from[i][k] + (a[k] - from[i][k]) * t; if (k === "width" || k === "height") v = Math.max(0, v); e.setAttribute(k, v); } }), done);
}
function countTo(node, to, dur = 900, d = 0, suffix = "") {
  node.textContent = fmt(to, d) + suffix;
  tween(dur, t => node.textContent = fmt(to * t, d) + suffix);
}
const tip = $("#tip");
function hoverTip(node, html) {
  node.addEventListener("pointermove", e => { tip.innerHTML = typeof html === "function" ? html() : html; tip.classList.add("on");
    const x = Math.min(e.clientX + 14, innerWidth - tip.offsetWidth - 8); tip.style.left = x + "px"; tip.style.top = (e.clientY + 14) + "px"; });
  node.addEventListener("pointerleave", () => tip.classList.remove("on"));
}
function segControl(host, onPick, init) {
  const btns = [...host.querySelectorAll("button")];
  const set = v => btns.forEach(b => b.setAttribute("aria-pressed", String(b.dataset.v === String(v))));
  btns.forEach(b => b.addEventListener("click", () => { set(b.dataset.v); onPick(b.dataset.v); }));
  set(init); return set;
}
function whenVisible(node, fn) {
  if (!("IntersectionObserver" in window)) return fn();
  const io = new IntersectionObserver(es => es.forEach(e => { if (e.isIntersecting) { io.disconnect(); fn(); } }), { threshold: .3 });
  io.observe(node);
}
const W = host => Math.max(280, host.clientWidth);
const redrawers = [];
let rt; addEventListener("resize", () => { clearTimeout(rt); rt = setTimeout(() => redrawers.forEach(f => f()), 150); });
function gridY(g, y, ticks, x0, x1, fmtT = String) {
  ticks.forEach(v => { el("line", { x1: x0, x2: x1, y1: y(v), y2: y(v), class: "gridl" }, g); txt(g, x0 - 8, y(v) + 4, fmtT(v), { "text-anchor": "end" }); });
}
function axisX(g, x, ticks, y, fmtT = String, label) {
  ticks.forEach(v => { el("line", { x1: x(v), x2: x(v), y1: y, y2: y + 4, class: "axis" }, g); txt(g, x(v), y + 17, fmtT(v), { "text-anchor": "middle" }); });
  if (label) txt(g, (x(ticks[0]) + x(ticks[ticks.length - 1])) / 2, y + 36, label, { "text-anchor": "middle", class: "t-muted" });
}
function niceStep(x) { const p = Math.pow(10, Math.floor(Math.log10(x))); for (const m of [1, 2, 2.5, 5, 10]) if (m * p >= x - 1e-12) return m * p; return 10 * p; }
function niceMax(v, n = 4) { const s = niceStep(v / n); return Math.ceil(v / s - 1e-9) * s; }
function ticksTo(max, n = 4) { const s = niceStep(max / n), out = []; for (let i = 0; i * s <= max + 1e-9; i++) out.push(+(i * s).toFixed(6)); return out; }
// draw a path in once it scrolls into view; a timer finishes it if animation frames do not run
function drawIn(p, host, dur) {
  if (reduce) return;
  whenVisible(host, () => { p.setAttribute("pathLength", 1); p.style.strokeDasharray = 1; p.style.strokeDashoffset = 1; p.getBoundingClientRect();
    p.style.transition = `stroke-dashoffset ${dur}s ease`; requestAnimationFrame(() => requestAnimationFrame(() => { p.style.strokeDashoffset = 0; }));
    setTimeout(() => { p.style.strokeDashoffset = 0; }, dur * 1000 + 400); });
}
function path(pts) { return pts.map((p, i) => (i ? "L" : "M") + p[0].toFixed(1) + "," + p[1].toFixed(1)).join(""); }
const css = name => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

/* ---------- hero ---------- */
const S2 = D.step2.top;
const hero = { exposures: [D.test.exposures, 0], users: [D.test.users, 0],
  lvStd: [100 * S2.trained_on_standard_feed.long_view.est, 1], lvRand: [100 * S2.trained_on_random_exposures.long_view.est, 1] };
document.querySelectorAll("[data-count]").forEach(n => { const [v, d] = hero[n.dataset.count]; whenVisible(n, () => countTo(n, v, 1400, d, n.dataset.suffix || "")); });

const SH = D.shares, shareLab = s => (s * 100 >= 1 ? fmt(100 * s, 0) : fmt(100 * s, 1)) + "%";

/* ---------- 1: outcome by list length ---------- */
(() => {
  const host = $("#c-curve"), lgd = $("#lg-curve"), rg = $("#rg-share"), out = $("#out-share");
  const SETS = {
    logs: [["transformer", "Trained on random exposures", "--navy", 3], ["trained_on_standard_feed", "Trained on the standard feed", "--orange", 3]],
    models: [["lgb_no_history", "Trees, no history", "--grey", 1.6], ["lgb_history_summaries", "Trees, history summaries", "--light", 1.6],
             ["network_no_sequence", "Network without the Transformer", "--blue", 1.8], ["transformer_no_ranking_loss", "Transformer, no within-user loss", "--orange", 1.8],
             ["transformer", "Transformer", "--navy", 3]] };
  let set = "logs", outc = "long_view", first = true;
  const f = v => outc === "long_view" ? pct(v) : fmt(v, 1) + " s";
  function draw(anim) {
    const w = W(host), h = Math.min(380, Math.max(280, w * .5)), L = 56, R = w - (w < 560 ? 16 : 150), T = 14, B = h - 44;
    const svg = svgIn(host, w, h), lines = SETS[set];
    const vals = lines.flatMap(([k]) => D.curves[k][outc]).concat([D.random[outc]]);
    const ymax = niceMax(Math.max(...vals) * 1.08), x = lg(SH[0], 1, L, R), y = lin(0, ymax, B, T);
    gridY(svg, y, ticksTo(ymax), L, R, v => outc === "long_view" ? fmt(100 * v, 0) + "%" : fmt(v, 0));
    axisX(svg, x, [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1], B, shareLab, "Share of each user's ranked list shown (log scale)");
    const ry = y(D.random[outc]);
    el("line", { x1: L, x2: R, y1: ry, y2: ry, class: "s-grey", "stroke-dasharray": "5 4", "stroke-width": 1.4 }, svg);
    txt(svg, L + 4, ry - 6, "shown at random", { class: "t-muted halo", "font-size": 11 });
    const i = +rg.value, xs = x(SH[i]);
    el("line", { x1: xs, x2: xs, y1: T, y2: B, stroke: css("--orange"), "stroke-width": 1, "stroke-dasharray": "2 3" }, svg);
    lines.forEach(([k, lab, col, sw]) => {
      const pts = SH.map((s, j) => [x(s), y(D.curves[k][outc][j])]);
      const p = el("path", { d: path(pts), fill: "none", stroke: css(col), "stroke-width": sw, "stroke-linejoin": "round", "stroke-linecap": "round" }, svg);
      if (anim) drawIn(p, host, 1.4);
      const c = el("circle", { cx: xs, cy: y(D.curves[k][outc][i]), r: 5.5, fill: css(col), class: "s-card", "stroke-width": 2 }, svg);
      hoverTip(c, `${lab}<br>top ${shareLab(SH[i])}: <b>${f(D.curves[k][outc][i])}</b>`);
      if (w >= 560) txt(svg, R + 8, y(D.curves[k][outc][0]) + 4, lab, { "font-size": 11, fill: css(col) });
    });
    // a transparent overlay: hover anywhere to move the slider
    const ov = el("rect", { x: L, y: T, width: R - L, height: B - T, fill: "transparent" }, svg);
    ov.addEventListener("pointermove", e => { const bx = svg.getBoundingClientRect().left, px = e.clientX - bx;
      let best = 0; SH.forEach((s, j) => { if (Math.abs(x(s) - px) < Math.abs(x(SH[best]) - px)) best = j; });
      if (best !== +rg.value) { rg.value = best; update(); } });
    lgd.innerHTML = lines.map(([k, lab, col]) => `<span><i style="background:${css(col)}"></i>${lab} <b>${f(D.curves[k][outc][i])}</b></span>`).join("")
      + `<span><i style="background:${css("--grey")}"></i>shown at random <b>${f(D.random[outc])}</b></span>`;
  }
  function update() { out.textContent = "top " + shareLab(SH[+rg.value]); draw(false); }
  segControl($("#seg-curve-set"), v => { set = v; draw(true); }, set);
  segControl($("#seg-curve-out"), v => { outc = v; draw(true); }, outc);
  rg.addEventListener("input", update); out.textContent = "top " + shareLab(SH[+rg.value]);
  draw(first); redrawers.push(() => draw(false));
})();

/* ---------- 2a: which ranker wins ---------- */
(() => {
  const host = $("#c-flip"), O = D.step2.offline;
  const M = {
    auc_standard: { lab: "AUC on held-out standard-feed exposures", v: k => O[k].auc_standard, f: v => fmt(v, 3), min: 0.5 },
    auc_random: { lab: "AUC on held-out random exposures (same metric, other data)", v: k => O[k].auc_random, f: v => fmt(v, 3), min: 0.5 },
    within_random: { lab: "Within-user AUC on random exposures", v: k => O[k].within_random, f: v => fmt(v, 3), min: 0.5 },
    long_view: { lab: "Long-view rate of each user's top 10%", v: k => S2[k].long_view.est, f: v => pct(v), min: 0 },
    watch_seconds: { lab: "Seconds watched per exposure, top 10%", v: k => S2[k].watch_seconds.est, f: v => fmt(v, 1) + " s", min: 0 } };
  const K = [["trained_on_random_exposures", "Trained on random exposures", "--navy"], ["trained_on_standard_feed", "Trained on the standard feed", "--orange"]];
  let m = "auc_standard"; const bars = {};
  function draw(anim) {
    const w = W(host), h = 170, L = w < 560 ? 12 : 210, R = w - 70, T = 18, rowH = 54, mm = M[m];
    const svg = svgIn(host, w, h), vals = K.map(([k]) => mm.v(k)), max = Math.max(...vals) * 1.08;
    const x = lin(mm.min, mm.min === 0.5 ? Math.max(0.75, max) : max, L, R), win = vals[0] > vals[1] ? 0 : 1;
    K.forEach(([k, lab, col], i) => {
      const yy = T + i * rowH;
      if (w >= 560) txt(svg, L - 12, yy + 22, lab, { "text-anchor": "end", class: "t-ink" });
      else txt(svg, L, yy + 2, lab, { class: "t-ink", "font-size": 12 });
      el("rect", { x: L, y: yy + 8, width: R - L, height: 26, rx: 6, class: "f-pale" }, svg);
      const b = el("rect", { x: L, y: yy + 8, width: anim ? (bars[k] || 0) : x(vals[i]) - L, height: 26, rx: 6, fill: css(col) }, svg);
      const t = txt(svg, x(vals[i]) + 8, yy + 26, mm.f(vals[i]), { class: "t-strong" });
      if (i === win) { const g = el("g", {}, svg); const bx = Math.min(R - 10, x(vals[i]) + 8 + t.getComputedTextLength() + 8);
        el("rect", { x: bx, y: yy + 12, width: 92, height: 18, rx: 9, class: "f-orange" }, g); txt(g, bx + 46, yy + 25, "WOULD SHIP", { "text-anchor": "middle", class: "t-on", "font-size": 10, "font-weight": 700 }); }
      if (anim) animate([[b, { width: x(vals[i]) - L }]], 800);
      bars[k] = x(vals[i]) - L;
      hoverTip(b, `${lab}<br>${mm.lab}: <b>${mm.f(vals[i])}</b>`);
    });
    txt(svg, L, h - 8, mm.lab + (mm.min === 0.5 ? " (axis starts at 0.5, chance)" : ""), { class: "t-muted", "font-size": 11 });
  }
  segControl($("#seg-flip"), v => { m = v; draw(true); }, m);
  whenVisible(host, () => draw(true)); draw(false); redrawers.push(() => draw(false));
})();

/* ---------- 2b: calibration ---------- */
(() => {
  const host = $("#c-calib"), ro = $("#ro-calib"), C = D.calibration;
  const K = [["transformer", "Trained on random exposures", "--navy", "trained_on_random_exposures"], ["trained_on_standard_feed", "Trained on the standard feed", "--orange", "trained_on_standard_feed"]];
  let show = "both";
  function draw(anim) {
    const w = W(host), h = Math.min(360, Math.max(260, w * .45)), L = 56, R = w - 20, T = 14, B = h - 44;
    const svg = svgIn(host, w, h), mx = niceMax(Math.max(...K.flatMap(([k]) => C[k].predicted)) * 1.05);
    const my = niceMax(Math.max(...K.flatMap(([k]) => C[k].observed)) * 1.15), x = lin(0, mx, L, R), y = lin(0, my, B, T);
    gridY(svg, y, ticksTo(my), L, R, v => fmt(100 * v, 0) + "%");
    axisX(svg, x, ticksTo(mx, 5), B, v => fmt(100 * v, 0) + "%", "Mean predicted long-view rate, by tenth of prediction");
    const d = Math.min(mx, my); el("line", { x1: x(0), y1: y(0), x2: x(d), y2: y(d), class: "s-grey", "stroke-dasharray": "4 4" }, svg);
    txt(svg, x(d) - 4, y(d) + 14, "predicted = observed", { "text-anchor": "end", class: "t-muted", "font-size": 11 });
    K.forEach(([k, lab, col]) => {
      const on = show === "both" || show === k, g = el("g", { opacity: on ? 1 : 0.12, class: "fade" }, svg);
      const pts = C[k].predicted.map((p, i) => [x(p), y(C[k].observed[i])]);
      el("path", { d: path(pts), fill: "none", stroke: css(col), "stroke-width": 2.4 }, g);
      pts.forEach(([px, py], i) => { const c = el("circle", { cx: px, cy: anim ? B : py, r: 5, fill: css(col), class: "s-card", "stroke-width": 1.5 }, g);
        if (anim) whenVisible(host, () => setTimeout(() => animate([[c, { cy: py }]], 700), 40 * i));
        hoverTip(c, `${lab}<br>tenth ${i + 1}: predicted ${pct(C[k].predicted[i])}, observed ${pct(C[k].observed[i])}`); });
    });
    ro.innerHTML = K.map(([k, lab, col, ko]) => `<div><div class="k">${lab}</div><div class="v" style="color:${css(col)}">${fmt(D.step2.offline[ko].pred_over_actual, 1)}× <small>mean predicted over observed</small></div></div>`).join("");
  }
  segControl($("#seg-calib"), v => { show = v; draw(false); }, show);
  draw(true); redrawers.push(() => draw(false));
})();

/* ---------- 3a: length of picks ---------- */
(() => {
  const host = $("#c-length"), ro = $("#ro-reward"), E = D.length_edges;
  const key = { completion: "reward_completion", long_view: "transformer", watch_time: "reward_watch_time" };
  const lab = E.map((e, i) => i === 0 ? "none" : i === E.length - 1 ? `${e}+` : `${e}–${E[i + 1]}`);
  let r = "completion"; let bars = [];
  function draw(anim) {
    const w = W(host), h = 280, L = 48, R = w - 10, T = 16, B = h - 48, n = E.length, bw = (R - L) / n;
    const svg = svgIn(host, w, h), cur = D.length_of_picks[key[r]], rnd = D.length_of_picks.shown_at_random;
    const my = niceMax(Math.max(...Object.values(D.length_of_picks).flat()) * 1.05), y = lin(0, my, B, T);
    gridY(svg, y, ticksTo(my), L, R, v => fmt(100 * v, 0) + "%");
    const prev = bars; bars = [];
    cur.forEach((v, i) => {
      const x0 = L + i * bw + 3, b = el("rect", { x: x0, y: anim ? (prev[i] ? prev[i][0] : B) : y(v), width: bw - 6, height: anim ? (prev[i] ? prev[i][1] : 0) : B - y(v), rx: 3,
        class: r === "completion" ? "f-orange" : r === "long_view" ? "f-navy" : "f-blue" }, svg);
      if (anim) animate([[b, { y: y(v), height: B - y(v) }]], 700);
      bars.push([y(v), B - y(v)]);
      el("rect", { x: x0, y: y(rnd[i]), width: bw - 6, height: B - y(rnd[i]), fill: "none", stroke: css("--ink2"), "stroke-width": 1, "stroke-dasharray": "3 2", rx: 3 }, svg);
      const hot = el("rect", { x: L + i * bw, y: T, width: bw, height: B - T, fill: "transparent" }, svg);
      hoverTip(hot, `${i === 0 ? "length recorded as zero" : lab[i] + " seconds"}<br>picks: <b>${pct(v)}</b><br>shown at random: ${pct(rnd[i])}`);
      if (w >= 640 || i % 2 === 0) txt(svg, L + i * bw + bw / 2, B + 16, lab[i], { "text-anchor": "middle", "font-size": 10.5 });
    });
    txt(svg, (L + R) / 2, B + 38, "Recorded video length (seconds; none = recorded as zero)", { "text-anchor": "middle", class: "t-muted", "font-size": 11 });
    const v = D.step3[r], base = D.step3.shown_at_random;
    ro.innerHTML = [["Seconds watched per exposure", v.watch_seconds.est, base.watch_seconds.est, x => fmt(x, 1) + " s"],
      ["Long-view rate", v.long_view.est, base.long_view.est, x => pct(x)], ["Share of the video watched", v.completion.est, base.completion.est, x => pct(x, 0)],
      ["Mean length of the picks", v.duration.est, base.duration.est, x => fmt(x, 0) + " s"]]
      .map(([k, a, b, f]) => `<div><div class="k">${k}</div><div class="v">${f(a)} <small>at random ${f(b)}</small></div></div>`).join("")
      + `<div><div class="k">Picks with no recorded length</div><div class="v">${pct(D.sensitivity.zero_top[r], 0)} <small>at random ${pct(D.sensitivity.zero_share, 0)}</small></div></div>`;
  }
  segControl($("#seg-reward"), v => { r = v; draw(true); }, r);
  whenVisible(host, () => draw(true)); draw(false); redrawers.push(() => draw(false));
})();

/* ---------- 3b: trade-off paths ---------- */
(() => {
  const host = $("#c-trade"), rg = $("#rg-trade"), out = $("#out-trade");
  const K = [["reward_completion", "Share watched", "--orange"], ["transformer", "Long view", "--navy"], ["reward_watch_time", "Seconds watched", "--blue"]];
  function draw(anim) {
    const w = W(host), h = Math.min(400, Math.max(300, w * .5)), L = 56, R = w - 24, T = 16, B = h - 46;
    const svg = svgIn(host, w, h), C = D.curves;
    const xs = K.flatMap(([k]) => C[k].watch_seconds), ys = K.flatMap(([k]) => C[k].long_view);
    const x = lin(0, niceMax(Math.max(...xs) * 1.05), L, R), y = lin(0, niceMax(Math.max(...ys) * 1.08), B, T);
    gridY(svg, y, ticksTo(niceMax(Math.max(...ys) * 1.08)), L, R, v => fmt(100 * v, 0) + "%");
    axisX(svg, x, ticksTo(niceMax(Math.max(...xs) * 1.05), 5), B, v => fmt(v, 0), "Seconds watched per exposure");
    txt(svg, L, T - 4, "Long-view rate", { class: "t-muted", "font-size": 11 });
    const rx = x(D.random.watch_seconds), ry = y(D.random.long_view);
    el("circle", { cx: rx, cy: ry, r: 6, fill: css("--grey") }, svg); txt(svg, rx + 9, ry + 15, "at random (100%)", { class: "t-muted halo", "font-size": 11 });
    const i = +rg.value;
    K.forEach(([k, lab, col]) => {
      const pts = SH.map((s, j) => [x(C[k].watch_seconds[j]), y(C[k].long_view[j])]);
      const p = el("path", { d: path(pts), fill: "none", stroke: css(col), "stroke-width": 2, opacity: .55 }, svg);
      if (anim) drawIn(p, host, 1.6);
      pts.forEach(([px, py], j) => el("circle", { cx: px, cy: py, r: 2.5, fill: css(col), opacity: .6 }, svg));
      const [cx, cy] = pts[i], c = el("circle", { cx, cy, r: 8, fill: css(col), class: "s-card", "stroke-width": 2.5 }, svg);
      txt(svg, cx + 11, cy - 8, lab, { class: "halo", "font-size": 12, "font-weight": 650, fill: css(col) });
      hoverTip(c, `Rewarded for ${lab.toLowerCase()}, top ${shareLab(SH[i])}<br>${fmt(C[k].watch_seconds[i], 1)} s watched, ${pct(C[k].long_view[i])} long views<br>mean length ${fmt(C[k].duration[i], 0)} s`);
    });
  }
  const upd = () => { out.textContent = "top " + shareLab(SH[+rg.value]); draw(false); };
  rg.addEventListener("input", upd); out.textContent = "top " + shareLab(SH[+rg.value]);
  draw(true); redrawers.push(() => draw(false));
})();

/* ---------- 4a: AUC of new-video scores ---------- */
(() => {
  const host = $("#c-auc"), A = D.step4.auc;
  const K = [["metadata", "Metadata (trees)", "--navy"], ["metadata_frozen_text", "Metadata + length-and-caption embedding", "--blue"],
             ["metadata_finetuned_text", "Metadata + fine-tuned model", "--blue"], ["zero_shot_judge", "Language-model judge, yes or no", "--orange"],
             ["zero_shot_judge_rating", "Language-model judge, rating", "--orange"], ["reference_rate_from_other_viewers", "Reference: rate among other viewers", "--grey"]];
  function draw(anim) {
    const w = W(host), rowH = 34, h = K.length * rowH + 48, L = w < 560 ? 12 : 250, R = w - 64, T = 10;
    const svg = svgIn(host, w, h), x = lin(0.45, 0.72, L, R);
    [0.5, 0.55, 0.6, 0.65, 0.7].forEach(v => { el("line", { x1: x(v), x2: x(v), y1: T, y2: h - 34, class: "gridl" }, svg); txt(svg, x(v), h - 18, fmt(v, 2), { "text-anchor": "middle" }); });
    el("line", { x1: x(0.5), x2: x(0.5), y1: T, y2: h - 34, class: "s-orange", "stroke-dasharray": "4 3" }, svg);
    K.forEach(([k, lab, col], i) => {
      const v = A[k], yy = T + i * rowH + 16;
      if (w >= 560) txt(svg, L - 12, yy + 4, lab, { "text-anchor": "end", class: "t-ink" }); else txt(svg, L, yy - 9, lab, { "font-size": 11 });
      const ln = el("line", { x1: x(v.lo), x2: anim ? x(v.lo) : x(v.hi), y1: yy, y2: yy, stroke: css(col), "stroke-width": 4, "stroke-linecap": "round", opacity: .45 }, svg);
      const c = el("circle", { cx: anim ? x(0.45) : x(v.est), cy: yy, r: 6.5, fill: css(col), class: "s-card", "stroke-width": 2 }, svg);
      txt(svg, x(v.hi) + 8, yy + 4, fmt(v.est, 3), { class: "t-strong" });
      if (anim) whenVisible(host, () => setTimeout(() => animate([[c, { cx: x(v.est) }], [ln, { x2: x(v.hi) }]], 800), 90 * i));
      hoverTip(c, `${lab}<br>within-user AUC ${fmt(v.est, 3)} (95% CI ${fmt(v.lo, 3)}–${fmt(v.hi, 3)})`);
    });
  }
  draw(true); redrawers.push(() => draw(false));
})();

/* ---------- 4b: videos by score, and deciles ---------- */
(() => {
  const host = $("#c-scatter"), dh = $("#c-decile"), lgd = $("#lg-decile"), S = D.step4.scatter;
  let by = "metadata", dots = [];
  const sqrt = v => Math.sqrt(v);
  function draw(anim) {
    const w = W(host), h = Math.min(360, Math.max(280, w * .8)), L = 44, R = w - 10, T = 12, B = h - 44;
    const svg = svgIn(host, w, h), ymax = 0.6, x = lin(0, 1, L, R), y = v => B - (B - T) * sqrt(Math.min(v, ymax) / ymax);
    [0, 0.05, 0.1, 0.2, 0.4, 0.6].forEach(v => { el("line", { x1: L, x2: R, y1: y(v), y2: y(v), class: "gridl" }, svg); txt(svg, L - 6, y(v) + 4, fmt(100 * v, 0) + "%", { "text-anchor": "end", "font-size": 11 }); });
    axisX(svg, x, [0, 0.25, 0.5, 0.75, 1], B, v => fmt(100 * v, 0), (by === "metadata" ? "Metadata model" : "Language-model judge") + ": score percentile");
    const col = by === "metadata" ? css("--navy") : css("--orange"), xs = S[by], prev = dots; dots = [];
    const g = el("g", {}, svg);
    S.observed.forEach((o, i) => {
      const r = 1.2 + Math.sqrt(S.exposures[i]) / 7, cx = x(xs[i]), cy = y(o);
      const c = el("circle", { cx: anim && prev[i] ? prev[i] : cx, cy, r, fill: col, "fill-opacity": .28 }, g);
      if (anim && prev[i] !== undefined) animate([[c, { cx }]], 900);
      dots.push(cx);
    });
    // weighted mean by tenth, drawn on top
    const dec = D.step4.by_decile[by === "metadata" ? "metadata" : "zero_shot_judge"];
    const pts = dec.map((v, i) => [x((i + 0.5) / 10), y(v)]);
    el("path", { d: path(pts), fill: "none", stroke: col, "stroke-width": 3 }, svg);
    pts.forEach(([px, py]) => el("circle", { cx: px, cy: py, r: 4.5, fill: col, class: "s-card", "stroke-width": 1.5 }, svg));
    txt(svg, L, T + 2, "Observed long-view rate (square-root scale)", { class: "t-muted", "font-size": 11 });
    const c = D.step4.corr[by === "metadata" ? "metadata" : "zero_shot_judge"];
    txt(svg, R, B - 8, `correlation with observed rate ${fmt(c, 2)}`, { "text-anchor": "end", class: "halo", "font-size": 12, "font-weight": 650, fill: col });
    drawDec();
  }
  function drawDec() {
    const w = W(dh), h = 260, L = 44, R = w - 10, T = 12, B = h - 40, Dd = D.step4.by_decile;
    const K = [["metadata", "Metadata", "--navy"], ["metadata_frozen_text", "+ caption embedding", "--blue"], ["zero_shot_judge", "Judge, yes or no", "--orange"], ["zero_shot_judge_rating", "Judge, rating", "--grey"]];
    const svg = svgIn(dh, w, h), my = niceMax(Math.max(...K.flatMap(([k]) => Dd[k])) * 1.1), x = lin(0.5, 10.5, L, R), y = lin(0, my, B, T);
    gridY(svg, y, ticksTo(my), L, R, v => fmt(100 * v, 0) + "%");
    axisX(svg, x, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], B, String, "Tenth of the score (1 = lowest)");
    const ov = y(D.step4.overall_rate); el("line", { x1: L, x2: R, y1: ov, y2: ov, class: "s-grey", "stroke-dasharray": "4 4" }, svg);
    K.forEach(([k, lab, col]) => {
      const hl = (by === "metadata" && k === "metadata") || (by === "judge" && k === "zero_shot_judge");
      el("path", { d: path(Dd[k].map((v, i) => [x(i + 1), y(v)])), fill: "none", stroke: css(col), "stroke-width": hl ? 3.2 : 1.6, opacity: hl ? 1 : .6 }, svg);
    });
    lgd.innerHTML = K.map(([k, lab, col]) => `<span><i style="background:${css(col)}"></i>${lab}</span>`).join("");
  }
  segControl($("#seg-cold"), v => { by = v; draw(true); }, by);
  draw(false); redrawers.push(() => draw(false));
})();

/* ---------- nav and theme ---------- */
$("#theme").addEventListener("click", () => {
  const r = document.documentElement, dark = r.dataset.theme ? r.dataset.theme === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
  r.dataset.theme = dark ? "light" : "dark"; redrawers.forEach(f => f());
});
})();
