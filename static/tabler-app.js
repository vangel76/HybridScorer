(() => {
  let state = JSON.parse(document.getElementById("initial-state").textContent);
  const $ = (id) => document.getElementById(id);
  const inputIds = [
    "method", "folder", "model_label", "pos_prompt", "neg_prompt", "pm_segment_mode",
    "ir_prompt", "ir_negative_prompt", "ir_penalty_weight", "llm_model_label",
    "llm_prompt", "llm_backend_id", "llm_shortlist_size", "tagmatch_tags",
    "main_threshold", "aux_threshold", "keep_pm_thresholds",
    "keep_ir_thresholds", "prompt_generator", "generated_prompt",
    "generated_prompt_detail", "export_left_enabled", "export_right_enabled",
    "export_move_enabled", "export_left_name", "export_right_name"
  ];

  const valueOf = (id) => {
    const el = $(id);
    if (!el) return undefined;
    if (el.type === "checkbox") return el.checked;
    if (el.type === "range") return Number(el.value);
    return el.value;
  };
  const collect = () => Object.fromEntries(inputIds.map((id) => [id, valueOf(id)]).filter(([, v]) => v !== undefined));
  const post = async (url, payload) => {
    const res = await fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload || collect()) });
    if (!res.ok) throw new Error(await res.text());
    return res.json();
  };
  const fillSelect = (id, choices, value) => {
    const el = $(id);
    const normalized = (choices || []).map((choice) => Array.isArray(choice) ? { label: choice[0], value: choice[1] } : { label: choice, value: choice });
    if (el.dataset.choices === JSON.stringify(normalized)) {
      el.value = value || el.value;
      return;
    }
    el.dataset.choices = JSON.stringify(normalized);
    el.innerHTML = "";
    for (const choice of normalized) {
      const opt = document.createElement("option");
      opt.value = choice.value;
      opt.textContent = choice.label;
      el.appendChild(opt);
    }
    el.value = value || normalized[0]?.value || "";
  };
  const setInput = (id, value) => {
    const el = $(id);
    if (!el || value === undefined || value === null) return;
    if (el.type === "checkbox") el.checked = Boolean(value);
    else el.value = value;
  };
  const show = (id, visible) => {
    const el = $(id);
    if (el) el.classList.toggle("d-none", !visible);
  };
  const setSlider = (id, cfg) => {
    const el = $(id);
    if (!el || !cfg) return;
    if (Number.isFinite(Number(cfg.minimum))) el.min = cfg.minimum;
    if (Number.isFinite(Number(cfg.maximum))) el.max = cfg.maximum;
    if (Number.isFinite(Number(cfg.value))) el.value = cfg.value;
    show(id, cfg.visible !== false);
  };
  const renderGallery = (id, items) => {
    const root = $(id);
    root.innerHTML = "";
    for (const item of items || []) {
      const fig = document.createElement("figure");
      fig.className = `hy-thumb${item.marked ? " marked" : ""}${item.preview ? " preview" : ""}${item.overridden ? " overridden" : ""}`;
      fig.draggable = true;
      fig.dataset.side = item.side;
      fig.dataset.index = item.index;
      fig.dataset.filename = item.filename;
      fig.title = hoverText(item);
      fig.innerHTML = `<img src="${item.url || ""}" loading="lazy" alt=""><figcaption>${escapeHtml(item.caption || item.filename)}</figcaption>`;
      fig.querySelector("img").addEventListener("error", () => fig.classList.add("image-error"));
      fig.addEventListener("mouseenter", () => { showHistogramHover(item.filename); renderSegmentPills(item.filename); renderTagPills(item.filename); });
      fig.addEventListener("mouseleave", () => { clearHistogramHover(); clearSegmentPills(); clearTagPills(); });
      fig.addEventListener("click", (event) => {
        if (event.shiftKey) {
          selection("mark", item.side, item.index);
          return;
        }
        openZoom(item);
        selection("preview", item.side, item.index);
      });
      fig.addEventListener("dragstart", (event) => event.dataTransfer.setData("application/json", JSON.stringify(item)));
      fig.addEventListener("dragover", (event) => event.preventDefault());
      fig.addEventListener("drop", async (event) => {
        event.preventDefault();
        const src = JSON.parse(event.dataTransfer.getData("application/json") || "{}");
        await selection("drop", item.side, src.index, { source_side: src.side, target_side: item.side, fnames: [src.filename] });
      });
      root.appendChild(fig);
    }
  };
  const hoverText = (item) => {
    const view = state.view || {};
    const parts = [item.filename, item.caption || ""];
    const score = view.score_lookup?.[item.filename];
    if (score) parts.push(`score: ${score.main}${score.neg !== null ? ` / neg ${score.neg}` : ""}`);
    const tags = view.tag_score_lookup?.[item.filename];
    if (tags) parts.push(Object.entries(tags).map(([k, v]) => `${k}: ${(v * 100).toFixed(1)}%`).join("\n"));
    const seg = view.segment_score_lookup?.[item.filename];
    if (seg) parts.push(Object.entries(seg).map(([k, v]) => `${k}: ${Number(v).toFixed(4)}`).join("\n"));
    const negSeg = view.neg_segment_score_lookup?.[item.filename];
    if (negSeg) parts.push(Object.entries(negSeg).map(([k, v]) => `neg ${k}: ${Number(v).toFixed(4)}`).join("\n"));
    return parts.filter(Boolean).join("\n");
  };
  const escapeHtml = (text) => String(text).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "\"": "&quot;", "'": "&#39;" }[c]));
  const openZoom = (item) => {
    $("hy-zoom-img").src = item.url || "";
    $("hy-zoom-caption").textContent = item.caption || item.filename;
    $("hy-zoom-overlay").classList.remove("d-none");
  };
  const closeZoom = () => {
    $("hy-zoom-overlay").classList.add("d-none");
    $("hy-zoom-img").src = "";
  };
  const scoreToColor = (norm) => {
    // pale transparent yellow (0) → bright opaque green (1)
    const h = Math.round(58 + 62 * norm);
    const s = Math.round(70 + 15 * norm);
    const l = Math.round(65 - 22 * norm);
    const a = (0.18 + 0.82 * norm).toFixed(2);
    return `hsla(${h}, ${s}%, ${l}%, ${a})`;
  };
  const renderSegmentPills = (filename) => {
    const root = $("pm-segment-pills");
    const ta = $("pos_prompt");
    if (!root || !ta) return;
    const seg = state.view?.segment_score_lookup?.[filename];
    if (!seg || !$("pm_segment_mode")?.checked) { root.classList.remove("active"); ta.classList.remove("pills-active"); return; }
    const tags = ta.value.split(",").map((t) => t.trim()).filter(Boolean);
    if (!tags.length) { root.classList.remove("active"); ta.classList.remove("pills-active"); return; }
    const scores = tags.map((tag) => seg[tag] ?? seg[tag.toLowerCase()] ?? null);
    const valid = scores.filter((s) => s !== null);
    const lo = valid.length ? Math.min(...valid) : 0;
    const hi = valid.length ? Math.max(...valid) : 1;
    const range = hi - lo || 1;
    root.innerHTML = "";
    for (let i = 0; i < tags.length; i++) {
      const span = document.createElement("span");
      span.className = "hy-seg-pill";
      span.textContent = tags[i];
      if (scores[i] !== null) {
        const norm = (scores[i] - lo) / range;
        span.style.background = scoreToColor(norm);
        span.style.color = norm > 0.55 ? "#061208" : "#8ca0a8";
        span.title = `${(scores[i] * 100).toFixed(1)}%`;
      } else {
        span.style.background = "rgba(80,90,110,0.3)";
        span.style.color = "#6a7888";
      }
      root.appendChild(span);
    }
    root.classList.add("active");
    ta.classList.add("pills-active");
  };
  const clearSegmentPills = () => {
    const root = $("pm-segment-pills");
    const ta = $("pos_prompt");
    if (root) root.classList.remove("active");
    if (ta) ta.classList.remove("pills-active");
  };
  const renderTagPills = (filename) => {
    const root = $("tm-segment-pills");
    const ta = $("tagmatch_tags");
    if (!root || !ta) return;
    const lookup = state.view?.tag_score_lookup?.[filename];
    if (!lookup) { root.classList.remove("active"); ta.classList.remove("pills-active"); return; }
    const tags = ta.value.split(",").map((t) => t.trim()).filter(Boolean);
    if (!tags.length) { root.classList.remove("active"); ta.classList.remove("pills-active"); return; }
    const scores = tags.map((tag) => lookup[tag] ?? lookup[tag.toLowerCase()] ?? null);
    const valid = scores.filter((s) => s !== null);
    const lo = valid.length ? Math.min(...valid) : 0;
    const hi = valid.length ? Math.max(...valid) : 1;
    const range = hi - lo || 1;
    root.innerHTML = "";
    for (let i = 0; i < tags.length; i++) {
      const span = document.createElement("span");
      span.className = "hy-seg-pill";
      span.textContent = tags[i];
      if (scores[i] !== null) {
        const norm = (scores[i] - lo) / range;
        span.style.background = scoreToColor(norm);
        span.style.color = norm > 0.55 ? "#061208" : "#8ca0a8";
        span.title = `${(scores[i] * 100).toFixed(1)}%`;
      } else {
        span.style.background = "rgba(80,90,110,0.3)";
        span.style.color = "#6a7888";
      }
      root.appendChild(span);
    }
    root.classList.add("active");
    ta.classList.add("pills-active");
  };
  const clearTagPills = () => {
    const root = $("tm-segment-pills");
    const ta = $("tagmatch_tags");
    if (root) root.classList.remove("active");
    if (ta) ta.classList.remove("pills-active");
  };
  const positionHistLine = (line, x, y, h) => {
    const img = $("histogram");
    const geom = state.view?.hist_geom;
    if (!img || !geom || !img.naturalWidth || !state.view?.histogram_url) return;
    const scaleX = img.clientWidth / geom.W;
    const scaleY = img.clientHeight / geom.H;
    line.style.left = `${x * scaleX}px`;
    line.style.top = `${y * scaleY}px`;
    line.style.height = `${h * scaleY}px`;
    line.style.display = "block";
  };
  const placeValueLine = (lineId, value, lo, hi, y, h, flip = false) => {
    const geom = state.view?.hist_geom;
    if (!geom || value === undefined || value === null || hi === lo) return;
    const cW = (geom.W || 0) - (geom.PAD_L || 0) - (geom.PAD_R || 0);
    const frac = (Number(value) - lo) / (hi - lo);
    const x = (geom.PAD_L || 0) + (flip ? (1 - frac) : frac) * cW;
    positionHistLine($(lineId), Math.max(geom.PAD_L, Math.min(geom.W - geom.PAD_R, x)), y, h);
  };
  const updateThresholdLines = () => {
    clearThresholdLines();
    const geom = state.view?.hist_geom;
    if (!geom || !$("histogram").complete) return;
    if (geom.pos_lo !== undefined) {
      placeValueLine("hist-threshold-main", valueOf("main_threshold"), geom.pos_lo, geom.pos_hi, geom.PAD_TOP || 0, geom.CH || geom.H || 0, geom.pos_flipped);
      if (geom.has_neg) {
        const y = (geom.PAD_TOP || 0) + (geom.CH || 0) + (geom.GAP || 0);
        placeValueLine("hist-threshold-neg", valueOf("aux_threshold"), geom.neg_lo, geom.neg_hi, y, geom.CH || 0);
      }
    } else if (geom.lo !== undefined) {
      placeValueLine("hist-threshold-main", valueOf("main_threshold"), geom.lo, geom.hi, geom.PAD_TOP || 0, (geom.H || 0) - (geom.PAD_TOP || 0) - (geom.PAD_BOT || 0), geom.flipped);
    }
  };
  const clearThresholdLines = () => {
    for (const id of ["hist-threshold-main", "hist-threshold-neg"]) {
      const line = $(id);
      if (line) line.style.display = "none";
    }
  };
  const showHistogramHover = (filename) => {
    clearHistogramHover();
    const geom = state.view?.hist_geom;
    const score = state.view?.score_lookup?.[filename];
    if (!geom || !score) return;
    const cW = (geom.W || 0) - (geom.PAD_L || 0) - (geom.PAD_R || 0);
    if (geom.pos_lo !== undefined && score.main !== undefined && geom.pos_hi !== geom.pos_lo) {
      const frac = (score.main - geom.pos_lo) / (geom.pos_hi - geom.pos_lo);
      const x = (geom.PAD_L || 0) + (geom.pos_flipped ? (1 - frac) : frac) * cW;
      positionHistLine($("hist-hover-main"), Math.max(geom.PAD_L, Math.min(geom.W - geom.PAD_R, x)), geom.PAD_TOP || 0, geom.CH || geom.H || 0);
    } else if (geom.lo !== undefined && score.main !== undefined && geom.hi !== geom.lo) {
      const frac = (score.main - geom.lo) / (geom.hi - geom.lo);
      const x = (geom.PAD_L || 0) + (geom.flipped ? (1 - frac) : frac) * (geom.cW || cW);
      positionHistLine($("hist-hover-main"), Math.max(geom.PAD_L, Math.min(geom.W - geom.PAD_R, x)), geom.PAD_TOP || 0, (geom.H || 0) - (geom.PAD_TOP || 0) - (geom.PAD_BOT || 0));
    }
    if (score.neg !== null && score.neg !== undefined && geom.has_neg && geom.neg_hi !== geom.neg_lo) {
      const x = (geom.PAD_L || 0) + ((score.neg - geom.neg_lo) / (geom.neg_hi - geom.neg_lo)) * cW;
      const y = (geom.PAD_TOP || 0) + (geom.CH || 0) + (geom.GAP || 0);
      positionHistLine($("hist-hover-neg"), Math.max(geom.PAD_L, Math.min(geom.W - geom.PAD_R, x)), y, geom.CH || 0);
    }
  };
  const clearHistogramHover = () => {
    for (const id of ["hist-hover-main", "hist-hover-neg"]) {
      const line = $(id);
      if (line) line.style.display = "none";
    }
  };
  const render = () => {
    const i = state.inputs || {};
    const c = state.controls || {};
    fillSelect("method", c.choices?.methods, i.method);
    fillSelect("model_label", c.choices?.promptmatch_models, i.model_label);
    fillSelect("llm_model_label", c.choices?.promptmatch_models, i.llm_model_label);
    fillSelect("llm_backend_id", c.choices?.llm_backends, i.llm_backend_id);
    fillSelect("prompt_generator", c.choices?.prompt_generators, i.prompt_generator);
    for (const id of inputIds) setInput(id, i[id]);
    show("group-promptmatch", c.visible?.promptmatch);
    show("group-imagereward", c.visible?.imagereward);
    show("group-llmsearch", c.visible?.llmsearch);
    show("group-tagmatch", c.visible?.tagmatch);
    show("keep-pm-row", c.visible?.keep_pm_thresholds);
    show("keep-ir-row", c.visible?.keep_ir_thresholds);
    setSlider("main_threshold", c.sliders?.main);
    setSlider("aux_threshold", c.sliders?.aux);
    $("main-label").textContent = c.sliders?.main?.label || "Threshold";
    $("aux-label").textContent = c.sliders?.aux?.label || "Aux threshold";
    $("method-note").textContent = c.method_note || "";
    $("status").textContent = state.view?.status || "";
    $("prompt-status").textContent = state.view?.query_status || "";
    $("left-title").textContent = stripMd(state.view?.left?.title || "");
    $("right-title").textContent = stripMd(state.view?.right?.title || "");
    renderGallery("left-gallery", state.view?.left?.items || []);
    renderGallery("right-gallery", state.view?.right?.items || []);
    const hist = $("histogram");
    hist.src = state.view?.histogram_url || "";
    hist.onload = updateThresholdLines;
    clearHistogramHover();
    updateThresholdLines();
    const query = $("query-drop");
    const qimg = $("query-image");
    qimg.src = state.view?.query_image_url || "";
    query.classList.toggle("has-image", Boolean(state.view?.query_image_url));
    $("changelog").textContent = decodeHtml(state.app?.changelog_html || "");
    renderTagSuggestions();
  };
  const stripMd = (text) => String(text).replace(/\*\*/g, "");
  const decodeHtml = (html) => {
    const txt = document.createElement("textarea");
    txt.innerHTML = html;
    return txt.value;
  };
  const runJob = async (url, payload) => {
    const started = await post(url, payload || collect());
    const box = $("job-status");
    const text = $("job-status-text");
    const bar = $("job-progress-bar");
    document.body.classList.add("hy-busy");
    box.classList.remove("d-none");
    text.textContent = `${started.job?.action || "job"}: starting...`;
    bar.style.width = "100%";
    bar.classList.add("progress-bar-indeterminate");
    const updateJobBox = (job) => {
      if (!job) return;
      const pct = Math.round((job.progress || 0) * 100);
      text.textContent = `${job.action}: ${pct}% ${job.message || ""}`;
      bar.classList.toggle("progress-bar-indeterminate", pct <= 0 || pct >= 100);
      bar.style.width = `${Math.max(3, Math.min(100, pct))}%`;
    };
    const finishJob = (msg) => {
      if (msg.state) {
        state = msg.state;
        render();
      }
      if (msg.error) text.textContent = msg.error;
      else text.textContent = "Done.";
      document.body.classList.remove("hy-busy");
      setTimeout(() => box.classList.add("d-none"), msg.error ? 6000 : 1200);
    };
    const pollJob = async () => {
      while (true) {
        const res = await fetch(`/api/jobs/${started.job_id}`);
        if (!res.ok) throw new Error(await res.text());
        const msg = await res.json();
        updateJobBox(msg.job);
        if (msg.job?.status === "done" || msg.job?.status === "error") {
          finishJob(msg);
          return;
        }
        await new Promise((resolve) => setTimeout(resolve, 500));
      }
    };
    let opened = false;
    let seenMessage = false;
    let fallbackStarted = false;
    const startPollingFallback = () => {
      if (fallbackStarted) return;
      fallbackStarted = true;
      pollJob().catch((error) => {
        text.textContent = String(error);
        document.body.classList.remove("hy-busy");
      });
    };
    const wsProto = location.protocol === "https:" ? "wss" : "ws";
    const ws = new WebSocket(`${wsProto}://${location.host}/ws/jobs/${started.job_id}`);
    ws.onopen = () => { opened = true; };
    ws.onmessage = (event) => {
      seenMessage = true;
      const msg = JSON.parse(event.data);
      updateJobBox(msg.job);
      if (msg.type === "done" || msg.type === "error") {
        finishJob({ state: msg.state, error: msg.type === "error" ? (msg.error || "Job failed") : null });
        ws.close();
      }
    };
    ws.onerror = () => {
      try { ws.close(); } catch {}
      startPollingFallback();
    };
    ws.onclose = () => {
      if (!opened) startPollingFallback();
    };
    setTimeout(() => {
      if (!seenMessage && !fallbackStarted) startPollingFallback();
    }, 1200);
  };
  const refreshControls = async () => {
    state = await post("/api/controls", collect());
    render();
  };
  const selection = async (action, side, index, extra = {}) => {
    state = await post("/api/selection", { ...collect(), action, side, index, ...extra });
    render();
  };
  const uploadFile = async (file) => {
    if (!file) return;
    const fd = new FormData();
    fd.append("image", file);
    const res = await fetch("/api/query-image", { method: "POST", body: fd });
    state = await res.json();
    render();
  };

  $("method").addEventListener("change", refreshControls);
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") closeZoom();
  });
  $("hy-zoom-overlay").addEventListener("click", (event) => {
    if (event.target === $("hy-zoom-overlay") || event.target.classList.contains("hy-zoom-backdrop")) closeZoom();
  });
  $("tagmatch_tags").addEventListener("input", renderTagSuggestions);
  $("load-folder").addEventListener("click", () => runJob("/api/folder/load"));
  $("run-score").addEventListener("click", () => runJob("/api/score"));
  $("find-similar").addEventListener("click", () => runJob("/api/search/similar"));
  $("find-same").addEventListener("click", () => runJob("/api/search/same-person"));
  $("find-object").addEventListener("click", () => runJob("/api/search/object"));
  $("main_threshold").addEventListener("input", updateThresholdLines);
  $("aux_threshold").addEventListener("input", updateThresholdLines);
  $("main_threshold").addEventListener("change", async () => { state = await post("/api/thresholds", { ...collect(), action: "split" }); render(); });
  $("aux_threshold").addEventListener("change", async () => { state = await post("/api/thresholds", { ...collect(), action: "split" }); render(); });
  $("histogram").addEventListener("click", async (event) => {
    const rect = event.currentTarget.getBoundingClientRect();
    const x = Math.round((event.clientX - rect.left) * (event.currentTarget.naturalWidth / rect.width));
    const y = Math.round((event.clientY - rect.top) * (event.currentTarget.naturalHeight / rect.height));
    state = await post("/api/thresholds", { ...collect(), action: "hist", x, y });
    render();
  });
  $("thumb-size").addEventListener("input", () => {
    const v = $("thumb-size").value;
    document.documentElement.style.setProperty("--thumb-size", `${v}px`);
    $("thumb-size-val").textContent = `${v}px`;
  });
  $("move-left").addEventListener("click", () => selection("move-left", "", -1));
  $("move-right").addEventListener("click", () => selection("move-right", "", -1));
  for (const [galleryId, targetSide] of [["left-gallery", "left"], ["right-gallery", "right"]]) {
    const el = $(galleryId);
    el.addEventListener("dragover", (event) => {
      if (!event.dataTransfer.types.includes("application/json")) return;
      event.preventDefault();
      el.classList.add("drag-over");
    });
    el.addEventListener("dragleave", (event) => {
      if (!el.contains(event.relatedTarget)) el.classList.remove("drag-over");
    });
    el.addEventListener("drop", async (event) => {
      el.classList.remove("drag-over");
      if (event.target.closest(".hy-thumb")) return;
      event.preventDefault();
      const src = JSON.parse(event.dataTransfer.getData("application/json") || "{}");
      if (!src.filename || src.side === targetSide) return;
      await selection("drop", targetSide, src.index, { source_side: src.side, target_side: targetSide, fnames: [src.filename] });
    });
  }
  $("pin").addEventListener("click", () => selection("pin", "", -1));
  $("fit-threshold").addEventListener("click", () => selection("fit-threshold", "", -1));
  $("clear-marked").addEventListener("click", () => selection("clear-marked", "", -1));
  $("clear-all").addEventListener("click", () => selection("clear-all", "", -1));
  $("query-file").addEventListener("change", (event) => uploadFile(event.target.files[0]));
  $("clear-query").addEventListener("click", async () => { const res = await fetch("/api/query-image", { method: "DELETE" }); state = await res.json(); render(); });
  $("generate-prompt").addEventListener("click", () => runJob("/api/prompt/generate"));
  $("insert-prompt").addEventListener("click", async () => { state = await post("/api/prompt/insert", collect()); render(); });
  $("generated_prompt_detail").addEventListener("change", async () => { state = await post("/api/prompt/detail", collect()); render(); });
  $("export").addEventListener("click", () => runJob("/api/export"));
  document.addEventListener("paste", (event) => {
    const file = Array.from(event.clipboardData?.items || []).find((i) => i.type.startsWith("image/"))?.getAsFile();
    if (file) uploadFile(file);
  });
  $("query-drop").addEventListener("dragover", (event) => event.preventDefault());
  $("query-drop").addEventListener("drop", (event) => {
    event.preventDefault();
    uploadFile(event.dataTransfer.files[0]);
  });
  for (const id of ["pos_prompt", "neg_prompt"]) {
    $(id).addEventListener("keydown", (event) => {
      if (!(event.ctrlKey || event.metaKey) || !["ArrowUp", "ArrowDown"].includes(event.key)) return;
      event.preventDefault();
      const el = event.currentTarget;
      const s = el.selectionStart, e = el.selectionEnd;
      if (s === e) return;
      const text = el.value.slice(s, e);
      const weight = event.key === "ArrowUp" ? "1.1" : "0.9";
      el.setRangeText(`(${text}:${weight})`, s, e, "select");
    });
  }
  render();
  function renderTagSuggestions() {
    const root = $("tag-suggestions");
    if (!root) return;
    root.innerHTML = "";
    const vocab = state.controls?.tagmatch_vocab || [];
    if (!vocab.length) return;
    const value = $("tagmatch_tags").value || "";
    const term = value.split(",").pop().trim().toLowerCase();
    if (!term) return;
    for (const tag of vocab.filter((t) => t.includes(term)).slice(0, 12)) {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.textContent = tag;
      btn.addEventListener("click", () => {
        const prefix = value.includes(",") ? value.slice(0, value.lastIndexOf(",") + 1).trimEnd() + " " : "";
        $("tagmatch_tags").value = `${prefix}${tag}, `;
        root.innerHTML = "";
      });
      root.appendChild(btn);
    }
  }
})();
