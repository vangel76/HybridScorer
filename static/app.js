(() => {
  const tooltips = __TOOLTIPS__;
  // Hidden text inputs are used as tiny event bridges from custom JS back into Gradio callbacks.
  const pushThumbAction = (value) => {
    const root = document.getElementById("hy-thumb-action");
    if (!root) return;
    const input = root.querySelector("input, textarea");
    if (!input) return;
    input.value = value;
    input.dispatchEvent(new Event("input", { bubbles: true }));
    input.dispatchEvent(new Event("change", { bubbles: true }));
  };
  const pushShortcutAction = (value) => {
    const root = document.getElementById("hy-shortcut-action");
    if (!root) return;
    const input = root.querySelector("input, textarea");
    if (!input) return;
    input.value = value;
    input.dispatchEvent(new Event("input", { bubbles: true }));
    input.dispatchEvent(new Event("change", { bubbles: true }));
  };
  const pushHistWidth = (value) => {
    const root = document.getElementById("hy-hist-width");
    if (!root) return;
    const input = root.querySelector("input, textarea");
    if (!input) return;
    const normalized = String(value);
    if (input.value === normalized) return;
    input.value = normalized;
    input.dispatchEvent(new Event("input", { bubbles: true }));
    input.dispatchEvent(new Event("change", { bubbles: true }));
  };
  const pushPreviewFname = (fname) => {
    if (!fname) return;
    pushThumbAction(`previewfname:${fname}:${Date.now()}`);
  };
  const readMarkedState = () => {
    const root = document.getElementById("hy-mark-state");
    if (!root) return { left: [], right: [] };
    const input = root.querySelector("input, textarea");
    if (!input || !input.value) return { left: [], right: [] };
    try {
      return JSON.parse(input.value);
    } catch {
      return { left: [], right: [] };
    }
  };
  const readPromptMatchModelStatus = () => {
    const root = document.getElementById("hy-model-status");
    if (!root) return {};
    const input = root.querySelector("input, textarea");
    if (!input || !input.value) return {};
    try {
      return JSON.parse(input.value);
    } catch {
      return {};
    }
  };
  const readTagMatchVocabulary = () => {
    const root = document.getElementById("hy-tagmatch-vocab");
    if (!root) return [];
    const input = root.querySelector("input, textarea");
    if (!input || !input.value) return [];
    try {
      const parsed = JSON.parse(input.value);
      return Array.isArray(parsed) ? parsed : [];
    } catch {
      return [];
    }
  };
  const paintPromptMatchModelNode = (node, entry, colors) => {
    if (!(node instanceof HTMLElement) || !entry) return;
    const color = entry.cached ? colors.cached : colors.download;
    node.style.setProperty("color", color, "important");
    node.style.setProperty("-webkit-text-fill-color", color, "important");
    node.style.setProperty("font-weight", entry.cached ? "600" : "500", "important");
  };
  const findPromptMatchModelStatusEntry = (text, statusMap, knownLabels) => {
    const raw = `${text || ""}`.trim();
    if (!raw) return null;
    const normalized = raw.replace(/\\s+/g, " ").trim();
    if (statusMap[normalized]) return statusMap[normalized];
    let bestLabel = "";
    for (const label of knownLabels) {
      if (normalized.includes(label) && label.length > bestLabel.length) {
        bestLabel = label;
      }
    }
    return bestLabel ? statusMap[bestLabel] : null;
  };
  const schedulePromptMatchModelAvailability = () => {
    applyPromptMatchModelAvailability();
    requestAnimationFrame(() => applyPromptMatchModelAvailability());
    setTimeout(applyPromptMatchModelAvailability, 0);
    setTimeout(applyPromptMatchModelAvailability, 60);
    setTimeout(applyPromptMatchModelAvailability, 180);
  };
  let repaintTimers = [];
  let activeDrag = null;
  let histResizeObserver = null;
  let activeHoverInfo = null;
  let activeDialogPreviewFname = "";
  let activeDialogSelection = { side: "", index: -1 };
  const findWeightedPromptSpan = (value, selectionStart, selectionEnd) => {
    const weightedRe = /\\(([^()]*?)\\s*:\\s*([0-9]*\\.?[0-9]+)\\)/g;
    let match;
    while ((match = weightedRe.exec(value)) !== null) {
      const fullStart = match.index;
      const fullEnd = fullStart + match[0].length;
      const hasSelection = selectionStart !== selectionEnd;
      const insideMatch = hasSelection
        ? (selectionStart >= fullStart && selectionEnd <= fullEnd)
        : (selectionStart >= fullStart && selectionStart <= fullEnd);
      if (!insideMatch) continue;
      return {
        fullStart,
        fullEnd,
        text: match[1],
        weight: Number.parseFloat(match[2]),
      };
    }
    return null;
  };
  const formatPromptWeight = (weight) => {
    const rounded = Math.round(weight * 10) / 10;
    return rounded.toFixed(1);
  };
  const dispatchTextboxEvents = (input) => {
    input.dispatchEvent(new Event("input", { bubbles: true }));
    input.dispatchEvent(new Event("change", { bubbles: true }));
  };
  const externalQueryBridgeInput = () => {
    const root = document.getElementById("hy-external-query-bridge");
    return root ? root.querySelector("input, textarea") : null;
  };
  const externalQueryButtons = () => {
    const root = document.getElementById("hy-external-query-image");
    return root ? Array.from(root.querySelectorAll("button")) : [];
  };
  const externalQueryRemoveButton = () => {
    for (const button of externalQueryButtons()) {
      const aria = `${button.getAttribute("aria-label") || ""} ${button.title || ""} ${button.textContent || ""}`.toLowerCase();
      if (aria.includes("remove") || aria.includes("clear")) return button;
    }
    return null;
  };
  const externalQueryClipboardButtons = () => {
    return externalQueryButtons().filter((button) => {
      const aria = `${button.getAttribute("aria-label") || ""} ${button.title || ""} ${button.textContent || ""}`.toLowerCase();
      return aria.includes("clipboard") || aria.includes("paste");
    });
  };
  const clearExternalQueryWidget = () => {
    const button = externalQueryRemoveButton();
    if (!button) return false;
    button.click();
    return true;
  };
  const blobToDataUrl = (blob) => new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result || ""));
    reader.onerror = () => reject(reader.error || new Error("Failed to read image blob."));
    reader.readAsDataURL(blob);
  });
  const pushExternalQueryPayload = (payload) => {
    const input = externalQueryBridgeInput();
    if (!input) return false;
    input.value = JSON.stringify({ ...payload, ts: Date.now() });
    dispatchTextboxEvents(input);
    return true;
  };
  const replaceExternalQueryFromBlob = async (blob, label) => {
    if (!blob) return false;
    try {
      clearExternalQueryWidget();
      const dataUrl = await blobToDataUrl(blob);
      return pushExternalQueryPayload({
        data_url: dataUrl,
        label: label || (blob.type && blob.type.includes("png") ? "clipboard-image.png" : "query-image"),
      });
    } catch (error) {
      console.warn("External query image import failed", error);
      return false;
    }
  };
  const clipboardImageBlobFromEvent = (event) => {
    const items = Array.from(event?.clipboardData?.items || []);
    for (const item of items) {
      if (item && item.kind === "file" && String(item.type || "").startsWith("image/")) {
        return item.getAsFile();
      }
    }
    return null;
  };
  const clipboardImageBlobFromNavigator = async () => {
    if (!navigator.clipboard || typeof navigator.clipboard.read !== "function") {
      throw new Error("Clipboard image read is unavailable in this browser.");
    }
    const items = await navigator.clipboard.read();
    for (const item of items) {
      for (const type of item.types || []) {
        if (String(type).startsWith("image/")) {
          return await item.getType(type);
        }
      }
    }
    throw new Error("Clipboard does not currently contain an image.");
  };
  const importExternalQueryFromClipboard = async () => {
    const blob = await clipboardImageBlobFromNavigator();
    const ext = (blob.type || "image/png").split("/")[1] || "png";
    return replaceExternalQueryFromBlob(blob, `clipboard-image.${ext}`);
  };
  const getPromptRootForElement = (element) => {
    if (!element || typeof element.closest !== "function") return null;
    return element.closest("#hy-pos, #hy-neg, #hy-ir-pos, #hy-ir-neg");
  };
  const adjustPromptWeight = (input, delta) => {
    if (!input) return false;
    const value = input.value || "";
    const selectionStart = input.selectionStart ?? 0;
    const selectionEnd = input.selectionEnd ?? selectionStart;
    const weighted = findWeightedPromptSpan(value, selectionStart, selectionEnd);
    if (weighted) {
      const newWeight = Math.max(0.1, (weighted.weight || 1.0) + delta);
      const replacement = Math.abs(newWeight - 1.0) < 1e-9
        ? weighted.text
        : `(${weighted.text}:${formatPromptWeight(newWeight)})`;
      input.value = value.slice(0, weighted.fullStart) + replacement + value.slice(weighted.fullEnd);
      const innerStart = weighted.fullStart + (replacement === weighted.text ? 0 : 1);
      const innerEnd = innerStart + weighted.text.length;
      input.setSelectionRange(innerStart, innerEnd);
      dispatchTextboxEvents(input);
      return true;
    }
    if (selectionStart === selectionEnd) return false;
    const selectedText = value.slice(selectionStart, selectionEnd);
    const leadingWs = (selectedText.match(/^\\s*/) || [""])[0];
    const trailingWs = (selectedText.match(/\\s*$/) || [""])[0];
    const coreText = selectedText.slice(leadingWs.length, selectedText.length - trailingWs.length);
    if (!coreText) return false;
    const baseWeight = delta >= 0 ? 1.1 : 0.9;
    const replacement = `${leadingWs}(${coreText}:${formatPromptWeight(baseWeight)})${trailingWs}`;
    input.value = value.slice(0, selectionStart) + replacement + value.slice(selectionEnd);
    const innerStart = selectionStart + leadingWs.length + 1;
    const innerEnd = innerStart + coreText.length;
    input.setSelectionRange(innerStart, innerEnd);
    dispatchTextboxEvents(input);
    return true;
  };
  const hookPromptWeightHotkeys = () => {
    for (const id of ["hy-pos", "hy-neg"]) {
      const root = document.getElementById(id);
      if (!root) continue;
      const input = root.querySelector("input, textarea");
      if (!input || input.dataset.hyWeightHooked) continue;
      input.addEventListener("keydown", (event) => {
        if (!(event.ctrlKey || event.metaKey) || event.altKey) return;
        let delta = null;
        const key = event.key || "";
        const code = event.code || "";
        if (key === "+" || key === "=" || code === "NumpadAdd") {
          delta = 0.1;
        } else if (key === "-" || key === "_" || code === "NumpadSubtract") {
          delta = -0.1;
        }
        if (delta === null) return;
        if (!adjustPromptWeight(input, delta)) return;
        event.preventDefault();
        event.stopPropagation();
      });
      input.dataset.hyWeightHooked = "1";
    }
  };
  const hookRunScoringHotkeys = () => {
    if (document.body.dataset.hyRunHotkeyHooked) return;
    document.addEventListener("keydown", (event) => {
      if (!(event.ctrlKey || event.metaKey) || event.altKey) return;
      const key = event.key || "";
      const code = event.code || "";
      if (key !== "Enter" && code !== "NumpadEnter") return;
      const promptRoot = getPromptRootForElement(event.target) || getPromptRootForElement(document.activeElement);
      const promptId = promptRoot ? promptRoot.id : "";
      if (!promptId) return;
      pushShortcutAction(`run:${promptId}:${Date.now()}`);
      event.preventDefault();
      event.stopPropagation();
    }, true);
    document.body.dataset.hyRunHotkeyHooked = "1";
  };
  const hookPreviewDialogTracking = () => {
    if (document.body.dataset.hyPreviewTrackHooked) return;
    document.addEventListener("click", (event) => {
      const target = event.target;
      if (!(target instanceof Element)) return;
      const dialogRoot = target.closest('[role="dialog"], [aria-modal="true"]');
      if (!dialogRoot) return;
      disablePreviewDialogNavigation(dialogRoot);
      if (isDialogMainImageClick(event, dialogRoot)) {
        event.preventDefault();
        event.stopPropagation();
        event.stopImmediatePropagation();
        closePreviewDialog(dialogRoot);
        return;
      }
      if (shouldBlockDialogNavigationClick(target, dialogRoot)) {
        event.preventDefault();
        event.stopPropagation();
        event.stopImmediatePropagation();
        return;
      }
      const thumbInfo = getDialogThumbTargetInfo(target);
      if (thumbInfo) {
        setTimeout(syncDialogPreviewTarget, 40);
        setTimeout(syncDialogPreviewTarget, 140);
        setTimeout(syncDialogPreviewTarget, 280);
        return;
      }
      setTimeout(syncDialogPreviewTarget, 40);
      setTimeout(syncDialogPreviewTarget, 140);
    }, true);
    document.addEventListener("keydown", (event) => {
      const dialogRoot = document.querySelector('[role="dialog"], [aria-modal="true"]');
      if (!dialogRoot) return;
      const key = event.key || "";
      const deltas = {
        ArrowLeft: -1,
        ArrowUp: -1,
        PageUp: -1,
        ArrowRight: 1,
        ArrowDown: 1,
        PageDown: 1,
        Home: 0,
        End: 0,
      };
      if (!(key in deltas)) return;
      disablePreviewDialogNavigation(dialogRoot);
      event.preventDefault();
      event.stopPropagation();
      event.stopImmediatePropagation();
    }, true);
    document.body.dataset.hyPreviewTrackHooked = "1";
  };
  const hookInlinePreviewNavigationLock = () => {
    if (document.body.dataset.hyInlinePreviewLockHooked) return;
    document.addEventListener("click", (event) => {
      const target = event.target;
      if (!(target instanceof Element)) return;
      const previewRoot = target.closest("#hy-left-gallery .preview, #hy-right-gallery .preview");
      if (!previewRoot) return;
      if (event.button !== 0) return;
      if (target.closest(".thumbnails, .thumbnail-item, .thumbnail-small, .thumbnail-lg")) return;
      const control = target.closest("button, a, input, textarea, select, label");
      if (control && !control.closest(".media-button")) return;
      const closeButton = findInlinePreviewCloseButton(previewRoot);
      if (!closeButton) return;
      event.preventDefault();
      event.stopPropagation();
      event.stopImmediatePropagation();
      closeButton.click();
    }, true);
    document.addEventListener("keydown", (event) => {
      const key = event.key || "";
      if (!["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown", "PageUp", "PageDown", "Home", "End"].includes(key)) return;
      const target = event.target instanceof Element ? event.target : null;
      const active = document.activeElement instanceof Element ? document.activeElement : null;
      const inPreview = !!(
        target?.closest?.("#hy-left-gallery .preview, #hy-right-gallery .preview")
        || active?.closest?.("#hy-left-gallery .preview, #hy-right-gallery .preview")
      );
      if (!inPreview) return;
      event.preventDefault();
      event.stopPropagation();
      event.stopImmediatePropagation();
    }, true);
    document.body.dataset.hyInlinePreviewLockHooked = "1";
  };
  const hookHistogramResize = () => {
    const root = document.getElementById("hy-hist");
    if (!root) return;
    const measure = () => {
      const rect = root.getBoundingClientRect();
      const width = Math.max(220, Math.round(rect.width || root.clientWidth || 300) - 2);
      pushHistWidth(width);
      syncHistogramHoverLine();
    };
    if (!histResizeObserver) {
      histResizeObserver = new ResizeObserver(() => {
        window.requestAnimationFrame(measure);
      });
    }
    if (root.dataset.hyResizeHooked !== "1") {
      histResizeObserver.observe(root);
      root.dataset.hyResizeHooked = "1";
    }
    measure();
  };
  const parseMainScoreFromCaption = (captionText) => {
    const cleaned = (captionText || "").replace(/^✋\s*/, "").trim();
    const match = cleaned.match(/^-?\d+(?:\.\d+)?/);
    return match ? Number.parseFloat(match[0]) : null;
  };
  const extractFnameFromCaption = (captionText) => {
    const text = (captionText || "").trim();
    if (!text) return "";
    const parts = text.split("|");
    return parts.length ? parts[parts.length - 1].trim() : "";
  };
  const resolveImageSourceTokens = (src) => {
    const tokens = new Set();
    if (!src) return tokens;
    const raw = String(src);
    const decoded = (() => {
      try {
        return decodeURIComponent(raw);
      } catch {
        return raw;
      }
    })();
    for (const candidate of [raw, decoded]) {
      if (!candidate) continue;
      tokens.add(candidate);
      for (const part of candidate.split(/[/?#&=]+/)) {
        const clean = (part || "").trim();
        if (clean) tokens.add(clean);
      }
      const slashParts = candidate.split("/");
      const tail = slashParts.length ? slashParts[slashParts.length - 1] : "";
      if (tail) tokens.add(tail);
    }
    return tokens;
  };
  const extractFnameFromDialogImage = (img, markedState) => {
    if (!img) return "";
    const direct = extractFnameFromCaption(
      img.getAttribute("alt")
      || img.getAttribute("aria-label")
      || img.getAttribute("title")
      || ""
    );
    if (direct) return direct;
    const mediaLookup = markedState.media_lookup || {};
    const sourceTokens = new Set([
      ...resolveImageSourceTokens(img.currentSrc || ""),
      ...resolveImageSourceTokens(img.src || ""),
    ]);
    for (const token of sourceTokens) {
      if (mediaLookup[token]) return mediaLookup[token];
    }
    return "";
  };
  const extractFnameFromDialogText = (dialogRoot, markedState) => {
    if (!dialogRoot) return "";
    const knownNames = [
      ...(Array.isArray(markedState.left_order) ? markedState.left_order : []),
      ...(Array.isArray(markedState.right_order) ? markedState.right_order : []),
    ];
    if (!knownNames.length) return "";
    const sortedNames = knownNames.slice().sort((a, b) => b.length - a.length);
    const dialogRect = dialogRoot.getBoundingClientRect();
    const thumbs = getDialogThumbImages(dialogRoot);
    const stripTop = thumbs.length
      ? Math.min(...thumbs.map((img) => img.getBoundingClientRect().top))
      : Number.POSITIVE_INFINITY;
    let bestFname = "";
    let bestScore = -Infinity;
    for (const node of Array.from(dialogRoot.querySelectorAll("div, span, p, figcaption, label"))) {
      if (!(node instanceof Element)) continue;
      if (node.closest("button, [role='button']")) continue;
      if (node.querySelector("img")) continue;
      const rect = node.getBoundingClientRect();
      const style = window.getComputedStyle(node);
      if (
        rect.width <= 0
        || rect.height <= 0
        || style.display === "none"
        || style.visibility === "hidden"
        || style.opacity === "0"
      ) continue;
      const text = (node.textContent || "").trim();
      if (!text || text.length > 300) continue;
      let fname = "";
      const parsed = extractFnameFromCaption(text);
      if (parsed && knownNames.includes(parsed)) {
        fname = parsed;
      } else {
        for (const candidate of sortedNames) {
          if (text.includes(candidate)) {
            fname = candidate;
            break;
          }
        }
      }
      if (!fname) continue;
      const centerX = rect.left + (rect.width / 2);
      const dialogCenterX = dialogRect.left + (dialogRect.width / 2);
      let score = 0;
      if (text.includes("|")) score += 800;
      if (Number.isFinite(stripTop) && rect.bottom <= stripTop + 18) {
        score += 500;
        score -= Math.abs(stripTop - rect.bottom);
      }
      if (rect.width < dialogRect.width * 0.9) score += 80;
      score += rect.top / 6;
      score -= Math.abs(centerX - dialogCenterX) / 4;
      if (score > bestScore) {
        bestScore = score;
        bestFname = fname;
      }
    }
    return bestFname;
  };
  const getDialogOrder = (markedState, side) => {
    if (side === "left") return Array.isArray(markedState.left_order) ? markedState.left_order : [];
    if (side === "right") return Array.isArray(markedState.right_order) ? markedState.right_order : [];
    return [];
  };
  const updateDialogSelectionFromFname = (markedState, fname) => {
    if (!fname) return false;
    for (const side of ["left", "right"]) {
      const order = getDialogOrder(markedState, side);
      const idx = order.indexOf(fname);
      if (idx >= 0) {
        activeDialogSelection = { side, index: idx };
        return true;
      }
    }
    return false;
  };
  const getActiveDialogImage = () => {
    const candidates = Array.from(document.querySelectorAll('[role="dialog"] img, [aria-modal="true"] img')).filter((img) => {
      const rect = img.getBoundingClientRect();
      const style = window.getComputedStyle(img);
      return rect.width >= 220 && rect.height >= 220 && style.display !== "none" && style.visibility !== "hidden" && style.opacity !== "0";
    });
    if (!candidates.length) return null;
    candidates.sort((a, b) => (b.getBoundingClientRect().width * b.getBoundingClientRect().height) - (a.getBoundingClientRect().width * a.getBoundingClientRect().height));
    return candidates[0];
  };
  const getDialogThumbImages = (dialogRoot) => {
    const root = dialogRoot || document.querySelector('[role="dialog"], [aria-modal="true"]');
    if (!root) return [];
    const candidates = Array.from(root.querySelectorAll("img")).filter((img) => {
      const rect = img.getBoundingClientRect();
      const style = window.getComputedStyle(img);
      return rect.width > 12
        && rect.height > 12
        && rect.width <= 180
        && rect.height <= 180
        && style.display !== "none"
        && style.visibility !== "hidden"
        && style.opacity !== "0";
    });
    if (candidates.length <= 1) return candidates;
    const rows = [];
    for (const img of candidates) {
      const rect = img.getBoundingClientRect();
      const centerY = rect.top + (rect.height / 2);
      let row = rows.find((entry) => Math.abs(entry.centerY - centerY) <= 22);
      if (!row) {
        row = { centerY, images: [] };
        rows.push(row);
      }
      row.images.push(img);
      row.centerY = row.images.reduce((sum, entry) => {
        const entryRect = entry.getBoundingClientRect();
        return sum + entryRect.top + (entryRect.height / 2);
      }, 0) / row.images.length;
    }
    rows.sort((a, b) => {
      if (b.images.length !== a.images.length) return b.images.length - a.images.length;
      return b.centerY - a.centerY;
    });
    const strip = rows[0]?.images || [];
    strip.sort((a, b) => a.getBoundingClientRect().left - b.getBoundingClientRect().left);
    return strip;
  };
  const getDialogThumbTargetInfo = (target) => {
    const dialogRoot = target.closest('[role="dialog"], [aria-modal="true"]');
    if (!dialogRoot) return null;
    const targetImg = target instanceof HTMLImageElement ? target : target.querySelector?.("img");
    if (!(targetImg instanceof HTMLImageElement)) return null;
    const thumbs = getDialogThumbImages(dialogRoot);
    const idx = thumbs.indexOf(targetImg);
    if (idx < 0) return null;
    return { dialogRoot, targetImg, idx };
  };
  const getDialogControlLabel = (element) => {
    if (!(element instanceof Element)) return "";
    return `${element.getAttribute("aria-label") || ""} ${element.getAttribute("title") || ""} ${(element.textContent || "").trim()}`.trim().toLowerCase();
  };
  const findInlinePreviewCloseButton = (previewRoot) => {
    if (!(previewRoot instanceof Element)) return null;
    const candidates = Array.from(previewRoot.querySelectorAll("button")).filter((button) => {
      return !button.matches(".media-button, .thumbnail-item, .thumbnail-small, .thumbnail-lg")
        && !button.closest(".thumbnails");
    });
    return candidates.length ? candidates[candidates.length - 1] : null;
  };
  const findCloseButton = (root) => {
    if (!(root instanceof Element)) return null;
    if (root.matches?.("#hy-left-gallery .preview, #hy-right-gallery .preview")) {
      const previewClose = findInlinePreviewCloseButton(root);
      if (previewClose) return previewClose;
    }
    return Array.from(root.querySelectorAll("button")).find((button) => {
      const label = getDialogControlLabel(button);
      return label === "x" || label === "×" || label.includes("close");
    }) || null;
  };
  const hideDialogThumbStrip = (dialogRoot) => {
    const thumbs = getDialogThumbImages(dialogRoot);
    if (!thumbs.length) return;
    const parentCounts = new Map();
    for (const img of thumbs) {
      const host = img.closest("button, [role='button'], .thumbnail-item, .gallery-item") || img;
      host.style.display = "none";
      host.style.pointerEvents = "none";
      host.setAttribute("aria-hidden", "true");
      if (host.parentElement) {
        parentCounts.set(host.parentElement, (parentCounts.get(host.parentElement) || 0) + 1);
      }
    }
    for (const [parent, count] of parentCounts.entries()) {
      if (count >= Math.max(3, Math.ceil(thumbs.length * 0.5))) {
        parent.style.display = "none";
        parent.setAttribute("aria-hidden", "true");
      }
    }
  };
  const disablePreviewDialogNavigation = (dialogRoot) => {
    const root = dialogRoot || document.querySelector('[role="dialog"], [aria-modal="true"]');
    if (!root) return;
    hideDialogThumbStrip(root);
    const mainImg = getActiveDialogImage();
    if (mainImg) {
      const host = mainImg.closest("button, [role='button']") || mainImg;
      host.style.pointerEvents = "none";
      host.style.cursor = "default";
      mainImg.style.pointerEvents = "none";
      mainImg.style.cursor = "default";
    }
  };
  const isDialogMainImageClick = (event, dialogRoot) => {
    if (!event || !(dialogRoot instanceof Element)) return false;
    if (event.button !== 0) return false;
    const target = event.target;
    if (!(target instanceof Element)) return false;
    if (target.closest("button, a, input, textarea, select, label")) return false;
    const dialogImg = getActiveDialogImage();
    if (!(dialogImg instanceof HTMLImageElement)) return false;
    const rect = dialogImg.getBoundingClientRect();
    if (rect.width <= 0 || rect.height <= 0) return false;
    return (
      event.clientX >= rect.left
      && event.clientX <= rect.right
      && event.clientY >= rect.top
      && event.clientY <= rect.bottom
    );
  };
  const shouldBlockDialogNavigationClick = (target, dialogRoot) => {
    if (!(target instanceof Element) || !dialogRoot) return false;
    if (target === dialogRoot) return false;
    const control = target.closest("button, a");
    if (control) {
      const label = getDialogControlLabel(control);
      const rect = control.getBoundingClientRect();
      const dialogRect = dialogRoot.getBoundingClientRect();
      if (rect.top <= dialogRect.top + 80) return false;
      if (
        label === "x"
        || label === "×"
        || label.includes("close")
        || label.includes("download")
        || label.includes("share")
        || label.includes("fullscreen")
        || label.includes("expand")
      ) {
        return false;
      }
    }
    return true;
  };
  const parseCssColor = (value) => {
    const match = String(value || "").match(/rgba?\\(([^)]+)\\)/i);
    if (!match) return null;
    const parts = match[1].split(",").map((part) => Number.parseFloat(part.trim()));
    if (parts.length < 3 || parts.some((part) => !Number.isFinite(part))) return null;
    return {
      r: parts[0],
      g: parts[1],
      b: parts[2],
      a: Number.isFinite(parts[3]) ? parts[3] : 1,
    };
  };
  const thumbSelectionScore = (img) => {
    const host = img.closest("button, [role='button'], .thumbnail-item, .gallery-item") || img;
    const hostStyle = window.getComputedStyle(host);
    const imgStyle = window.getComputedStyle(img);
    const hostClasses = String(host.className || "").toLowerCase();
    const imgClasses = String(img.className || "").toLowerCase();
    let score = 0;
    if (host.getAttribute("aria-selected") === "true" || img.getAttribute("aria-selected") === "true") score += 1000;
    if (host.getAttribute("aria-current") === "true" || img.getAttribute("aria-current") === "true") score += 1000;
    if (host.dataset.selected === "true" || img.dataset.selected === "true") score += 1000;
    if (/(^|\\s)(selected|active|current)(\\s|$)/.test(hostClasses)) score += 300;
    if (/(^|\\s)(selected|active|current)(\\s|$)/.test(imgClasses)) score += 300;
    const outlineWidth = Number.parseFloat(hostStyle.outlineWidth || "0") || 0;
    const borderWidth = Number.parseFloat(hostStyle.borderTopWidth || "0") || 0;
    const boxShadow = `${hostStyle.boxShadow || ""} ${imgStyle.boxShadow || ""}`.toLowerCase();
    if (outlineWidth > 0) score += outlineWidth * 25;
    if (borderWidth > 0) score += borderWidth * 15;
    if (boxShadow && boxShadow !== "none") score += 30;
    const colors = [
      parseCssColor(hostStyle.outlineColor),
      parseCssColor(hostStyle.borderTopColor),
      parseCssColor(hostStyle.boxShadow),
      parseCssColor(imgStyle.outlineColor),
      parseCssColor(imgStyle.borderTopColor),
      parseCssColor(imgStyle.boxShadow),
    ].filter(Boolean);
    for (const color of colors) {
      const blueBias = color.b - Math.max(color.r, color.g);
      const cyanBias = Math.min(color.g, color.b) - color.r;
      if (blueBias > 20) score += 40 + blueBias;
      if (cyanBias > 20) score += 40 + cyanBias;
      score += (color.a || 0) * 10;
    }
    return score;
  };
  const getSelectedDialogThumbIndex = (dialogRoot) => {
    const thumbs = getDialogThumbImages(dialogRoot);
    if (!thumbs.length) return -1;
    let bestIndex = -1;
    let bestScore = -1;
    thumbs.forEach((img, idx) => {
      const score = thumbSelectionScore(img);
      if (score > bestScore) {
        bestScore = score;
        bestIndex = idx;
      }
    });
    return bestScore > 0 ? bestIndex : -1;
  };
  const syncDialogPreviewTarget = () => {
    const dialogRoot = document.querySelector('[role="dialog"], [aria-modal="true"]');
    const dialogImg = getActiveDialogImage();
    const markedState = readMarkedState();
    if (!activeDialogSelection.side) {
      updateDialogSelectionFromFname(markedState, markedState.preview || "");
    }
    if (!dialogRoot || !dialogImg) {
      activeDialogPreviewFname = "";
      activeDialogSelection = { side: "", index: -1 };
      return;
    }
    disablePreviewDialogNavigation(dialogRoot);
    const fname = extractFnameFromDialogText(dialogRoot, markedState) || extractFnameFromDialogImage(dialogImg, markedState);
    if (!fname || fname === activeDialogPreviewFname) return;
    updateDialogSelectionFromFname(markedState, fname);
    activeDialogPreviewFname = fname;
    pushPreviewFname(fname);
  };
  const resolveDialogPreviewFnameForAction = (dialogRoot, markedState) => {
    const fromText = extractFnameFromDialogText(dialogRoot, markedState);
    if (fromText) {
      updateDialogSelectionFromFname(markedState, fromText);
      return fromText;
    }
    const dialogImg = getActiveDialogImage();
    const fromImage = extractFnameFromDialogImage(dialogImg, markedState);
    if (fromImage) {
      updateDialogSelectionFromFname(markedState, fromImage);
      return fromImage;
    }
    const selectedThumbIndex = getSelectedDialogThumbIndex(dialogRoot);
    if (selectedThumbIndex >= 0) {
      let side = activeDialogSelection.side || "";
      if (!side) {
        side = updateDialogSelectionFromFname(markedState, markedState.preview || "") ? activeDialogSelection.side : "";
      }
      const order = getDialogOrder(markedState, side);
      if (selectedThumbIndex < order.length) {
        activeDialogSelection = { side, index: selectedThumbIndex };
        return order[selectedThumbIndex] || "";
      }
    }
    const order = getDialogOrder(markedState, activeDialogSelection.side);
    if (order.length && activeDialogSelection.index >= 0 && activeDialogSelection.index < order.length) {
      return order[activeDialogSelection.index] || "";
    }
    return activeDialogPreviewFname || markedState.preview || "";
  };
  const closePreviewDialog = (dialogRoot) => {
    const root = dialogRoot || document.querySelector('[role="dialog"], [aria-modal="true"]');
    if (!root) return false;
    const closeButton = findCloseButton(root);
    if (closeButton) {
      closeButton.click();
      return true;
    }
    const escapeEvent = new KeyboardEvent("keydown", {
      key: "Escape",
      code: "Escape",
      keyCode: 27,
      which: 27,
      bubbles: true,
    });
    document.dispatchEvent(escapeEvent);
    return true;
  };
  const closeAllGalleryPreviews = () => {
    let closedAny = false;
    const inlinePreviews = Array.from(document.querySelectorAll("#hy-left-gallery .preview, #hy-right-gallery .preview"));
    for (const previewRoot of inlinePreviews) {
      if (!(previewRoot instanceof Element)) continue;
      const closeButton = findInlinePreviewCloseButton(previewRoot) || findCloseButton(previewRoot);
      if (!closeButton) continue;
      closeButton.click();
      closedAny = true;
    }
    const dialogRoots = Array.from(document.querySelectorAll('[role="dialog"], [aria-modal="true"]'));
    for (const dialogRoot of dialogRoots) {
      closedAny = closePreviewDialog(dialogRoot) || closedAny;
    }
    if (closedAny) {
      activeDialogPreviewFname = "";
      activeDialogSelection = { side: "", index: -1 };
      setTimeout(scheduleRepaint, 40);
      setTimeout(scheduleRepaint, 140);
    }
    return closedAny;
  };
  const hookButtonPreviewReset = () => {
    if (document.body.dataset.hyButtonPreviewResetHooked) return;
    document.addEventListener("click", (event) => {
      const target = event.target;
      if (!(target instanceof Element)) return;
      const actionRoot = target.closest("button, [role='button']");
      if (!actionRoot) return;
      if (actionRoot.closest("#hy-left-gallery, #hy-right-gallery")) return;
      if (actionRoot.closest('[role="dialog"], [aria-modal="true"]')) return;
      closeAllGalleryPreviews();
    }, true);
    document.body.dataset.hyButtonPreviewResetHooked = "1";
  };
  const getAccordionToggle = (root) => {
    if (!(root instanceof Element)) return null;
    return root.querySelector("button.label-wrap") || root.querySelector("summary, button");
  };
  const isAccordionOpen = (root) => {
    if (!(root instanceof Element)) return false;
    const btn = root.querySelector("button.label-wrap");
    if (btn) return btn.classList.contains("open");
    const details = root.querySelector("details");
    if (details) return !!details.open;
    return false;
  };
  const _progToggle = new Set();
  const setAccordionOpen = (root, shouldOpen) => {
    if (!(root instanceof Element)) return false;
    const currentlyOpen = isAccordionOpen(root);
    if (currentlyOpen === shouldOpen) return false;
    const toggle = getAccordionToggle(root);
    if (toggle) {
      _progToggle.add(root);
      toggle.click();
      setTimeout(() => _progToggle.delete(root), 300);
      return true;
    }
    return false;
  };
  const hookChangelogOverlay = () => {
    const btn = document.getElementById("hy-version-btn");
    const overlay = document.getElementById("hy-changelog-overlay");
    const closeBtn = document.getElementById("hy-changelog-close");
    if (!btn || !overlay || btn.dataset.changelogHooked) return;
    btn.addEventListener("click", () => overlay.classList.add("open"));
    closeBtn?.addEventListener("click", () => overlay.classList.remove("open"));
    overlay.addEventListener("click", (e) => { if (e.target === overlay) overlay.classList.remove("open"); });
    document.addEventListener("keydown", (e) => { if (e.key === "Escape") overlay.classList.remove("open"); });
    btn.dataset.changelogHooked = "1";
  };
  const hookSidebarAccordionBehavior = () => {
    const setupRoot = document.getElementById("hy-acc-setup");
    const scoringRoot = document.getElementById("hy-acc-scoring");
    const searchRoot = document.getElementById("hy-acc-search-image");
    const exportRoot = document.getElementById("hy-acc-export");
    const managedRoots = [setupRoot, scoringRoot, searchRoot, exportRoot].filter((root) => root instanceof Element);
    if (!managedRoots.length) return;
    for (const root of managedRoots) {
      if (!(root instanceof Element)) continue;
      const toggle = getAccordionToggle(root);
      if (!(toggle instanceof Element)) continue;
      if (toggle.dataset.hySidebarAccordionClickHooked === "1") continue;
      toggle.addEventListener("click", () => {
        if (_progToggle.has(root)) return;
        const willOpen = !isAccordionOpen(root);
        if (willOpen) {
          setTimeout(() => {
            for (const r of managedRoots) {
              if (r !== root) setAccordionOpen(r, false);
            }
          }, 60);
        }
      }, true);
      toggle.dataset.hySidebarAccordionClickHooked = "1";
    }
  };
  const hookDialogActionHandoff = () => {
    if (document.body.dataset.hyDialogActionHooked) return;
    document.addEventListener("click", (event) => {
      const target = event.target;
      if (!(target instanceof Element)) return;
      const actionRoot = target.closest("#hy-fit-threshold, #hy-move-right, #hy-move-left");
      if (!actionRoot) return;
      const dialogRoot = document.querySelector('[role="dialog"], [aria-modal="true"]');
      if (!dialogRoot) return;
      const markedState = readMarkedState();
      const fname = resolveDialogPreviewFnameForAction(dialogRoot, markedState);
      event.preventDefault();
      event.stopPropagation();
      event.stopImmediatePropagation();
      closePreviewDialog(dialogRoot);
      pushThumbAction(`dialogactionjson:${JSON.stringify({
        action: actionRoot.id || "",
        fname: fname || "",
        ts: Date.now(),
      })}`);
      setTimeout(scheduleRepaint, 40);
      setTimeout(scheduleRepaint, 140);
    }, true);
    document.body.dataset.hyDialogActionHooked = "1";
  };
  const getHoverScores = (markedState) => {
    if (!activeHoverInfo || !activeHoverInfo.fname) return null;
    const lookup = markedState.score_lookup || {};
    const scored = lookup[activeHoverInfo.fname] || null;
    if (scored) return scored;
    if (Number.isFinite(activeHoverInfo.main)) {
      return {
        main: activeHoverInfo.main,
        neg: Number.isFinite(activeHoverInfo.neg) ? activeHoverInfo.neg : null,
      };
    }
    return null;
  };
  const syncHistogramHoverLine = () => {
    const root = document.getElementById("hy-hist");
    if (!root) return;
    root.style.position = "relative";
    const markedState = readMarkedState();
    const geom = markedState.hist_geom || null;
    const img = root.querySelector("img");
    let line = root.querySelector(".hy-hover-line-main");
    if (!line) {
      line = document.createElement("div");
      line.className = "hy-hover-line hy-hover-line-main";
      root.appendChild(line);
    }
    let negLine = root.querySelector(".hy-hover-line-neg");
    if (!negLine) {
      negLine = document.createElement("div");
      negLine.className = "hy-hover-line hy-hover-line-neg";
      root.appendChild(negLine);
    }
    const hoverScores = getHoverScores(markedState);
    if (!geom || !img || !hoverScores || !Number.isFinite(hoverScores.main)) {
      line.style.opacity = "0";
      negLine.style.opacity = "0";
      return;
    }
    const usesPositiveSimilarityChart = ["PromptMatch", "Similarity", "SamePerson", "TagMatch", "LM Search"].includes(geom.method);
    const chartLo = usesPositiveSimilarityChart ? geom.pos_lo : geom.lo;
    const chartHi = usesPositiveSimilarityChart ? geom.pos_hi : geom.hi;
    if (!Number.isFinite(chartLo) || !Number.isFinite(chartHi) || Math.abs(chartHi - chartLo) < 1e-9) {
      line.style.opacity = "0";
      negLine.style.opacity = "0";
      return;
    }
    const scaleX = img.clientWidth / Math.max(geom.W || 1, 1);
    const scaleY = img.clientHeight / Math.max(geom.H || 1, 1);
    const chartX = (geom.PAD_L + (((hoverScores.main - chartLo) / (chartHi - chartLo)) * (geom.W - geom.PAD_L - geom.PAD_R))) * scaleX;
    const clampedX = Math.max(geom.PAD_L * scaleX, Math.min((geom.W - geom.PAD_R) * scaleX, chartX));
    const chartTop = (geom.PAD_TOP || 0) * scaleY;
    const chartHeight = (usesPositiveSimilarityChart ? geom.CH : (geom.H - geom.PAD_TOP - geom.PAD_BOT)) * scaleY;
    line.style.left = `${Math.round(img.offsetLeft + clampedX)}px`;
    line.style.top = `${Math.round(img.offsetTop + chartTop)}px`;
    line.style.height = `${Math.max(12, Math.round(chartHeight))}px`;
    line.style.opacity = "1";
    negLine.style.opacity = "0";
    if (usesPositiveSimilarityChart && geom.has_neg && Number.isFinite(hoverScores.neg) && Number.isFinite(geom.neg_lo) && Number.isFinite(geom.neg_hi) && Math.abs(geom.neg_hi - geom.neg_lo) >= 1e-9) {
      const negX = (geom.PAD_L + (((hoverScores.neg - geom.neg_lo) / (geom.neg_hi - geom.neg_lo)) * (geom.W - geom.PAD_L - geom.PAD_R))) * scaleX;
      const clampedNegX = Math.max(geom.PAD_L * scaleX, Math.min((geom.W - geom.PAD_R) * scaleX, negX));
      const negTop = (geom.PAD_TOP + geom.CH + geom.GAP) * scaleY;
      negLine.style.left = `${Math.round(img.offsetLeft + clampedNegX)}px`;
      negLine.style.top = `${Math.round(img.offsetTop + negTop)}px`;
      negLine.style.height = `${Math.max(12, Math.round(geom.CH * scaleY))}px`;
      negLine.style.opacity = "1";
    }
  };
  const ensureOverlay = (elemId) => {
    const root = document.getElementById(elemId);
    if (!root) return null;
    let ov = root.querySelector(".hy-tag-overlay");
    if (!ov) {
      ov = document.createElement("div");
      ov.className = "hy-tag-overlay";
      root.appendChild(ov);
    }
    return ov;
  };
  const hideOverlay = (elemId) => {
    const root = document.getElementById(elemId);
    if (!root) return;
    const ov = root.querySelector(".hy-tag-overlay");
    if (ov) ov.classList.remove("active");
    const ta = root.querySelector("textarea");
    if (ta) ta.classList.remove("hy-tag-overlay-active");
  };
  const ensureTagMatchSuggestBox = () => {
    let box = document.getElementById("hy-tagmatch-suggest-box");
    if (!box) {
      box = document.createElement("div");
      box.className = "hy-tagmatch-suggest";
      box.id = "hy-tagmatch-suggest-box";
      document.body.appendChild(box);
    }
    return box;
  };
  const positionTagMatchSuggestBox = (box, input) => {
    if (!box || !input) return;
    const rect = input.getBoundingClientRect();
    const viewportWidth = Math.max(document.documentElement.clientWidth || 0, window.innerWidth || 0);
    const left = Math.max(8, Math.round(rect.left));
    const width = Math.max(220, Math.round(rect.width));
    const maxWidth = Math.max(220, viewportWidth - left - 8);
    box.style.left = `${left}px`;
    box.style.top = `${Math.round(rect.bottom - 2)}px`;
    box.style.width = `${Math.min(width, maxWidth)}px`;
  };
  const hideTagMatchAutocomplete = () => {
    const box = ensureTagMatchSuggestBox();
    if (!box) return;
    box.classList.remove("active");
    box.innerHTML = "";
    box.dataset.activeIndex = "-1";
  };
  const escapeHtml = (value) => {
    return String(value || "").replace(/[&<>\"']/g, (ch) => {
      return {
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        "\"": "&quot;",
        "'": "&#39;",
      }[ch] || ch;
    });
  };
  const getTagMatchTokenInfo = (input) => {
    if (!input) return null;
    const value = input.value || "";
    const caret = input.selectionStart ?? value.length;
    const lastComma = value.lastIndexOf(",", Math.max(0, caret - 1));
    const tokenStart = lastComma >= 0 ? lastComma + 1 : 0;
    const nextComma = value.indexOf(",", caret);
    const tokenEnd = nextComma >= 0 ? nextComma : value.length;
    const tokenRaw = value.slice(tokenStart, tokenEnd);
    const leadingMatch = tokenRaw.match(/^\\s*/);
    const trailingMatch = tokenRaw.match(/\\s*$/);
    const leadingLen = leadingMatch ? leadingMatch[0].length : 0;
    const trailingLen = trailingMatch ? trailingMatch[0].length : 0;
    const replaceStart = tokenStart + leadingLen;
    const replaceEnd = Math.max(replaceStart, tokenEnd - trailingLen);
    const fragment = value.slice(replaceStart, caret).trim().toLowerCase();
    return {
      value,
      caret,
      tokenStart,
      tokenEnd,
      replaceStart,
      replaceEnd,
      fragment,
      currentTag: value.slice(replaceStart, replaceEnd).trim().toLowerCase(),
    };
  };
  const rankTagMatchSuggestions = (vocab, fragment, currentTag) => {
    const needle = String(fragment || "").trim().toLowerCase();
    if (!needle) return [];
    const prefix = [];
    const contains = [];
    for (const tag of vocab) {
      if (!tag || tag === currentTag) continue;
      if (tag.startsWith(needle)) {
        prefix.push(tag);
      } else if (tag.includes(needle)) {
        contains.push(tag);
      }
      if ((prefix.length + contains.length) >= 24) {
        break;
      }
    }
    return prefix.concat(contains).slice(0, 12);
  };
  const highlightTagSuggestion = (box, nextIndex) => {
    if (!box) return;
    const items = Array.from(box.querySelectorAll(".hy-tagmatch-suggest-item"));
    if (!items.length) {
      box.dataset.activeIndex = "-1";
      return;
    }
    let index = Number.parseInt(String(nextIndex), 10);
    if (!Number.isFinite(index)) index = 0;
    index = ((index % items.length) + items.length) % items.length;
    items.forEach((item, itemIndex) => {
      item.classList.toggle("active", itemIndex === index);
    });
    box.dataset.activeIndex = String(index);
  };
  const applyTagMatchSuggestion = (tag) => {
    const root = document.getElementById("hy-tagmatch-tags");
    const input = root ? root.querySelector("textarea") : null;
    if (!input || !tag) return false;
    const info = getTagMatchTokenInfo(input);
    if (!info) return false;
    const before = info.value.slice(0, info.replaceStart);
    const after = info.value.slice(info.tokenEnd);
    const needsTrailingComma = info.tokenEnd >= info.value.length;
    const insertion = needsTrailingComma ? `${tag}, ` : tag;
    input.value = before + insertion + after;
    const caretPos = before.length + insertion.length;
    input.focus();
    input.setSelectionRange(caretPos, caretPos);
    dispatchTextboxEvents(input);
    hideTagMatchAutocomplete();
    return true;
  };
  const syncTagMatchAutocomplete = () => {
    const root = document.getElementById("hy-tagmatch-tags");
    const input = root ? root.querySelector("textarea") : null;
    const box = ensureTagMatchSuggestBox();
    if (!input || !box) return;
    const vocab = readTagMatchVocabulary();
    const info = getTagMatchTokenInfo(input);
    if (!info || !vocab.length || !info.fragment) {
      hideTagMatchAutocomplete();
      return;
    }
    const matches = rankTagMatchSuggestions(vocab, info.fragment, info.currentTag);
    if (!matches.length) {
      hideTagMatchAutocomplete();
      return;
    }
    positionTagMatchSuggestBox(box, input);
    box.innerHTML = matches.map((tag, idx) => {
      const activeClass = idx === 0 ? " active" : "";
      return `<button type="button" class="hy-tagmatch-suggest-item${activeClass}" data-tag="${escapeHtml(tag)}">${escapeHtml(tag)}</button>`;
    }).join("");
    box.classList.add("active");
    box.dataset.activeIndex = "0";
  };
  const hslStyle = (t) => {
    const hue  = Math.round(55 + t * 65);
    const sat  = Math.round(60 + t * 30);
    const lght = Math.round(38 - t * 5);
    const alph = (0.35 + t * 0.55).toFixed(2);
    const col  = t > 0.35 ? "#fff" : "#aaa";
    return `background:hsla(${hue},${sat}%,${lght}%,${alph});color:${col};border:none`;
  };
  const syncTagMatchPills = () => {
    const markedState = readMarkedState();
    const method = (markedState.hist_geom || {}).method || "";

    // --- TagMatch: tag probability pills over #hy-tagmatch-tags ---
    if (method === "TagMatch" && activeHoverInfo && activeHoverInfo.fname) {
      const tmRoot = document.getElementById("hy-tagmatch-tags");
      const tmTa = tmRoot ? tmRoot.querySelector("textarea") : null;
      const tmOv = ensureOverlay("hy-tagmatch-tags");
      const imgProbs = (markedState.tag_score_lookup || {})[activeHoverInfo.fname] || null;
      const rawText = tmTa ? tmTa.value : "";
      const tags = rawText.split(",").map(t => t.trim().toLowerCase()).filter(Boolean);
      if (tmOv && tags.length) {
        const probToStyle = (prob) => {
          if (!prob || prob < 0.01)
            return `background:rgba(35,35,42,0.9);color:#555;border:1px solid rgba(60,60,80,0.3)`;
          return hslStyle(Math.min(1, prob));
        };
        let html = "";
        for (const tag of tags) {
          const prob = imgProbs ? (imgProbs[tag] || 0) : 0;
          const pct  = prob > 0 ? `<span class="tag-prob">${Math.round(prob * 100)}%</span>` : "";
          html += `<span class="hy-tag-pill" style="${probToStyle(prob)}">${tag}${pct}</span>`;
        }
        tmOv.innerHTML = html;
        tmOv.classList.add("active");
        if (tmTa) tmTa.classList.add("hy-tag-overlay-active");
      } else {
        hideOverlay("hy-tagmatch-tags");
      }
    } else {
      hideOverlay("hy-tagmatch-tags");
    }

    // --- PromptMatch per-segment: positive similarity pills over #hy-pos ---
    const segLookup = markedState.segment_score_lookup || {};
    const hasSegData = Object.keys(segLookup).length > 0;
    if (method === "PromptMatch" && hasSegData && activeHoverInfo && activeHoverInfo.fname) {
      const pmRoot = document.getElementById("hy-pos");
      const pmTa = pmRoot ? pmRoot.querySelector("textarea") : null;
      const pmOv = ensureOverlay("hy-pos");
      const imgSegs = segLookup[activeHoverInfo.fname] || null;
      const rawText = pmTa ? pmTa.value : "";
      const segs = rawText.split(",").map(s => s.trim()).filter(Boolean);
      if (pmOv && segs.length && imgSegs) {
        const vals = segs.map(s => imgSegs[s] !== undefined ? imgSegs[s] : null);
        const defined = vals.filter(v => v !== null);
        const minV = defined.length ? Math.min(...defined) : 0;
        const maxV = defined.length ? Math.max(...defined) : 1;
        const range = Math.max(maxV - minV, 0.001);
        let html = "";
        for (let i = 0; i < segs.length; i++) {
          const v = vals[i];
          const style = v !== null
            ? hslStyle((v - minV) / range)
            : `background:rgba(35,35,42,0.9);color:#555;border:1px solid rgba(60,60,80,0.3)`;
          const label = v !== null ? `<span class="tag-prob">${v.toFixed(2)}</span>` : "";
          html += `<span class="hy-tag-pill" style="${style}">${segs[i]}${label}</span>`;
        }
        pmOv.innerHTML = html;
        pmOv.classList.add("active");
        if (pmTa) pmTa.classList.add("hy-tag-overlay-active");
      } else {
        hideOverlay("hy-pos");
      }
    } else {
      hideOverlay("hy-pos");
    }

    // --- PromptMatch per-segment: negative similarity pills over #hy-neg (yellow→red) ---
    const negSegLookup = markedState.neg_segment_score_lookup || {};
    const hasNegSegData = Object.keys(negSegLookup).length > 0;
    if (method === "PromptMatch" && hasNegSegData && activeHoverInfo && activeHoverInfo.fname) {
      const ngRoot = document.getElementById("hy-neg");
      const ngTa = ngRoot ? ngRoot.querySelector("textarea") : null;
      const ngOv = ensureOverlay("hy-neg");
      const imgNegSegs = negSegLookup[activeHoverInfo.fname] || null;
      const negRawText = ngTa ? ngTa.value : "";
      const negSegs = negRawText.split(",").map(s => s.trim()).filter(Boolean);
      if (ngOv && negSegs.length && imgNegSegs) {
        const negVals = negSegs.map(s => imgNegSegs[s] !== undefined ? imgNegSegs[s] : null);
        const negDefined = negVals.filter(v => v !== null);
        const negMinV = negDefined.length ? Math.min(...negDefined) : 0;
        const negMaxV = negDefined.length ? Math.max(...negDefined) : 1;
        const negRange = Math.max(negMaxV - negMinV, 0.001);
        let html = "";
        for (let i = 0; i < negSegs.length; i++) {
          const v = negVals[i];
          let style;
          if (v !== null) {
            const t = (v - negMinV) / negRange;
            const hue  = Math.round(55 - t * 55);          // yellow(55) → red(0)
            const sat  = Math.round(60 + t * 30);           // 60% → 90%
            const lght = Math.round(38 - t * 5);            // 38% → 33%
            const alph = (0.35 + t * 0.55).toFixed(2);     // 0.35 → 0.90
            const col  = t > 0.35 ? "#fff" : "#aaa";
            style = `background:hsla(${hue},${sat}%,${lght}%,${alph});color:${col};border:none`;
          } else {
            style = `background:rgba(35,35,42,0.9);color:#555;border:1px solid rgba(60,60,80,0.3)`;
          }
          const label = v !== null ? `<span class="tag-prob">${v.toFixed(2)}</span>` : "";
          html += `<span class="hy-tag-pill" style="${style}">${negSegs[i]}${label}</span>`;
        }
        ngOv.innerHTML = html;
        ngOv.classList.add("active");
        if (ngTa) ngTa.classList.add("hy-tag-overlay-active");
      } else {
        hideOverlay("hy-neg");
      }
    } else {
      hideOverlay("hy-neg");
    }
  };
  const scheduleRepaint = () => {
    // Gallery DOM mutates a moment after clicks; repaint a few times to catch the final layout.
    ensureThumbBehavior();
    hookHistogramResize();
    syncHistogramHoverLine();
    syncDialogPreviewTarget();
    disablePreviewDialogNavigation();
    for (const timer of repaintTimers) clearTimeout(timer);
    repaintTimers = [
      setTimeout(ensureThumbBehavior, 40),
      setTimeout(syncHistogramHoverLine, 60),
      setTimeout(syncDialogPreviewTarget, 80),
      setTimeout(disablePreviewDialogNavigation, 100),
      setTimeout(ensureThumbBehavior, 140),
      setTimeout(syncHistogramHoverLine, 170),
      setTimeout(syncDialogPreviewTarget, 200),
      setTimeout(disablePreviewDialogNavigation, 220),
      setTimeout(ensureThumbBehavior, 320),
      setTimeout(syncHistogramHoverLine, 350),
      setTimeout(syncDialogPreviewTarget, 380),
      setTimeout(disablePreviewDialogNavigation, 400),
    ];
  };
  const ensureThumbBehavior = () => {
    // Re-apply green/red borders after every gallery rerender and preview change.
    const markedState = readMarkedState();
    const allMarked = new Set([...(markedState.left || []), ...(markedState.right || [])]);
    const heldSet = new Set(markedState.held || []);
    const clearDropTarget = (root) => {
      if (!root) return;
      root.style.boxShadow = "";
      root.style.borderColor = "";
    };
    for (const [galleryId, side] of [["hy-left-gallery", "left"], ["hy-right-gallery", "right"]]) {
      const root = document.getElementById(galleryId);
      if (!root) continue;
      if (!root.dataset.hyShiftHooked) {
        root.addEventListener("click", (event) => {
          const card = event.target.closest("button");
          if (!card || !root.contains(card)) return;
          if (!event.shiftKey) {
            const thumbButtons = Array.from(root.querySelectorAll("button")).filter((btn) => {
              const img = btn.querySelector("img");
              const inDialog = !!btn.closest('[role="dialog"], [aria-modal="true"]');
              const hasCaption = !!(btn.querySelector(".caption-label span") || btn.querySelector('[class*="caption"]'));
              return !inDialog && !!img && hasCaption;
            });
            const index = thumbButtons.indexOf(card);
            if (index >= 0) {
              activeDialogSelection = { side, index };
              activeDialogPreviewFname = card.dataset.hyFname || "";
              pushThumbAction(`preview:${side}:${index}:${Date.now()}`);
            }
            setTimeout(scheduleRepaint, 30);
            setTimeout(scheduleRepaint, 140);
            setTimeout(scheduleRepaint, 320);
            return;
          }
          const thumbButtons = Array.from(root.querySelectorAll("button")).filter((btn) => {
            const img = btn.querySelector("img");
            const inDialog = !!btn.closest('[role="dialog"], [aria-modal="true"]');
            const hasCaption = !!(btn.querySelector(".caption-label span") || btn.querySelector('[class*="caption"]'));
            return !inDialog && !!img && hasCaption;
          });
          const index = thumbButtons.indexOf(card);
          if (index < 0) return;
          event.preventDefault();
          event.stopPropagation();
          event.stopImmediatePropagation();
          pushThumbAction(`mark:${side}:${index}:${Date.now()}`);
        }, true);
        root.dataset.hyShiftHooked = "1";
      }
      if (!root.dataset.hyDropHooked) {
        root.addEventListener("dragenter", (event) => {
          if (!activeDrag || activeDrag.side === side) return;
          event.preventDefault();
          root.style.boxShadow = "0 0 0 3px rgba(125, 168, 255, 0.35)";
          root.style.borderColor = "#7da8ff";
        });
        root.addEventListener("dragover", (event) => {
          if (!activeDrag || activeDrag.side === side) return;
          event.preventDefault();
          if (event.dataTransfer) event.dataTransfer.dropEffect = "move";
        });
        root.addEventListener("dragleave", (event) => {
          if (!root.contains(event.relatedTarget)) {
            clearDropTarget(root);
          }
        });
        root.addEventListener("drop", (event) => {
          if (!activeDrag || activeDrag.side === side) return;
          event.preventDefault();
          clearDropTarget(root);
          pushThumbAction(`dropjson:${JSON.stringify({
            source_side: activeDrag.side,
            source_index: activeDrag.index,
            target_side: side,
            fnames: activeDrag.fnames || [],
            ts: Date.now(),
          })}`);
          activeDrag = null;
          scheduleRepaint();
        });
        root.dataset.hyDropHooked = "1";
      }
      const thumbButtons = Array.from(root.querySelectorAll("button")).filter((btn) => {
        const img = btn.querySelector("img");
        const inDialog = !!btn.closest('[role="dialog"], [aria-modal="true"]');
        const hasCaption = !!(btn.querySelector(".caption-label span") || btn.querySelector('[class*="caption"]'));
        return !inDialog && !!img && hasCaption;
      });
      thumbButtons.forEach((card, index) => {
        const img = card.querySelector("img");
        if (!img) return;
        const captionEl = card.querySelector(".caption-label span") || card.querySelector('[class*="caption"]');
        const captionText = captionEl ? (captionEl.textContent || "") : "";
        const held = captionText.includes("✋ ");
        const parts = captionText.split("|");
        const fname = parts.length ? parts[parts.length - 1].trim() : "";
        const marked = (markedState[side] || []).includes(fname);
        card.style.position = "relative";
        card.style.boxSizing = "border-box";
        card.style.outline = marked ? "3px solid #58bb73" : (held ? "3px solid #dd3322" : "");
        card.style.outlineOffset = (marked || held) ? "-3px" : "";
        card.style.boxShadow = marked ? "inset 0 0 0 1px rgba(88,187,115,0.35)" : "";
        card.style.cursor = "grab";
        card.draggable = true;
        card.dataset.hySide = side;
        card.dataset.hyIndex = String(index);
        card.dataset.hyFname = fname;
        const scoreLookup = markedState.score_lookup || {};
        const hoverScores = scoreLookup[fname] || null;
        const hoverMain = hoverScores && Number.isFinite(hoverScores.main)
          ? hoverScores.main
          : parseMainScoreFromCaption(captionText);
        card.dataset.hyHoverMain = Number.isFinite(hoverMain) ? String(hoverMain) : "";
        card.dataset.hyHoverNeg = hoverScores && Number.isFinite(hoverScores.neg) ? String(hoverScores.neg) : "";
        if (!card.dataset.hyHoverHooked) {
          card.addEventListener("mouseenter", (event) => {
            const target = event.currentTarget;
            if (!target) return;
            const main = Number.parseFloat(target.dataset.hyHoverMain || "");
            const neg = Number.parseFloat(target.dataset.hyHoverNeg || "");
            activeHoverInfo = {
              fname: target.dataset.hyFname || "",
              main: Number.isFinite(main) ? main : null,
              neg: Number.isFinite(neg) ? neg : null,
            };
            syncHistogramHoverLine();
            syncTagMatchPills();
          });
          card.addEventListener("mouseleave", () => {
            activeHoverInfo = null;
            syncHistogramHoverLine();
            syncTagMatchPills();
          });
          card.dataset.hyHoverHooked = "1";
        }
        if (!card.dataset.hyDragHooked) {
          card.addEventListener("dragstart", (event) => {
            const dragSide = card.dataset.hySide || side;
            const dragIndex = Number.parseInt(card.dataset.hyIndex || String(index), 10);
            const dragFname = card.dataset.hyFname || fname;
            const currentMarkedState = readMarkedState();
            const markedNames = Array.isArray(currentMarkedState[dragSide]) ? currentMarkedState[dragSide] : [];
            const isMarked = markedNames.includes(dragFname);
            const dragNames = isMarked && markedNames.length > 1 ? markedNames.slice() : [dragFname];
            activeDrag = { side: dragSide, index: dragIndex, fnames: dragNames };
            card.style.opacity = "0.55";
            if (event.dataTransfer) {
              event.dataTransfer.effectAllowed = "move";
              event.dataTransfer.setData("text/plain", dragNames.join("\\n"));
            }
          });
          card.addEventListener("dragend", () => {
            card.style.opacity = "";
            activeDrag = null;
            clearDropTarget(document.getElementById("hy-left-gallery"));
            clearDropTarget(document.getElementById("hy-right-gallery"));
          });
          card.dataset.hyDragHooked = "1";
        }
        img.style.outline = "";
        img.style.outlineOffset = "";
      });
    }
    for (const el of document.querySelectorAll('[data-hy-preview-border="1"]')) {
      el.style.outline = "";
      el.style.outlineOffset = "";
      el.style.boxShadow = "";
      el.style.border = "";
      el.style.borderRadius = "";
      el.style.background = "";
      el.style.padding = "";
      el.style.boxSizing = "";
      el.style.overflow = "";
      el.removeAttribute("data-hy-preview-border");
    }
    const previewImages = Array.from(document.querySelectorAll(
      "#hy-left-gallery span.preview img, #hy-right-gallery span.preview img, #hy-left-gallery .preview img, #hy-right-gallery .preview img"
    )).filter((img) => {
      const rect = img.getBoundingClientRect();
      const style = window.getComputedStyle(img);
      return rect.width >= 220 && rect.height >= 220 && style.display !== "none" && style.visibility !== "hidden" && style.opacity !== "0";
    });
    for (const chosen of previewImages) {
      const captionText = chosen.getAttribute("alt") || "";
      const parts = captionText.split("|");
      const fname = parts.length ? parts[parts.length - 1].trim() : "";
      const marked = allMarked.has(fname);
      const held = heldSet.has(fname);
      if (!(marked || held)) continue;
      const color = marked ? "#58bb73" : "#dd3322";
      const mediaButton = chosen.closest("button.media-button");
      if (mediaButton && !mediaButton.matches("button.thumbnail-item, .thumbnail-item")) {
        mediaButton.style.outline = "";
        mediaButton.style.outlineOffset = "";
        mediaButton.style.border = `4px solid ${color}`;
        mediaButton.style.boxShadow = "0 0 0 1px rgba(0, 0, 0, 0.28)";
        mediaButton.style.borderRadius = "10px";
        mediaButton.style.background = "#0a0a12";
        mediaButton.style.padding = "0";
        mediaButton.style.boxSizing = "border-box";
        mediaButton.style.overflow = "hidden";
        mediaButton.setAttribute("data-hy-preview-border", "1");
      }
      chosen.style.border = "0";
      chosen.style.boxShadow = "none";
      chosen.style.borderRadius = "6px";
      chosen.setAttribute("data-hy-preview-border", "1");
    }
  };
  const applyTooltips = () => {
    for (const [id, text] of Object.entries(tooltips)) {
      const root = document.getElementById(id);
      if (!root) continue;
      root.title = text;
      root.setAttribute("aria-label", text);
      if (id === "hy-external-query-image") continue;
      const targets = root.querySelectorAll("button, input, textarea, select, img");
      for (const el of targets) {
        el.title = text;
        el.setAttribute("aria-label", text);
      }
    }
    hookRunScoringHotkeys();
    hookPromptWeightHotkeys();
    ensureThumbBehavior();
    applyPromptMatchModelAvailability();
  };
  const applyPromptMatchModelAvailability = () => {
    const statusMap = readPromptMatchModelStatus();
    const knownLabels = Object.keys(statusMap || {});
    if (!knownLabels.length) return;
    const colors = {
      cached: "#8fdc7e",
      download: "#e7b062",
    };
    const applyToNode = (node) => {
      if (!(node instanceof HTMLElement)) return;
      const text = node instanceof HTMLInputElement
        ? (node.value || "").trim()
        : ((node.textContent || "").trim());
      const entry = findPromptMatchModelStatusEntry(text, statusMap, knownLabels);
      if (!entry) return;
      paintPromptMatchModelNode(node, entry, colors);
      for (const child of node.querySelectorAll("*")) {
        paintPromptMatchModelNode(child, entry, colors);
      }
    };
    for (const rootId of ["hy-model", "hy-llm-model", "hy-llm-backend", "hy-prompt-generator"]) {
      const root = document.getElementById(rootId);
      if (!root) continue;
      const input = root.querySelector("input");
      if (input) applyToNode(input);
      for (const node of root.querySelectorAll("span, div")) applyToNode(node);
    }
    for (const node of document.querySelectorAll('[role="option"], [role="listbox"], [role="listbox"] *')) {
      applyToNode(node);
    }
  };
  const hookMarkState = () => {
    const root = document.getElementById("hy-mark-state");
    if (!root || root.dataset.hyStateHooked) return;
    const input = root.querySelector("input, textarea");
    if (!input) return;
    input.addEventListener("input", scheduleRepaint);
    input.addEventListener("change", scheduleRepaint);
    root.dataset.hyStateHooked = "1";
  };
  const hookModelStatusState = () => {
    const root = document.getElementById("hy-model-status");
    if (!root || root.dataset.hyStateHooked) return;
    const input = root.querySelector("input, textarea");
    if (!input) return;
    const repaint = () => schedulePromptMatchModelAvailability();
    input.addEventListener("input", repaint);
    input.addEventListener("change", repaint);
    root.dataset.hyStateHooked = "1";
  };
  const hookPromptMatchModelAvailability = () => {
    const repaint = () => schedulePromptMatchModelAvailability();
    for (const rootId of ["hy-model", "hy-llm-model", "hy-llm-backend", "hy-prompt-generator"]) {
      const root = document.getElementById(rootId);
      if (!root || root.dataset.hyAvailabilityHooked) continue;
      for (const eventName of ["click", "pointerdown", "mousedown", "focusin", "focusout", "input", "change", "keydown", "keyup"]) {
        root.addEventListener(eventName, repaint);
      }
      root.dataset.hyAvailabilityHooked = "1";
    }
    if (document.body.dataset.hyAvailabilityDocHooked) return;
    const docRepaint = (event) => {
      const target = event.target;
      if (!(target instanceof Element)) return;
      if (target.closest('#hy-model, #hy-llm-model, #hy-llm-backend, #hy-prompt-generator, [role="option"], [role="listbox"]')) {
        schedulePromptMatchModelAvailability();
      }
    };
    document.addEventListener("click", docRepaint, true);
    document.addEventListener("pointerup", docRepaint, true);
    document.addEventListener("keyup", docRepaint, true);
    document.body.dataset.hyAvailabilityDocHooked = "1";
  };
  const hookTagMatchAutocomplete = () => {
    const root = document.getElementById("hy-tagmatch-tags");
    if (!root || root.dataset.hyAutocompleteHooked) return;
    const input = root.querySelector("textarea");
    const box = ensureTagMatchSuggestBox();
    if (!input || !box) return;
    const refresh = () => syncTagMatchAutocomplete();
    input.addEventListener("input", refresh);
    input.addEventListener("click", refresh);
    input.addEventListener("focus", refresh);
    input.addEventListener("blur", () => setTimeout(hideTagMatchAutocomplete, 120));
    window.addEventListener("resize", refresh);
    window.addEventListener("scroll", refresh, true);
    input.addEventListener("keydown", (event) => {
      if (!box.classList.contains("active")) return;
      const items = Array.from(box.querySelectorAll(".hy-tagmatch-suggest-item"));
      if (!items.length) return;
      let activeIndex = Number.parseInt(box.dataset.activeIndex || "0", 10);
      if (!Number.isFinite(activeIndex)) activeIndex = 0;
      if (event.key === "ArrowDown") {
        highlightTagSuggestion(box, activeIndex + 1);
        event.preventDefault();
        return;
      }
      if (event.key === "ArrowUp") {
        highlightTagSuggestion(box, activeIndex - 1);
        event.preventDefault();
        return;
      }
      if (event.key === "Enter" || event.key === "Tab") {
        const selected = items[activeIndex] || items[0];
        if (selected && applyTagMatchSuggestion(selected.dataset.tag || "")) {
          event.preventDefault();
        }
        return;
      }
      if (event.key === "Escape") {
        hideTagMatchAutocomplete();
      }
    });
    box.addEventListener("mousedown", (event) => {
      const target = event.target instanceof Element ? event.target.closest(".hy-tagmatch-suggest-item") : null;
      if (!target) return;
      event.preventDefault();
      applyTagMatchSuggestion(target.dataset.tag || "");
    });
    root.dataset.hyAutocompleteHooked = "1";
  };
  const hookTagMatchVocabularyState = () => {
    const root = document.getElementById("hy-tagmatch-vocab");
    if (!root || root.dataset.hyStateHooked) return;
    const input = root.querySelector("input, textarea");
    if (!input) return;
    const refresh = () => syncTagMatchAutocomplete();
    input.addEventListener("input", refresh);
    input.addEventListener("change", refresh);
    root.dataset.hyStateHooked = "1";
  };
  const hookExternalQueryArea = () => {
    const root = document.getElementById("hy-external-query-image");
    if (!root || root.dataset.hyExternalHooked) return;
    root.tabIndex = 0;
    root.addEventListener("paste", async (event) => {
      const blob = clipboardImageBlobFromEvent(event);
      if (!blob) return;
      event.preventDefault();
      event.stopPropagation();
      await replaceExternalQueryFromBlob(blob, blob.name || "clipboard-image.png");
    }, true);
    root.addEventListener("dragover", (event) => {
      const dt = event.dataTransfer;
      const hasImageFile = Array.from(dt?.items || []).some((item) => item.kind === "file" && String(item.type || "").startsWith("image/"));
      if (!hasImageFile) return;
      event.preventDefault();
      if (dt) dt.dropEffect = "copy";
    });
    root.addEventListener("drop", async (event) => {
      const files = Array.from(event.dataTransfer?.files || []);
      const imageFile = files.find((file) => String(file.type || "").startsWith("image/"));
      if (!imageFile) return;
      event.preventDefault();
      event.stopPropagation();
      await replaceExternalQueryFromBlob(imageFile, imageFile.name || "dropped-image.png");
    }, true);
    root.addEventListener("click", () => {
      if (typeof root.focus === "function") root.focus();
    });
    if (!document.body.dataset.hyExternalClipboardDocHooked) {
      document.addEventListener("paste", async (event) => {
        const queryRoot = document.getElementById("hy-external-query-image");
        if (!queryRoot) return;
        const target = event.target instanceof Element ? event.target : null;
        const isRelevant = !!(target && queryRoot.contains(target)) || queryRoot.matches?.(":hover") || document.activeElement === queryRoot;
        if (!isRelevant) return;
        const blob = clipboardImageBlobFromEvent(event);
        if (!blob) return;
        event.preventDefault();
        event.stopPropagation();
        await replaceExternalQueryFromBlob(blob, blob.name || "clipboard-image.png");
      }, true);
      document.body.dataset.hyExternalClipboardDocHooked = "1";
    }
    root.dataset.hyExternalHooked = "1";
  };
  applyTooltips();
  hookMarkState();
  hookModelStatusState();
  hookPromptMatchModelAvailability();
  hookTagMatchAutocomplete();
  hookTagMatchVocabularyState();
  hookExternalQueryArea();
  hookHistogramResize();
  hookChangelogOverlay();
  hookSidebarAccordionBehavior();
  hookPreviewDialogTracking();
  hookInlinePreviewNavigationLock();
  hookButtonPreviewReset();
  hookDialogActionHandoff();
  scheduleRepaint();
  new MutationObserver((mutations) => {
    let dialogMutation = false;
    const relevantMutation = mutations.some((mutation) => {
      const nodes = [...mutation.addedNodes, ...mutation.removedNodes];
      return nodes.some((node) => {
        if (!(node instanceof Element)) return false;
        if (node.closest('[role="dialog"], [aria-modal="true"]') || node.matches?.('[role="dialog"], [aria-modal="true"]')) {
          dialogMutation = true;
          return false;
        }
        return !!node.closest('#hy-left-gallery, #hy-right-gallery, #hy-hist, .sidebar-box');
      });
    });
    if (dialogMutation) {
      setTimeout(syncDialogPreviewTarget, 30);
      setTimeout(syncDialogPreviewTarget, 120);
      setTimeout(disablePreviewDialogNavigation, 45);
      setTimeout(disablePreviewDialogNavigation, 135);
    }
    if (!relevantMutation) return;
    applyTooltips();
    hookMarkState();
    hookModelStatusState();
    hookPromptMatchModelAvailability();
    hookTagMatchAutocomplete();
    hookTagMatchVocabularyState();
    hookExternalQueryArea();
    hookHistogramResize();
    hookChangelogOverlay();
  hookSidebarAccordionBehavior();
    hookPreviewDialogTracking();
    hookInlinePreviewNavigationLock();
    hookButtonPreviewReset();
    hookDialogActionHandoff();
    scheduleRepaint();
  }).observe(document.body, { childList: true, subtree: true });
})();
