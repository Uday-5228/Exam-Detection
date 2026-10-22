"use strict";

const feedGrid = document.getElementById("feed-grid");
const liveAlerts = document.getElementById("live-alerts");
const eventRows = document.getElementById("event-rows");
const cameraFilter = document.getElementById("camera-filter");
const typeFilter = document.getElementById("type-filter");
const liveIndicator = document.getElementById("live-indicator");
const modal = document.getElementById("snapshot-modal");
const streamCards = new Map();
const streamNames = new Map();
const knownAlerts = new Set();
const alertTimers = new Map();

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, (character) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
  })[character]);
}

function timestampLabel(value) {
  const date = new Date(Number(value) * 1000);
  return Number.isNaN(date.getTime()) ? "—" : date.toLocaleString();
}

function snapshotUrl(path) {
  const filename = String(path || "").replaceAll("\\", "/").split("/").pop();
  return filename ? `/snapshots/${encodeURIComponent(filename)}` : "";
}

function buildFeed(stream) {
  const card = document.createElement("article");
  card.className = "feed-card";
  card.dataset.streamId = stream.id;
  card.innerHTML = `
    <div class="feed-image-wrap">
      <img class="feed-image" src="/stream/${encodeURIComponent(stream.id)}" alt="Live feed: ${escapeHtml(stream.name)}">
      <div class="offline-overlay"><strong>Offline</strong><span>Waiting for camera</span></div>
    </div>
    <div class="feed-info"><span class="feed-name">${escapeHtml(stream.name)}</span><span class="feed-status">Starting</span></div>`;
  const overlay = card.querySelector(".offline-overlay");
  const status = card.querySelector(".feed-status");
  streamCards.set(stream.id, { card, overlay, status });
  streamNames.set(stream.id, stream.name);
  return card;
}

function updateStreams(streams) {
  const incoming = new Set(streams.map((stream) => stream.id));
  for (const [id, item] of streamCards.entries()) {
    if (!incoming.has(id)) {
      item.card.remove();
      streamCards.delete(id);
    }
  }
  for (const stream of streams) {
    if (!streamCards.has(stream.id)) {
      feedGrid.append(buildFeed(stream));
      cameraFilter.add(new Option(stream.name, stream.id));
    }
    streamNames.set(stream.id, stream.name);
    const item = streamCards.get(stream.id);
    const online = stream.status === "online";
    item.overlay.hidden = online;
    item.overlay.querySelector("span").textContent = stream.error || "Waiting for camera";
    item.status.classList.toggle("online", online);
    item.status.textContent = online ? `${Number(stream.fps || 0).toFixed(1)} FPS` : "Offline";
  }
  document.getElementById("feed-summary").textContent = `${streams.length} configured`;
}

async function refreshMetrics() {
  try {
    const response = await fetch("/api/metrics", { cache: "no-store" });
    if (!response.ok) throw new Error("Metrics unavailable");
    const metrics = await response.json();
    document.getElementById("online-count").textContent = `${metrics.streams_online || 0} / ${Object.keys(metrics.streams || {}).length}`;
    document.getElementById("alert-count").textContent = String(metrics.total_alerts_today || 0);
    const latency = Number(metrics.average_latency_ms || 0);
    document.getElementById("latency-value").textContent = `${latency.toFixed(0)} ms`;
    updateStreams(Object.values(metrics.streams || {}));
  } catch (error) {
    document.getElementById("feed-summary").textContent = "Backend reconnecting";
  }
}

function flashStream(streamId) {
  const item = streamCards.get(streamId);
  if (!item) return;
  if (alertTimers.has(streamId)) window.clearTimeout(alertTimers.get(streamId));
  item.card.classList.add("alerting");
  alertTimers.set(streamId, window.setTimeout(() => {
    item.card.classList.remove("alerting");
    alertTimers.delete(streamId);
  }, 5000));
}

function showSnapshot(alert) {
  const url = snapshotUrl(alert.snapshot_path);
  if (!url) return;
  document.getElementById("modal-image").src = url;
  document.getElementById("modal-caption").textContent = `${streamNames.get(alert.stream_id) || alert.stream_id} · ${alert.type} · ${alert.detail || "Suspicious activity for review"}`;
  modal.showModal();
}

function addLiveAlert(alert) {
  if (!alert || knownAlerts.has(alert.id)) return;
  knownAlerts.add(alert.id);
  if (knownAlerts.size > 500) knownAlerts.delete(knownAlerts.values().next().value);
  flashStream(alert.stream_id);
  const empty = liveAlerts.querySelector(".empty-state");
  if (empty) empty.remove();
  const button = document.createElement("button");
  button.type = "button";
  button.className = "alert-card";
  button.setAttribute("aria-label", `Open ${alert.type} snapshot`);
  const image = snapshotUrl(alert.snapshot_path);
  button.innerHTML = `
    <img class="alert-thumb" src="${escapeHtml(image)}" alt="Snapshot thumbnail">
    <span class="alert-copy">
      <span class="alert-title"><span>Suspicious activity</span><span>${escapeHtml(alert.type)}</span></span>
      <span class="alert-camera">${escapeHtml(streamNames.get(alert.stream_id) || alert.stream_id)} · ${escapeHtml(timestampLabel(alert.timestamp))}</span>
      <span class="alert-reason">${escapeHtml(alert.detail)}</span>
    </span>`;
  button.addEventListener("click", () => showSnapshot(alert));
  liveAlerts.prepend(button);
  while (liveAlerts.children.length > 60) liveAlerts.lastElementChild.remove();
  refreshEvents();
  refreshMetrics();
}

async function refreshStreams() {
  const response = await fetch("/api/streams", { cache: "no-store" });
  if (!response.ok) throw new Error("Stream status unavailable");
  updateStreams(await response.json());
}

async function refreshEvents() {
  const query = new URLSearchParams({ limit: "100" });
  if (cameraFilter.value) query.set("stream_id", cameraFilter.value);
  if (typeFilter.value) query.set("type", typeFilter.value);
  try {
    const response = await fetch(`/api/events?${query.toString()}`, { cache: "no-store" });
    if (!response.ok) throw new Error("Events unavailable");
    renderEvents(await response.json());
  } catch (error) {
    eventRows.innerHTML = '<tr><td colspan="5" class="empty-cell">Event log is unavailable.</td></tr>';
  }
}

function renderEvents(events) {
  if (!events.length) {
    eventRows.innerHTML = '<tr><td colspan="5" class="empty-cell">No events match these filters.</td></tr>';
    return;
  }
  eventRows.replaceChildren();
  for (const event of events) {
    const row = document.createElement("tr");
    const streamName = streamNames.get(event.stream_id) || event.stream_id;
    row.innerHTML = `
      <td>${escapeHtml(timestampLabel(event.timestamp))}</td>
      <td>${escapeHtml(streamName)}</td>
      <td><span class="type-pill">${escapeHtml(event.type)}</span></td>
      <td>${escapeHtml(event.detail)}</td>
      <td>${event.reviewed ? '<span class="reviewed-label">Reviewed</span>' : `<button class="secondary-button review-button" type="button" data-event-id="${escapeHtml(event.id)}">Mark reviewed</button>`}</td>`;
    eventRows.append(row);
  }
  eventRows.querySelectorAll(".review-button").forEach((button) => {
    button.addEventListener("click", async () => {
      button.disabled = true;
      try {
        const response = await fetch(`/api/events/${encodeURIComponent(button.dataset.eventId)}/review`, { method: "POST" });
        if (!response.ok) throw new Error("Review update failed");
        await refreshEvents();
      } catch (error) {
        button.disabled = false;
      }
    });
  });
}

function connectAlerts() {
  const scheme = location.protocol === "https:" ? "wss:" : "ws:";
  const socket = new WebSocket(`${scheme}//${location.host}/ws/alerts`);
  socket.addEventListener("open", () => liveIndicator.classList.add("connected"));
  socket.addEventListener("message", (message) => {
    try { addLiveAlert(JSON.parse(message.data)); } catch (error) { /* Ignore malformed alert payloads. */ }
  });
  socket.addEventListener("close", () => {
    liveIndicator.classList.remove("connected");
    window.setTimeout(connectAlerts, 1500);
  });
  socket.addEventListener("error", () => socket.close());
}

cameraFilter.addEventListener("change", refreshEvents);
typeFilter.addEventListener("change", refreshEvents);
document.getElementById("refresh-events").addEventListener("click", refreshEvents);
document.getElementById("close-modal").addEventListener("click", () => modal.close());
modal.addEventListener("click", (event) => { if (event.target === modal) modal.close(); });

refreshMetrics();
refreshStreams().catch(() => { document.getElementById("feed-summary").textContent = "Backend reconnecting"; });
refreshEvents();
connectAlerts();
window.setInterval(refreshMetrics, 2000);
window.setInterval(refreshStreams, 2000);
window.setInterval(refreshEvents, 5000);
