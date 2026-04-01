const defaults = {
  company_age_months: "24",
  monthly_burn_usd: "$120,000",
  runway_months: "10",
  team_size: "18",
  founder_exits: "0",
  customer_growth_pct: "12%",
  support_tickets_last_30_days: "14",
  annual_revenue_usd: "$1,900,000",
  market_segment: "SaaS",
  growth_stage: "Series-A",
  remote_friendly: "yes",
  investor_tier: "tier-2-vc",
};

const form = document.getElementById("prediction-form");
const formError = document.getElementById("form-error");
const resultPanel = document.getElementById("prediction-result");
const predictionBadge = document.getElementById("prediction-badge");
const predictionProbability = document.getElementById("prediction-probability");
const predictionLabel = document.getElementById("prediction-label");
const predictionRequestId = document.getElementById("prediction-request-id");

function formatPercent(value) {
  return `${(Number(value) * 100).toFixed(1)}%`;
}

function formatMetric(value) {
  return Number(value ?? 0).toFixed(4);
}

function setText(id, value) {
  document.getElementById(id).textContent = value;
}

function setLatencyBar(id, value) {
  const width = Math.min(Number(value || 0), 250);
  document.getElementById(id).style.width = `${(width / 250) * 100}%`;
}

async function loadSummary() {
  const [healthResponse, summaryResponse] = await Promise.all([
    fetch("/health"),
    fetch("/dashboard/summary"),
  ]);

  const health = await healthResponse.json();
  const summary = await summaryResponse.json();
  const metrics = summary.model_metadata.metrics;
  const leaderboard = summary.artifact_metrics.results || {};

  setText("health-status", health.status.toUpperCase());
  setText("selected-model", summary.selected_model.replaceAll("_", " "));
  setText("artifact-version", summary.artifact_metrics.experiment_tracking?.artifact_version || "n/a");

  setText("metric-precision", formatMetric(metrics.precision));
  setText("metric-recall", formatMetric(metrics.recall));
  setText("metric-roc-auc", formatMetric(metrics.roc_auc));
  setText("metric-selection-score", formatMetric(metrics.selection_score));

  const leaderboardRows = document.getElementById("leaderboard-rows");
  leaderboardRows.innerHTML = "";
  Object.entries(leaderboard)
    .sort(([, a], [, b]) => b.selection_score - a.selection_score)
    .forEach(([name, values]) => {
      const row = document.createElement("div");
      row.className = "leaderboard-row";
      const isSelected = name === summary.selected_model;
      row.innerHTML = `
        <strong>${name.replaceAll("_", " ")} ${isSelected ? '<span class="chip">deployed</span>' : ""}</strong>
        <span>${formatMetric(values.precision)}</span>
        <span>${formatMetric(values.recall)}</span>
        <span>${formatMetric(values.roc_auc)}</span>
        <span>${formatMetric(values.selection_score)}</span>
      `;
      leaderboardRows.appendChild(row);
    });

  const featurePills = document.getElementById("feature-pills");
  featurePills.innerHTML = "";
  (summary.model_metadata.model_features || summary.features).forEach((feature) => {
    const pill = document.createElement("span");
    pill.className = "pill";
    pill.textContent = feature.replaceAll("_", " ");
    featurePills.appendChild(pill);
  });
}

async function loadMetrics() {
  const response = await fetch("/metrics");
  const metrics = await response.json();

  setText("requests-total", metrics.requests_total);
  setText("request-error-rate", formatPercent(metrics.request_error_rate));
  setText("predictions-total", metrics.predictions_total);
  setText("prediction-error-rate", formatPercent(metrics.prediction_error_rate));
  setText("avg-request-latency", `${Number(metrics.avg_request_latency_ms).toFixed(2)} ms`);
  setText("avg-prediction-latency", `${Number(metrics.avg_prediction_latency_ms).toFixed(2)} ms`);
  setLatencyBar("request-latency-bar", metrics.avg_request_latency_ms);
  setLatencyBar("prediction-latency-bar", metrics.avg_prediction_latency_ms);
}

function resetForm() {
  Object.entries(defaults).forEach(([name, value]) => {
    const element = form.elements.namedItem(name);
    if (element) {
      element.value = value;
    }
  });
  formError.hidden = true;
}

async function submitPrediction(event) {
  event.preventDefault();
  formError.hidden = true;

  const payload = Object.fromEntries(new FormData(form).entries());
  const response = await fetch("/predict", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    const error = await response.json();
    const details = (error.detail || [])
      .map((item) => `${item.loc[item.loc.length - 1]}: ${item.msg}`)
      .join(" | ");
    formError.textContent = details || "Prediction request failed.";
    formError.hidden = false;
    resultPanel.hidden = true;
    return;
  }

  const result = await response.json();
  const probability = Number(result.churn_probability);
  const requestId = response.headers.get("X-Request-ID") || "-";
  const highRisk = probability >= 0.5;

  predictionBadge.textContent = highRisk ? "High churn risk" : "Lower churn risk";
  predictionBadge.classList.toggle("high-risk", highRisk);
  predictionProbability.textContent = `${(probability * 100).toFixed(1)}%`;
  predictionLabel.textContent = String(result.predicted_label);
  predictionRequestId.textContent = requestId;
  resultPanel.hidden = false;

  await loadMetrics();
}

document.getElementById("reset-form").addEventListener("click", resetForm);
form.addEventListener("submit", submitPrediction);

resetForm();
loadSummary();
loadMetrics();
setInterval(loadMetrics, 10000);
