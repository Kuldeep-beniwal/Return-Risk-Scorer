const API_BASE = ""; // same-origin: Flask serves both API and frontend

const form = document.getElementById("order-form");
const submitBtn = document.getElementById("submit-btn");
const formError = document.getElementById("form-error");

const certificateEmpty = document.getElementById("certificate-empty");
const certificateResult = document.getElementById("certificate-result");
const stampEl = document.getElementById("stamp");
const stampTextEl = document.getElementById("stamp-text");
const probabilityNumberEl = document.getElementById("probability-number");
const thresholdValueEl = document.getElementById("threshold-value");
const explainerProbabilityEl = document.getElementById("explainer-probability");
const riskUpList = document.getElementById("risk-up-list");
const riskDownList = document.getElementById("risk-down-list");
const costFpEl = document.getElementById("cost-fp");
const benefitTpEl = document.getElementById("benefit-tp");

async function loadOptions() {
  const res = await fetch(`${API_BASE}/api/options`);
  if (!res.ok) {
    formError.textContent = "Could not load form options from the server.";
    return;
  }
  const data = await res.json();

  populateSelect("category", data.categorical.category);
  populateSelect("payment_method", data.categorical.payment_method);
  populateSelect("device", data.categorical.device);

  // Pre-fill numeric fields with sensible medians from the training data
  for (const [field, range] of Object.entries(data.numeric_ranges)) {
    const el = document.getElementById(field);
    if (el) el.value = roundSensibly(field, range.median);
  }
}

function roundSensibly(field, value) {
  if (["past_orders", "days_to_deliver", "size_variants_in_order"].includes(field)) {
    return Math.round(value);
  }
  if (field === "past_return_rate") return Number(value.toFixed(2));
  return Number(value.toFixed(1));
}

function populateSelect(id, options) {
  const select = document.getElementById(id);
  select.innerHTML = "";
  for (const opt of options) {
    const option = document.createElement("option");
    option.value = opt;
    option.textContent = opt;
    select.appendChild(option);
  }
}

function collectOrderPayload() {
  const formData = new FormData(form);
  const payload = {};
  for (const [key, value] of formData.entries()) {
    payload[key] = value;
  }
  payload.is_gift_wrapped = document.getElementById("is_gift_wrapped").checked ? 1 : 0;
  return payload;
}

function renderResult(result) {
  certificateEmpty.classList.add("hidden");
  certificateResult.classList.remove("hidden");

  const pct = Math.round(result.deployed_model_probability * 100);
  probabilityNumberEl.textContent = pct;
  thresholdValueEl.textContent = `${Math.round(result.threshold_used * 100)}%`;
  explainerProbabilityEl.textContent = Math.round(result.explainer_probability * 100);
  costFpEl.textContent = result.cost_assumptions.cost_false_positive_inr;
  benefitTpEl.textContent = result.cost_assumptions.benefit_true_positive_inr;

  // Stamp: restart animation each time by cloning the node
  stampEl.classList.remove("settle");
  void stampEl.offsetWidth; // force reflow so the animation restarts
  if (result.flagged_for_review) {
    stampEl.classList.add("flagged");
    stampTextEl.textContent = "FLAG";
  } else {
    stampEl.classList.remove("flagged");
    stampTextEl.textContent = "CLEAR";
  }
  stampEl.classList.add("settle");

  renderLedger(riskUpList, result.risk_increasing_factors, "up");
  renderLedger(riskDownList, result.risk_decreasing_factors, "down");
}

function renderLedger(listEl, factors, direction) {
  listEl.innerHTML = "";
  if (!factors || factors.length === 0) {
    const li = document.createElement("li");
    li.className = "ledger-empty";
    li.textContent = "none material";
    listEl.appendChild(li);
    return;
  }
  for (const f of factors) {
    const li = document.createElement("li");
    const nameSpan = document.createElement("span");
    nameSpan.textContent = f.feature;
    const valueSpan = document.createElement("span");
    valueSpan.className = direction === "up" ? "contribution-up" : "contribution-down";
    const sign = f.log_odds_contribution > 0 ? "+" : "";
    valueSpan.textContent = `${sign}${f.log_odds_contribution.toFixed(2)}`;
    li.appendChild(nameSpan);
    li.appendChild(valueSpan);
    listEl.appendChild(li);
  }
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  formError.textContent = "";
  submitBtn.disabled = true;
  submitBtn.textContent = "Scoring...";

  try {
    const payload = collectOrderPayload();
    const res = await fetch(`${API_BASE}/api/score`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await res.json();

    if (!res.ok) {
      formError.textContent = (data.errors || ["Something went wrong."]).join(" ");
      return;
    }
    renderResult(data);
  } catch (err) {
    formError.textContent = "Could not reach the scoring service. Is the backend running?";
  } finally {
    submitBtn.disabled = false;
    submitBtn.textContent = "Score this order";
  }
});

loadOptions();
