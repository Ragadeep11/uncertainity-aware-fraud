// SafeEscalate Interactive Dashboard Controller

let autoStreamInterval = null;
let streamCount = 0;
let tier0Count = 0;
let tier1Count = 0;
let tier2Count = 0;

let chartCost = null;
let chartReviews = null;

// Initialize when DOM ready
document.addEventListener("DOMContentLoaded", () => {
  if (window.lucide) {
    lucide.createIcons();
  }
  refreshQueue();
  loadBenchmarkData();
});

// Tab Switcher
function switchTab(tabId) {
  document.querySelectorAll(".tab-content").forEach((el) => el.classList.add("hidden"));
  document.querySelectorAll(".tab-btn").forEach((el) => el.classList.remove("active"));

  const targetContent = document.getElementById(`tab-${tabId}`);
  const targetBtn = document.getElementById(`tab-btn-${tabId}`);

  if (targetContent) targetContent.classList.remove("hidden");
  if (targetBtn) targetBtn.classList.add("active");

  if (tabId === "workbench") {
    refreshQueue();
  } else if (tabId === "benchmarks") {
    loadBenchmarkData();
  }

  if (window.lucide) {
    lucide.createIcons();
  }
}

// -------------------------------------------------------------
// TAB 1: LIVE SIMULATOR & CASCADE FLOW
// -------------------------------------------------------------

async function streamNextTransaction() {
  try {
    const res = await fetch("/api/simulate/next");
    if (!res.ok) throw new Error("Stream request failed");
    const packet = await res.json();
    renderStreamUpdate(packet);
  } catch (err) {
    console.error("Error streaming transaction:", err);
  }
}

function toggleAutoStream() {
  const btn = document.getElementById("btn-auto-stream");
  const text = document.getElementById("auto-stream-text");

  if (autoStreamInterval) {
    clearInterval(autoStreamInterval);
    autoStreamInterval = null;
    btn.classList.remove("bg-indigo-600", "text-white");
    btn.classList.add("bg-slate-800", "text-slate-200");
    text.innerText = "Auto Stream";
  } else {
    streamNextTransaction();
    autoStreamInterval = setInterval(streamNextTransaction, 1600);
    btn.classList.add("bg-indigo-600", "text-white");
    btn.classList.remove("bg-slate-800", "text-slate-200");
    text.innerText = "Pause Stream";
  }
}

function renderStreamUpdate(packet) {
  streamCount++;
  if (packet.escalation_tier === 0) tier0Count++;
  else if (packet.escalation_tier === 1) tier1Count++;
  else if (packet.escalation_tier === 2) {
    tier2Count++;
    refreshQueue(); // update HITL queue badge
  }

  // Update Counters
  document.getElementById("stat-total-stream").innerText = streamCount;
  document.getElementById("stat-tier0-count").innerHTML = `${tier0Count} <span class="text-xs text-slate-400 font-normal">(${Math.round(tier0Count/streamCount*100)}%)</span>`;
  document.getElementById("stat-tier1-count").innerHTML = `${tier1Count} <span class="text-xs text-slate-400 font-normal">(${Math.round(tier1Count/streamCount*100)}%)</span>`;
  document.getElementById("stat-tier2-count").innerHTML = `${tier2Count} <span class="text-xs text-slate-400 font-normal">(${Math.round(tier2Count/streamCount*100)}%)</span>`;

  // Update Stage Flow Cards
  document.getElementById("live-tx-amount").innerText = `$${packet.amount.toFixed(2)}`;
  document.getElementById("live-tx-prob").innerText = `${(packet.p_fraud_final * 100).toFixed(1)}%`;

  const setEl = document.getElementById("live-tx-set");
  const setStr = packet.conformal_set.join(", ");
  setEl.innerText = `[${setStr}]`;
  if (packet.conformal_set.length > 1) {
    setEl.className = "text-amber-400 font-bold";
  } else if (packet.conformal_set[0] === "Fraud") {
    setEl.className = "text-rose-400 font-bold";
  } else {
    setEl.className = "text-emerald-400 font-bold";
  }

  document.getElementById("live-tx-uq").innerText = `${packet.aleatoric_uncertainty.toFixed(2)} / ${packet.epistemic_uncertainty.toFixed(2)}`;

  // Stage 3 (Dynamic Evidence)
  const stepupEl = document.getElementById("live-tx-stepup");
  const deviceEl = document.getElementById("live-tx-devicetrust");
  if (packet.evidence_collected) {
    const s2fa = packet.evidence_collected.two_factor_auth_success === 1 ? "PASSED (SMS 2FA)" : "FAILED / TIMEOUT";
    stepupEl.innerText = s2fa;
    stepupEl.className = packet.evidence_collected.two_factor_auth_success === 1 ? "text-emerald-400 font-bold" : "text-rose-400 font-bold";
    deviceEl.innerText = `Trust Score: ${packet.evidence_collected.device_trust_score.toFixed(2)}`;
  } else {
    stepupEl.innerText = "Bypassed / Not Triggered";
    stepupEl.className = "text-slate-500 font-normal";
    deviceEl.innerText = "--";
  }

  // Stage 4 (Action & Tier)
  const actionEl = document.getElementById("live-tx-action");
  actionEl.innerText = packet.final_action;
  if (packet.final_action.startsWith("APPROVE")) {
    actionEl.className = "font-bold text-white px-1.5 py-0.5 rounded text-[11px] bg-emerald-500/30 border border-emerald-500/50";
  } else if (packet.final_action.startsWith("DECLINE")) {
    actionEl.className = "font-bold text-white px-1.5 py-0.5 rounded text-[11px] bg-rose-500/30 border border-rose-500/50";
  } else {
    actionEl.className = "font-bold text-white px-1.5 py-0.5 rounded text-[11px] bg-purple-500/30 border border-purple-500/50";
  }
  document.getElementById("live-tx-tier").innerText = `Tier ${packet.escalation_tier} (${packet.escalation_tier === 0 ? 'Autonomous' : packet.escalation_tier === 1 ? 'Dynamic Evidence' : 'Human Review'})`;

  // Prepend Row to Stream Table
  const tableBody = document.getElementById("stream-table-body");
  if (streamCount === 1) {
    tableBody.innerHTML = ""; // Clear placeholder
  }

  let badgeAction = "";
  if (packet.final_action.startsWith("APPROVE")) {
    badgeAction = `<span class="px-2 py-0.5 rounded bg-emerald-500/20 text-emerald-300 border border-emerald-500/30">${packet.final_action}</span>`;
  } else if (packet.final_action.startsWith("DECLINE")) {
    badgeAction = `<span class="px-2 py-0.5 rounded bg-rose-500/20 text-rose-300 border border-rose-500/30">${packet.final_action}</span>`;
  } else {
    badgeAction = `<span class="px-2 py-0.5 rounded bg-purple-500/20 text-purple-300 border border-purple-500/30 font-bold">ESCALATE TO HITL</span>`;
  }

  const row = document.createElement("tr");
  row.className = "hover:bg-slate-800/40 transition new-row-animation";
  row.innerHTML = `
    <td class="px-4 py-2.5 font-bold text-slate-200">${packet.transaction_id}</td>
    <td class="px-4 py-2.5 text-white font-semibold">$${packet.amount.toFixed(2)}</td>
    <td class="px-4 py-2.5 text-cyan-400 font-bold">${(packet.p_fraud_final * 100).toFixed(1)}%</td>
    <td class="px-4 py-2.5">[${packet.conformal_set.join(", ")}]</td>
    <td class="px-4 py-2.5 text-slate-400">${packet.aleatoric_uncertainty.toFixed(2)} / ${packet.epistemic_uncertainty.toFixed(2)}</td>
    <td class="px-4 py-2.5 text-slate-300">Tier ${packet.escalation_tier}</td>
    <td class="px-4 py-2.5">${badgeAction}</td>
    <td class="px-4 py-2.5 text-slate-400">$${packet.operational_cost.toFixed(2)}</td>
  `;

  tableBody.insertBefore(row, tableBody.firstChild);

  // Keep max 15 rows
  while (tableBody.children.length > 15) {
    tableBody.removeChild(tableBody.lastChild);
  }
}

// -------------------------------------------------------------
// TAB 2: HUMAN-IN-THE-LOOP (HITL) WORKBENCH
// -------------------------------------------------------------

async function refreshQueue() {
  try {
    const res = await fetch("/api/queue");
    if (!res.ok) return;
    const data = await res.json();

    const pending = data.pending_cases;
    const stats = data.stats;

    // Header badge
    const badge = document.getElementById("queue-badge");
    if (stats.pending_count > 0) {
      badge.innerText = stats.pending_count;
      badge.classList.remove("hidden");
    } else {
      badge.classList.add("hidden");
    }

    document.getElementById("queue-stat-pending").innerText = `${stats.pending_count} cases`;
    document.getElementById("queue-stat-exposure").innerText = `$${stats.pending_exposure_usd.toFixed(2)}`;

    const container = document.getElementById("queue-cards-container");
    if (pending.length === 0) {
      container.innerHTML = `
        <div class="col-span-2 text-center py-12 bg-slate-900 border border-slate-800 rounded-2xl">
          <i data-lucide="check-circle-2" class="w-10 h-10 text-emerald-500/80 mx-auto mb-2"></i>
          <p class="text-sm text-slate-400">Queue is clear! All escalated cases have been reviewed.</p>
        </div>
      `;
      if (window.lucide) lucide.createIcons();
      return;
    }

    container.innerHTML = "";
    pending.forEach((item) => {
      const qItem = item.queue_item;
      const dossier = item.dossier;

      const card = document.createElement("div");
      card.className = "bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl space-y-4";
      
      let checklistHtml = dossier.investigator_checklist
        .map((c) => `<li class="flex items-start space-x-2 text-xs text-slate-300"><span class="text-indigo-400 font-bold">&bull;</span><span>${c}</span></li>`)
        .join("");

      let featuresHtml = dossier.top_features
        .map(
          ([name, pct]) => `
            <div>
              <div class="flex justify-between text-[11px] text-slate-400 mb-0.5">
                <span>${name.replace(/_/g, ' ')}</span>
                <span class="font-mono">${pct}%</span>
              </div>
              <div class="w-full bg-slate-800 rounded-full h-1.5 overflow-hidden">
                <div class="bg-indigo-500 h-1.5 rounded-full" style="width: ${Math.min(100, pct * 2)}%"></div>
              </div>
            </div>
          `
        )
        .join("");

      card.innerHTML = `
        <div class="flex items-center justify-between pb-3 border-b border-slate-800">
          <div>
            <div class="flex items-center space-x-2">
              <span class="font-bold text-white text-base font-mono">${dossier.transaction_id}</span>
              <span class="px-2 py-0.5 text-[10px] rounded bg-purple-500/20 text-purple-300 font-bold border border-purple-500/30">Priority: ${qItem.priority}</span>
            </div>
            <div class="text-xs text-slate-400 mt-0.5">Amount: <span class="text-white font-bold">$${dossier.amount.toFixed(2)}</span></div>
          </div>
          <div class="text-right">
            <div class="text-xs text-slate-400">Risk Score</div>
            <div class="text-lg font-bold text-rose-400 font-mono">${dossier.risk_score_pct}%</div>
          </div>
        </div>

        <div class="grid grid-cols-2 gap-3 text-xs bg-slate-950/60 p-3 rounded-xl border border-slate-800/80 font-mono">
          <div>Conformal Set: <span class="text-amber-400 font-bold">[${dossier.conformal_set.join(", ")}]</span></div>
          <div>Uncertainty: <span class="text-indigo-300 font-semibold">${dossier.uncertainty_profile.type}</span></div>
          <div>Aleatoric: <span class="text-slate-300">${dossier.uncertainty_profile.aleatoric.toFixed(2)}</span></div>
          <div>Epistemic: <span class="text-slate-300">${dossier.uncertainty_profile.epistemic.toFixed(2)}</span></div>
        </div>

        <div>
          <div class="text-xs font-bold uppercase tracking-wider text-slate-400 mb-1.5">Anomaly Feature Drivers</div>
          <div class="space-y-2 bg-slate-950/40 p-3 rounded-xl border border-slate-800/60">
            ${featuresHtml}
          </div>
        </div>

        <div>
          <div class="text-xs font-bold uppercase tracking-wider text-slate-400 mb-1.5">Investigator Checklist</div>
          <ul class="space-y-1 bg-slate-950/40 p-3 rounded-xl border border-slate-800/60">
            ${checklistHtml}
          </ul>
        </div>

        <div class="text-xs text-slate-400 italic bg-slate-800/40 p-2.5 rounded-lg border border-slate-700/40">
          "${dossier.rationale}"
        </div>

        <div class="pt-2 flex items-center space-x-3">
          <button onclick="resolveCase('${dossier.transaction_id}', 'APPROVE')" class="flex-1 py-2 bg-emerald-600 hover:bg-emerald-500 text-white text-xs font-bold rounded-xl transition shadow-lg shadow-emerald-600/20">
            Approve Transaction
          </button>
          <button onclick="resolveCase('${dossier.transaction_id}', 'DECLINE')" class="flex-1 py-2 bg-rose-600 hover:bg-rose-500 text-white text-xs font-bold rounded-xl transition shadow-lg shadow-rose-600/20">
            Decline Fraud
          </button>
        </div>
      `;

      container.appendChild(card);
    });

    if (window.lucide) lucide.createIcons();
  } catch (err) {
    console.error("Error refreshing queue:", err);
  }
}

async function resolveCase(txId, decision) {
  try {
    const res = await fetch("/api/queue/resolve", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        transaction_id: txId,
        decision: decision,
        notes: `Investigator resolved via workbench: ${decision}`,
      }),
    });
    if (res.ok) {
      refreshQueue();
    }
  } catch (err) {
    console.error("Error resolving case:", err);
  }
}

// -------------------------------------------------------------
// TAB 3: RESEARCH EVALUATION & BENCHMARKS
// -------------------------------------------------------------

async function loadBenchmarkData() {
  try {
    const res = await fetch("/api/benchmark/latest");
    if (!res.ok) return;
    const data = await res.json();
    renderBenchmarkResults(data);
  } catch (err) {
    console.error("Error loading benchmark:", err);
  }
}

async function runBenchmarkExperiment() {
  const btn = document.getElementById("btn-run-benchmark");
  const text = document.getElementById("benchmark-btn-text");
  btn.disabled = true;
  text.innerText = "Simulating 6,000 Transactions...";

  try {
    const res = await fetch("/api/benchmark/run", { method: "POST" });
    if (res.ok) {
      const data = await res.json();
      renderBenchmarkResults(data);
    }
  } catch (err) {
    console.error("Benchmark error:", err);
  } finally {
    btn.disabled = false;
    text.innerText = "Re-run Benchmark";
  }
}

function renderBenchmarkResults(data) {
  const models = data.models;
  const se = models["SafeEscalate_Proposed"];
  const b1 = models["Static_Threshold_0.5"];
  const b2 = models["Dual_Threshold_Abstention"];
  const b3 = models["Cost_Sensitive_Threshold"];

  if (!se) return;

  document.getElementById("bm-kpi-workload").innerText = `${se.human_workload_reduction_pct}%`;
  document.getElementById("bm-kpi-cost").innerText = `${se.cost_savings_pct}%`;
  document.getElementById("bm-kpi-recall").innerText = `${se.fraud_recall_pct}%`;

  // Render Table
  const tbody = document.getElementById("benchmark-table-body");
  tbody.innerHTML = "";

  const allModels = [b1, b2, b3, se];
  allModels.forEach((m) => {
    if (!m) return;
    const isProposed = m.name.includes("SafeEscalate");
    const row = document.createElement("tr");
    row.className = isProposed ? "bg-indigo-950/40 border-indigo-500/30 font-bold" : "hover:bg-slate-800/40";
    row.innerHTML = `
      <td class="px-4 py-3 text-white flex items-center space-x-2">
        ${isProposed ? '<span class="px-1.5 py-0.5 rounded bg-indigo-500 text-white text-[10px]">Ours</span>' : ''}
        <span>${m.name}</span>
      </td>
      <td class="px-4 py-3 text-cyan-400">$${m.total_cost.toLocaleString()}</td>
      <td class="px-4 py-3 text-rose-400">$${m.fraud_loss.toLocaleString()}</td>
      <td class="px-4 py-3 text-amber-400">$${m.friction_cost.toLocaleString()}</td>
      <td class="px-4 py-3 text-purple-400">$${m.human_cost.toLocaleString()}</td>
      <td class="px-4 py-3">${m.human_reviews.toLocaleString()}</td>
      <td class="px-4 py-3 text-emerald-400">${m.fraud_recall_pct}%</td>
      <td class="px-4 py-3 text-slate-300">$${m.cost_per_tx.toFixed(2)}</td>
    `;
    tbody.appendChild(row);
  });

  // Render Charts
  renderBenchmarkCharts(allModels);
}

function renderBenchmarkCharts(models) {
  const labels = models.map((m) => m.name.replace(/ \(.*\)/, ''));

  // Cost breakdown
  const fraudLosses = models.map((m) => m.fraud_loss);
  const frictionCosts = models.map((m) => m.friction_cost);
  const humanCosts = models.map((m) => m.human_cost);
  const queryCosts = models.map((m) => m.query_cost);

  const ctxCost = document.getElementById("chart-cost-breakdown");
  if (chartCost) chartCost.destroy();

  chartCost = new Chart(ctxCost, {
    type: "bar",
    data: {
      labels: labels,
      datasets: [
        { label: "Fraud Loss ($)", data: fraudLosses, backgroundColor: "#f43f5e" },
        { label: "Customer Friction ($)", data: frictionCosts, backgroundColor: "#f59e0b" },
        { label: "Human Review ($)", data: humanCosts, backgroundColor: "#a855f7" },
        { label: "Dynamic Evidence ($)", data: queryCosts, backgroundColor: "#06b6d4" },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: { stacked: true, ticks: { color: "#94a3b8", font: { size: 10 } }, grid: { display: false } },
        y: { stacked: true, ticks: { color: "#94a3b8", font: { size: 10 } }, grid: { color: "#334155" } },
      },
      plugins: {
        legend: { labels: { color: "#cbd5e1", font: { size: 11 } } },
      },
    },
  });

  // Reviews chart
  const reviews = models.map((m) => m.human_reviews);
  const ctxReviews = document.getElementById("chart-reviews");
  if (chartReviews) chartReviews.destroy();

  chartReviews = new Chart(ctxReviews, {
    type: "bar",
    data: {
      labels: labels,
      datasets: [
        {
          label: "Cases Escalated to Human Investigator",
          data: reviews,
          backgroundColor: ["#64748b", "#a855f7", "#64748b", "#4f46e5"],
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: { ticks: { color: "#94a3b8", font: { size: 10 } }, grid: { display: false } },
        y: { ticks: { color: "#94a3b8", font: { size: 10 } }, grid: { color: "#334155" } },
      },
      plugins: {
        legend: { labels: { color: "#cbd5e1", font: { size: 11 } } },
      },
    },
  });
}

// -------------------------------------------------------------
// TAB 4: COST MATRIX & POLICY TUNER
// -------------------------------------------------------------

function updateCostSliders() {
  const fp = document.getElementById("slider-cost-fp").value;
  const human = document.getElementById("slider-cost-human").value;
  const evidence = document.getElementById("slider-cost-evidence").value;
  const highVal = document.getElementById("slider-high-value").value;

  document.getElementById("val-cost-fp").innerText = `$${parseFloat(fp).toFixed(2)}`;
  document.getElementById("val-cost-human").innerText = `$${parseFloat(human).toFixed(2)}`;
  document.getElementById("val-cost-evidence").innerText = `$${parseFloat(evidence).toFixed(2)}`;
  document.getElementById("val-high-value").innerText = `$${parseInt(highVal).toLocaleString()}`;
}

async function savePolicyConfig() {
  const fp = parseFloat(document.getElementById("slider-cost-fp").value);
  const human = parseFloat(document.getElementById("slider-cost-human").value);
  const evidence = parseFloat(document.getElementById("slider-cost-evidence").value);
  const highVal = parseFloat(document.getElementById("slider-high-value").value);

  try {
    const res = await fetch("/api/config/update", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        cost_false_positive: fp,
        cost_human_review: human,
        cost_evidence_acquisition: evidence,
        high_value_amount_threshold: highVal,
      }),
    });
    if (res.ok) {
      alert("Policy parameters updated successfully! Future transactions will be routed with the updated cost thresholds.");
    }
  } catch (err) {
    console.error("Error updating policy config:", err);
  }
}
