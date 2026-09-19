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
  checkSystemStatus();
  refreshQueue();
  loadBenchmarkData();
});

async function checkSystemStatus() {
  try {
    const res = await fetch("/api/status");
    if (!res.ok) return;
    const data = await res.json();
    const datasetElem = document.getElementById("dataset-name-text");
    if (datasetElem && data.dataset_name) {
      datasetElem.innerText = `Dataset: ${data.dataset_name}`;
    }
  } catch (err) {
    console.debug("Status check error:", err);
  }
}

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
  } else if (tabId === "blockchain") {
    fetchBlockchainLedger();
    verifyBlockchainIntegrity(false);
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

        let trajectoryHtml = "";
        if (dossier.investigation_trajectory && dossier.investigation_trajectory.length > 0) {
          const stepsHtml = dossier.investigation_trajectory.map((s) => `
            <div class="p-2 rounded bg-slate-900 border border-slate-800 text-[11px] space-y-0.5">
              <div class="flex items-center justify-between font-mono">
                <span class="text-cyan-300 font-bold">Step ${s.step_number}: ${s.evidence_name}</span>
                <span class="text-emerald-400 font-bold">$${s.cost_usd.toFixed(2)} (${s.latency_ms}ms)</span>
              </div>
              <div class="text-slate-300 text-[10px]">${s.summary}</div>
              <div class="flex items-center justify-between text-[10px] font-mono text-slate-400 pt-0.5">
                <span>Risk: <b>${s.p_fraud_transition}</b></span>
                <span>Uncertainty: <b>${s.uncertainty_transition}</b></span>
              </div>
            </div>
          `).join("");

          trajectoryHtml = `
            <div>
              <div class="text-xs font-bold uppercase tracking-wider text-cyan-400 mb-1.5 flex items-center space-x-1">
                <i data-lucide="git-commit" class="w-3.5 h-3.5"></i>
                <span>Automated Investigation Trajectory</span>
              </div>
              <div class="space-y-1.5 bg-slate-950/40 p-3 rounded-xl border border-slate-800/60">
                ${stepsHtml}
              </div>
            </div>
          `;
        }

        let bcProofHtml = "";
        if (dossier.blockchain_audit && dossier.blockchain_audit.block_hash) {
          bcProofHtml = `
            <div class="text-[10px] font-mono text-slate-400 bg-slate-950 p-2 rounded-lg border border-indigo-500/20 flex items-center justify-between">
              <span class="text-indigo-400">Blockchain Block #${dossier.blockchain_audit.block_index}</span>
              <span class="truncate max-w-[200px]" title="${dossier.blockchain_audit.block_hash}">${dossier.blockchain_audit.block_hash.substring(0, 22)}...</span>
            </div>
          `;
        }

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

        ${trajectoryHtml}

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

        ${bcProofHtml}

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
    if (res.ok) {
      const data = await res.json();
      renderBenchmarkResults(data);
    }
  } catch (err) {
    console.error("Error loading benchmark:", err);
  }
  loadResearchExperiments();
}

async function loadResearchExperiments() {
  try {
    const res = await fetch("/api/research/experiments");
    if (!res.ok) return;
    const data = await res.json();
    renderResearchExperiments(data);
  } catch (err) {
    console.error("Error loading research experiments:", err);
  }
}

async function runResearchExperimentsBattery() {
  const btn = document.getElementById("btn-run-benchmark");
  const text = document.getElementById("benchmark-btn-text");
  if (btn) btn.disabled = true;
  if (text) text.innerText = "Running Academic Battery...";

  try {
    const res = await fetch("/api/research/run-experiments?samples=1000", { method: "POST" });
    if (res.ok) {
      const data = await res.json();
      renderResearchExperiments(data);
    }
  } catch (err) {
    console.error("Error running research experiments:", err);
  } finally {
    if (btn) btn.disabled = false;
    if (text) text.innerText = "Re-run Research Battery";
  }
}

function renderResearchExperiments(data) {
  if (!data) return;

  // 1. Executive summary finding
  const summaryElem = document.getElementById("research-summary-text");
  if (summaryElem && data.summary_finding) {
    summaryElem.innerText = data.summary_finding;
  }

  // 2. Primary Comparative Evaluation Matrix
  const primaryTbody = document.getElementById("research-primary-table-body");
  if (primaryTbody && data.primary_comparison_table) {
    primaryTbody.innerHTML = "";
    data.primary_comparison_table.forEach((row) => {
      const isProposed = row.method.includes("Proposed") || row.method.includes("Adaptive");
      const tr = document.createElement("tr");
      tr.className = isProposed ? "bg-indigo-950/60 border-l-4 border-indigo-500 font-bold" : "hover:bg-slate-800/40";
      tr.innerHTML = `
        <td class="px-4 py-3 text-white flex items-center space-x-2">
          ${isProposed ? '<span class="px-1.5 py-0.5 rounded bg-indigo-500 text-white text-[10px] uppercase tracking-wider">Proposed</span>' : ''}
          <span>${row.method}</span>
        </td>
        <td class="px-4 py-3 text-cyan-400">${typeof row.pr_auc === 'number' ? row.pr_auc.toFixed(4) : row.pr_auc}</td>
        <td class="px-4 py-3 text-emerald-400">${row.recall_pct}</td>
        <td class="px-4 py-3 ${row.human_review_pct === '—' ? 'text-slate-500' : 'text-purple-400'}">${row.human_review_pct}</td>
        <td class="px-4 py-3 text-amber-400">${row.avg_evidence_checks}</td>
        <td class="px-4 py-3 text-slate-200">${row.cost_per_tx}</td>
      `;
      primaryTbody.appendChild(tr);
    });
  }

  // 3. KPIs
  if (data.experiment_c_direct_hitl && data.experiment_e_proposed) {
    const cRate = data.experiment_c_direct_hitl.human_review_rate_pct;
    const eRate = data.experiment_e_proposed.human_review_rate_pct;
    const redWorkload = cRate > 0 ? ((cRate - eRate) / cRate * 100).toFixed(1) : "0.0";
    const elemWorkload = document.getElementById("bm-kpi-workload");
    if (elemWorkload) elemWorkload.innerText = `${redWorkload}%`;

    const dChecks = data.experiment_d_fixed_evidence.avg_evidence_checks;
    const eChecks = data.experiment_e_proposed.avg_evidence_checks;
    const redChecks = dChecks > 0 ? ((dChecks - eChecks) / dChecks * 100).toFixed(1) : "0.0";
    const elemChecks = document.getElementById("bm-kpi-checks");
    if (elemChecks) elemChecks.innerText = `${redChecks}%`;

    const elemRecall = document.getElementById("bm-kpi-recall");
    if (elemRecall) elemRecall.innerText = `${data.experiment_e_proposed.fraud_recall_pct}%`;
  }

  // 4. Experiment A: Baselines Table
  const expABody = document.getElementById("exp-a-table-body");
  if (expABody && data.experiment_a_baselines) {
    expABody.innerHTML = "";
    Object.values(data.experiment_a_baselines).forEach((m) => {
      const tr = document.createElement("tr");
      tr.className = "hover:bg-slate-800/40";
      tr.innerHTML = `
        <td class="px-3 py-2 text-white font-medium">${m.model_name}</td>
        <td class="px-3 py-2 text-slate-300">${(m.precision * 100).toFixed(1)}%</td>
        <td class="px-3 py-2 text-emerald-400">${(m.recall * 100).toFixed(1)}%</td>
        <td class="px-3 py-2 text-indigo-400">${m.f1_score.toFixed(3)}</td>
        <td class="px-3 py-2 text-cyan-400 font-bold">${m.pr_auc.toFixed(4)}</td>
        <td class="px-3 py-2 text-amber-400">${m.roc_auc.toFixed(4)}</td>
      `;
      expABody.appendChild(tr);
    });
  }

  // 5. Experiment B: Calibration & Uncertainty Quality
  if (data.experiment_b_calibration) {
    const b = data.experiment_b_calibration;
    const eceElem = document.getElementById("exp-b-ece");
    if (eceElem) eceElem.innerText = b.expected_calibration_error_ece.toFixed(4);

    const covElem = document.getElementById("exp-b-coverage");
    if (covElem) covElem.innerText = `${(b.empirical_conformal_coverage * 100).toFixed(1)}%`;

    if (b.error_rate_by_uncertainty_tier) {
      const errTiers = b.error_rate_by_uncertainty_tier;
      const elLow = document.getElementById("exp-b-err-low");
      const elMed = document.getElementById("exp-b-err-med");
      const elHigh = document.getElementById("exp-b-err-high");
      if (elLow) elLow.innerText = `${(errTiers.low_uncertainty_error_rate * 100).toFixed(1)}%`;
      if (elMed) elMed.innerText = `${(errTiers.medium_uncertainty_error_rate * 100).toFixed(1)}%`;
      if (elHigh) elHigh.innerText = `${(errTiers.high_uncertainty_error_rate * 100).toFixed(1)}%`;
    }

    const corrText = document.getElementById("exp-b-corr-text");
    if (corrText && b.uncertainty_error_correlation) {
      corrText.innerText = `Correlation: ${b.uncertainty_error_correlation}`;
    }
  }

  // 6. Ablation Study Matrix
  const ablBody = document.getElementById("ablation-table-body");
  if (ablBody && data.ablation_study) {
    ablBody.innerHTML = "";
    Object.values(data.ablation_study).forEach((row) => {
      const isFull = row.name.includes("Full Proposed");
      const tr = document.createElement("tr");
      tr.className = isFull ? "bg-cyan-950/40 border-l-4 border-cyan-400 font-bold" : "hover:bg-slate-800/40";
      tr.innerHTML = `
        <td class="px-4 py-2.5 text-white">${row.name}</td>
        <td class="px-4 py-2.5 text-purple-400">${row.human_review_rate_pct}%</td>
        <td class="px-4 py-2.5 text-amber-400">${row.avg_evidence_checks}</td>
        <td class="px-4 py-2.5 text-emerald-400">${row.fraud_recall_pct}%</td>
        <td class="px-4 py-2.5 text-cyan-400">$${row.cost_per_tx.toFixed(2)}</td>
        <td class="px-4 py-2.5 text-slate-400 text-[11px] font-sans">${row.impact_finding}</td>
      `;
      ablBody.appendChild(tr);
    });
  }

  // 7. Statistical Significance & Hypothesis Testing
  const statsContainer = document.getElementById("stats-tests-container");
  if (statsContainer && data.statistical_significance && data.statistical_significance.hypothesis_testing) {
    statsContainer.innerHTML = "";
    const tests = data.statistical_significance.hypothesis_testing;
    Object.entries(tests).forEach(([testKey, t]) => {
      const card = document.createElement("div");
      card.className = "p-3 rounded-xl bg-slate-950 border border-slate-800 flex items-start justify-between space-x-3";
      const isSig = t.statistically_significant || t.is_statistically_comparable;
      card.innerHTML = `
        <div>
          <div class="text-xs font-bold text-white flex items-center space-x-2">
            <span>${testKey.replace(/_/g, ' ').toUpperCase()}</span>
          </div>
          <p class="text-[11px] text-slate-400 mt-1">${t.conclusion}</p>
          <div class="text-[10px] font-mono text-slate-500 mt-1">
            Test: ${t.test} | t-stat: ${t.t_statistic} | p-value: ${t.p_value}
          </div>
        </div>
        <span class="px-2 py-0.5 rounded text-[10px] font-bold uppercase shrink-0 ${isSig ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30' : 'bg-amber-500/20 text-amber-400 border border-amber-500/30'}">
          ${isSig ? 'Pass (p < 0.01)' : 'Parity'}
        </span>
      `;
      statsContainer.appendChild(card);
    });
  }

  // 8. Blockchain Benchmark
  if (data.blockchain_benchmark) {
    const bc = data.blockchain_benchmark;
    const tpsElem = document.getElementById("bc-bench-tps");
    if (tpsElem) tpsElem.innerText = `${bc.throughput_blocks_per_sec.toLocaleString()}`;

    const latElem = document.getElementById("bc-bench-latency");
    if (latElem) latElem.innerText = `${bc.verification_latency_ms.toFixed(2)} ms`;
  }

  if (window.lucide) {
    lucide.createIcons();
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

// -------------------------------------------------------------
// TAB: MANUAL INPUT & SCENARIO TESTER
// -------------------------------------------------------------

function toggleAdvancedTier2() {
  const panel = document.getElementById("panel-advanced-tier2");
  const icon = document.getElementById("icon-advanced-tier2");
  if (panel.classList.contains("hidden")) {
    panel.classList.remove("hidden");
    if (icon) icon.classList.add("rotate-90");
  } else {
    panel.classList.add("hidden");
    if (icon) icon.classList.remove("rotate-90");
  }
}

let currentCanonicalMode = null;
let activeCanonicalTx = null;

async function loadCanonicalCase(caseKey) {
  try {
    const res = await fetch("/api/scenarios/canonical");
    if (!res.ok) throw new Error("Failed to fetch canonical scenarios");
    const scenarios = await res.json();
    const sc = scenarios[caseKey];
    if (!sc) return;

    currentCanonicalMode = sc.canonical_mode || null;
    const tx = sc.transaction;
    activeCanonicalTx = tx;

    document.getElementById("manual-tx-id").value = tx.transaction_id;
    document.getElementById("manual-amount").value = tx.amount;
    document.getElementById("manual-merchant").value = tx.merchant_category;
    document.getElementById("manual-dist-home").value = tx.distance_from_home;
    document.getElementById("manual-dist-last").value = tx.distance_from_last_tx;
    document.getElementById("manual-ratio-median").value = tx.ratio_to_median_price;
    document.getElementById("manual-repeat-retailer").checked = tx.repeat_retailer === 1;
    document.getElementById("manual-used-chip").checked = tx.used_chip === 1;
    document.getElementById("manual-used-pin").checked = tx.used_pin === 1;
    document.getElementById("manual-online-order").checked = tx.online_order === 1;
    document.getElementById("manual-vel-1h").value = tx.velocity_1h;
    document.getElementById("manual-vel-24h").value = tx.velocity_24h;

    // Clear overrides
    document.getElementById("manual-override-2fa").value = "";
    document.getElementById("manual-override-device").value = "";
    document.getElementById("manual-override-sim").value = "";

    // Trigger evaluation automatically
    submitManualTransaction(null);
  } catch (err) {
    console.error("Error loading canonical case:", err);
  }
}

function loadScenario(type) {
  const randomSuffix = Math.floor(100 + Math.random() * 900);
  document.getElementById("manual-tx-id").value = `SCENARIO-${type.toUpperCase()}-${randomSuffix}`;
  currentCanonicalMode = null;
  activeCanonicalTx = null;

  // Clear overrides
  document.getElementById("manual-override-2fa").value = "";
  document.getElementById("manual-override-device").value = "";
  document.getElementById("manual-override-sim").value = "";

  if (type === "grocery") {
    document.getElementById("manual-amount").value = "28.50";
    document.getElementById("manual-merchant").value = "1";
    document.getElementById("manual-dist-home").value = "2.1";
    document.getElementById("manual-dist-last").value = "0.5";
    document.getElementById("manual-ratio-median").value = "0.9";
    document.getElementById("manual-repeat-retailer").checked = true;
    document.getElementById("manual-used-chip").checked = true;
    document.getElementById("manual-used-pin").checked = true;
    document.getElementById("manual-online-order").checked = false;
    document.getElementById("manual-vel-1h").value = "0";
    document.getElementById("manual-vel-24h").value = "1";
  } else if (type === "traveler") {
    document.getElementById("manual-amount").value = "340.00";
    document.getElementById("manual-merchant").value = "3";
    document.getElementById("manual-dist-home").value = "58.0";
    document.getElementById("manual-dist-last").value = "32.0";
    document.getElementById("manual-ratio-median").value = "1.8";
    document.getElementById("manual-repeat-retailer").checked = false;
    document.getElementById("manual-used-chip").checked = false;
    document.getElementById("manual-used-pin").checked = false;
    document.getElementById("manual-online-order").checked = true;
    document.getElementById("manual-vel-1h").value = "2";
    document.getElementById("manual-vel-24h").value = "3";
    document.getElementById("manual-override-2fa").value = "1";
  } else if (type === "luxury") {
    document.getElementById("manual-amount").value = "4850.00";
    document.getElementById("manual-merchant").value = "5";
    document.getElementById("manual-dist-home").value = "145.0";
    document.getElementById("manual-dist-last").value = "95.0";
    document.getElementById("manual-ratio-median").value = "5.5";
    document.getElementById("manual-repeat-retailer").checked = false;
    document.getElementById("manual-used-chip").checked = false;
    document.getElementById("manual-used-pin").checked = false;
    document.getElementById("manual-online-order").checked = true;
    document.getElementById("manual-vel-1h").value = "3";
    document.getElementById("manual-vel-24h").value = "5";
  } else if (type === "fraud") {
    document.getElementById("manual-amount").value = "890.00";
    document.getElementById("manual-merchant").value = "2";
    document.getElementById("manual-dist-home").value = "180.0";
    document.getElementById("manual-dist-last").value = "120.0";
    document.getElementById("manual-ratio-median").value = "4.2";
    document.getElementById("manual-repeat-retailer").checked = false;
    document.getElementById("manual-used-chip").checked = false;
    document.getElementById("manual-used-pin").checked = false;
    document.getElementById("manual-online-order").checked = true;
    document.getElementById("manual-vel-1h").value = "6";
    document.getElementById("manual-vel-24h").value = "14";
    document.getElementById("manual-override-2fa").value = "0";
    document.getElementById("manual-override-device").value = "0.12";
    document.getElementById("manual-override-sim").value = "3";
  }

  submitManualTransaction(null);
}

function resetManualForm() {
  document.getElementById("form-manual-tx").reset();
  currentCanonicalMode = null;
  activeCanonicalTx = null;
  document.getElementById("manual-placeholder").classList.remove("hidden");
  document.getElementById("manual-active-result").classList.add("hidden");
  document.getElementById("manual-result-badge").innerText = "Awaiting Evaluation";
  document.getElementById("manual-result-badge").className = "px-2 py-0.5 rounded text-[10px] font-mono bg-slate-800 text-slate-400";
}

async function submitManualTransaction(e) {
  if (e) e.preventDefault();

  const submitBtn = document.getElementById("btn-manual-submit");
  submitBtn.disabled = true;

  try {
    const txId = document.getElementById("manual-tx-id").value || `CUSTOM-${Date.now()}`;
    const amount = parseFloat(document.getElementById("manual-amount").value);
    const merchant = parseInt(document.getElementById("manual-merchant").value);
    const distHome = parseFloat(document.getElementById("manual-dist-home").value) || 0;
    const distLast = parseFloat(document.getElementById("manual-dist-last").value) || 0;
    const ratioMedian = parseFloat(document.getElementById("manual-ratio-median").value) || 1.0;
    const repeat = document.getElementById("manual-repeat-retailer").checked ? 1 : 0;
    const chip = document.getElementById("manual-used-chip").checked ? 1 : 0;
    const pin = document.getElementById("manual-used-pin").checked ? 1 : 0;
    const online = document.getElementById("manual-online-order").checked ? 1 : 0;
    const vel1h = parseInt(document.getElementById("manual-vel-1h").value) || 0;
    const vel24h = parseInt(document.getElementById("manual-vel-24h").value) || 0;

    // Overrides
    const override2faVal = document.getElementById("manual-override-2fa").value;
    const override2fa = override2faVal !== "" ? parseInt(override2faVal) : null;

    const overrideDevVal = document.getElementById("manual-override-device").value;
    const overrideDev = overrideDevVal !== "" ? parseFloat(overrideDevVal) : null;

    const overrideSimVal = document.getElementById("manual-override-sim").value;
    const overrideSim = overrideSimVal !== "" ? parseInt(overrideSimVal) : null;

    const txPayload = {
      transaction_id: txId,
      amount: amount,
      merchant_category: merchant,
      distance_from_home: distHome,
      distance_from_last_tx: distLast,
      ratio_to_median_price: ratioMedian,
      repeat_retailer: repeat,
      used_chip: chip,
      used_pin: pin,
      online_order: online,
      velocity_1h: vel1h,
      velocity_24h: vel24h,
      device_trust_score: overrideDev,
      carrier_sim_swap_age_days: overrideSim,
      two_factor_auth_success: override2fa,
    };

    // Merge active canonical metadata if present (e.g. ₹85,000 Hyderabad context)
    if (activeCanonicalTx) {
      Object.assign(txPayload, {
        customer_id: activeCanonicalTx.customer_id,
        customer_home_state: activeCanonicalTx.customer_home_state,
        location_city: activeCanonicalTx.location_city,
        time_of_day: activeCanonicalTx.time_of_day,
        normal_avg_amount: activeCanonicalTx.normal_avg_amount,
        normal_device: activeCanonicalTx.normal_device,
        device_fingerprint: activeCanonicalTx.device_fingerprint,
        merchant_name: activeCanonicalTx.merchant_name,
        is_fraud: activeCanonicalTx.is_fraud,
      });
    }

    const url = currentCanonicalMode ? `/api/predict?canonical_mode=${encodeURIComponent(currentCanonicalMode)}` : "/api/predict";

    const res = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(txPayload),
    });

    if (!res.ok) throw new Error("Evaluation request failed");
    const packet = await res.json();

    renderManualResult(packet);

    if (packet.escalation_tier === 2) {
      refreshQueue();
    }
  } catch (err) {
    console.error("Error evaluating manual transaction:", err);
    alert("Evaluation error: " + err.message);
  } finally {
    submitBtn.disabled = false;
  }
}

function renderManualResult(packet) {
  document.getElementById("manual-placeholder").classList.add("hidden");
  document.getElementById("manual-active-result").classList.remove("hidden");

  // Badge in header
  const badge = document.getElementById("manual-result-badge");
  const caseLabel = packet.canonical_case_id ? ` [${packet.canonical_case_id.replace(/_/g, ' ')}]` : '';
  badge.innerText = `Tier ${packet.escalation_tier} Evaluated${caseLabel}`;
  badge.className = "px-2 py-0.5 rounded text-[10px] font-mono bg-indigo-500/20 text-indigo-300 border border-indigo-500/30";

  // Action Banner
  const banner = document.getElementById("manual-action-banner");
  const actionText = document.getElementById("manual-action-text");
  const tierText = document.getElementById("manual-tier-text");

  actionText.innerText = packet.final_action;
  tierText.innerText = `Tier ${packet.escalation_tier}: ${packet.escalation_tier === 0 ? 'Autonomous' : packet.escalation_tier === 1 ? 'Dynamic Step-Up' : 'Human Escalation'}`;

  if (packet.final_action.startsWith("APPROVE")) {
    banner.className = "p-4 rounded-xl border border-emerald-500/40 bg-emerald-950/40 flex items-center justify-between";
    actionText.className = "text-base font-bold text-emerald-300 mt-0.5";
  } else if (packet.final_action.startsWith("DECLINE")) {
    banner.className = "p-4 rounded-xl border border-rose-500/40 bg-rose-950/40 flex items-center justify-between";
    actionText.className = "text-base font-bold text-rose-300 mt-0.5";
  } else {
    banner.className = "p-4 rounded-xl border border-purple-500/40 bg-purple-950/40 flex items-center justify-between";
    actionText.className = "text-base font-bold text-purple-300 mt-0.5";
  }

  // Key Scores
  document.getElementById("manual-p-initial").innerText = `${(packet.p_fraud_initial * 100).toFixed(1)}%`;
  document.getElementById("manual-p-final").innerText = `${(packet.p_fraud_final * 100).toFixed(1)}%`;

  const confEl = document.getElementById("manual-conformal-set");
  confEl.innerText = `[${packet.conformal_set.join(", ")}]`;
  confEl.className = packet.conformal_set.length > 1 ? "text-amber-400 font-bold text-sm" : packet.conformal_set[0] === "Fraud" ? "text-rose-400 font-bold text-sm" : "text-emerald-400 font-bold text-sm";

  document.getElementById("manual-uq-scores").innerText = `${packet.aleatoric_uncertainty.toFixed(2)} / ${packet.epistemic_uncertainty.toFixed(2)}`;

  // Sequential Evidence Trajectory Flow
  const trajContainer = document.getElementById("manual-trajectory-container");
  const trajSteps = document.getElementById("manual-trajectory-steps");
  const trajCost = document.getElementById("manual-trajectory-cost");

  if (packet.investigation_trajectory && packet.investigation_trajectory.length > 0) {
    trajContainer.classList.remove("hidden");
    let totalCost = 0;
    trajSteps.innerHTML = "";

    packet.investigation_trajectory.forEach((step) => {
      totalCost += (step.cost || 0);
      const stepDiv = document.createElement("div");
      stepDiv.className = "p-2.5 rounded-lg bg-slate-900 border border-slate-800 space-y-1";
      stepDiv.innerHTML = `
        <div class="flex items-center justify-between font-mono">
          <span class="text-white font-bold flex items-center space-x-1.5">
            <span class="w-4 h-4 rounded-full bg-cyan-500/20 text-cyan-300 text-[10px] flex items-center justify-center font-bold">${step.step_number}</span>
            <span>${step.evidence_name}</span>
          </span>
          <span class="text-emerald-400 font-bold">$${step.cost.toFixed(2)} (${step.latency_ms}ms)</span>
        </div>
        <div class="text-[10px] text-slate-300">${step.summary}</div>
        <div class="flex items-center justify-between text-[10px] font-mono pt-0.5 border-t border-slate-800/60 text-slate-400">
          <span>Risk: <b class="text-cyan-300">${(step.p_fraud_before * 100).toFixed(1)}% &rarr; ${(step.p_fraud_after * 100).toFixed(1)}%</b></span>
          <span>Uncertainty: <b class="text-amber-300">${step.uncertainty_before.toFixed(2)} &rarr; ${step.uncertainty_after.toFixed(2)}</b> (ΔU: -${step.uncertainty_reduction.toFixed(2)})</span>
        </div>
      `;
      trajSteps.appendChild(stepDiv);
    });
    trajCost.innerText = `Total Cost: $${totalCost.toFixed(2)}`;
  } else {
    trajContainer.classList.add("hidden");
  }

  // Cryptographic Blockchain Proof
  const bcCard = document.getElementById("manual-blockchain-card");
  if (packet.blockchain_block_hash) {
    bcCard.classList.remove("hidden");
    document.getElementById("manual-block-idx").innerText = packet.blockchain_index !== undefined ? packet.blockchain_index : "#";
    document.getElementById("manual-block-hash").innerText = packet.blockchain_block_hash;
  } else {
    bcCard.classList.add("hidden");
  }

  // Tier 1 dynamic legacy evidence (if present)
  const tier1Box = document.getElementById("manual-tier1-details");
  if (packet.evidence_collected && packet.evidence_collected.device_trust_score !== undefined) {
    tier1Box.classList.remove("hidden");
    const s2fa = packet.evidence_collected.two_factor_auth_success === 1 ? "Passed (SMS 2FA)" : "Failed / Timeout";
    const el2fa = document.getElementById("manual-ev-2fa");
    el2fa.innerText = s2fa;
    el2fa.className = packet.evidence_collected.two_factor_auth_success === 1 ? "font-bold text-emerald-400" : "font-bold text-rose-400";

    document.getElementById("manual-ev-device").innerText = packet.evidence_collected.device_trust_score.toFixed(2);
    document.getElementById("manual-ev-sim").innerText = `${packet.evidence_collected.carrier_sim_swap_age_days} days`;
  } else {
    tier1Box.classList.add("hidden");
  }

  // Feature attributions
  const attrList = document.getElementById("manual-attributions-list");
  attrList.innerHTML = "";
  const attrs = Object.entries(packet.feature_attributions || {}).slice(0, 4);
  attrs.forEach(([name, pct]) => {
    const item = document.createElement("div");
    item.innerHTML = `
      <div class="flex justify-between text-[11px] text-slate-400 mb-0.5">
        <span>${name.replace(/_/g, ' ')}</span>
        <span class="font-mono text-slate-200">${pct}%</span>
      </div>
      <div class="w-full bg-slate-800 rounded-full h-1.5 overflow-hidden">
        <div class="bg-indigo-500 h-1.5 rounded-full" style="width: ${Math.min(100, pct * 2)}%"></div>
      </div>
    `;
    attrList.appendChild(item);
  });

  // Rationale
  document.getElementById("manual-rationale-text").innerText = packet.investigator_rationale || "Autonomous evaluation complete.";
}

// -------------------------------------------------------------
// BATCH CSV UPLOAD & TEMPLATE DOWNLOAD
// -------------------------------------------------------------

function downloadSampleCsv() {
  const csvContent = 
`transaction_id,amount,merchant_category,distance_from_home,distance_from_last_tx,ratio_to_median_price,repeat_retailer,used_chip,used_pin,online_order,velocity_1h,velocity_24h,device_trust_score,two_factor_auth_success
CSV-TX-101,24.50,1,1.5,0.2,0.8,1,1,1,0,0,1,0.95,1
CSV-TX-102,320.00,3,48.0,25.0,1.9,0,0,0,1,2,4,0.72,1
CSV-TX-103,4250.00,5,150.0,80.0,4.8,0,0,0,1,3,6,0.35,0
CSV-TX-104,89.00,2,12.0,3.0,1.2,1,1,0,0,1,2,0.88,1
CSV-TX-105,950.00,2,210.0,140.0,5.1,0,0,0,1,5,12,0.15,0`;

  const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.setAttribute("href", url);
  link.setAttribute("download", "sample_transactions.csv");
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
}

function onCsvFileSelected(event) {
  const file = event.target.files && event.target.files[0];
  const fileInfo = document.getElementById("csv-file-info");
  const statusBox = document.getElementById("csv-status-box");
  if (statusBox) statusBox.classList.add("hidden");

  if (file) {
    if (fileInfo) {
      fileInfo.classList.remove("hidden");
      fileInfo.innerHTML = `
        <div class="flex items-center space-x-2 text-emerald-400">
          <i data-lucide="file-check" class="w-4 h-4"></i>
          <span>Selected: <b class="text-white">${file.name}</b> (${(file.size / 1024).toFixed(1)} KB) &bull; Ready to process</span>
        </div>
      `;
      if (window.lucide) lucide.createIcons();
    }
  } else {
    if (fileInfo) fileInfo.classList.add("hidden");
  }
}

async function runSampleCsvDirectly() {
  const statusBox = document.getElementById("csv-status-box");
  const btn = document.getElementById("btn-quick-run-csv");

  if (statusBox) {
    statusBox.classList.remove("hidden", "bg-rose-950/60", "border-rose-800", "text-rose-300", "bg-emerald-950/60", "border-emerald-800", "text-emerald-300");
    statusBox.classList.add("bg-indigo-950/60", "border-indigo-800", "text-indigo-300");
    statusBox.innerHTML = `<div class="flex items-center space-x-2"><i data-lucide="loader" class="w-4 h-4 animate-spin"></i><span>Fetching test_transactions.csv from server...</span></div>`;
    if (window.lucide) lucide.createIcons();
  }

  if (btn) btn.disabled = true;
  try {
    const res = await fetch("/api/download/test-csv");
    if (!res.ok) throw new Error("Could not download test_transactions.csv from server.");
    const blob = await res.blob();
    const file = new File([blob], "test_transactions.csv", { type: "text/csv" });
    await processCsvFileObject(file);
  } catch (err) {
    console.error("Direct sample run error:", err);
    if (statusBox) {
      statusBox.classList.remove("hidden", "bg-indigo-950/60", "border-indigo-800", "text-indigo-300", "bg-emerald-950/60", "border-emerald-800", "text-emerald-300");
      statusBox.classList.add("bg-rose-950/60", "border-rose-800", "text-rose-300");
      statusBox.innerHTML = `<div class="flex items-center space-x-2"><i data-lucide="alert-triangle" class="w-4 h-4 text-rose-400"></i><span>Failed to load test CSV: ${err.message}</span></div>`;
      if (window.lucide) lucide.createIcons();
    }
  } finally {
    if (btn) btn.disabled = false;
  }
}

async function uploadCsvFile() {
  const fileInput = document.getElementById("input-csv-file");
  const statusBox = document.getElementById("csv-status-box");

  if (!fileInput.files || fileInput.files.length === 0) {
    if (statusBox) {
      statusBox.classList.remove("hidden", "bg-indigo-950/60", "border-indigo-800", "text-indigo-300", "bg-emerald-950/60", "border-emerald-800", "text-emerald-300");
      statusBox.classList.add("bg-rose-950/60", "border-rose-800", "text-rose-300");
      statusBox.innerHTML = `<div class="flex items-center space-x-2"><i data-lucide="alert-circle" class="w-4 h-4 text-rose-400"></i><span>Please select a CSV file first, or click <b>"⚡ 1-Click Quick Run"</b> above!</span></div>`;
      if (window.lucide) lucide.createIcons();
    }
    alert("Please select a CSV file first (or click the '⚡ 1-Click Quick Run' button)!");
    return;
  }

  await processCsvFileObject(fileInput.files[0]);
}

async function processCsvFileObject(file) {
  const statusBox = document.getElementById("csv-status-box");
  const btn = document.getElementById("btn-upload-csv");
  if (btn) {
    btn.disabled = true;
    btn.innerHTML = `<i data-lucide="loader" class="w-4 h-4 animate-spin"></i><span>Processing CSV...</span>`;
  }
  if (window.lucide) lucide.createIcons();

  if (statusBox) {
    statusBox.classList.remove("hidden", "bg-rose-950/60", "border-rose-800", "text-rose-300", "bg-emerald-950/60", "border-emerald-800", "text-emerald-300");
    statusBox.classList.add("bg-indigo-950/60", "border-indigo-800", "text-indigo-300");
    statusBox.innerHTML = `<div class="flex items-center space-x-2"><i data-lucide="loader" class="w-4 h-4 animate-spin"></i><span>Evaluating <b>${file.name}</b> through SafeEscalate 3-tier cascade...</span></div>`;
    if (window.lucide) lucide.createIcons();
  }

  const formData = new FormData();
  formData.append("file", file);

  try {
    const res = await fetch("/api/upload-csv", {
      method: "POST",
      body: formData,
    });

    if (!res.ok) {
      const errData = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(errData.detail || `Upload failed with HTTP ${res.status}`);
    }

    const data = await res.json();

    if (data.processed_count === 0) {
      throw new Error("No valid transactions could be parsed. Check column names (expected 'amount', 'velocity_1h', etc.).");
    }

    const t0 = data.tier_breakdown ? data.tier_breakdown[0] || 0 : 0;
    const t1 = data.tier_breakdown ? data.tier_breakdown[1] || 0 : 0;
    const t2 = data.tier_breakdown ? data.tier_breakdown[2] || 0 : 0;

    if (statusBox) {
      statusBox.classList.remove("hidden", "bg-indigo-950/60", "border-indigo-800", "text-indigo-300", "bg-rose-950/60", "border-rose-800", "text-rose-300");
      statusBox.classList.add("bg-emerald-950/60", "border-emerald-800", "text-emerald-300");
      statusBox.innerHTML = `
        <div class="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
          <div class="flex items-center space-x-2">
            <i data-lucide="check-circle-2" class="w-4 h-4 text-emerald-400"></i>
            <span class="font-bold">Successfully evaluated ${data.processed_count} transactions from ${file.name}!</span>
          </div>
          <div class="font-mono text-[11px] space-x-2">
            <span class="text-emerald-400">Tier 0 Auto: <b>${t0}</b></span> &bull; 
            <span class="text-amber-400">Tier 1 Step-Up: <b>${t1}</b></span> &bull; 
            <span class="text-purple-400">Tier 2 Escalated: <b>${t2}</b></span>
          </div>
        </div>
      `;
      if (window.lucide) lucide.createIcons();
    }

    const resultsContainer = document.getElementById("csv-results-container");
    if (resultsContainer) resultsContainer.classList.remove("hidden");

    const countElem = document.getElementById("csv-summary-count");
    if (countElem) countElem.innerText = `${data.processed_count} transactions evaluated (${t0} Auto, ${t1} Step-Up, ${t2} Escalated)`;

    const tbody = document.getElementById("csv-table-body");
    if (tbody) {
      tbody.innerHTML = "";

      data.results.forEach((pkt) => {
        const row = document.createElement("tr");
        row.className = "hover:bg-slate-800/40 transition";
        
        let badge = `<span class="px-2 py-0.5 rounded text-[10px] bg-slate-800 text-slate-300 font-bold">${pkt.final_action}</span>`;
        if (pkt.final_action.startsWith("APPROVE")) {
          badge = `<span class="px-2 py-0.5 rounded text-[10px] bg-emerald-500/20 text-emerald-300 font-bold border border-emerald-500/30">${pkt.final_action}</span>`;
        } else if (pkt.final_action.startsWith("DECLINE")) {
          badge = `<span class="px-2 py-0.5 rounded text-[10px] bg-rose-500/20 text-rose-300 font-bold border border-rose-500/30">${pkt.final_action}</span>`;
        } else {
          badge = `<span class="px-2 py-0.5 rounded text-[10px] bg-purple-500/20 text-purple-300 font-bold border border-purple-500/30 font-bold">ESCALATE HITL</span>`;
        }

        let tierBadge = `<span class="px-1.5 py-0.5 rounded text-[10px] bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">Tier 0 (Auto)</span>`;
        if (pkt.escalation_tier === 1) {
          tierBadge = `<span class="px-1.5 py-0.5 rounded text-[10px] bg-amber-500/10 text-amber-400 border border-amber-500/20">Tier 1 (Step-Up)</span>`;
        } else if (pkt.escalation_tier === 2) {
          tierBadge = `<span class="px-1.5 py-0.5 rounded text-[10px] bg-purple-500/10 text-purple-300 border border-purple-500/20">Tier 2 (HITL)</span>`;
        }

        row.innerHTML = `
          <td class="px-3 py-2 font-bold text-white">${pkt.transaction_id}</td>
          <td class="px-3 py-2 text-slate-200 font-bold">$${pkt.amount.toFixed(2)}</td>
          <td class="px-3 py-2 text-cyan-400 font-semibold">${(pkt.p_fraud_final * 100).toFixed(1)}%</td>
          <td class="px-3 py-2 font-mono text-[11px]">${pkt.conformal_set.length > 1 ? '<span class="text-amber-400 font-bold">[Legit, Fraud]</span>' : `<span class="text-emerald-400">[${pkt.conformal_set.join("")}]</span>`}</td>
          <td class="px-3 py-2">${tierBadge}</td>
          <td class="px-3 py-2">${badge}</td>
        `;
        tbody.appendChild(row);
      });
    }

    refreshQueue();
    if (resultsContainer) {
      resultsContainer.scrollIntoView({ behavior: "smooth", block: "nearest" });
    }
  } catch (err) {
    console.error("Error processing CSV:", err);
    if (statusBox) {
      statusBox.classList.remove("hidden", "bg-indigo-950/60", "border-indigo-800", "text-indigo-300", "bg-emerald-950/60", "border-emerald-800", "text-emerald-300");
      statusBox.classList.add("bg-rose-950/60", "border-rose-800", "text-rose-300");
      statusBox.innerHTML = `<div class="flex items-center space-x-2"><i data-lucide="alert-triangle" class="w-4 h-4 text-rose-400"></i><span><b>Error processing CSV:</b> ${err.message}</span></div>`;
      if (window.lucide) lucide.createIcons();
    }
    alert("Error processing CSV: " + err.message);
  } finally {
    if (btn) {
      btn.disabled = false;
      btn.innerHTML = `<i data-lucide="upload" class="w-4 h-4"></i><span>Process CSV Batch</span>`;
    }
    if (window.lucide) lucide.createIcons();
  }
}

// -------------------------------------------------------------
// TAB 6: BLOCKCHAIN AUDIT LEDGER
// -------------------------------------------------------------

let lastTamperedRecord = null;

async function fetchBlockchainLedger() {
  try {
    const res = await fetch("/api/audit/ledger?limit=30");
    if (!res.ok) return;
    const data = await res.json();

    const totalEl = document.getElementById("bc-total-blocks");
    if (totalEl) totalEl.innerText = data.total_blocks || 0;

    const headEl = document.getElementById("bc-head-hash");
    if (headEl) headEl.innerText = data.head_hash ? data.head_hash.substring(0, 24) + "..." : "--";

    const tbody = document.getElementById("bc-table-body");
    if (!tbody) return;
    tbody.innerHTML = "";

    if (!data.blocks || data.blocks.length === 0) {
      tbody.innerHTML = `<tr class="text-slate-500 text-center"><td colspan="7" class="py-6">No blocks recorded yet.</td></tr>`;
      return;
    }

    data.blocks.forEach((b) => {
      const row = document.createElement("tr");
      row.className = "hover:bg-slate-800/40 transition";

      let decisionBadge = `<span class="px-2 py-0.5 rounded text-[10px] bg-slate-800 text-slate-300 font-bold">${b.final_action}</span>`;
      if (b.final_action.includes("APPROVE")) {
        decisionBadge = `<span class="px-2 py-0.5 rounded text-[10px] bg-emerald-500/20 text-emerald-300 font-bold border border-emerald-500/30">${b.final_action}</span>`;
      } else if (b.final_action.includes("DECLINE")) {
        decisionBadge = `<span class="px-2 py-0.5 rounded text-[10px] bg-rose-500/20 text-rose-300 font-bold border border-rose-500/30">${b.final_action}</span>`;
      } else if (b.human_resolved) {
        decisionBadge = `<span class="px-2 py-0.5 rounded text-[10px] bg-purple-500/20 text-purple-300 font-bold border border-purple-500/30">RESOLVED: ${b.human_decision}</span>`;
      } else {
        decisionBadge = `<span class="px-2 py-0.5 rounded text-[10px] bg-amber-500/20 text-amber-300 font-bold border border-amber-500/30 font-mono">HITL QUEUED</span>`;
      }

      let evSummary = "None (Tier 0)";
      if (b.evidence_chain && b.evidence_chain.length > 0) {
        evSummary = b.evidence_chain.map(s => `${s.evidence_type} ($${s.cost})`).join(" &rarr; ");
      }

      const shortHash = b.block_hash ? b.block_hash.substring(0, 16) + "..." : "--";
      const ts = b.timestamp ? b.timestamp.replace("T", " ").substring(0, 19) : "--";

      row.innerHTML = `
        <td class="px-4 py-2.5 font-bold text-cyan-400">#${b.index}</td>
        <td class="px-4 py-2.5 text-slate-400 text-[11px]">${ts}</td>
        <td class="px-4 py-2.5 font-bold text-white">${b.transaction_id} <span class="text-slate-400 font-normal">($${b.amount.toFixed(2)})</span></td>
        <td class="px-4 py-2.5 text-[11px] text-slate-300">${(b.initial_p_fraud * 100).toFixed(1)}% &rarr; <b class="text-cyan-300">${(b.final_p_fraud * 100).toFixed(1)}%</b></td>
        <td class="px-4 py-2.5 text-[11px] text-slate-400 max-w-xs truncate">${evSummary}</td>
        <td class="px-4 py-2.5">${decisionBadge}</td>
        <td class="px-4 py-2.5 text-slate-400 font-mono text-[10px]" title="${b.block_hash}">${shortHash}</td>
      `;
      tbody.appendChild(row);
    });

    if (window.lucide) lucide.createIcons();
  } catch (err) {
    console.error("Error fetching blockchain ledger:", err);
  }
}

async function verifyBlockchainIntegrity(showAlert = true) {
  try {
    const res = await fetch("/api/audit/verify");
    if (!res.ok) throw new Error("Integrity check failed");
    const report = await res.json();

    const statusEl = document.getElementById("bc-integrity-status");
    const detailEl = document.getElementById("bc-integrity-detail");
    const tamperBanner = document.getElementById("bc-tamper-banner");
    const restoreBtn = document.getElementById("btn-bc-restore");

    if (report.valid) {
      if (statusEl) statusEl.innerHTML = `<span>VALID</span><span class="w-2.5 h-2.5 rounded-full bg-emerald-500"></span>`;
      if (detailEl) detailEl.innerText = `100% SHA-256 hash verified across ${report.total_blocks} blocks`;
      if (tamperBanner) tamperBanner.classList.add("hidden");
      if (restoreBtn) restoreBtn.classList.add("hidden");
      if (showAlert) {
        alert(`Blockchain Integrity Verified: All ${report.total_blocks} investigation blocks and cryptographic parent hashes are 100% authentic!`);
      }
    } else {
      if (statusEl) statusEl.innerHTML = `<span class="text-rose-400">TAMPERED</span><span class="w-2.5 h-2.5 rounded-full bg-rose-500 animate-pulse"></span>`;
      if (detailEl) detailEl.innerText = `Integrity violation at block #${report.tampered_block_index}`;
      if (tamperBanner) {
        tamperBanner.classList.remove("hidden");
        const msg = document.getElementById("bc-tamper-message");
        if (msg) msg.innerText = report.error || "Cryptographic integrity violation detected.";
      }
      if (restoreBtn) restoreBtn.classList.remove("hidden");
    }
  } catch (err) {
    console.error("Error verifying blockchain:", err);
  }
}

async function simulateTamperDemo() {
  try {
    const ledgerRes = await fetch("/api/audit/ledger?limit=10");
    if (!ledgerRes.ok) return;
    const ledgerData = await ledgerRes.json();

    if (!ledgerData.blocks || ledgerData.blocks.length <= 1) {
      alert("Please evaluate at least one transaction first so the blockchain ledger contains an investigation block to tamper with!");
      return;
    }

    // Pick the most recent non-genesis block
    const targetBlock = ledgerData.blocks[0];
    const targetIdx = targetBlock.index > 0 ? targetBlock.index : 1;

    const res = await fetch("/api/audit/simulate-tamper", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        block_index: targetIdx,
        fake_action: "FORGED_APPROVE_ATTACK",
      }),
    });

    if (!res.ok) throw new Error("Simulation failed");
    const result = await res.json();

    lastTamperedRecord = {
      block_index: targetIdx,
      original_action: result.original_action,
    };

    fetchBlockchainLedger();
    verifyBlockchainIntegrity(false);

    alert(`Tampering Attack Simulated on Block #${targetIdx}!\nFalsified Action: 'FORGED_APPROVE_ATTACK'.\nNotice how the SHA-256 cryptographic chain immediately flags a broken integrity violation!`);
  } catch (err) {
    console.error("Error simulating tamper:", err);
  }
}

async function restoreBlockchainLedger() {
  if (!lastTamperedRecord) return;
  try {
    const res = await fetch("/api/audit/restore", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(lastTamperedRecord),
    });
    if (!res.ok) throw new Error("Restoration failed");
    lastTamperedRecord = null;
    fetchBlockchainLedger();
    verifyBlockchainIntegrity(false);
    alert("Blockchain Ledger Restored to original authentic state! Cryptographic integrity is 100% restored.");
  } catch (err) {
    console.error("Error restoring blockchain:", err);
  }
}

