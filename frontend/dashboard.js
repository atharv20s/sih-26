// PRAHARI dashboard — pulls mission history from Postgres and live model
// benchmark numbers, renders summary cards + a recent-missions table.

(async function () {
  const userEmailEl = document.getElementById('user-email');
  const logoutBtn = document.getElementById('logout-btn');

  logoutBtn.addEventListener('click', async () => {
    await fetch('/api/auth/logout', { method: 'POST', credentials: 'include' });
    window.location.href = '/login';
  });

  async function loadMe() {
    const res = await fetch('/api/auth/me', { credentials: 'include' });
    const data = await res.json();
    if (!data.authenticated) {
      window.location.href = '/login';
      return null;
    }
    userEmailEl.textContent = data.full_name ? `${data.full_name} · ${data.email}` : data.email;
    return data;
  }

  function fmtHours(totalSeconds) {
    const h = totalSeconds / 3600;
    return h < 0.01 ? '0.00' : h.toFixed(2);
  }

  function fmtDuration(seconds) {
    if (seconds === null || seconds === undefined) return 'LIVE';
    const m = Math.floor(seconds / 60);
    const s = Math.round(seconds % 60);
    return `${m}m ${s.toString().padStart(2, '0')}s`;
  }

  function fmtDate(iso) {
    if (!iso) return '—';
    const d = new Date(iso);
    return d.toLocaleString(undefined, { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });
  }

  async function loadSummary() {
    try {
      const res = await fetch('/api/dashboard/summary', { credentials: 'include' });
      const s = await res.json();

      document.getElementById('stat-missions').textContent = s.total_missions;
      document.getElementById('stat-hours').textContent = fmtHours(s.total_telemetry_seconds);
      document.getElementById('stat-critical').textContent = s.critical_events;

      const bm = s.benchmarks || {};
      if (bm.clf_ensemble_accuracy !== undefined && bm.clf_ensemble_accuracy !== null) {
        document.getElementById('stat-clf-acc').textContent = `${bm.clf_ensemble_accuracy.toFixed(1)}%`;
      } else if (bm.clf_test_acc !== undefined) {
        document.getElementById('stat-clf-acc').textContent = `${bm.clf_test_acc.toFixed(1)}%`;
        document.getElementById('stat-clf-sub').textContent = 'Self-reported (see MODEL_CARD.md)';
      }
      if (bm.pinn_mae !== undefined) {
        document.getElementById('stat-pinn-mae').textContent = bm.pinn_mae.toFixed(1);
      }
      if (bm.drl_avg_endurance !== undefined) {
        document.getElementById('stat-drl').textContent = `+${bm.drl_avg_endurance.toFixed(1)}`;
      }
    } catch (err) {
      console.error('[Dashboard] summary load failed', err);
    }
  }

  async function loadMissions() {
    const wrap = document.getElementById('missions-table-wrap');
    const badge = document.getElementById('mission-count-badge');
    try {
      const res = await fetch('/api/dashboard/missions?limit=25', { credentials: 'include' });
      const missions = await res.json();
      badge.textContent = `${missions.length} SHOWN`;

      if (missions.length === 0) {
        wrap.innerHTML = `<div class="empty-state">
          No missions logged yet. Launch the Live Ops Console and run a session —
          it will appear here automatically once you connect.
        </div>`;
        return;
      }

      const rows = missions.map((m) => {
        const isLive = !m.ended_at;
        const faultChip = m.final_fault_archetype === 'nominal'
          ? `<span class="chip chip-nominal">NOMINAL</span>`
          : `<span class="chip chip-fault">${m.final_fault_archetype.toUpperCase()}</span>`;
        const statusChip = isLive ? `<span class="chip chip-live">LIVE</span>` : '';
        return `<tr>
          <td>${fmtDate(m.started_at)}</td>
          <td>${fmtDuration(m.duration_seconds)}</td>
          <td>${m.frame_count.toLocaleString()}</td>
          <td>${faultChip} ${statusChip}</td>
          <td>${m.final_rul_cycles !== null ? m.final_rul_cycles.toFixed(0) : '—'}</td>
          <td>${m.event_count}</td>
          <td>${m.had_critical_event ? '<span class="chip chip-fault">YES</span>' : '—'}</td>
        </tr>`;
      }).join('');

      wrap.innerHTML = `<table>
        <thead>
          <tr>
            <th>Started</th><th>Duration</th><th>Frames</th><th>Final State</th>
            <th>Final RUL</th><th>Events</th><th>Critical?</th>
          </tr>
        </thead>
        <tbody>${rows}</tbody>
      </table>`;
    } catch (err) {
      wrap.innerHTML = `<div class="empty-state">Could not load mission history.</div>`;
      console.error('[Dashboard] missions load failed', err);
    }
  }

  const me = await loadMe();
  if (me) {
    loadSummary();
    loadMissions();
  }
})();
