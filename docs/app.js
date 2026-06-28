/* ────────────────────────────────────────────────────────────────────────
 * Soybean Supply Chain Dashboard — interactive logic
 * Reads `DATA` (set by data.js), renders Leaflet map + 3 tab panels.
 * ──────────────────────────────────────────────────────────────────────── */

(function() {
    'use strict';
    if (typeof DATA === 'undefined') {
        document.body.innerHTML =
            '<div style="padding:40px;font-family:sans-serif;color:#a00">' +
            '<h2>⚠️ data.js belum di-generate</h2>' +
            '<p>Jalankan <code>python build_dashboard.py</code> terlebih dahulu.</p>' +
            '</div>';
        return;
    }

    // ── State ────────────────────────────────────────────────────────────
    let currentMonth    = 0;
    let currentScenario = 'optimized';
    let selectedProvIdx = DATA.decision?.default_province_idx ?? 0;
    let selectedPortIdx = DATA.decision?.default_port_idx ?? 0;
    let activeTab       = 'provinces';
    let transferSort    = {key: 'volume', dir: 'desc'};

    // ── DOM refs ─────────────────────────────────────────────────────────
    const $month     = document.getElementById('month-select');
    const $scenario  = document.getElementById('scenario-select');
    const $showFlows = document.getElementById('show-flows');
    const $showTrns  = document.getElementById('show-transfers');
    const $showPorts = document.getElementById('show-ports');
    const $showAdj   = document.getElementById('show-adjacency');
    const $status    = document.getElementById('status-bar');
    const $decisionTitle = document.getElementById('decision-title');
    const $decisionMode = document.getElementById('decision-mode');
    const $decisionNarrative = document.getElementById('decision-narrative');
    const $monthQuick = document.getElementById('month-quick');
    const $decisionKpis = document.getElementById('decision-kpis');
    const $provincePlanSelect = document.getElementById('province-plan-select');
    const $provincePlanTitle = document.getElementById('province-plan-title');
    const $provincePlanSummary = document.getElementById('province-plan-summary');
    const $provinceCurrentFlow = document.getElementById('province-current-flow');
    const $provinceMonthPlanBody = document.querySelector('#province-month-plan tbody');
    const $portPlanSelect = document.getElementById('port-plan-select');
    const $portPlanTitle = document.getElementById('port-plan-title');
    const $portPlanSummary = document.getElementById('port-plan-summary');
    const $portCurrentFlow = document.getElementById('port-current-flow');
    const $portMonthPlanBody = document.querySelector('#port-month-plan tbody');
    const $countrySummary = document.getElementById('country-summary');
    const $topPorts = document.getElementById('top-ports');
    const $topReceivers = document.getElementById('top-receivers');
    const $transferActionSummary = document.getElementById('transfer-action-summary');
    const $topTransferActions = document.getElementById('top-transfer-actions');
    const $riskActions = document.getElementById('risk-actions');
    const $transferScenario = document.getElementById('transfer-scenario');
    const $transferMonth = document.getElementById('transfer-month');
    const $transferSummary = document.getElementById('transfer-summary');

    // Populate month dropdown
    DATA.meta.months.forEach((m, i) => {
        const opt = document.createElement('option');
        opt.value = i; opt.textContent = m;
        $month.appendChild(opt);
    });
    if ($transferMonth) {
        DATA.meta.months.forEach((m, i) => {
            const opt = document.createElement('option');
            opt.value = i; opt.textContent = m;
            $transferMonth.appendChild(opt);
        });
    }

    // ── Helpers ──────────────────────────────────────────────────────────
    const fmt = (n, dp) => (n == null || isNaN(n)) ? '—' :
        n.toLocaleString('id-ID', {maximumFractionDigits: (dp ?? 0), minimumFractionDigits: 0});
    const fmtPct = (n) => (n == null) ? '—' : (n * 100).toFixed(1) + '%';
    const scenarioLabel = (scen) => scen === 'optimized' ? 'Rekomendasi Model' : 'Solusi Awal ALNS';
    const provName = (i) => DATA.provinces[i].name;
    const portName = (h) => DATA.ports[h].name;
    const impName  = (s) => DATA.meta.imp_names[s];

    function svcRate(prov, t, scen) {
        const m = DATA[scen].provinces[prov].monthly[t];
        const d = m.demand;
        return d > 0 ? 1 - m.shortage / d : 1.0;
    }
    function svcColor(rate) {
        if (rate >= 0.999) return '#2ca02c';
        if (rate >= 0.90)  return '#98df8a';
        if (rate >= 0.75)  return '#ffbb78';
        return '#d62728';
    }
    function svcTag(rate) {
        const cls = rate >= 0.999 ? 'tag-ok' :
                    rate >= 0.90  ? 'tag-ok' :
                    rate >= 0.75  ? 'tag-warn' : 'tag-bad';
        return `<span class="tag ${cls}">${fmtPct(rate)}</span>`;
    }
    function delta(a, b) {
        const d = b - a;
        if (Math.abs(d) < 0.5) return `<span class="stat-delta">±0</span>`;
        const cls = d >= 0 ? 'delta-pos' : 'delta-neg';
        const sign = d >= 0 ? '+' : '';
        return `<span class="stat-delta ${cls}">${sign}${fmt(d, 0)}</span>`;
    }

    // ── Decision cockpit helpers ─────────────────────────────────────────
    function renderMiniTable(rows, columns, emptyText) {
        if (!rows || rows.length === 0) {
            return `<div class="empty-state">${emptyText}</div>`;
        }
        return `<table class="action-table">
            <thead><tr>${columns.map(c => `<th>${c.label}</th>`).join('')}</tr></thead>
            <tbody>${rows.map(r => `
                <tr ${r.attrs || ''}>
                    ${columns.map(c => `<td>${c.value(r)}</td>`).join('')}
                </tr>
            `).join('')}</tbody>
        </table>`;
    }

    function bindDecisionClicks() {
        document.querySelectorAll('#decision-cockpit [data-prov-idx]').forEach(row => {
            if (row.dataset.boundDecision === '1') return;
            row.dataset.boundDecision = '1';
            row.addEventListener('click', () => {
                selectedProvIdx = parseInt(row.dataset.provIdx);
                if ($provincePlanSelect) $provincePlanSelect.value = selectedProvIdx;
                switchTab('provinces');
                renderProvinceTable();
                renderProvinceDetail();
                renderProvinceInspector();
                redrawMap();
            });
        });
        document.querySelectorAll('#decision-cockpit [data-port-idx]').forEach(row => {
            if (row.dataset.boundDecision === '1') return;
            row.dataset.boundDecision = '1';
            row.addEventListener('click', () => {
                selectedPortIdx = parseInt(row.dataset.portIdx);
                if ($portPlanSelect) $portPlanSelect.value = selectedPortIdx;
                switchTab('ports');
                renderPortTable();
                renderPortDetail();
                renderPortInspector();
                redrawMap();
            });
        });
    }
    if ($provincePlanSelect) {
        const order = DATA.decision?.province_order || DATA.provinces.map(p => p.idx);
        order.forEach(i => {
            const opt = document.createElement('option');
            opt.value = i;
            opt.textContent = DATA.provinces[i].name;
            $provincePlanSelect.appendChild(opt);
        });
        $provincePlanSelect.value = selectedProvIdx;
        $provincePlanSelect.addEventListener('change', () => {
            selectedProvIdx = parseInt($provincePlanSelect.value);
            renderDecisionCockpit();
            renderProvinceTable();
            renderProvinceDetail();
            redrawMap();
        });
    }
    if ($portPlanSelect) {
        const order = DATA.decision?.port_order || DATA.ports.map(p => p.idx);
        order.forEach(h => {
            const opt = document.createElement('option');
            opt.value = h;
            opt.textContent = DATA.ports[h].name;
            $portPlanSelect.appendChild(opt);
        });
        $portPlanSelect.value = selectedPortIdx;
        $portPlanSelect.addEventListener('change', () => {
            selectedPortIdx = parseInt($portPlanSelect.value);
            renderDecisionCockpit();
            renderPortTable();
            renderPortDetail();
            redrawMap();
        });
    }

    function refreshAllViews() {
        renderDecisionCockpit();
        redrawMap();
        renderProvinceTable();
        renderPortTable();
        renderTransferTable();
        renderClusterTable();
        renderProvinceDetail();
        renderPortDetail();
        if (activeTab === 'analytics') renderAnalytics();
    }

    function renderMonthQuick() {
        if (!$monthQuick) return;
        $monthQuick.innerHTML = DATA.meta.months.map((m, i) => `
            <button type="button" class="month-pill ${i === currentMonth ? 'active' : ''}" data-month="${i}">
                ${m.slice(0, 3)}
            </button>
        `).join('');
        $monthQuick.querySelectorAll('button').forEach(btn => {
            btn.addEventListener('click', () => {
                currentMonth = parseInt(btn.dataset.month);
                $month.value = currentMonth;
                refreshAllViews();
            });
        });
    }

    function provincePlanRows(i, scen) {
        const fromPayload = DATA.decision?.province_plan?.[scen]?.[String(i)];
        if (fromPayload) return fromPayload.monthly;
        return DATA.meta.months.map((month, t) => {
            const m = DATA[scen].provinces[i].monthly[t];
            const demand = m.demand;
            const shortage = m.shortage;
            return {
                month_idx: t,
                month,
                demand,
                local: m.local,
                import: m.import,
                transfer_in: m.transfer_in,
                transfer_out: m.transfer_out,
                inventory: m.inventory,
                shortage,
                service_rate: demand > 0 ? 1 - shortage / demand : 1.0,
            };
        });
    }

    function portPlanRows(h, scen) {
        const fromPayload = DATA.decision?.port_plan?.[scen]?.[String(h)];
        if (fromPayload) return fromPayload.monthly;
        return DATA.meta.months.map((month, t) => {
            const m = DATA[scen].ports[h].monthly[t];
            const imports = Object.entries(m.imports || {})
                .map(([s, volume]) => ({
                    idx: parseInt(s),
                    name: impName(parseInt(s)),
                    volume,
                }))
                .filter(r => r.volume > 0.1)
                .sort((a, b) => b.volume - a.volume);
            const destinations = Object.entries(m.distribution || {})
                .map(([i, volume]) => ({
                    idx: parseInt(i),
                    name: provName(parseInt(i)),
                    volume,
                }))
                .filter(r => r.volume > 0.1)
                .sort((a, b) => b.volume - a.volume);
            const capacity = m.thru_cap;
            return {
                month_idx: t,
                month,
                total_in: m.total_in,
                total_out: m.total_out,
                capacity,
                utilization: capacity > 0 ? m.total_in / capacity : 0,
                imports,
                destinations,
                top_country: imports[0]?.name || '',
                top_destination: destinations[0]?.name || '',
            };
        });
    }

    function renderProvinceInspector() {
        if (!$provinceMonthPlanBody || selectedProvIdx == null) return;
        const scen = currentScenario;
        const prov = DATA.provinces[selectedProvIdx];
        const rows = provincePlanRows(selectedProvIdx, scen);
        const totals = rows.reduce((acc, r) => {
            acc.demand += r.demand;
            acc.local += r.local;
            acc.import += r.import;
            acc.transfer_in += r.transfer_in;
            acc.transfer_out += r.transfer_out;
            acc.inventory += r.inventory;
            acc.shortage += r.shortage;
            return acc;
        }, {demand:0, local:0, import:0, transfer_in:0, transfer_out:0, inventory:0, shortage:0});
        const service = totals.demand > 0 ? 1 - totals.shortage / totals.demand : 1.0;

        if ($provincePlanSelect) $provincePlanSelect.value = selectedProvIdx;
        if ($provincePlanTitle) $provincePlanTitle.textContent = `${prov.name} - ${scenarioLabel(scen)}`;
        if ($provinceCurrentFlow) {
            const active = DATA[scen].provinces[selectedProvIdx].monthly[currentMonth];
            const localInRaw = active.local_from_province || {};
            const localInRows = Object.entries(localInRaw)
                .map(([src, volume]) => ({
                    idx: parseInt(src),
                    name: provName(parseInt(src)),
                    volume,
                    attrs: `data-prov-idx="${parseInt(src)}" class="action-row"`,
                }))
                .sort((a, b) => b.volume - a.volume);
            const localOutRows = Object.entries(active.local_out_to || {})
                .map(([dst, volume]) => ({
                    idx: parseInt(dst),
                    name: provName(parseInt(dst)),
                    volume,
                    attrs: `data-prov-idx="${parseInt(dst)}" class="action-row"`,
                }))
                .sort((a, b) => b.volume - a.volume);
            const importRows = Object.entries(active.import_by_port || {})
                .map(([h, volume]) => ({
                    idx: parseInt(h),
                    name: portName(parseInt(h)),
                    volume,
                    attrs: `data-port-idx="${parseInt(h)}" class="action-row"`,
                }))
                .sort((a, b) => b.volume - a.volume);
            const transferInRows = Object.entries(active.transfer_in_from || {})
                .map(([src, volume]) => ({
                    idx: parseInt(src),
                    name: provName(parseInt(src)),
                    volume,
                    attrs: `data-prov-idx="${parseInt(src)}" class="action-row"`,
                }))
                .sort((a, b) => b.volume - a.volume);
            const transferOutRows = Object.entries(active.transfer_out_to || {})
                .map(([dst, volume]) => ({
                    idx: parseInt(dst),
                    name: provName(parseInt(dst)),
                    volume,
                    attrs: `data-prov-idx="${parseInt(dst)}" class="action-row"`,
                }))
                .sort((a, b) => b.volume - a.volume);

            $provinceCurrentFlow.innerHTML = `
                <div class="flow-card primary">
                    <h4>${DATA.meta.months[currentMonth]}: Lokal diterima dari mana?</h4>
                    ${renderMiniTable(localInRows, [
                        {label: 'Asal produsen', value: r => r.name},
                        {label: 'Ton', value: r => fmt(r.volume)},
                    ], 'Tidak ada pasokan lokal masuk ke provinsi ini pada bulan aktif.')}
                </div>
                <div class="flow-card">
                    <h4>Impor masuk lewat pelabuhan mana?</h4>
                    ${renderMiniTable(importRows, [
                        {label: 'Pelabuhan', value: r => r.name},
                        {label: 'Ton', value: r => fmt(r.volume)},
                    ], 'Tidak ada impor masuk ke provinsi ini pada bulan aktif.')}
                </div>
                <div class="flow-card">
                    <h4>Transfer masuk dari mana?</h4>
                    ${renderMiniTable(transferInRows, [
                        {label: 'Asal provinsi', value: r => r.name},
                        {label: 'Ton', value: r => fmt(r.volume)},
                    ], 'Tidak ada transfer masuk pada bulan aktif.')}
                </div>
                <div class="flow-card">
                    <h4>Provinsi ini mengirim ke mana?</h4>
                    ${renderMiniTable([...localOutRows.map(r => ({...r, kind: 'Lokal'})), ...transferOutRows.map(r => ({...r, kind: 'Transfer'}))], [
                        {label: 'Jenis', value: r => r.kind},
                        {label: 'Tujuan', value: r => r.name},
                        {label: 'Ton', value: r => fmt(r.volume)},
                    ], 'Provinsi ini tidak mengirim lokal/transfer keluar pada bulan aktif.')}
                </div>
            `;
            bindDecisionClicks();
        }
        if ($provincePlanSummary) {
            $provincePlanSummary.innerHTML = [
                {label: 'Demand tahunan', value: `${fmt(totals.demand)} ton`},
                {label: 'Lokal', value: `${fmt(totals.local)} ton`},
                {label: 'Impor', value: `${fmt(totals.import)} ton`},
                {label: 'Transfer in/out', value: `${fmt(totals.transfer_in)} / ${fmt(totals.transfer_out)} ton`},
                {label: 'Shortage', value: `${fmt(totals.shortage)} ton`},
                {label: 'Service', value: fmtPct(service)},
            ].map(item => `<div class="province-plan-chip">
                <span>${item.label}</span><strong>${item.value}</strong>
            </div>`).join('');
        }

        $provinceMonthPlanBody.innerHTML = rows.map(r => `
            <tr data-month="${r.month_idx}" class="${r.month_idx === currentMonth ? 'selected' : ''}">
                <td>${r.month}</td>
                <td>${fmt(r.demand)}</td>
                <td>${fmt(r.local)}</td>
                <td>${fmt(r.import)}</td>
                <td>${fmt(r.transfer_in)}</td>
                <td>${fmt(r.transfer_out)}</td>
                <td>${fmt(r.inventory)}</td>
                <td class="${r.shortage > 0.5 ? 'risk-text' : ''}">${fmt(r.shortage)}</td>
                <td>${svcTag(r.service_rate)}</td>
            </tr>
        `).join('');
        $provinceMonthPlanBody.querySelectorAll('tr').forEach(row => {
            row.addEventListener('click', () => {
                currentMonth = parseInt(row.dataset.month);
                $month.value = currentMonth;
                refreshAllViews();
            });
        });
    }

    function renderPortInspector() {
        if (!$portMonthPlanBody || selectedPortIdx == null) return;
        const scen = currentScenario;
        const port = DATA.ports[selectedPortIdx];
        const rows = portPlanRows(selectedPortIdx, scen);
        const totals = rows.reduce((acc, r) => {
            acc.total_in += r.total_in;
            acc.total_out += r.total_out;
            acc.capacity += r.capacity;
            return acc;
        }, {total_in:0, total_out:0, capacity:0});
        totals.utilization = totals.capacity > 0 ? totals.total_in / totals.capacity : 0;

        const active = rows[currentMonth] || rows[0];
        const importRows = (active?.imports || []).map(r => ({
            ...r,
            attrs: '',
        }));
        const destinationRows = (active?.destinations || []).map(r => ({
            ...r,
            attrs: `data-prov-idx="${r.idx}" class="action-row"`,
        }));
        const topImport = importRows[0];
        const topDestination = destinationRows[0];
        const capacityTone = (active?.utilization || 0) >= 0.85 ? 'risk-text' : '';

        if ($portPlanSelect) $portPlanSelect.value = selectedPortIdx;
        if ($portPlanTitle) $portPlanTitle.textContent = `${port.name} - ${scenarioLabel(scen)}`;
        if ($portCurrentFlow) {
            $portCurrentFlow.innerHTML = `
                <div class="flow-card primary">
                    <h4>${DATA.meta.months[currentMonth]}: impor masuk dari negara mana?</h4>
                    ${renderMiniTable(importRows, [
                        {label: 'Negara asal', value: r => r.name},
                        {label: 'Ton', value: r => fmt(r.volume)},
                    ], 'Tidak ada impor masuk ke pelabuhan ini pada bulan aktif.')}
                </div>
                <div class="flow-card">
                    <h4>Dikirim ke provinsi mana?</h4>
                    ${renderMiniTable(destinationRows, [
                        {label: 'Provinsi tujuan', value: r => r.name},
                        {label: 'Ton', value: r => fmt(r.volume)},
                    ], 'Tidak ada distribusi keluar dari pelabuhan ini pada bulan aktif.')}
                </div>
                <div class="flow-card">
                    <h4>Kapasitas bulan ini</h4>
                    <div class="flow-metrics">
                        <div><span>Masuk</span><strong>${fmt(active?.total_in || 0)} ton</strong></div>
                        <div><span>Keluar</span><strong>${fmt(active?.total_out || 0)} ton</strong></div>
                        <div><span>Kapasitas</span><strong>${fmt(active?.capacity || 0)} ton</strong></div>
                        <div><span>Utilisasi</span><strong class="${capacityTone}">${fmtPct(active?.utilization || 0)}</strong></div>
                    </div>
                </div>
                <div class="flow-card">
                    <h4>Ringkasan keputusan</h4>
                    <div class="flow-note">
                        <strong>${topImport ? topImport.name : 'Tidak ada negara aktif'}</strong>
                        ${topImport ? ` memasok ${fmt(topImport.volume)} ton.` : ''}
                        ${topDestination ? `<br><strong>${topDestination.name}</strong> menerima ${fmt(topDestination.volume)} ton terbesar dari pelabuhan ini.` : '<br>Tidak ada provinsi tujuan aktif.'}
                    </div>
                </div>
            `;
            bindDecisionClicks();
        }
        if ($portPlanSummary) {
            $portPlanSummary.innerHTML = [
                {label: 'Masuk tahunan', value: `${fmt(totals.total_in)} ton`},
                {label: 'Keluar tahunan', value: `${fmt(totals.total_out)} ton`},
                {label: 'Kapasitas tahunan', value: `${fmt(totals.capacity)} ton`},
                {label: 'Utilisasi tahunan', value: fmtPct(totals.utilization)},
                {label: 'Negara bulan aktif', value: topImport ? `${topImport.name} (${fmt(topImport.volume)} ton)` : 'tidak ada'},
                {label: 'Tujuan bulan aktif', value: topDestination ? `${topDestination.name} (${fmt(topDestination.volume)} ton)` : 'tidak ada'},
            ].map(item => `<div class="province-plan-chip">
                <span>${item.label}</span><strong>${item.value}</strong>
            </div>`).join('');
        }

        $portMonthPlanBody.innerHTML = rows.map(r => {
            const topCountry = r.imports?.[0]
                ? `${r.imports[0].name} (${fmt(r.imports[0].volume)})`
                : '-';
            const topDestinationText = r.destinations?.[0]
                ? `${r.destinations[0].name} (${fmt(r.destinations[0].volume)})`
                : '-';
            return `
                <tr data-month="${r.month_idx}" class="${r.month_idx === currentMonth ? 'selected' : ''}">
                    <td>${r.month}</td>
                    <td>${fmt(r.total_in)}</td>
                    <td>${fmt(r.total_out)}</td>
                    <td>${fmt(r.capacity)}</td>
                    <td class="${r.utilization >= 0.85 ? 'risk-text' : ''}">${fmtPct(r.utilization)}</td>
                    <td>${topCountry}</td>
                    <td>${topDestinationText}</td>
                </tr>
            `;
        }).join('');
        $portMonthPlanBody.querySelectorAll('tr').forEach(row => {
            row.addEventListener('click', () => {
                currentMonth = parseInt(row.dataset.month);
                $month.value = currentMonth;
                refreshAllViews();
            });
        });
    }

    function renderDecisionCockpit() {
        if (!$decisionTitle) return;
        const scen = currentScenario;
        const t = currentMonth;
        const monthName = DATA.meta.months[t];
        const label = scenarioLabel(scen);
        renderMonthQuick();

        const provRows = DATA.provinces.map((p, i) => {
            const m = DATA[scen].provinces[i].monthly[t];
            return {
                idx: i,
                name: p.name,
                demand: m.demand,
                local: m.local,
                import: m.import,
                transferIn: m.transfer_in,
                transferOut: m.transfer_out,
                shortage: m.shortage,
                inventory: m.inventory,
                service: svcRate(i, t, scen),
            };
        });

        const totalDemand = provRows.reduce((s, r) => s + r.demand, 0);
        const totalLocal = provRows.reduce((s, r) => s + r.local, 0);
        const totalImport = provRows.reduce((s, r) => s + r.import, 0);
        const totalShortage = provRows.reduce((s, r) => s + r.shortage, 0);
        const totalInventory = provRows.reduce((s, r) => s + r.inventory, 0);
        const service = totalDemand > 0 ? 1 - totalShortage / totalDemand : 1.0;
        const importDep = (totalImport + totalLocal) > 0 ? totalImport / (totalImport + totalLocal) : 0;

        const portRows = DATA.ports.map((p, h) => {
            const m = DATA[scen].ports[h].monthly[t];
            return {
                idx: h,
                name: p.name,
                totalIn: m.total_in,
                totalOut: m.total_out,
                cap: m.thru_cap,
                util: m.thru_cap > 0 ? m.total_in / m.thru_cap : 0,
                imports: m.imports || {},
                distribution: m.distribution || {},
            };
        }).filter(r => r.totalIn > 0.5 || r.totalOut > 0.5)
          .sort((a, b) => b.totalIn - a.totalIn);

        const countryTotals = DATA.meta.imp_names.map((name, s) => ({name, volume: 0}));
        portRows.forEach(port => {
            Object.entries(port.imports).forEach(([s, volume]) => {
                const idx = parseInt(s);
                if (countryTotals[idx]) countryTotals[idx].volume += volume;
            });
        });
        const countryRows = countryTotals.filter(r => r.volume > 0.5)
            .sort((a, b) => b.volume - a.volume);

        const receiverRows = provRows.filter(r => r.import > 0.5)
            .sort((a, b) => b.import - a.import)
            .slice(0, 7)
            .map(r => ({...r, attrs: `data-prov-idx="${r.idx}" class="action-row"`}));

        const transferRows = [];
        DATA.provinces.forEach((p, i) => {
            const m = DATA[scen].provinces[i].monthly[t];
            Object.entries(m.transfer_out_to || {}).forEach(([j, volume]) => {
                if (volume > 0.1) {
                    transferRows.push({
                        fromIdx: i,
                        toIdx: parseInt(j),
                        from: p.name,
                        to: provName(parseInt(j)),
                        volume,
                        attrs: `data-prov-idx="${parseInt(j)}" class="action-row"`,
                    });
                }
            });
        });
        transferRows.sort((a, b) => b.volume - a.volume);
        const transferTotal = transferRows.reduce((s, r) => s + r.volume, 0);

        const riskRows = [];
        provRows.filter(r => r.shortage > 0.5)
            .sort((a, b) => b.shortage - a.shortage)
            .slice(0, 3)
            .forEach(r => riskRows.push({
                kind: 'Shortage',
                location: r.name,
                value: `${fmt(r.shortage)} ton`,
                action: 'prioritaskan pasokan',
                attrs: `data-prov-idx="${r.idx}" class="action-row risk-bad"`,
            }));
        portRows.filter(r => r.util >= 0.85)
            .sort((a, b) => b.util - a.util)
            .slice(0, 3)
            .forEach(r => riskRows.push({
                kind: 'Pelabuhan',
                location: r.name,
                value: fmtPct(r.util),
                action: 'cek kapasitas',
                attrs: `data-port-idx="${r.idx}" class="action-row risk-warn"`,
            }));
        provRows.filter(r => r.demand > 0 && r.inventory < 0.10 * r.demand)
            .sort((a, b) => (a.inventory / Math.max(a.demand, 1)) - (b.inventory / Math.max(b.demand, 1)))
            .slice(0, 3)
            .forEach(r => riskRows.push({
                kind: 'Inventori',
                location: r.name,
                value: `${fmt(r.inventory)} ton`,
                action: 'pantau stok',
                attrs: `data-prov-idx="${r.idx}" class="action-row risk-warn"`,
            }));

        const topPortText = portRows.length ? `${portRows[0].name} (${fmt(portRows[0].totalIn)} ton)` : 'tidak ada pelabuhan aktif';
        const topReceiverText = receiverRows.length ? `${receiverRows[0].name} (${fmt(receiverRows[0].import)} ton)` : 'tidak ada impor diterima';
        const topTransferText = transferRows.length ? `${transferRows[0].from} ke ${transferRows[0].to} (${fmt(transferRows[0].volume)} ton)` : 'tidak ada transfer';
        const shortageText = totalShortage > 0.5 ? `masih ada shortage ${fmt(totalShortage)} ton` : 'shortage tertutup';

        $decisionTitle.textContent = `${label} - ${monthName}`;
        $decisionMode.textContent = label;
        $decisionNarrative.innerHTML =
            `<strong>Inti keputusan:</strong> pada bulan ${monthName}, model memenuhi demand ` +
            `${fmt(totalDemand)} ton dengan ${fmt(totalLocal)} ton pasokan lokal dan ` +
            `${fmt(totalImport)} ton pasokan impor. Pelabuhan utama adalah ` +
            `<strong>${topPortText}</strong>, penerima impor terbesar adalah ` +
            `<strong>${topReceiverText}</strong>, dan transfer utama adalah ` +
            `<strong>${topTransferText}</strong>. Status layanan: <strong>${shortageText}</strong>.`;

        $decisionKpis.innerHTML = [
            {label: 'Demand bulan ini', value: `${fmt(totalDemand)} ton`, sub: `service ${fmtPct(service)}`},
            {label: 'Pasokan lokal', value: `${fmt(totalLocal)} ton`, sub: `${fmtPct(totalLocal / Math.max(totalLocal + totalImport, 1))} dari supply`},
            {label: 'Pasokan impor', value: `${fmt(totalImport)} ton`, sub: `dependency ${fmtPct(importDep)}`},
            {label: 'Shortage', value: `${fmt(totalShortage)} ton`, sub: totalShortage > 0.5 ? 'perlu perhatian' : 'tidak ada defisit', tone: totalShortage > 0.5 ? 'bad' : 'good'},
            {label: 'Inventori akhir', value: `${fmt(totalInventory)} ton`, sub: 'akumulasi provinsi'},
            {label: 'Transfer', value: `${fmt(transferTotal)} ton`, sub: `${transferRows.length} relasi aktif`},
        ].map(k => `<div class="decision-kpi ${k.tone || ''}">
            <div class="decision-kpi-label">${k.label}</div>
            <div class="decision-kpi-value">${k.value}</div>
            <div class="decision-kpi-sub">${k.sub}</div>
        </div>`).join('');

        $countrySummary.textContent = countryRows.length ? `${countryRows[0].name} terbesar` : 'tidak ada impor';
        $transferActionSummary.textContent = transferRows.length ? `${transferRows.length} relasi aktif` : 'tidak ada transfer';

        $topPorts.innerHTML = renderMiniTable(portRows.slice(0, 7).map(r => ({
            ...r,
            attrs: `data-port-idx="${r.idx}" class="action-row"`,
        })), [
            {label: 'Pelabuhan', value: r => r.name},
            {label: 'Masuk', value: r => fmt(r.totalIn)},
            {label: 'Keluar', value: r => fmt(r.totalOut)},
            {label: 'Util.', value: r => fmtPct(r.util)},
        ], 'Tidak ada aliran impor pada bulan ini.');

        $topReceivers.innerHTML = renderMiniTable(receiverRows, [
            {label: 'Provinsi', value: r => r.name},
            {label: 'Impor', value: r => fmt(r.import)},
            {label: 'Lokal', value: r => fmt(r.local)},
            {label: 'Trn In', value: r => fmt(r.transferIn)},
        ], 'Tidak ada provinsi penerima impor pada bulan ini.');

        $topTransferActions.innerHTML = renderMiniTable(transferRows.slice(0, 7), [
            {label: 'Dari', value: r => r.from},
            {label: 'Ke', value: r => r.to},
            {label: 'Volume', value: r => fmt(r.volume)},
        ], 'Tidak ada transfer antarprovinsi pada bulan ini.');

        $riskActions.innerHTML = renderMiniTable(riskRows.slice(0, 7), [
            {label: 'Jenis', value: r => r.kind},
            {label: 'Lokasi', value: r => r.location},
            {label: 'Nilai', value: r => r.value},
            {label: 'Aksi', value: r => r.action},
        ], 'Tidak ada shortage besar, bottleneck pelabuhan >= 85%, atau inventori sangat rendah.');

        bindDecisionClicks();
        renderProvinceInspector();
        renderPortInspector();
    }

    // ── Map setup ────────────────────────────────────────────────────────
    function bearingDeg(from, to) {
        const p1 = map.latLngToLayerPoint(from);
        const p2 = map.latLngToLayerPoint(to);
        return Math.atan2(p2.y - p1.y, p2.x - p1.x) * 180 / Math.PI;
    }

    const map = L.map('map').setView([-2.5, 118], 5);
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        maxZoom: 18,
        attribution: '© OpenStreetMap'
    }).addTo(map);

    const provLayer     = L.layerGroup().addTo(map);
    const portLayer     = L.layerGroup().addTo(map);
    const flowLayer     = L.layerGroup().addTo(map);
    const transferLayer = L.layerGroup().addTo(map);
    const adjLayer      = L.layerGroup().addTo(map);

    function redrawMap() {
        provLayer.clearLayers();
        portLayer.clearLayers();
        flowLayer.clearLayers();
        transferLayer.clearLayers();
        adjLayer.clearLayers();

        // ── Adjacency edges (W + W_sea) ──────────────────────────────
        if ($showAdj && $showAdj.checked && DATA.adjacency) {
            DATA.adjacency.forEach(edge => {
                const pi = DATA.provinces[edge.i];
                const pj = DATA.provinces[edge.j];
                if (!pi || !pj) return;
                const isSea = edge.type === 'sea';
                const line = L.polyline(
                    [pi.coord, pj.coord],
                    { color: isSea ? '#e44' : '#4a9',
                      weight: 2, opacity: 0.5,
                      dashArray: isSea ? '6,4' : null }
                );
                line.bindTooltip(
                    `${pi.name} ↔ ${pj.name}<br/>${isSea ? 'W_sea (ferry)' : 'W (land border)'}`
                );
                adjLayer.addLayer(line);
            });
        }

        const t = currentMonth;
        const scen = currentScenario;
        const otherScen = scen === 'initial' ? 'optimized' : 'initial';

        // ── Province circles ─────────────────────────────────────────
        DATA.provinces.forEach((prov, i) => {
            const m = DATA[scen].provinces[i].monthly[t];
            const sr = svcRate(i, t, scen);
            const color = svcColor(sr);
            const r = Math.max(5, Math.sqrt(m.demand / 200));
            const marker = L.circleMarker([prov.coord[0], prov.coord[1]], {
                radius: r,
                fillColor: color,
                color: i === selectedProvIdx ? '#000' : '#333',
                weight: i === selectedProvIdx ? 3 : 1,
                fillOpacity: 0.75,
            });
            marker.bindTooltip(
                `<strong>${prov.name}</strong><br/>` +
                `Demand: ${fmt(m.demand)} ton<br/>` +
                `Lokal: ${fmt(m.local)} | Impor: ${fmt(m.import)}<br/>` +
                `Shortage: ${fmt(m.shortage)} ton<br/>` +
                `Service: ${fmtPct(sr)}`
            );
            marker.on('click', () => {
                selectedProvIdx = i;
                if ($provincePlanSelect) $provincePlanSelect.value = selectedProvIdx;
                switchTab('provinces');
                renderProvinceTable();
                renderProvinceDetail();
                renderProvinceInspector();
                redrawMap();
            });
            provLayer.addLayer(marker);
        });

        // ── Port markers (squares) ───────────────────────────────────
        if ($showPorts.checked) {
            DATA.ports.forEach((port, h) => {
                const m = DATA[scen].ports[h].monthly[t];
                const totalIn = m.total_in;
                const r = Math.max(6, Math.sqrt(totalIn / 100));
                const icon = L.divIcon({
                    className: 'port-icon',
                    iconSize: [r * 2, r * 2],
                    html: `<div style="width:${r * 2}px;height:${r * 2}px;` +
                          `background:#444;border:${h === selectedPortIdx ? '3px' : '1px'} solid ` +
                          `${h === selectedPortIdx ? '#000' : '#fff'};` +
                          `box-shadow:0 0 4px rgba(0,0,0,0.4);"></div>`,
                });
                const marker = L.marker([port.coord[0], port.coord[1]], {icon: icon});
                marker.bindTooltip(
                    `<strong>⚓ ${port.name}</strong><br/>` +
                    `Masuk: ${fmt(totalIn)} ton<br/>` +
                    `Keluar: ${fmt(m.total_out)} ton<br/>` +
                    `Util: ${fmtPct(totalIn / Math.max(m.thru_cap, 1))}`
                );
                marker.on('click', () => {
                    selectedPortIdx = h;
                    if ($portPlanSelect) $portPlanSelect.value = selectedPortIdx;
                    switchTab('ports');
                    renderPortTable();
                    renderPortDetail();
                    renderPortInspector();
                    redrawMap();
                });
                portLayer.addLayer(marker);
            });
        }

        // ── Import flows (port → province) ───────────────────────────
        if ($showFlows.checked && $showPorts.checked) {
            DATA.ports.forEach((port, h) => {
                const m = DATA[scen].ports[h].monthly[t];
                Object.entries(m.distribution).forEach(([provIdx, vol]) => {
                    if (vol < 50) return;     // skip trivial flows
                    const i = parseInt(provIdx);
                    const w = Math.max(1, Math.min(6, Math.sqrt(vol / 200)));
                    const line = L.polyline([port.coord, DATA.provinces[i].coord], {
                        color: '#1f77b4',
                        weight: w,
                        opacity: 0.55,
                    });
                    line.bindTooltip(
                        `${port.name} → ${provName(i)}<br/>` +
                        `${fmt(vol)} ton`
                    );
                    flowLayer.addLayer(line);
                });
            });
        }

        // ── Inter-provincial transfers ───────────────────────────────
        if ($showTrns.checked) {
            DATA.provinces.forEach((prov, i) => {
                const m = DATA[scen].provinces[i].monthly[t];
                Object.entries(m.transfer_out_to).forEach(([toIdx, vol]) => {
                    if (vol < 0.1) return;
                    const j = parseInt(toIdx);
                    const fromCoord = prov.coord;
                    const toCoord = DATA.provinces[j].coord;
                    const midCoord = [
                        (fromCoord[0] + toCoord[0]) / 2,
                        (fromCoord[1] + toCoord[1]) / 2,
                    ];
                    const w = Math.max(1, Math.min(5, Math.sqrt(vol / 100)));
                    const line = L.polyline([fromCoord, toCoord], {
                        color: '#f2c300',
                        weight: w,
                        opacity: 0.9,
                        dashArray: '6,4',
                    });
                    line.bindTooltip(`${prov.name} &rarr; ${provName(j)}`);
                    transferLayer.addLayer(line);
                    const arrow = L.marker(midCoord, {
                        interactive: true,
                        icon: L.divIcon({
                            className: 'transfer-arrow-icon',
                            iconSize: [18, 18],
                            iconAnchor: [9, 9],
                            html: `<div class="transfer-arrow" style="transform: rotate(${bearingDeg(fromCoord, toCoord)}deg);"></div>`,
                        }),
                    });
                    arrow.bindTooltip(`${prov.name} &rarr; ${provName(j)}`);
                    transferLayer.addLayer(arrow);
                });
            });
        }

        // Status bar
        const total_demand = DATA.provinces.reduce((s, _, i) =>
            s + DATA[scen].provinces[i].monthly[t].demand, 0);
        const total_short = DATA.provinces.reduce((s, _, i) =>
            s + DATA[scen].provinces[i].monthly[t].shortage, 0);
        $status.innerHTML =
            `Bulan: <strong>${DATA.meta.months[t]}</strong> · ` +
            `Skenario: <strong>${scenarioLabel(scen)}</strong> · ` +
            `Demand: ${fmt(total_demand)} t · Shortage: ${fmt(total_short)} t`;
    }

    // ── Province table + detail ──────────────────────────────────────────
    function renderProvinceTable() {
        const tbody = document.querySelector('#province-table tbody');
        const t = currentMonth;
        const rows = DATA.provinces.map((p, i) => {
            const m = DATA[currentScenario].provinces[i].monthly[t];
            const sr = svcRate(i, t, currentScenario);
            return {
                idx: i, name: p.name,
                demand: m.demand, local: m.local, import: m.import,
                trnIn: m.transfer_in, trnOut: m.transfer_out,
                shortage: m.shortage, service: sr, inv: m.inventory,
            };
        });
        rows.sort((a, b) => b.demand - a.demand);   // default: desc demand

        tbody.innerHTML = rows.map(r => `
            <tr data-idx="${r.idx}" class="${r.idx === selectedProvIdx ? 'selected' : ''}">
                <td>${r.name}</td>
                <td>${fmt(r.demand)}</td>
                <td>${fmt(r.local)}</td>
                <td>${fmt(r.import)}</td>
                <td>${fmt(r.trnIn)}</td>
                <td>${fmt(r.trnOut)}</td>
                <td>${fmt(r.shortage)}</td>
                <td>${svcTag(r.service)}</td>
                <td>${fmt(r.inv)}</td>
            </tr>
        `).join('');
        tbody.querySelectorAll('tr').forEach(row => {
            row.addEventListener('click', () => {
                selectedProvIdx = parseInt(row.dataset.idx);
                if ($provincePlanSelect) $provincePlanSelect.value = selectedProvIdx;
                renderProvinceTable();
                renderProvinceDetail();
                renderProvinceInspector();
                redrawMap();
            });
        });
    }

    function renderProvinceDetail() {
        const $det = document.getElementById('province-detail');
        if (selectedProvIdx == null) {
            $det.innerHTML = '<em>Klik baris atau marker provinsi untuk melihat detail.</em>';
            return;
        }
        const i = selectedProvIdx;
        const t = currentMonth;
        const prov = DATA.provinces[i];
        const mI = DATA.initial.provinces[i].monthly[t];
        const mO = DATA.optimized.provinces[i].monthly[t];
        const m = currentScenario === 'initial' ? mI : mO;
        const srI = svcRate(i, t, 'initial');
        const srO = svcRate(i, t, 'optimized');

        // Build flow tables
        const portFlows = (mm) => {
            const entries = Object.entries(mm.import_by_port)
                .map(([h, v]) => ({h: parseInt(h), v}))
                .sort((a, b) => b.v - a.v);
            if (entries.length === 0) return '<em>—</em>';
            return `<table class="flow-table">
                <thead><tr><th>Pelabuhan</th><th>Volume (ton)</th></tr></thead>
                <tbody>${entries.map(e =>
                    `<tr><td>${portName(e.h)}</td><td>${fmt(e.v)}</td></tr>`).join('')}</tbody>
            </table>`;
        };
        const trnInFlows = (mm) => {
            const entries = Object.entries(mm.transfer_in_from)
                .map(([j, v]) => ({j: parseInt(j), v}))
                .sort((a, b) => b.v - a.v);
            if (entries.length === 0) return '<em>—</em>';
            return `<table class="flow-table">
                <thead><tr><th>Dari Provinsi</th><th>Volume</th></tr></thead>
                <tbody>${entries.map(e =>
                    `<tr><td>${provName(e.j)}</td><td>${fmt(e.v)}</td></tr>`).join('')}</tbody>
            </table>`;
        };
        const trnOutFlows = (mm) => {
            const entries = Object.entries(mm.transfer_out_to)
                .map(([j, v]) => ({j: parseInt(j), v}))
                .sort((a, b) => b.v - a.v);
            if (entries.length === 0) return '<em>—</em>';
            return `<table class="flow-table">
                <thead><tr><th>Ke Provinsi</th><th>Volume</th></tr></thead>
                <tbody>${entries.map(e =>
                    `<tr><td>${provName(e.j)}</td><td>${fmt(e.v)}</td></tr>`).join('')}</tbody>
            </table>`;
        };

        $det.innerHTML = `
            <h3>📍 ${prov.name}
                ${prov.is_producer ? '<span class="tag tag-ok">PRODUSEN</span>' : ''}
                <span class="tag tag-warn">Cluster ${DATA.clusters[prov.cluster].name}</span>
            </h3>
            <div class="stat-grid">
                <div class="stat">
                    <div class="stat-label">Demand (${DATA.meta.months[t]})</div>
                    <div class="stat-value">${fmt(m.demand)} ton</div>
                </div>
                <div class="stat">
                    <div class="stat-label">Lokal Diterima</div>
                    <div class="stat-value">${fmt(m.local)} ton ${delta(mI.local, mO.local)}</div>
                </div>
                <div class="stat">
                    <div class="stat-label">Impor Diterima</div>
                    <div class="stat-value">${fmt(m.import)} ton ${delta(mI.import, mO.import)}</div>
                </div>
                <div class="stat">
                    <div class="stat-label">Transfer Masuk</div>
                    <div class="stat-value">${fmt(m.transfer_in)} ton ${delta(mI.transfer_in, mO.transfer_in)}</div>
                </div>
                <div class="stat">
                    <div class="stat-label">Transfer Keluar</div>
                    <div class="stat-value">${fmt(m.transfer_out)} ton ${delta(mI.transfer_out, mO.transfer_out)}</div>
                </div>
                <div class="stat">
                    <div class="stat-label">Inventori Akhir</div>
                    <div class="stat-value">${fmt(m.inventory)} ton ${delta(mI.inventory, mO.inventory)}</div>
                </div>
                <div class="stat">
                    <div class="stat-label">Shortage</div>
                    <div class="stat-value">${fmt(m.shortage)} ton ${delta(mI.shortage, mO.shortage)}</div>
                </div>
                <div class="stat">
                    <div class="stat-label">Service Rate</div>
                    <div class="stat-value">${svcTag(currentScenario === 'initial' ? srI : srO)}</div>
                </div>
            </div>

            <h4>Aliran Bulan Ini — Solusi Awal vs Rekomendasi</h4>
            <div class="compare-grid">
                <div class="compare-col initial">
                    <h5>Solusi Awal ALNS</h5>
                    <strong>Impor masuk via pelabuhan:</strong> ${portFlows(mI)}
                    <strong>Transfer masuk dari:</strong> ${trnInFlows(mI)}
                    <strong>Transfer keluar ke:</strong> ${trnOutFlows(mI)}
                </div>
                <div class="compare-col optimized">
                    <h5>Rekomendasi Model</h5>
                    <strong>Impor masuk via pelabuhan:</strong> ${portFlows(mO)}
                    <strong>Transfer masuk dari:</strong> ${trnInFlows(mO)}
                    <strong>Transfer keluar ke:</strong> ${trnOutFlows(mO)}
                </div>
            </div>
        `;
    }

    // ── Port table + detail ──────────────────────────────────────────────
    function renderPortTable() {
        const tbody = document.querySelector('#port-table tbody');
        const t = currentMonth;
        const rows = DATA.ports.map((p, h) => {
            const m = DATA[currentScenario].ports[h].monthly[t];
            return {
                idx: h, name: p.name,
                totalIn: m.total_in, totalOut: m.total_out,
                cap: m.thru_cap,
                util: m.thru_cap > 0 ? m.total_in / m.thru_cap : 0,
            };
        });
        rows.sort((a, b) => b.totalIn - a.totalIn);

        tbody.innerHTML = rows.map(r => `
            <tr data-idx="${r.idx}" class="${r.idx === selectedPortIdx ? 'selected' : ''}">
                <td>${r.name}</td>
                <td>${fmt(r.totalIn)}</td>
                <td>${fmt(r.totalOut)}</td>
                <td>${fmt(r.cap)}</td>
                <td>${fmtPct(r.util)}</td>
            </tr>
        `).join('');
        tbody.querySelectorAll('tr').forEach(row => {
            row.addEventListener('click', () => {
                selectedPortIdx = parseInt(row.dataset.idx);
                if ($portPlanSelect) $portPlanSelect.value = selectedPortIdx;
                renderPortTable();
                renderPortDetail();
                renderPortInspector();
                redrawMap();
            });
        });
    }

    function renderPortDetail() {
        const $det = document.getElementById('port-detail');
        if (selectedPortIdx == null) {
            $det.innerHTML = '<em>Klik baris atau marker pelabuhan untuk melihat detail.</em>';
            return;
        }
        const h = selectedPortIdx;
        const t = currentMonth;
        const port = DATA.ports[h];
        const mI = DATA.initial.ports[h].monthly[t];
        const mO = DATA.optimized.ports[h].monthly[t];
        const m  = currentScenario === 'initial' ? mI : mO;

        const importsTable = (mm) => {
            const rows = Object.entries(mm.imports)
                .map(([s, v]) => ({s: parseInt(s), v}))
                .filter(r => r.v > 0.5)
                .sort((a, b) => b.v - a.v);
            if (rows.length === 0) return '<em>—</em>';
            return `<table class="flow-table">
                <thead><tr><th>Negara Asal</th><th>Volume</th></tr></thead>
                <tbody>${rows.map(r =>
                    `<tr><td>${impName(r.s)}</td><td>${fmt(r.v)}</td></tr>`).join('')}</tbody>
            </table>`;
        };
        const distTable = (mm) => {
            const rows = Object.entries(mm.distribution)
                .map(([i, v]) => ({i: parseInt(i), v}))
                .filter(r => r.v > 0.5)
                .sort((a, b) => b.v - a.v);
            if (rows.length === 0) return '<em>—</em>';
            return `<table class="flow-table">
                <thead><tr><th>Ke Provinsi</th><th>Volume</th></tr></thead>
                <tbody>${rows.map(r =>
                    `<tr><td>${provName(r.i)}</td><td>${fmt(r.v)}</td></tr>`).join('')}</tbody>
            </table>`;
        };

        $det.innerHTML = `
            <h3>⚓ ${port.name}</h3>
            <div class="stat-grid">
                <div class="stat">
                    <div class="stat-label">Total Masuk (${DATA.meta.months[t]})</div>
                    <div class="stat-value">${fmt(m.total_in)} ton ${delta(mI.total_in, mO.total_in)}</div>
                </div>
                <div class="stat">
                    <div class="stat-label">Total Keluar</div>
                    <div class="stat-value">${fmt(m.total_out)} ton ${delta(mI.total_out, mO.total_out)}</div>
                </div>
                <div class="stat">
                    <div class="stat-label">Kapasitas Throughput</div>
                    <div class="stat-value">${fmt(m.thru_cap)} ton</div>
                    <div class="stat-mini">Util: ${fmtPct(m.total_in / Math.max(m.thru_cap, 1))}</div>
                </div>
            </div>
            <div class="stat-mini">Provinsi yang dilayani: ${port.services.map(provName).join(', ')}</div>

            <h4>Aliran Bulan Ini — Solusi Awal vs Rekomendasi</h4>
            <div class="compare-grid">
                <div class="compare-col initial">
                    <h5>Solusi Awal ALNS</h5>
                    <strong>Impor masuk dari negara:</strong> ${importsTable(mI)}
                    <strong>Distribusi keluar ke provinsi:</strong> ${distTable(mI)}
                </div>
                <div class="compare-col optimized">
                    <h5>Rekomendasi Model</h5>
                    <strong>Impor masuk dari negara:</strong> ${importsTable(mO)}
                    <strong>Distribusi keluar ke provinsi:</strong> ${distTable(mO)}
                </div>
            </div>
        `;
    }

    // ── Cluster table + detail ───────────────────────────────────────────
    // Transfer page
    function getTransferRows() {
        const scen = $transferScenario ? $transferScenario.value : currentScenario;
        const month = $transferMonth ? $transferMonth.value : 'all';
        let rows = [];
        if (scen === 'both') {
            rows = [...(DATA.transfers?.initial || []), ...(DATA.transfers?.optimized || [])];
        } else {
            rows = [...(DATA.transfers?.[scen] || [])];
        }
        if (month !== 'all') {
            const monthIdx = parseInt(month);
            rows = rows.filter(r => r.month_idx === monthIdx);
        }
        return rows;
    }

    function renderTransferTable() {
        const tbody = document.querySelector('#transfer-table tbody');
        if (!tbody) return;
        const rows = getTransferRows();
        const key = transferSort.key;
        const dir = transferSort.dir === 'asc' ? 1 : -1;
        rows.sort((a, b) => {
            const av = a[key], bv = b[key];
            if (typeof av === 'number' && typeof bv === 'number') return (av - bv) * dir;
            return String(av).localeCompare(String(bv), 'id') * dir;
        });

        const total = rows.reduce((s, r) => s + r.volume, 0);
        if ($transferSummary) {
            $transferSummary.textContent = `${rows.length} instance, total ${fmt(total)} ton`;
        }
        tbody.innerHTML = rows.map(r => `
            <tr>
                <td>${scenarioLabel(r.scenario)}</td>
                <td>${DATA.meta.months[r.month_idx]}</td>
                <td>${r.from}</td>
                <td>${r.to}</td>
                <td>${fmt(r.volume)}</td>
            </tr>
        `).join('') || '<tr><td colspan="5"><em>Tidak ada transfer pada filter ini.</em></td></tr>';
    }

    document.querySelectorAll('#transfer-table th[data-sort]').forEach(th => {
        th.addEventListener('click', () => {
            const key = th.dataset.sort;
            if (transferSort.key === key) {
                transferSort.dir = transferSort.dir === 'asc' ? 'desc' : 'asc';
            } else {
                transferSort = {key, dir: key === 'volume' ? 'desc' : 'asc'};
            }
            renderTransferTable();
        });
    });

    [$transferScenario, $transferMonth].filter(Boolean).forEach(el => {
        el.addEventListener('change', renderTransferTable);
    });

    function clusterAggregate(cidx, t, scen) {
        const provs = DATA.clusters[cidx].provinces;
        let demand = 0, local = 0, imp = 0, sh = 0, inv = 0;
        provs.forEach(i => {
            const m = DATA[scen].provinces[i].monthly[t];
            demand += m.demand; local += m.local; imp += m.import;
            sh += m.shortage; inv += m.inventory;
        });
        return {demand, local, imp, sh, inv,
                service: demand > 0 ? 1 - sh / demand : 1.0};
    }

    function renderClusterTable() {
        const tbody = document.querySelector('#cluster-table tbody');
        const t = currentMonth;
        const rows = DATA.clusters.map((c, ci) => {
            const a = clusterAggregate(ci, t, currentScenario);
            return {
                idx: ci, name: c.name, provs: c.provinces.length,
                ...a, import: a.imp, shortage: a.sh,
            };
        });
        rows.sort((a, b) => b.demand - a.demand);

        tbody.innerHTML = rows.map(r => `
            <tr data-idx="${r.idx}">
                <td>${r.name}</td>
                <td>${r.provs}</td>
                <td>${fmt(r.demand)}</td>
                <td>${fmt(r.local)}</td>
                <td>${fmt(r.import)}</td>
                <td>${fmt(r.shortage)}</td>
                <td>${fmt(r.inv)}</td>
                <td>${svcTag(r.service)}</td>
            </tr>
        `).join('');
        tbody.querySelectorAll('tr').forEach(row => {
            row.addEventListener('click', () => {
                renderClusterDetail(parseInt(row.dataset.idx));
            });
        });
    }

    function renderClusterDetail(cidx) {
        const $det = document.getElementById('cluster-detail');
        const t = currentMonth;
        const c  = DATA.clusters[cidx];
        const aI = clusterAggregate(cidx, t, 'initial');
        const aO = clusterAggregate(cidx, t, 'optimized');

        const provRows = c.provinces.map(i => {
            const mI = DATA.initial.provinces[i].monthly[t];
            const mO = DATA.optimized.provinces[i].monthly[t];
            const srI = svcRate(i, t, 'initial');
            const srO = svcRate(i, t, 'optimized');
            return `<tr>
                <td>${provName(i)}</td>
                <td>${fmt(mO.demand)}</td>
                <td>${fmt(mI.local)} → ${fmt(mO.local)}</td>
                <td>${fmt(mI.import)} → ${fmt(mO.import)}</td>
                <td>${fmt(mI.shortage)} → ${fmt(mO.shortage)}</td>
                <td>${fmt(mI.inventory)} -> ${fmt(mO.inventory)}</td>
                <td>${svcTag(srO)}</td>
            </tr>`;
        }).join('');

        $det.innerHTML = `
            <h3>🏝️ ${c.name} (${c.provinces.length} provinsi)</h3>
            <div class="stat-grid">
                <div class="stat">
                    <div class="stat-label">Total Demand (${DATA.meta.months[t]})</div>
                    <div class="stat-value">${fmt(aO.demand)} ton</div>
                </div>
                <div class="stat">
                    <div class="stat-label">Lokal Diterima</div>
                    <div class="stat-value">${fmt(aO.local)} ton ${delta(aI.local, aO.local)}</div>
                </div>
                <div class="stat">
                    <div class="stat-label">Impor Diterima</div>
                    <div class="stat-value">${fmt(aO.imp)} ton ${delta(aI.imp, aO.imp)}</div>
                </div>
                <div class="stat">
                    <div class="stat-label">Shortage</div>
                    <div class="stat-value">${fmt(aO.sh)} ton ${delta(aI.sh, aO.sh)}</div>
                </div>
                <div class="stat">
                    <div class="stat-label">Inventori Cluster</div>
                    <div class="stat-value">${fmt(aO.inv)} ton ${delta(aI.inv, aO.inv)}</div>
                </div>
                <div class="stat">
                    <div class="stat-label">Service Rate</div>
                    <div class="stat-value">${svcTag(aO.service)}
                        <span class="stat-mini">(awal: ${fmtPct(aI.service)})</span></div>
                </div>
            </div>

            <h4>Provinsi dalam cluster ini (Solusi Awal ke Rekomendasi)</h4>
            <table class="flow-table">
                <thead>
                    <tr><th>Provinsi</th><th>Demand</th><th>Lokal</th><th>Impor</th><th>Shortage</th><th>Inventori</th><th>Service</th></tr>
                </thead>
                <tbody>${provRows}</tbody>
            </table>
        `;
    }

    // ── Tabs ─────────────────────────────────────────────────────────────
    function switchTab(name) {
        activeTab = name;
        document.querySelectorAll('.tab').forEach(t =>
            t.classList.toggle('active', t.dataset.tab === name));
        document.querySelectorAll('.panel').forEach(p =>
            p.classList.toggle('active', p.id === name + '-panel'));
    }
    document.querySelectorAll('.tab').forEach(t => {
        t.addEventListener('click', () => switchTab(t.dataset.tab));
    });


    [$month, $scenario, $showFlows, $showTrns, $showPorts, $showAdj]
        .filter(Boolean).forEach(el => el.addEventListener('change', refresh));

    // ── Analytics Tab (Chart.js) ─────────────────────────────────────────
    let chartInstances = {};
    function destroyChart(id) { if (chartInstances[id]) { chartInstances[id].destroy(); delete chartInstances[id]; } }

    function fmtRp(n) {
        if (n >= 1e12) return (n/1e12).toFixed(2) + ' T';
        if (n >= 1e9)  return (n/1e9).toFixed(1) + ' M';
        if (n >= 1e6)  return (n/1e6).toFixed(1) + ' Jt';
        return fmt(n);
    }

    function renderAnalytics() {
        if (typeof Chart === 'undefined' || !DATA.cost_initial) return;
        const cI = DATA.cost_initial, cO = DATA.cost_optimized;

        // KPI Cards
        const savings = cI.z_cost - cO.z_cost;
        const savPct = cI.z_cost > 0 ? (savings / cI.z_cost * 100).toFixed(1) : '0';
        document.getElementById('kpi-cards').innerHTML = [
            { label:'Z_cost Solusi Awal', value: 'Rp ' + fmtRp(cI.z_cost), sub:'Total biaya awal ALNS' },
            { label:'Z_cost Rekomendasi', value: 'Rp ' + fmtRp(cO.z_cost), sub:'Total biaya rencana model',
              delta: savings > 0 ? `↓ ${savPct}% penghematan` : null, cls: 'positive' },
            { label:'Import Dependency', value: (cO.import_dep*100).toFixed(1)+'%',
              sub: `Limit ≤ ${(DATA.meta.eps_import_dep*100).toFixed(0)}%` },
            { label:'Total Shortage', value: fmt(cO.total_shortage) + ' ton',
              sub: cO.total_shortage < 1 ? '✓ Zero shortage' : '⚠ Ada shortage' },
            { label:'Produksi Lokal', value: fmt(cO.total_local, 0) + ' ton',
              sub: `Min: ${fmt(DATA.meta.eps_local_min)} ton` },
        ].map(k => `<div class="kpi-card">
            <div class="kpi-label">${k.label}</div>
            <div class="kpi-value">${k.value}</div>
            <div class="kpi-sub">${k.sub}</div>
            ${k.delta ? `<div class="kpi-delta ${k.cls||''}">${k.delta}</div>` : ''}
        </div>`).join('');

        // 1) Cost Breakdown Bar Chart
        destroyChart('cost-breakdown');
        const termKeys = ['loc_cost','imp_cost','emg_cost','dist_cost','trns_cost','hold_prov','fix_act','fix_emg'];
        const termLabels = ['Produksi+Ship','Impor','Emergency','Distribusi','Transfer','Hold Prov','Fix Aktivasi','Fix Emergency'];
        const termKeysI = termKeys.map(k => cI.terms[k] || 0);
        const termKeysO = termKeys.map(k => cO.terms[k] || 0);
        chartInstances['cost-breakdown'] = new Chart(
            document.getElementById('chart-cost-breakdown'), {
            type: 'bar',
            data: {
                labels: termLabels,
                datasets: [
                    { label: 'Solusi Awal', data: termKeysI, backgroundColor: 'rgba(255,127,14,0.7)', borderRadius: 4 },
                    { label: 'Rekomendasi', data: termKeysO, backgroundColor: 'rgba(44,160,44,0.7)', borderRadius: 4 },
                ]
            },
            options: { responsive:true, plugins:{legend:{position:'top'}},
                scales:{y:{ticks:{callback:v=>fmtRp(v)}}}, indexAxis:'x' }
        });

        // 2) Convergence Line Chart
        destroyChart('convergence');
        if (DATA.obj_history && DATA.obj_history.length > 0) {
            const step = Math.max(1, Math.floor(DATA.obj_history.length / 500));
            const sampled = DATA.obj_history.filter((_,i) => i % step === 0);
            chartInstances['convergence'] = new Chart(
                document.getElementById('chart-convergence'), {
                type: 'line',
                data: {
                    labels: sampled.map((_,i) => i*step),
                    datasets: [{ label:'Best Objective (Rp)', data: sampled,
                        borderColor:'#1f77b4', fill:false, pointRadius:0, borderWidth:2 }]
                },
                options: { responsive:true, plugins:{legend:{display:false}},
                    scales:{y:{ticks:{callback:v=>fmtRp(v)}}, x:{title:{display:true,text:'Iterasi'}}} }
            });
        }

        function renderHistoryChart(id, history, label, color) {
            destroyChart(id);
            if (!history || history.length === 0) return;
            const step = Math.max(1, Math.floor(history.length / 500));
            const sampled = history.filter((_, i) => i % step === 0);
            chartInstances[id] = new Chart(document.getElementById('chart-' + id), {
                type: 'line',
                data: {
                    labels: sampled.map((_, i) => i * step),
                    datasets: [{
                        label,
                        data: sampled,
                        borderColor: color,
                        fill: false,
                        pointRadius: 0,
                        borderWidth: 2,
                    }]
                },
                options: { responsive:true, plugins:{legend:{display:false}},
                    scales:{y:{ticks:{callback:v=>fmtRp(v)}}, x:{title:{display:true,text:'Iterasi'}}} }
            });
        }
        renderHistoryChart('cost-convergence', DATA.cost_history, 'Z_cost (Rp)', '#2ca02c');
        renderHistoryChart('penalty-convergence', DATA.penalty_history, 'Penalty (Rp)', '#d62728');

        // 3) Monthly Supply vs Demand (stacked area)
        destroyChart('monthly-supply');
        const mO = cO.monthly;
        chartInstances['monthly-supply'] = new Chart(
            document.getElementById('chart-monthly-supply'), {
            type: 'bar',
            data: {
                labels: DATA.meta.months,
                datasets: [
                    { label:'Lokal', data: mO.local, backgroundColor:'rgba(44,160,44,0.7)', stack:'supply' },
                    { label:'Impor', data: mO.import, backgroundColor:'rgba(31,119,180,0.7)', stack:'supply' },
                    { label:'Demand', data: mO.demand, type:'line', borderColor:'#d62728',
                      fill:false, pointRadius:3, borderWidth:2, borderDash:[5,3] },
                ]
            },
            options: { responsive:true, plugins:{legend:{position:'top'}},
                scales:{y:{stacked:false, ticks:{callback:v=>fmt(v)}}} }
        });

        // 4) Source Mix (Solusi Awal vs Rekomendasi stacked %)
        destroyChart('source-mix');
        const mI = cI.monthly;
        const pctLocal = mO.local.map((v,i) => { const tot=v+mO.import[i]; return tot>0?v/tot*100:0; });
        const pctImport = pctLocal.map(v => 100 - v);
        chartInstances['source-mix'] = new Chart(
            document.getElementById('chart-source-mix'), {
            type: 'bar',
            data: {
                labels: DATA.meta.months,
                datasets: [
                    { label:'% Lokal', data: pctLocal, backgroundColor:'rgba(44,160,44,0.7)', stack:'s' },
                    { label:'% Impor', data: pctImport, backgroundColor:'rgba(31,119,180,0.7)', stack:'s' },
                ]
            },
            options: { responsive:true, plugins:{legend:{position:'top'}},
                scales:{y:{stacked:true, max:100, ticks:{callback:v=>v+'%'}}, x:{stacked:true}} }
        });
    }

    // ── Re-render everything when controls change ────────────────────────
    function refresh() {
        currentMonth = parseInt($month.value);
        currentScenario = $scenario.value;
        renderDecisionCockpit();
        redrawMap();
        renderProvinceTable();
        renderPortTable();
        renderTransferTable();
        renderClusterTable();
        renderProvinceDetail();
        renderPortDetail();
        if (activeTab === 'analytics') renderAnalytics();
    }

    // Also render analytics when switching to that tab
    const origSwitchTab = switchTab;
    switchTab = function(name) {
        origSwitchTab(name);
        if (name === 'analytics') renderAnalytics();
        if (name === 'transfers') renderTransferTable();
    };

    // Initial render
    refresh();
})();
