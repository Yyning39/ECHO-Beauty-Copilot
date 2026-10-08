let cases = [];
let currentCase = null;
let riskEvents = [];
let latestAnalysisSchemaVersion = null;

const $ = (id) => document.getElementById(id);

function showToast(message) {
  const toast = $('toast');
  toast.textContent = message;
  toast.classList.add('show');
  setTimeout(() => toast.classList.remove('show'), 2200);
}

function renderCaseList() {
  const inboxCount = $('inbox-count');
  if (inboxCount) inboxCount.textContent = cases.length;
  $('case-list').innerHTML = cases.map((item) => `
    <div class="case-item ${item.id === currentCase.id ? 'active' : ''}" data-id="${item.id}">
      <div class="avatar ${item.riskTone === 'red' ? 'coral' : 'lavender'}">${item.name.slice(0,1)}</div>
      <div class="case-copy"><strong>${item.name}</strong><small>${item.issue} · ${item.messages[item.messages.length - 1].text}</small><div class="case-meta"><small>${item.lastSeen}</small><span class="case-status">${item.risk}</span></div></div>
    </div>`).join('');
  document.querySelectorAll('.case-item').forEach((node) => node.addEventListener('click', async () => {
    const fallback = cases.find((item) => item.id === node.dataset.id);
    currentCase = fallback;
    renderAll();
    try {
      const response = await fetch(`/api/cases/${encodeURIComponent(node.dataset.id)}`);
      if (response.ok) currentCase = await response.json();
      renderAll();
    } catch (_) {
      showToast('详细数据加载失败，已保留列表摘要');
    }
  }));
}

function renderChat() {
  $('customer-name').textContent = currentCase.name;
  $('customer-avatar').textContent = currentCase.name.slice(0,1);
  $('customer-scene').textContent = currentCase.scene;
  $('customer-scene').className = `tag ${currentCase.riskTone === 'red' ? 'red-tag' : 'blue-tag'}`;
  $('customer-last').textContent = `${currentCase.lastSeen}活跃`;
  $('order-product').textContent = currentCase.order;
  $('order-meta').textContent = `${currentCase.orderStatus} · ${currentCase.orderAmount}`;
  $('chat-messages').innerHTML = currentCase.messages.map((message) => `
    <div class="message ${message.role}"><div class="message-avatar">${message.role === 'buyer' ? currentCase.name.slice(0,1) : 'A'}</div><div class="message-body"><div class="message-meta">${message.role === 'buyer' ? currentCase.name : '测试客服'} · ${message.time}</div><div class="message-text">${message.text}</div></div></div>`).join('');
}

function renderCopilot() {
  $('issue-type').textContent = currentCase.issue;
  $('emotion-state').textContent = currentCase.emotion;
  $('risk-level').textContent = currentCase.risk;
  $('risk-reason').textContent = currentCase.riskReason;
  $('risk-card').className = `risk-card ${currentCase.riskTone === 'red' ? 'red-card' : 'amber-card'}`;
  $('risk-card').querySelector('.risk-icon').textContent = currentCase.riskTone === 'red' ? '!' : '!';
  $('reply-suggestion').textContent = currentCase.reply;
  $('evidence-list').innerHTML = currentCase.evidence.map((item) => `<span>${item}</span>`).join('');
  $('emotion-bars').innerHTML = Array.from({length:5}, (_, i) => `<i style="background:${i < Math.round(currentCase.emotionScore / 20) ? (currentCase.riskTone === 'red' ? '#e78a93' : '#8b9aff') : '#e5e9f5'}"></i>`).join('');
  $('profile-contacts').textContent = currentCase.id === 'S00024' ? '3' : '2';
  $('profile-tickets').textContent = '1';
  $('profile-note').textContent = currentCase.riskTone === 'red' ? '该场景需要人工升级，避免承诺无法确认的结果。' : '回复中应明确下一步和时间预期，减少消费者重复追问。';
}

function renderTimeline() {
  $('timeline').innerHTML = currentCase.timeline.map((item) => `<div class="timeline-item"><i class="timeline-dot ${item.tone}"></i><div class="timeline-copy"><div><strong>${item.title}</strong><small>${item.detail}</small></div><span class="timeline-time">${item.time}</span></div></div>`).join('');
}

function renderRiskTable() {
  const highRisk = riskEvents.filter((item) => item.level === '高风险').length;
  const processing = riskEvents.filter((item) => item.status === '处理中').length;
  [['risk-nav-count', riskEvents.length], ['risk-metric-count', riskEvents.length], ['risk-kpi-count', riskEvents.length], ['high-risk-count', highRisk], ['risk-filter-count', riskEvents.length], ['high-risk-filter-count', highRisk], ['processing-risk-filter-count', processing]].forEach(([id, value]) => { if ($(id)) $(id).textContent = value; });
  $('risk-table').innerHTML = `<div class="risk-row header"><div>事件 / 用户</div><div>风险类型</div><div>触发依据</div><div>负责人</div><div>状态</div></div>` + riskEvents.map((event) => {
    const canUpdate = String(event.id).startsWith('R-');
    return `<div class="risk-row"><div><strong>${event.id}</strong><small>${event.customer}</small></div><div><span class="risk-badge ${event.tone}">${event.level}</span><small>${event.issue}</small></div><div>${event.reason}</div><div>${event.owner}</div><div>${canUpdate ? `<button class="status-badge risk-status-button ${event.status === '已解决' ? 'done' : ''}" data-risk-id="${event.id}" data-status="${event.status}">${event.status}</button>` : `<span class="status-badge">${event.status}</span>`}</div></div>`;
  }).join('');
  document.querySelectorAll('.risk-status-button').forEach((button) => button.addEventListener('click', async () => {
    const nextStatus = button.dataset.status === '待处理' ? '处理中' : button.dataset.status === '处理中' ? '已解决' : '已解决';
    if (button.dataset.status === '已解决') return;
    try {
      const response = await fetch(`/api/risk-events/${encodeURIComponent(button.dataset.riskId)}`, {method:'PATCH', headers:{'Content-Type':'application/json'}, body:JSON.stringify({status:nextStatus, owner:'当前客服'})});
      if (!response.ok) throw new Error('update failed');
      riskEvents = await (await fetch('/api/risk-events')).json();
      renderRiskTable();
      showToast(`风险事件已更新为${nextStatus}`);
    } catch (_) { showToast('风险状态更新失败'); }
  }));
}

function renderAll() { renderCaseList(); renderChat(); renderCopilot(); renderTimeline(); renderRiskTable(); }

function applyCopilotResult(result) {
  const final = result.analysis?.final || {};
  const intent = final.intent || {};
  const emotion = final.emotion || {};
  const risk = final.risk || {};
  const reply = result.reply?.suggested_reply || {};
  latestAnalysisSchemaVersion = result.analysis?.schema_version || null;
  const levelMap = {high: '高风险', medium: '中风险', low: '低风险'};
  const intensityMap = {high: 90, medium: 60, low: 30, unknown: 20};

  currentCase.scene = intent.scene_major || currentCase.scene;
  currentCase.issue = intent.scene_minor || currentCase.issue;
  currentCase.emotion = emotion.label || currentCase.emotion;
  currentCase.emotionScore = intensityMap[emotion.intensity] || currentCase.emotionScore;
  currentCase.risk = levelMap[risk.level] || currentCase.risk;
  currentCase.riskTone = risk.level === 'high' ? 'red' : risk.level === 'medium' ? 'amber' : 'blue';
  currentCase.riskReason = (risk.reasons || []).join('；') || currentCase.riskReason;
  currentCase.reply = reply.text || currentCase.reply;

  const basis = result.reply?.reply_basis || [];
  currentCase.evidence = basis.map((item) => {
    const label = item.field || item.field_code || '业务事实';
    return item.value ? `${label}：${item.value}` : label;
  });
  if (!currentCase.evidence.length) {
    const binding = result.order_binding || {};
    currentCase.evidence = binding.target_order_no
      ? [`目标订单：${binding.target_order_no}`]
      : ['当前会话消息'];
  }

  const total = result.observability?.stage_timings_ms?.total_ms;
  const traceTime = document.querySelector('.trace-time');
  if (traceTime && total !== undefined) traceTime.textContent = `本轮 ${(total / 1000).toFixed(1)}s`;
  const sourceCount = document.querySelector('.source-count');
  if (sourceCount) sourceCount.textContent = `${currentCase.evidence.length} 个核验依据`;
  renderAll();
}

async function saveReplyFeedback(action, finalReply = null, reason = null) {
  const response = await fetch('/api/reply-feedback', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
      session_id: currentCase.id,
      action,
      original_reply: currentCase.reply,
      final_reply: finalReply,
      reason,
      analysis_schema_version: latestAnalysisSchemaVersion,
    }),
  });
  if (!response.ok) throw new Error('feedback failed');
}

async function analyzeCurrentCase() {
  const button = $('regenerate-button');
  const originalText = button.textContent;
  button.disabled = true;
  button.textContent = '分析中…';
  showToast('正在检索历史记录并分析当前会话');

  const latestBuyer = [...currentCase.messages].reverse().find((item) => item.role === 'buyer');
  const latestMessage = currentCase.messages[currentCase.messages.length - 1];
  try {
    if (currentCase.aiGenerated) {
      await saveReplyFeedback('rejected', null, '客服选择重新生成');
    }
    const response = await fetch('/api/copilot/analyze', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        session_id: currentCase.id,
        query: latestBuyer?.text || '请分析当前消费者会话',
        // 当前工作台分析到“最新一条可见消息”为止，避免离线数据看到未来。
        as_of_message_id: latestMessage?.message_id || null,
      }),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.detail || result.error || '分析失败');
    applyCopilotResult(result);
    currentCase.aiGenerated = true;
    showToast('AI 分析已更新，请人工确认后使用回复');
  } catch (error) {
    showToast(`分析失败：${error.message}`);
  } finally {
    button.disabled = false;
    button.textContent = originalText;
  }
}

function setupNavigation() {
  document.querySelectorAll('.nav-item').forEach((button) => button.addEventListener('click', () => {
    document.querySelectorAll('.nav-item').forEach((item) => item.classList.remove('active'));
    button.classList.add('active');
    document.querySelectorAll('.view').forEach((view) => view.classList.remove('active-view'));
    const target = button.dataset.view === 'copilot' ? 'copilot-view' : `${button.dataset.view}-view`;
    $(target).classList.add('active-view');
    $('page-title').textContent = button.dataset.view === 'copilot' ? '接待辅助' : button.dataset.view === 'risk' ? '风险中心' : '服务洞察';
  }));
}

function setupActions() {
  $('use-button').addEventListener('click', () => { $('reply-box').value = currentCase.reply; showToast('回复已放入输入框，请确认后发送'); });
  $('copy-button').addEventListener('click', async () => { try { await navigator.clipboard.writeText(currentCase.reply); } catch (_) {} showToast('回复已复制'); });
  $('regenerate-button').addEventListener('click', analyzeCurrentCase);
  $('event-button').addEventListener('click', async () => {
    try {
      const response = await fetch('/api/risk-events', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({session_id:currentCase.id})});
      if (!response.ok) throw new Error('risk event failed');
      riskEvents = await (await fetch('/api/risk-events')).json();
      renderRiskTable();
      showToast('风险事件已保存并同步到风险中心');
    } catch (_) {
      showToast('风险事件保存失败，请稍后重试');
    }
  });
  $('handoff-button').addEventListener('click', () => { showToast('已生成升级摘要，等待主管接管'); });
  document.querySelector('.send-button').addEventListener('click', async () => {
    const finalReply = $('reply-box').value.trim();
    if (!finalReply) { showToast('请先填写或使用建议回复'); return; }
    const action = finalReply === currentCase.reply ? 'accepted' : 'edited';
    try {
      await saveReplyFeedback(action, finalReply, null);
      showToast('人工确认结果已记录；演示环境未连接真实消息发送');
    } catch (_) {
      showToast('人工反馈记录失败，请稍后重试');
    }
  });
}

async function init() {
  try {
    const [casesResponse, riskResponse] = await Promise.all([fetch('/api/cases?limit=20'), fetch('/api/risk-events')]);
    cases = await casesResponse.json();
    riskEvents = await riskResponse.json();
  } catch (_) { cases = []; riskEvents = []; }
  if (!cases.length) return;
  currentCase = cases[0];
  setupNavigation(); setupActions(); renderAll();
}

init();
