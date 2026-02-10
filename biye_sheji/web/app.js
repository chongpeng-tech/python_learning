const $ = (id) => document.getElementById(id);

function escapeHtml(text) {
  return text
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function setButtonLoading(btn, loading, text) {
  if (!btn) return;
  if (loading) {
    btn.dataset.rawText = btn.textContent;
    btn.textContent = text;
    btn.disabled = true;
    return;
  }
  btn.textContent = btn.dataset.rawText || btn.textContent;
  btn.disabled = false;
}

async function postJson(url, payload) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    const detail = data.detail || "请求失败";
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return data;
}

function parseRelations(text) {
  return text
    .split(/[，,]/)
    .map((v) => v.trim())
    .filter(Boolean);
}

function labelClass(label) {
  return `label-${label.toLowerCase()}`;
}

function renderNER(data) {
  const entities = data.entities || [];
  const tokenLabels = data.token_labels || [];

  if (!entities.length) {
    $("ner-entities").innerHTML = '<span class="muted">未识别到实体</span>';
  } else {
    $("ner-entities").innerHTML = entities
      .map(
        (item) =>
          `<span class="chip"><span class="type">${escapeHtml(item.type)}</span>${escapeHtml(item.text)}</span>`
      )
      .join("");
  }

  if (!tokenLabels.length) {
    $("ner-tokens").innerHTML = '<span class="muted">无 token 结果</span>';
  } else {
    $("ner-tokens").innerHTML = tokenLabels
      .map(
        (item) =>
          `<span class="token-chip ${labelClass(item.label)}">${escapeHtml(item.token)} <em>${escapeHtml(item.label)}</em></span>`
      )
      .join("");
  }
}

function renderRanking(ranking) {
  if (!Array.isArray(ranking) || !ranking.length) {
    return '<p class="muted">暂无结果</p>';
  }

  return ranking
    .map((item) => {
      const score = Number(item.score || 0);
      const width = Math.max(0, Math.min(100, score * 100));
      return `
        <div class="bar-row">
          <span>${escapeHtml(item.relation)}</span>
          <div class="bar-track"><div class="bar-fill" style="width:${width.toFixed(2)}%"></div></div>
          <span>${score.toFixed(4)}</span>
        </div>
      `;
    })
    .join("");
}

function renderTable(rows) {
  const tbody = $("qa-table").querySelector("tbody");
  if (!Array.isArray(rows) || !rows.length) {
    tbody.innerHTML = '<tr><td colspan="3" class="muted">无查询结果</td></tr>';
    return;
  }

  tbody.innerHTML = rows
    .map(
      (row) => `
      <tr>
        <td>${escapeHtml(String(row.subject ?? ""))}</td>
        <td>${escapeHtml(String(row.relation ?? ""))}</td>
        <td>${escapeHtml(String(row.object ?? ""))}</td>
      </tr>
    `
    )
    .join("");
}

function renderQA(data) {
  const answers = data.answers || [];
  const answerSource = data.answer_source || "";
  $("qa-answer").innerHTML = answers.length
    ? answers.map((name) => `<span class="chip">${escapeHtml(name)}</span>`).join("")
    : '<span class="muted">未找到明确实体答案</span>';
  $("qa-answer-source").textContent = answerSource
    ? `答案来源: ${answerSource}`
    : "";

  const names = data.person_names || [];
  $("qa-names").innerHTML = names.length
    ? names.map((name) => `<span class="chip">${escapeHtml(name)}</span>`).join("")
    : '<span class="muted">未识别到人名</span>';

  $("qa-ranking").innerHTML = renderRanking(data.relation_ranking || []);
  $("qa-cypher").textContent = data.cypher || "无";
  $("qa-params").textContent = JSON.stringify(data.params || {}, null, 2);

  const errorEl = $("qa-neo4j-error");
  if (data.neo4j_error) {
    errorEl.textContent = `Neo4j 错误: ${data.neo4j_error}`;
  } else {
    errorEl.textContent = "";
  }

  renderTable(data.neo4j_rows || []);
}

async function checkHealth() {
  const box = $("health-text");
  try {
    const res = await fetch("/api/health");
    const data = await res.json();
    box.textContent = `在线 | MPS: ${data.mps_available ? "可用" : "不可用"} | 缓存 NER ${data.ner_cache_count}`;
    box.style.color = "var(--good)";
  } catch (_err) {
    box.textContent = "服务不可用";
    box.style.color = "var(--warn)";
  }
}

function bindNER() {
  const form = $("ner-form");
  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const btn = $("ner-submit");
    setButtonLoading(btn, true, "NER 运行中...");
    try {
      const data = await postJson("/api/ner", {
        text: $("ner-text").value,
        ner_model_dir: $("ner-model-dir").value,
        device: $("ner-device").value,
      });
      renderNER(data);
    } catch (err) {
      $("ner-entities").innerHTML = `<span class="error">${escapeHtml(err.message)}</span>`;
      $("ner-tokens").innerHTML = '<span class="muted">无结果</span>';
    } finally {
      setButtonLoading(btn, false, "");
    }
  });
}

function bindQA() {
  const form = $("qa-form");
  const runNeo4j = $("qa-run-neo4j");
  const neo4jFields = $("neo4j-fields");

  runNeo4j.addEventListener("change", () => {
    neo4jFields.classList.toggle("active", runNeo4j.checked);
  });
  neo4jFields.classList.toggle("active", runNeo4j.checked);

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const btn = $("qa-submit");
    setButtonLoading(btn, true, "问答运行中...");
    try {
      const data = await postJson("/api/qa", {
        question: $("qa-question").value,
        relations: parseRelations($("qa-relations").value),
        sbert_model: $("qa-sbert-model").value,
        ner_model_dir: $("qa-model-dir").value,
        device: $("qa-device").value,
        run_neo4j: runNeo4j.checked,
        neo4j_uri: $("qa-neo4j-uri").value,
        neo4j_user: $("qa-neo4j-user").value,
        neo4j_password: $("qa-neo4j-password").value,
        neo4j_database: $("qa-neo4j-database").value,
      });
      renderQA(data);
    } catch (err) {
      $("qa-answer").innerHTML = '<span class="muted">无结果</span>';
      $("qa-answer-source").textContent = "";
      $("qa-neo4j-error").textContent = err.message;
      $("qa-ranking").innerHTML = '<p class="muted">无结果</p>';
      $("qa-cypher").textContent = "无";
      $("qa-params").textContent = "{}";
      renderTable([]);
    } finally {
      setButtonLoading(btn, false, "");
    }
  });

  document.querySelectorAll(".ghost[data-question]").forEach((button) => {
    button.addEventListener("click", () => {
      $("qa-question").value = button.getAttribute("data-question") || "";
    });
  });
}

checkHealth();
bindNER();
bindQA();
