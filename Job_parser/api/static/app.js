"use strict";

// ---------------------------------------------------------------- helpers

// Build DOM nodes. Strings become text nodes, so LLM output is never parsed as HTML.
function el(tag, attrs, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v == null || v === false) continue;
    if (k === "class") node.className = v;
    else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
    else node.setAttribute(k, v === true ? "" : v);
  }
  for (const child of children.flat(Infinity)) {
    if (child == null || child === false) continue;
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return node;
}

const chips = (items, cls = "") =>
  items && items.length
    ? el("div", { class: "chips" }, items.map((t) => el("span", { class: `chip ${cls}` }, t)))
    : el("span", { class: "empty" }, "none");

const meter = (fraction) =>
  el("div", { class: "meter" }, el("span", { style: `width:${Math.max(0, Math.min(1, fraction)) * 100}%` }));

const rawJson = (data) => el("details", { class: "raw" }, el("summary", {}, "View JSON"), el("pre", {}, JSON.stringify(data, null, 2)));

function field(label, input, hint) {
  return el("div", { class: "field" }, el("label", { for: input.id }, label, hint ? el("span", { class: "hint" }, ` — ${hint}`) : null), input);
}
const textInput = (id, placeholder = "") => el("input", { type: "text", id, placeholder });
const textArea = (id, placeholder = "", rows) => el("textarea", { id, placeholder, rows });
const val = (id) => document.getElementById(id).value.trim();
const setVal = (id, v) => { document.getElementById(id).value = v; };

const splitList = (s) => s.split(/[,\n]/).map((x) => x.trim()).filter(Boolean);

// "Skill: description" per line -> {skill: description}
function parseSkills(text) {
  const out = {};
  for (const line of text.split("\n")) {
    const t = line.trim();
    if (!t) continue;
    const i = t.indexOf(":");
    if (i === -1) out[t] = "";
    else out[t.slice(0, i).trim()] = t.slice(i + 1).trim();
  }
  return out;
}

class ApiError extends Error {
  constructor(status, body) {
    super(typeof body.detail === "string" ? body.detail : "Request failed");
    this.status = status;
    this.body = body;
  }
}

async function api(path, body) {
  const res = await fetch(path, body === undefined ? {} : {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  let data;
  try { data = await res.json(); } catch { data = { detail: `HTTP ${res.status}` }; }
  if (!res.ok) throw new ApiError(res.status, data);
  return data;
}

function errorBox(err) {
  if (!(err instanceof ApiError)) return el("div", { class: "alert" }, el("strong", {}, "Request failed"), String(err));
  if (err.status === 422 && Array.isArray(err.body.detail)) {
    return el("div", { class: "alert" }, el("strong", {}, "Please check the form"),
      err.body.detail.map((d) => el("div", {}, `${(d.loc || []).slice(1).join(".")}: ${d.msg}`)));
  }
  const title = err.status === 502 ? "The LLM failed to produce a usable answer" : `Error ${err.status}`;
  return el("div", { class: "alert" }, el("strong", {}, title), err.message,
    err.body.raw_response ? el("details", {}, el("summary", {}, "Raw model output"), el("pre", {}, err.body.raw_response)) : null);
}

// Disable the button, show a running timer, render the result or the error.
async function run(button, target, task) {
  const timer = button.parentElement.querySelector(".timer");
  const started = performance.now();
  button.disabled = true;
  const tick = setInterval(() => { timer.textContent = `${((performance.now() - started) / 1000).toFixed(1)}s`; }, 100);
  target.replaceChildren();
  try {
    target.replaceChildren(...[].concat(await task()));
  } catch (err) {
    target.replaceChildren(errorBox(err));
  } finally {
    clearInterval(tick);
    timer.textContent = `done in ${((performance.now() - started) / 1000).toFixed(1)}s`;
    button.disabled = false;
  }
}

function actions(onRun, onExample, label = "Analyze") {
  const btn = el("button", { class: "btn primary", type: "button" }, label);
  btn.addEventListener("click", () => onRun(btn));
  return el("div", { class: "actions" }, btn,
    el("button", { class: "btn", type: "button", onclick: onExample }, "Load example"),
    el("span", { class: "timer" }));
}

// ---------------------------------------------------------------- examples

const EXAMPLES = {
  scraperJob: {
    title: "Real-estate listings scraper + price dashboard",
    description:
`We're a small property investment firm in Texas. We need a Python developer to build a scraper that pulls new listings from Zillow and Redfin every morning for 6 ZIP codes, cleans and de-duplicates them, and stores them in PostgreSQL.

On top of that we want a simple dashboard (Streamlit or similar) showing median price per sq ft, days on market and price drops per ZIP code, plus a weekly email summary to our team.

The scraper must handle rate limits and rotating proxies, and run on a schedule on our AWS account. Please describe similar scrapers you've built. Experience with Scrapy or Playwright is a plus.`,
    skills: "Python, Web Scraping, Scrapy, PostgreSQL, Data Visualization, AWS",
    duration: "1 to 3 months",
    weekly: "Less than 30 hrs/week",
  },
  chatbotJob:
`Looking for an AI engineer to build a customer-support chatbot for our Shopify store. It should answer questions about orders, shipping and returns using our help-center articles (RAG with OpenAI or a similar model), and hand off to a human agent in Gorgias when it isn't confident.

All conversations should be logged to a PostgreSQL database so we can review them weekly. Deploy with Docker on our AWS account. Bonus: a small React admin page to review flagged conversations.`,
  responsibilities: {
    paragraph:
`• Partner with cross-functional teams to design and implement data-driven solutions that align with business goals
• Gather, clean, and integrate data from disparate sources to support analytical workflows
• Utilize predictive modeling techniques to identify trends and inform strategic planning
• Establish secure data pipelines and manage cloud infrastructure access
• Build dashboards that track weekly KPIs for leadership
• Lead technical initiatives and coach junior analysts in project execution`,
    skills: "Python, SQL, Machine Learning, Scikit-learn, TensorFlow, Power BI, AWS, Airflow",
  },
  freelancerSkills:
`Python: 5 years building ETL jobs and scraping tools
Scrapy: built crawlers for 20+ e-commerce and real-estate sites
PostgreSQL: schema design and query tuning
React: admin dashboards and internal tools
Figma: wireframes for client projects`,
  otherSkills:
`Python: data pipelines and automation
Airflow: scheduling ETL workflows
PostgreSQL: data warehousing
Tableau: executive dashboards`,
};

// ---------------------------------------------------------------- tools

const TOOLS = [];
const tool = (id, name, render) => TOOLS.push({ id, name, render });

// 1. Analyze a job posting -------------------------------------------------
tool("job", "Analyze job", (root) => {
  const out = el("div", { class: "result" });
  const duration = el("select", { id: "job-duration" },
    ["", "Less than 1 month", "1 to 3 months", "3 to 6 months", "More than 6 months"]
      .map((d) => el("option", { value: d }, d || "Not specified")));
  const weekly = el("select", { id: "job-weekly" },
    ["Less than 30 hrs/week", "30+ hrs/week", "Hours to be determined"].map((d) => el("option", { value: d }, d)));

  root.append(
    el("h1", {}, "Analyze a job posting"),
    el("p", { class: "lede" }, "Rewrites the posting, splits it into four subtasks with categories and, if a duration is given, an hour estimate per subtask. Each analysis is saved to the jobs database."),
    el("div", { class: "card" },
      field("Title", textInput("job-title", "e.g. Shopify chatbot with RAG")),
      field("Description", textArea("job-desc", "Paste the full job posting", 9)),
      field("Required skills", textInput("job-skills", "Python, Scrapy, PostgreSQL"), "comma separated"),
      el("div", { class: "grid-2" },
        field("Project length", duration),
        field("Weekly hours", weekly)),
      actions(
        (btn) => run(btn, out, async () => {
          const length = val("job-duration");
          const data = await api("/analyze_jobs", { text: {
            title: val("job-title") || "Untitled job",
            summary: val("job-desc"),
            skills_list: splitList(val("job-skills")),
            duration: length ? `${length}, ${val("job-weekly")}` : null,
          } });
          return renderJob(data.jobs);
        }),
        () => {
          const ex = EXAMPLES.scraperJob;
          setVal("job-title", ex.title); setVal("job-desc", ex.description);
          setVal("job-skills", ex.skills); setVal("job-duration", ex.duration); setVal("job-weekly", ex.weekly);
        })),
    out);
});

const PRIORITY_CLASS = { High: "bad", Medium: "warn", Low: "good" };

function renderJob(job) {
  const hasHours = job.estimated_hours != null;
  const follow = el("div", { class: "result" });

  const table = el("div", { class: "table-wrap" }, el("table", {},
    el("thead", {}, el("tr", {}, el("th", {}, "Subtask"), el("th", {}, "Group"), el("th", {}, "Effort"),
      hasHours ? el("th", {}, "Hours") : null, el("th", {}, "Priority"))),
    el("tbody", {}, job.subtasks.map((s) => {
      const a = s.attributes || {};
      return el("tr", {},
        el("td", {}, el("div", {}, s.task), el("div", { class: "sub" }, s.explanation)),
        el("td", {}, el("span", { class: "chip accent" }, s.group)),
        el("td", { class: "num" }, a.priority_score != null
          ? [`${Math.round(a.priority_score * 100)}%`, meter(a.priority_score)] : "—"),
        hasHours ? el("td", { class: "num" }, a.inherited_time != null ? `${a.inherited_time} h` : "—") : null,
        el("td", {}, a.remark ? el("span", { class: `chip ${PRIORITY_CLASS[a.remark] || ""}` }, a.remark) : "—"));
    }))));

  const profileBtn = el("button", { class: "btn", type: "button" }, "Build skill profile");
  profileBtn.addEventListener("click", () => run(profileBtn, follow, async () => {
    const data = await api("/build_profiles", { jobs: { summary: job.summary, subtasks: job.subtasks } });
    const p = data.profiles[0];
    return el("div", { class: "card" }, el("h2", {}, "Skill profile"), el("pre", { class: "tree" }, p.profile_breakdown.join("\n")));
  }));

  const graphBtn = el("button", { class: "btn", type: "button" }, "Add subtasks to knowledge graph");
  graphBtn.addEventListener("click", () => run(graphBtn, follow, async () => {
    const data = await api("/graph/ingest_subtasks", {
      subtasks: job.subtasks.map((s) => ({ task: s.task, explanation: s.explanation, group: s.group })),
    });
    return el("div", { class: "card" }, el("h2", {}, `Added to graph (${data.ingested} nodes)`),
      data.results.map((n) => el("div", { class: "task-row" },
        el("div", {}, n.name, n.synonyms.length ? el("div", { class: "sub" }, `also: ${n.synonyms.join("; ")}`) : null),
        el("span", { class: "weight" }, `weight ${n.weight}`))));
  }));

  return [
    el("div", { class: "card" },
      el("h2", {}, `Job #${job.job_id}`),
      el("dl", { class: "kv" },
        el("dt", {}, "Summary"), el("dd", {}, job.summary),
        el("dt", {}, "Domain"), el("dd", {}, job.tech_domain || "—"),
        el("dt", {}, "Technologies"), el("dd", {}, chips(job.suggested_technologies)),
        el("dt", {}, "Industry"), el("dd", {}, chips(job.industry_metadata.map(String))),
        el("dt", {}, "Budget"), el("dd", {}, hasHours
          ? [`≈ ${Math.round(job.estimated_hours)} hours`, el("div", { class: "sub" }, job.estimated_hours_basis)]
          : el("span", { class: "empty" }, "no duration given")),
        el("dt", {}, "Assumptions"), el("dd", {}, job.assumptions || "—"))),
    el("div", { class: "card" }, el("h2", {}, "Subtasks"), table,
      el("div", { class: "actions", style: "margin-top:14px" }, profileBtn, graphBtn, el("span", { class: "timer" })),
      rawJson(job)),
    follow,
  ];
}

// 2. Intent + categories ---------------------------------------------------
tool("intent", "Intent & categories", (root) => {
  const out = el("div", { class: "result" });
  root.append(
    el("h1", {}, "Client intent & categories"),
    el("p", { class: "lede" }, "Two analyses side by side: what the client is really after and whether they understand the technical ask, plus a cleaned-up description with general work categories (the input to category clustering)."),
    el("div", { class: "card" },
      field("Job description", textArea("intent-desc", "Paste the job posting", 9)),
      actions(
        (btn) => run(btn, out, async () => {
          const description = val("intent-desc");
          const [intent, transform] = await Promise.all([
            api("/analyze_intent", { description }),
            api("/transform_job", { description }),
          ]);
          const a = intent.analysis;
          const t = transform.transformation;
          const knows = String(a.client_technical_expertise).toUpperCase().startsWith("Y");
          return el("div", { class: "grid-2" },
            el("div", { class: "card" }, el("h2", {}, "Client intent"),
              el("dl", { class: "kv" },
                el("dt", {}, "Technical client"), el("dd", {}, el("span", { class: `chip ${knows ? "good" : "warn"}` }, knows ? "Yes" : "No")),
                el("dt", {}, "Intent"), el("dd", {}, a.purpose_analysis.raw_intent),
                el("dt", {}, "Broad category"), el("dd", {}, a.purpose_analysis.broad_category),
                el("dt", {}, "Specific"), el("dd", {}, a.purpose_analysis.specific_category),
                el("dt", {}, "Confidence"), el("dd", {}, meter(a.confidence_score), el("span", { class: "sub" }, `${Math.round(a.confidence_score * 100)}%`))),
              rawJson(intent)),
            el("div", { class: "card" }, el("h2", {}, "Cleaned description"),
              el("p", { style: "margin-top:0" }, t.cleaned_job_description),
              el("div", { class: "col-title" }, "Key categories"), chips(t.key_categories, "accent"),
              rawJson(transform)));
        }),
        () => setVal("intent-desc", EXAMPLES.chatbotJob))),
    out);
});

// 3. Responsibilities ------------------------------------------------------
tool("parse", "Responsibilities", (root) => {
  const out = el("div", { class: "result" });
  root.append(
    el("h1", {}, "Break down responsibilities"),
    el("p", { class: "lede" }, "Turns bullet-point responsibilities into short action summaries with technical subtasks, and maps each to the skills you list (fuzzy-matched, with confidence)."),
    el("div", { class: "card" },
      field("Responsibilities", textArea("parse-text", "One responsibility per line or bullet", 9)),
      field("Skills to map to", textInput("parse-skills", "Python, SQL, AWS"), "comma separated"),
      actions(
        (btn) => run(btn, out, async () => {
          const data = await api("/parse", { paragraph: val("parse-text"), skills: splitList(val("parse-skills")) });
          if (!data.result.length) return el("div", { class: "card empty" }, "No responsibilities found. Each needs at least three words.");
          return el("div", { class: "card" }, data.result.map((r) => el("div", { class: "resp" },
            el("div", { class: "resp-title" }, r.summary),
            r.tools.length
              ? el("div", { class: "chips" }, r.tools.map((t, i) => el("span", { class: "chip accent", title: "How essential this skill is to the responsibility" },
                  t, r.tool_confidence[i] != null ? el("small", {}, `${Math.round(r.tool_confidence[i])}% relevant`) : null)))
              : el("span", { class: "empty" }, "no matching skills"),
            el("ol", { class: "steps" }, r.subtasks.map((s) => el("li", {}, s.task, s.explanation ? el("div", { class: "sub" }, s.explanation) : null))))),
          rawJson(data));
        }),
        () => { setVal("parse-text", EXAMPLES.responsibilities.paragraph); setVal("parse-skills", EXAMPLES.responsibilities.skills); })),
    out);
});

// 4. Compare two jobs ------------------------------------------------------
tool("compare", "Compare jobs", (root) => {
  const out = el("div", { class: "result" });
  root.append(
    el("h1", {}, "Compare two jobs"),
    el("p", { class: "lede" }, "Finds the shared foundation skills, what each job needs on its own, and how transferable experience is from one to the other."),
    el("div", { class: "card" },
      el("div", { class: "grid-2" },
        field("Job 1", textArea("cmp-1", "", 8)),
        field("Job 2", textArea("cmp-2", "", 8))),
      actions(
        (btn) => run(btn, out, async () => {
          const data = await api("/compare_jobs", { job1: val("cmp-1"), job2: val("cmp-2") });
          const c = data.comparison;
          return el("div", { class: "card" },
            el("div", { class: "col-title" }, "Transferability"),
            el("div", { class: "big-number" }, `${Math.round(c.transferability_score)}`, el("span", { class: "sub" }, " / 100")),
            meter(c.transferability_score / 100),
            el("div", { class: "cols-3", style: "margin-top:18px" },
              el("div", {}, el("div", { class: "col-title" }, "Shared foundation"), chips(c.base_foundation_skills, "good")),
              el("div", {}, el("div", { class: "col-title" }, "Only job 1"), chips(c.job1_specific_expertise)),
              el("div", {}, el("div", { class: "col-title" }, "Only job 2"), chips(c.job2_specific_expertise))),
            rawJson(data));
        }),
        () => { setVal("cmp-1", EXAMPLES.scraperJob.description); setVal("cmp-2", EXAMPLES.chatbotJob); })),
    out);
});

// 5. My skills vs a job ----------------------------------------------------
tool("skills", "My skills vs job", (root) => {
  const out = el("div", { class: "result" });
  const strictness = el("input", { type: "range", id: "sk-strict", min: 0, max: 100, step: 5, value: 70 });
  const strictLabel = el("span", { class: "hint" }, " — 70");
  strictness.addEventListener("input", () => { strictLabel.textContent = ` — ${strictness.value}`; });

  root.append(
    el("h1", {}, "Check your skills against a job"),
    el("p", { class: "lede" }, "Shows which of your skills the job actually uses, which are irrelevant, and what the job needs that you didn't list."),
    el("div", { class: "card" },
      el("div", { class: "grid-2" },
        field("Job description", textArea("sk-job", "", 9)),
        field("Your skills", textArea("sk-list", "Skill: what you did with it", 9), "one per line")),
      el("div", { class: "field" }, el("label", { for: "sk-strict" }, "Strictness", strictLabel), strictness),
      actions(
        (btn) => run(btn, out, async () => {
          const data = await api("/validate_job_skills", {
            job: val("sk-job"), skills: parseSkills(val("sk-list")), strictness: Number(strictness.value),
          });
          const v = data.validation;
          const listed = v.relevant_skills.length + v.unnecessary_skills.length;
          return el("div", { class: "card" },
            el("div", { class: "col-title" }, "Fit"),
            el("div", { class: "big-number" }, `${v.relevant_skills.length}`, el("span", { class: "sub" }, ` of ${listed} skills relevant`)),
            meter(listed ? v.relevant_skills.length / listed : 0),
            el("div", { class: "cols-3", style: "margin-top:18px" },
              el("div", {}, el("div", { class: "col-title" }, "Relevant"), chips(v.relevant_skills, "good")),
              el("div", {}, el("div", { class: "col-title" }, "Not needed for this job"), chips(v.unnecessary_skills, "warn")),
              el("div", {}, el("div", { class: "col-title" }, "Missing from your list"), chips(v.missing_skills, "bad"))),
            el("p", { style: "margin:16px 0 0" }, v.explanation || ""),
            rawJson(data));
        }),
        () => { setVal("sk-job", EXAMPLES.scraperJob.description); setVal("sk-list", EXAMPLES.freelancerSkills); })),
    out);
});

// 6. Compare two skill lists -----------------------------------------------
tool("lists", "Compare skill lists", (root) => {
  const out = el("div", { class: "result" });
  root.append(
    el("h1", {}, "Compare two skill lists"),
    el("p", { class: "lede" }, "E.g. two freelancers, or your profile against a team's. Finds the overlap and what is unique to each list."),
    el("div", { class: "card" },
      el("div", { class: "grid-2" },
        field("List 1", textArea("ls-1", "Skill: description", 7), "one per line"),
        field("List 2", textArea("ls-2", "Skill: description", 7), "one per line")),
      actions(
        (btn) => run(btn, out, async () => {
          const data = await api("/compare_skills_lists", { list1: parseSkills(val("ls-1")), list2: parseSkills(val("ls-2")) });
          const c = data.comparison;
          return el("div", { class: "card" },
            el("div", { class: "cols-3" },
              el("div", {}, el("div", { class: "col-title" }, "In both"), chips(c.foundation_skills, "good")),
              el("div", {}, el("div", { class: "col-title" }, "Only list 1"), chips(c.list1_specific_expertise)),
              el("div", {}, el("div", { class: "col-title" }, "Only list 2"), chips(c.list2_specific_expertise))),
            rawJson(data));
        }),
        () => { setVal("ls-1", EXAMPLES.freelancerSkills); setVal("ls-2", EXAMPLES.otherSkills); },
        "Compare")),
    out);
});

// 7. Knowledge graph -------------------------------------------------------
tool("graph", "Knowledge graph", (root) => {
  const out = el("div", { class: "result" });
  const filter = textInput("graph-filter", "Search tasks and synonyms…");
  const LIMIT = 50;

  // Filtering and per-group limits happen on the server, so this stays fast on large graphs.
  function render(data, q) {
    const tasks = new Map(data.nodes.filter((n) => n.type === "task").map((n) => [n.id, n]));
    const byGroup = new Map();
    for (const e of data.edges) {
      if (e.type !== "belongs_to") continue;
      if (!byGroup.has(e.target)) byGroup.set(e.target, []);
      byGroup.get(e.target).push(tasks.get(e.source));
    }
    const cards = data.nodes.filter((n) => n.type === "group").map((g) => {
      const items = byGroup.get(g.id) || [];
      if (!items.length) return null;
      const more = g.task_count - items.length;
      return el("div", { class: "card group-card" },
        el("h2", {}, g.name, el("span", { class: "sub" }, `  ${g.task_count} tasks`)),
        items.map((t) => el("div", { class: "task-row" },
          el("div", {}, t.name, t.synonyms && t.synonyms.length ? el("div", { class: "sub" }, `also: ${t.synonyms.join("; ")}`) : null),
          el("span", { class: "weight" }, `×${t.weight}`))),
        !q && more > 0 ? el("div", { class: "sub" }, `+ ${more} lighter tasks — search to find them`) : null);
    }).filter(Boolean);
    const summary = el("p", { class: "notice" },
      `${data.total_tasks} tasks in the graph` + (q ? ` · showing matches for “${q}”` : "") + (data.truncated ? ` · top ${LIMIT} per group` : ""));
    if (!cards.length) {
      return [summary, el("div", { class: "card empty" },
        data.total_tasks ? "No tasks match the search." : "The graph is empty. Analyze a job and click “Add subtasks to knowledge graph”.")];
    }
    return [summary, el("div", { class: "groups" }, cards)];
  }

  const btn = el("button", { class: "btn primary", type: "button" }, "Refresh");
  const load = () => run(btn, out, async () => {
    const q = filter.value.trim();
    const params = new URLSearchParams({ limit_per_group: LIMIT });
    if (q) params.set("q", q);
    return render(await api(`/graph?${params}`), q);
  });
  btn.addEventListener("click", load);
  let debounce;
  filter.addEventListener("input", () => { clearTimeout(debounce); debounce = setTimeout(load, 300); });

  root.append(
    el("h1", {}, "Task knowledge graph"),
    el("p", { class: "lede" }, "Every subtask you add is merged with the matching existing task if there is one (closest candidates shortlisted by text similarity, then judged by the LLM in a single call). Weight counts how often a task has appeared across jobs."),
    el("div", { class: "card" }, field("Search", filter),
      el("div", { class: "actions" }, btn, el("a", { class: "btn", href: "/export_graph", target: "_blank" }, "Export JSON to disk"), el("span", { class: "timer" }))),
    out);
  load();
});

// ---------------------------------------------------------------- shell

const nav = document.getElementById("nav");
const main = document.getElementById("main");
const panels = {};

function show(id) {
  const t = TOOLS.find((x) => x.id === id) || TOOLS[0];
  if (!panels[t.id]) {
    panels[t.id] = el("section", {});
    t.render(panels[t.id]);
  }
  main.replaceChildren(panels[t.id]);
  for (const b of nav.children) b.classList.toggle("active", b.dataset.id === t.id);
  if (location.hash !== `#${t.id}`) history.replaceState(null, "", `#${t.id}`);
}

for (const t of TOOLS) {
  nav.append(el("button", { type: "button", "data-id": t.id, onclick: () => show(t.id) }, t.name));
}
window.addEventListener("hashchange", () => show(location.hash.slice(1)));
show(location.hash.slice(1));

(async function checkStatus() {
  const box = document.getElementById("status");
  const text = box.querySelector(".status-text");
  try {
    const { llm } = await api("/health");
    const models = llm.models.fast === llm.models.reasoning ? llm.models.fast : `${llm.models.fast} + ${llm.models.reasoning}`;
    box.className = `status ${llm.reachable ? "ok" : "down"}`;
    const problem = llm.error && llm.error.startsWith("model") ? "model not available" : "unreachable";
    text.textContent = `${llm.provider} · ${models}${llm.reachable ? "" : ` · ${problem}`}`;
    box.title = llm.error || llm.base_url;
  } catch {
    box.className = "status down";
    text.textContent = "API unreachable";
  }
})();
