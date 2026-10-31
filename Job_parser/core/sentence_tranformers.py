import json, os, re, spacy, torch
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline

# === Config ===
NODES_FILE = 'tech_synonyms.json'
TAGS_FILE = 'semantic_tags.json'
SYN_THRESHOLD = 0.45
REL_THRESHOLD = 0.40
MAIN_MODEL = 'sentence-transformers/all-mpnet-base-v2'
DOMAIN_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
ENTAIL_MODEL = 'cross-encoder/nli-deberta-v3-base'

model = SentenceTransformer(MAIN_MODEL)
code_model = SentenceTransformer(DOMAIN_MODEL)
entailer = pipeline("text-classification", model=ENTAIL_MODEL, device=0)
nlp = spacy.load("en_core_web_sm")

with open(TAGS_FILE, 'r', encoding='utf-8') as f:
    TAG_MAP = json.load(f)

TOOL_TO_DOMAIN = {tool.lower(): domain for domain, tools in TAG_MAP.items() for tool in tools}

DOMAIN_ALIAS_MAP = {
    "dl_concepts": "deep_learning", "cloud_providers": "cloud_services",
    "networking": "cloud_services", "security": "cloud_services",
    "observability": "cloud_services", "development_tools": "devops",
    "nlp": "machine_learning", "programming_languages": "machine_learning"
}

DOMAIN_PRIORITY = [
    "mlops", "machine_learning", "deep_learning",
    "data_engineering", "big_data", "cloud_services",
    "devops", "containers", "infrastructure_as_code",
    "databases", "data_architecture"
]

def normalize_domain(domain):
    return DOMAIN_ALIAS_MAP.get(domain.lower(), domain.lower())

def are_domains_compatible(d1, d2):
    d1, d2 = normalize_domain(d1), normalize_domain(d2)
    compatible = {
        ("machine_learning", "deep_learning"), ("data_engineering", "big_data"),
        ("mlops", "machine_learning"), ("data_engineering", "databases"),
        ("cloud_services", "infrastructure_as_code"), ("cloud_services", "containers"),
        ("mlops", "cloud_services"), ("devops", "cloud_services")
    }
    return d1 == d2 or (d1, d2) in compatible or (d2, d1) in compatible

def extract_technical_terms(phrase):
    return {chunk.text.lower() for chunk in nlp(phrase).noun_chunks if len(chunk.text) > 3}

def get_primary_domain(phrase):
    phrase_lower = phrase.lower()
    domain_scores = {}
    for tool, domain in TOOL_TO_DOMAIN.items():
        if re.search(r'\b' + re.escape(tool) + r'\b', phrase_lower):
            domain_scores[domain] = domain_scores.get(domain, 0) + 3
        elif tool in phrase_lower:
            domain_scores[domain] = domain_scores.get(domain, 0) + 1
    for domain in DOMAIN_PRIORITY:
        if domain in domain_scores:
            return domain
    return normalize_domain(max(domain_scores.items(), key=lambda x: x[1])[0]) if domain_scores else "other"

def check_entailment(premise, hypothesis):
    result = entailer(f"{premise} [SEP] {hypothesis}", truncation=True)
    return max((res['score'] for res in result if res['label'] == 'entailment'), default=0.0)

def noun_chunk_similarity(t1, t2):
    c1, c2 = list(extract_technical_terms(t1)), list(extract_technical_terms(t2))
    if not c1 or not c2: return 0
    e1, e2 = model.encode(" ".join(c1), convert_to_tensor=True), model.encode(" ".join(c2), convert_to_tensor=True)
    return util.cos_sim(e1, e2).item()

def tag_overlap_score(t1, t2):
    tags1 = {tag for tag, tools in TAG_MAP.items() for tool in tools if tool.lower() in t1.lower()}
    tags2 = {tag for tag, tools in TAG_MAP.items() for tool in tools if tool.lower() in t2.lower()}
    return len(tags1 & tags2) / max(len(tags1 | tags2), 1) if tags1 and tags2 else 0

def get_all_aliases(node):
    return [node['name']] + node.get('synonyms', []) + node.get('relationships', [])

# === Load data ===
if os.path.exists(NODES_FILE):
    with open(NODES_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
else:
    data = {'version': '1.0', 'nodes': []}

major, minor = map(int, data['version'].split('.'))
data['version'] = f"{major}.{minor + 1}"

# === New responsibilities ===
new_responsibilities = [
    "Train a CNN for image classification",
"Use PyTorch Lightning to train a model",
"Build an image classifier using TensorFlow",
"Optimize model accuracy using hyperparameter tuning",
"Use cross-validation for model selection",
"Train a model for entity recognition",
"Extract named entities from text using spaCy",
"Create a text summarization model with transformers",
"Build a sentiment analysis classifier",
"Implement zero-shot text classification",
"Serve models with TensorFlow Serving",
"Deploy machine learning models using Docker",
"Monitor deployed models for performance drift",
"Track ML experiments with Weights & Biases",
"Set up CI/CD for machine learning pipelines",
"Process streaming data with Apache Kafka",
"Run ETL pipelines with Apache Airflow",
"Build a data lakehouse with Delta Lake",
"Query big data with BigQuery",
"Monitor cloud infrastructure using Grafana"





]

# === Precompute embeddings ===
for node in data['nodes']:
    node['embedding'] = model.encode(node['name'], convert_to_tensor=True).tolist()

alias_embeddings = {
    alias: model.encode(alias, convert_to_tensor=True)
    for node in data['nodes'] for alias in get_all_aliases(node)
}

# === Main Matching Logic ===
for phrase in new_responsibilities:
    emb = model.encode(phrase, convert_to_tensor=True)
    tech1 = extract_technical_terms(phrase)
    domain1 = get_primary_domain(phrase)
    best_syn, best_rel = None, None
    best_syn_score = best_rel_score = 0

    for node in data['nodes']:
        node_domain = node.get('domain', get_primary_domain(node['name']))
        node['domain'] = node_domain
        if not are_domains_compatible(domain1, node_domain):
            continue

        for alias in get_all_aliases(node):
            if alias not in alias_embeddings:
                alias_embeddings[alias] = model.encode(alias, convert_to_tensor=True)

            alias_emb = alias_embeddings[alias]
            if alias_emb is None:
                continue  # Skip if encoding failed
            sim = util.cos_sim(emb, alias_emb).item()
            if sim < 0.5: continue

            tech2 = extract_technical_terms(alias)
            tech_overlap = len(tech1 & tech2) / max(len(tech1 | tech2), 1)
            entail = check_entailment(phrase, alias)
            noun_sim = noun_chunk_similarity(phrase, alias)
            tag_sim = tag_overlap_score(phrase, alias)

            syn_score = (sim * 0.5 + tech_overlap * 0.2 + tag_sim * 0.3)
            rel_score = (tag_sim * 0.5 + entail * 0.3 + sim * 0.2)

            if syn_score >= SYN_THRESHOLD and syn_score > best_syn_score:
                best_syn, best_syn_score = node, syn_score
            elif rel_score >= REL_THRESHOLD and rel_score > best_rel_score:
                best_rel, best_rel_score = node, rel_score

    if best_syn:
        if phrase not in best_syn['synonyms']:
            best_syn.setdefault('synonyms', []).append(phrase)
            best_syn['weight'] = best_syn.get('weight', 1) + 1
    elif best_rel:
        if phrase not in best_rel.get('relationships', []):
            best_rel.setdefault('relationships', []).append(phrase)
            best_rel['weight'] = best_rel.get('weight', 1) + 0.5
    else:
        data['nodes'].append({
            'name': phrase,
            'domain': domain1,
            'weight': 1.0,
            'synonyms': [],
            'relationships': [],
            'embedding': emb.tolist()
        })

# === Save data ===
for node in data['nodes']:
    node.pop('embedding', None)

with open(NODES_FILE, 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print(f"✅ Enrichment complete. Added {len(new_responsibilities)} phrases. Version: {data['version']}")
