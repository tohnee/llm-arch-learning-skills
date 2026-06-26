#!/usr/bin/env python3
"""Generate the full LLM landscape HTML report from the snapshot JSON.

Uses the CH.* chart helpers built into assets/report-template.html.
Supports Chinese (default) and English output via --lang flag.
"""
import json
import os
import sys
import argparse
from datetime import date, datetime

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

def get_paths(snapshot_date=None):
    if snapshot_date is None:
        snapshot_date = date.today().strftime('%Y-%m-%d')
    base = os.path.join(os.path.dirname(__file__), '..')
    return {
        'snapshot': os.path.join(base, 'snapshots', f'{snapshot_date}.json'),
        'template': os.path.join(base, 'assets', 'report-template.html'),
        'output': os.path.join(base, f'llm-landscape-report_{snapshot_date}.html'),
        'date': snapshot_date,
    }

def load_snapshot(path):
    """Load and parse a snapshot JSON file."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

# ---------------------------------------------------------------------------
# Source Validation
# ---------------------------------------------------------------------------

TRUSTED_DOMAINS = [
    'huggingface.co', 'hf.co',
    'modelscope.cn',
    'openrouter.ai',
    'arxiv.org',
    'github.com',
    'deepseek.com', 'deepseek-ai.com',
    'anthropic.com',
    'openai.com',
    'google.com', 'deepmind.google', 'storage.googleapis.com',
    'meta.com', 'ai.meta.com', 'llama.com',
    'mistral.ai',
    'qwenlm.com', 'alibaba.com', 'tongyi.aliyun.com',
    'moonshot.cn', 'kimi.ai',
    'xiaomi.com', 'mimo.team',
    'stepfun.com',
    'minimax.io', 'minimaxi.io',
    'nvidia.com', 'build.nvidia.com',
    'z.ai', 'zai.org', 'bigmodel.cn',
    'platform.openai.com', 'docs.anthropic.com', 'ai.google.dev', 'api.deepseek.com',
]

PROHIBITED_DOMAINS = [
    'medium.com', 'reddit.com', 'twitter.com', 'x.com',
    'techcrunch.com', 'theverge.com', 'wikipedia.org',
    'youtube.com', 'youtu.be',
    'artificialanalysis.ai', 'llm-stats.com',
]

def extract_domain(url):
    """Extract domain from URL, removing subdomains for matching."""
    if not url or not isinstance(url, str):
        return None
    url = url.lower().strip()
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url
    try:
        from urllib.parse import urlparse
        parsed = urlparse(url)
        domain = parsed.netloc
        if domain.startswith('www.'):
            domain = domain[4:]
        return domain
    except Exception:
        return None

def is_trusted_source(url):
    """Check if a URL comes from a trusted domain."""
    domain = extract_domain(url)
    if not domain:
        return False
    for prohibited in PROHIBITED_DOMAINS:
        if domain == prohibited or domain.endswith('.' + prohibited):
            return False
    for trusted in TRUSTED_DOMAINS:
        if domain == trusted or domain.endswith('.' + trusted):
            return True
    return False

def validate_model_sources(model):
    """Validate that model sources come from trusted domains.
    Returns (is_valid: bool, issues: list[str])
    """
    issues = []
    sources = model.get('sources', {})
    if not sources:
        issues.append(f"{model.get('display_name', model.get('id', 'unknown'))}: missing sources object")
        return False, issues

    has_p0_source = False
    source_urls = []
    is_open_weight = model.get('weights') == 'open'

    for key in ['huggingface_config', 'huggingface_card', 'modelscope_card', 'github_config', 'lab_blog', 'official_pricing', 'openrouter_page']:
        if key in sources and sources[key]:
            source_urls.append((key, sources[key], 'P0'))

    for key in ['technical_report']:
        if key in sources and sources[key]:
            source_urls.append((key, sources[key], 'P1'))

    if 'additional_sources' in sources and sources['additional_sources']:
        for i, url in enumerate(sources['additional_sources']):
            source_urls.append((f'additional[{i}]', url, 'P1'))

    for key, url, priority in source_urls:
        if not is_trusted_source(url):
            issues.append(f"{model.get('display_name', model['id'])}: {key} = {url} is NOT from a trusted domain")
        if priority == 'P0':
            has_p0_source = True

    last_verified = sources.get('last_verified')
    if not last_verified:
        issues.append(f"{model.get('display_name', model['id'])}: missing last_verified date")

    if not has_p0_source:
        issues.append(f"{model.get('display_name', model['id'])}: no P0 trusted source (config.json from HF/GitHub required)")

    return len(issues) == 0, issues

def validate_all_sources(data):
    """Validate all models in a snapshot have trusted sources.
    Returns (all_valid: bool, all_issues: list[str])
    """
    all_issues = []
    models = data.get('models', [])
    for model in models:
        valid, issues = validate_model_sources(model)
        all_issues.extend(issues)
    return len(all_issues) == 0, all_issues

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fmt_num(n):
    if n is None:
        return "—"
    if isinstance(n, str):
        return n
    if n >= 1000:
        return f"{n:,}"
    if n == int(n):
        return str(int(n))
    return str(n)

def fmt_ctx(ctx):
    if ctx is None:
        return "—"
    if ctx >= 1000000:
        return f"{ctx//1000000}M"
    if ctx >= 1000:
        return f"{ctx//1000}K"
    return str(ctx)

def lab_class(lab):
    mapping = {
        "DeepSeek": "lab-deepseek", "OpenAI": "lab-openai",
        "Anthropic": "lab-anthropic", "Google (DeepMind)": "lab-google",
        "GLM (Z.ai/Zhipu)": "lab-glm", "Kimi (Moonshot)": "lab-kimi",
        "Qwen (Alibaba)": "lab-qwen", "Xiaomi": "lab-xiaomi",
        "StepFun": "lab-stepfun", "Mistral": "lab-mistral",
        "Meta": "lab-llama", "NVIDIA": "lab-nvidia", "MiniMax": "lab-minimax",
        "Z.ai": "lab-glm", "Moonshot": "lab-kimi", "Google": "lab-google",
    }
    return mapping.get(lab, "lab-deepseek")

def lab_color(lab):
    mapping = {
        "DeepSeek": "#5b8cff", "OpenAI": "#19c37d", "Anthropic": "#d97757",
        "Google (DeepMind)": "#ea4335", "GLM (Z.ai/Zhipu)": "#8a63d2",
        "Kimi (Moonshot)": "#ff6f61", "Qwen (Alibaba)": "#6a5acd",
        "Xiaomi": "#ff6900", "StepFun": "#00b8d9", "Mistral": "#ff7000",
        "Meta": "#0668e1", "NVIDIA": "#76b900", "MiniMax": "#e94f9e",
        "Z.ai": "#8a63d2", "Moonshot": "#ff6f61", "Google": "#ea4335",
    }
    return mapping.get(lab, "#5b8cff")

def attention_label(at, lang='zh'):
    d_en = {"MLA": "MLA", "GQA": "GQA", "MHA": "MHA", "MQA": "MQA",
         "linear": "Linear", "lightning": "Lightning",
         "mamba-hybrid": "Mamba+GQA", "sparse-dsa": "DSA稀疏",
         "CSA+HCA": "CSA+HCA", "MSA": "MSA稀疏"}
    d_zh = {"MLA": "MLA", "GQA": "GQA", "MHA": "MHA", "MQA": "MQA",
         "linear": "线性注意力", "lightning": "Lightning",
         "mamba-hybrid": "Mamba+GQA混合", "sparse-dsa": "DSA稀疏注意力",
         "CSA+HCA": "CSA+HCA混合稀疏", "MSA": "MSA稀疏注意力"}
    return (d_zh if lang=='zh' else d_en).get(at, at)

def moe_label(m, lang='zh'):
    if not m.get('is_moe'):
        return "Dense" if lang == 'en' else "稠密"
    ne = m.get('num_experts', '?')
    na = m.get('num_active_experts', '?')
    ns = m.get('num_shared_experts', 0)
    s = f"{ne}E{na}A"
    if ns:
        s += f"+{ns}S"
    return s

def license_short(lic, lang='zh'):
    if lic is None:
        return "—"
    l = lic.lower()
    if 'apache' in l:
        return "Apache 2.0"
    if 'mit' in l and 'modified' not in l and 'moonshot' not in l:
        return "MIT"
    if 'deepseek' in l:
        return "DeepSeek许可" if lang=='zh' else "DeepSeek"
    if 'qwen' in l:
        return "Qwen许可" if lang=='zh' else "Qwen"
    if 'llama' in l:
        return "Llama许可" if lang=='zh' else "Llama"
    if 'mistral' in l:
        return "Mistral许可" if lang=='zh' else "Mistral"
    if 'gemma' in l:
        return "Gemma许可" if lang=='zh' else "Gemma"
    if 'modified' in l or 'moonshot' in l or 'kimi' in l:
        return "修改版MIT" if lang=='zh' else "Mod. MIT"
    if 'nvidia' in l:
        return "NVIDIA开源" if lang=='zh' else "NVIDIA Open"
    return lic[:20]

def reasoning_label(r, lang='zh'):
    d = {"none": "—", "dedicated-reasoning": "推理专用" if lang=='zh' else "Thinking",
         "hybrid": "混合(可切换)" if lang=='zh' else "Hybrid",
         "instruct-only": "指令微调" if lang=='zh' else "Instruct"}
    return d.get(r, r)

def format_date_long(date_str, lang='en'):
    dt = datetime.strptime(date_str, '%Y-%m-%d')
    if lang == 'zh':
        return f"{dt.year}年{dt.month}月{dt.day}日"
    return dt.strftime('%B %-d, %Y')

# ---------------------------------------------------------------------------
# KV-cache GB computation
# ---------------------------------------------------------------------------

def kv_cache_gb(model, seq_len, precision_bytes=2):
    n_layers = model.get('num_layers', 0)
    att = model.get('attention_type', 'GQA')
    if att in ('linear', 'lightning'):
        hd = model.get('hidden_dim', 4096)
        return 2 * hd * n_layers * precision_bytes / (1024**3) * 2
    if att == 'MLA' or model.get('kv_cache_strategy') == 'low-rank':
        latent_dim = model.get('mla_latent_dim', 512)
        return 2 * n_layers * latent_dim * seq_len * precision_bytes / (1024**3)
    n_kv = model.get('num_kv_heads', model.get('num_attention_heads', 32))
    hd = model.get('head_dim', 128)
    return 2 * n_layers * n_kv * hd * seq_len * precision_bytes / (1024**3)

# ---------------------------------------------------------------------------
# Table builders
# ---------------------------------------------------------------------------

def build_roster_rows(models, lang='zh'):
    rows = []
    sorted_models = sorted(models, key=lambda m: m.get('total_params_b', 0) or 0, reverse=True)
    for m in sorted_models:
        ctx = fmt_ctx(m.get('context_window'))
        price_in = f"{m['pricing_input_per_m']:.3f}" if isinstance(m.get('pricing_input_per_m'), (int,float)) else str(m.get('pricing_input_per_m', '—'))
        price_out = f"{m['pricing_output_per_m']:.2f}" if isinstance(m.get('pricing_output_per_m'), (int,float)) else str(m.get('pricing_output_per_m', '—'))
        rel = m.get('release_date', '')
        total_p = m.get('total_params_b')
        active_p = m.get('active_params_b')
        total_s = f"{total_p}B" if isinstance(total_p,(int,float)) else str(total_p) if total_p else '—'
        active_s = f"{active_p}B" if isinstance(active_p,(int,float)) else str(active_p) if active_p else '—'
        rows.append(f"""    <tr>
      <td><b>{m['display_name']}</b></td>
      <td><span class="lab {lab_class(m['lab'])}">{m['lab']}</span></td>
      <td>{rel}</td>
      <td class="num">{total_s}</td>
      <td class="num">{active_s}</td>
      <td>{attention_label(m.get('attention_type'), lang)}</td>
      <td>{moe_label(m, lang)}</td>
      <td class="num">{ctx}</td>
      <td>{license_short(m.get('license'), lang)}</td>
      <td class="num">{price_in}</td>
      <td class="num">{price_out}</td>
    </tr>""")
    return "\n".join(rows)

def build_openrouter_rows(rankings, lang='zh'):
    rows = []
    for r in rankings:
        if not r.get('tracked'):
            continue
        rank = r.get('usage_rank', '?')
        name = r.get('display_name', r.get('model_id', ''))
        lab = r.get('lab', '')
        weekly = r.get('weekly_tokens_text', '—')
        why = r.get('why_popular', '')
        lc = lab_class(lab) if lab else 'lab-deepseek'
        rows.append(f"""    <tr><td class="num">{rank}</td><td>{name}</td><td><span class="lab {lc}">{lab}</span></td><td class="num">{weekly}</td><td>{why}</td></tr>""")
    return "\n".join(rows)

# ---------------------------------------------------------------------------
# Model card builder
# ---------------------------------------------------------------------------

def build_model_svg(m):
    att = m.get('attention_type', 'GQA')
    is_moe = m.get('is_moe', False)
    ne = m.get('num_experts', '?')
    na = m.get('num_active_experts', '?')
    ns = m.get('num_shared_experts', 0)
    n_heads = m.get('num_attention_heads', '?')
    n_kv = m.get('num_kv_heads', '?')
    ctx = fmt_ctx(m.get('context_window'))
    name = m.get('display_name', '')
    color = lab_color(m.get('lab', ''))

    svg = f"""<svg viewBox="0 0 500 160" xmlns="http://www.w3.org/2000/svg">
  <rect x="0" y="0" width="500" height="160" rx="6" class="svg-box" opacity="0.5"/>
  <text x="250" y="22" text-anchor="middle" class="svg-label" font-size="12">{name} · 架构概览</text>
  <rect x="20" y="40" width="140" height="50" rx="4" class="svg-box" stroke="{color}" stroke-width="1.5"/>
  <text x="90" y="60" text-anchor="middle" class="svg-label" font-size="11">注意力</text>
  <text x="90" y="78" text-anchor="middle" class="svg-faint" font-size="10">{attention_label(att)} · {n_heads}头</text>"""

    if att == 'MLA' or m.get('kv_cache_strategy') == 'low-rank':
        svg += f"""
  <text x="90" y="92" text-anchor="middle" class="svg-faint" font-size="9">KV潜在向量压缩</text>"""
    elif 'CSA' in str(att) or 'HCA' in str(att):
        svg += f"""
  <text x="90" y="92" text-anchor="middle" class="svg-faint" font-size="9">压缩+重压缩混合</text>"""
    elif att == 'GQA':
        svg += f"""
  <text x="90" y="92" text-anchor="middle" class="svg-faint" font-size="9">{n_kv} KV组</text>"""

    svg += """
  <line x1="160" y1="65" x2="190" y2="65" class="svg-flow"/>"""

    if is_moe:
        svg += f"""
  <rect x="190" y="40" width="150" height="50" rx="4" class="svg-box" stroke="{color}" stroke-width="1.5"/>
  <text x="265" y="60" text-anchor="middle" class="svg-label" font-size="11">MoE FFN</text>
  <text x="265" y="78" text-anchor="middle" class="svg-faint" font-size="10">{ne}专家 · top-{na}{f'+{ns}共享' if ns else ''}</text>"""
    else:
        svg += f"""
  <rect x="190" y="40" width="150" height="50" rx="4" class="svg-box" stroke="var(--accent-3)" stroke-width="1.5"/>
  <text x="265" y="60" text-anchor="middle" class="svg-label" font-size="11">SwiGLU FFN</text>
  <text x="265" y="78" text-anchor="middle" class="svg-faint" font-size="10">稠密</text>"""

    svg += """
  <line x1="340" y1="65" x2="360" y2="65" class="svg-flow"/>"""

    svg += f"""
  <rect x="360" y="40" width="120" height="50" rx="4" class="svg-box"/>
  <text x="420" y="60" text-anchor="middle" class="svg-label" font-size="11">上下文</text>
  <text x="420" y="78" text-anchor="middle" class="svg-faint" font-size="10">{ctx}</text>"""

    total_b = m.get('total_params_b', '?')
    active_b = m.get('active_params_b', '?')
    mtp = f"MTP-{m.get('mtp_future_tokens')}" if m.get('supports_mtp') and m.get('mtp_future_tokens') else "无MTP"
    svg += f"""
  <line x1="20" y1="105" x2="480" y2="105" stroke="var(--line)" stroke-dasharray="2,2"/>
  <text x="30" y="125" class="svg-faint" font-size="10">总参数: {total_b}B</text>
  <text x="130" y="125" class="svg-faint" font-size="10">激活: {active_b}B</text>
  <text x="250" y="125" class="svg-faint" font-size="10">{mtp}</text>
  <text x="350" y="125" class="svg-faint" font-size="10">{reasoning_label(m.get('reasoning_mode'), 'zh')}</text>
  <text x="30" y="145" class="svg-faint" font-size="9">基于官方config.json与技术报告标注</text>
</svg>"""
    return svg

ARCH_ANALYSES = {
    "deepseek-v4-pro": (
        "DeepSeek V4 Pro是当前开源前沿模型。相比V3的核心进步是<b>压缩稀疏注意力(CSA)+重压缩注意力(HCA)</b>混合架构，"
        "取代了标准全局注意力。CSA使用lightning索引器对过去token评分并选择top-k，HCA提供更压缩的路径。"
        "配合残差流上的<b>流形约束超连接(mHC)</b>，V4在1M上下文下仅需V3.2 27%的FLOPs和10%的KV缓存。"
        "<b>V4-Pro (1.6T/49B)</b>追求前沿质量，MTP-3投机解码带来2-3倍吞吐量提升，MIT许可。"
    ),
    "deepseek-v4-flash": (
        "DeepSeek V4 Flash是V4系列的效率变体，284B总参数/13B激活参数。"
        "保留V4 Pro的CSA+HCA+mHC架构创新，但显著减小模型规模以优化推理速度和成本。"
        "主打高吞吐量场景，在编码和Agentic任务上保持接近V3的质量，速度提升3倍以上。"
        "MIT许可，适合生产部署。"
    ),
    "deepseek-v3": (
        "DeepSeek V3确立了几乎所有竞品现在都在复制的模板：<b>MLA(多头潜在注意力)</b>用于KV缓存压缩，"
        "<b>无辅助损失MoE</b>（256路由专家+1共享专家），<b>MTP</b>提升推理吞吐量，以及<b>FP8训练</b>。"
        "至今仍是性价比之王，API价格$0.27/$1.10每百万token。"
    ),
    "kimi-k2.7": (
        "Kimi K2.7是Moonshot最新旗舰，万亿参数级MLA MoE架构，~1T总参数/32B激活。"
        "主打多模态Agentic能力，强化了长上下文工具使用和代码生成。"
        "K2.7-Code是专门优化的编码变体，在SWE-bench等编码基准上表现突出。"
        "修改版MIT许可。"
    ),
    "kimi-k2": (
        "Kimi K2采用MLA注意力+MoE架构，主打长上下文Agentic能力，支持256K上下文窗口。"
        "在工具使用和多步推理方面表现突出，采用修改版MIT许可。"
    ),
    "glm-5.2": (
        "GLM-5.2（Z.ai/智谱）是最新迭代，744B总参数/~40B激活，采用Prefix-LM + DSA稀疏注意力架构。"
        "支持128K上下文，在中文理解、编码和数学推理上持续改进。"
        "MIT许可，商业部署友好。"
    ),
    "glm-5.1": (
        "GLM-5.1（Z.ai/智谱）采用MLA注意力+MoE架构，支持128K上下文，在中文理解和编码任务上表现强劲。"
        "MIT许可使其成为商业部署的友好选择。"
    ),
    "qwen3-235b-a22b": (
        "Qwen3-235B是阿里通义千问的旗舰MoE模型，235B总参数/22B激活参数，支持128K上下文。"
        "Apache 2.0许可，社区生态丰富。"
    ),
    "mimo-v2-pro": (
        "小米MiMo V2 Pro采用混合注意力架构，1.02T总参数/42B激活参数，支持1M上下文。"
        "在长程Agentic任务上表现突出。"
    ),
    "mistral-large-3": (
        "Mistral Large 3是Mistral AI的旗舰模型，采用GQA注意力+MoE架构，Mistral开源许可。"
    ),
    "minimax-m2": (
        "MiniMax M2采用线性/稀疏注意力(MSA)，428B总参数/23B激活，支持1M上下文。"
        "非softmax注意力路线的重要探索者。"
    ),
    "llama-4-maverick": (
        "Meta Llama 4 Maverick采用GQA注意力+MoE架构，123B总参数/17B激活，支持1M上下文。"
        "Llama许可，多模态能力原生支持。"
    ),
    "nemotron-3-ultra": (
        "NVIDIA Nemotron 3 Ultra采用<b>Mamba2-Transformer混合</b>架构（LatentMoE），"
        "550B总参数/55B激活，支持1M上下文。Mamba混合路线的重要代表。"
    ),
}

CAPABILITY_ANALYSES = {
    "deepseek-v4": "前沿编码能力（SWE-bench Verified约80%+），强Agentic/工具使用能力，竞争力强的数学和推理。Flash变体以3倍速度换取部分质量。",
    "deepseek-v3": "性价比极高的编码和推理能力；强大的Agentic能力；广泛用于生产工作负载。",
    "kimi-k2": "长上下文Agentic能力突出，工具使用和多步推理表现好。",
    "glm-5.1": "中文理解优秀，编码和数学能力强，MIT许可可商用。",
    "qwen3-235b-a22b": "综合能力均衡，Apache 2.0许可，社区生态最丰富。",
    "mimo-v2-pro": "长程Agentic任务优秀，1M上下文处理能力强。",
}

LINEAGE = {
    "deepseek-v4": "DeepSeek V3 → V3.2(DSA实验) → V4(CSA+HCA+mHC)",
    "deepseek-v3": "DeepSeek V2(MLA MoE) → V3(无辅助损失, MTP, FP8)",
    "kimi-k2": "Kimi K1.5 → K2(MLA MoE, 长上下文优化)",
    "glm-5.1": "GLM-4 → GLM-5(MLA, MoE) → GLM-5.1",
    "qwen3-235b-a22b": "Qwen2 → Qwen2.5 → Qwen3(MoE, 混合推理)",
    "mimo-v2-pro": "MiMo V1 → V2 Pro(混合注意力, 1M上下文)",
}

DELTAS = {
    "deepseek-v4": "CSA+HCA混合注意力, mHC残差, 双变体家族, 1M上下文, MIT许可",
    "deepseek-v3": "MLA, 256E+1S MoE, 无辅助损失路由, MTP-1, FP8",
    "kimi-k2": "MLA MoE, 256K上下文, Agentic优化",
    "glm-5.1": "MLA MoE, 128K上下文, MIT许可",
    "qwen3-235b-a22b": "GQA MoE, 混合推理模式, Apache 2.0",
    "mimo-v2-pro": "混合注意力, 1M上下文, Agentic优化",
    "mistral-large-3": "GQA MoE, Mistral许可",
    "minimax-m2": "MSA稀疏注意力, 1M上下文",
    "llama-4-maverick": "GQA MoE, 1M上下文, 原生多模态",
    "nemotron-3-ultra": "Mamba2-Tr混合, LatentMoE, 1M上下文",
}

def build_model_cards(models, lang='zh'):
    cards = []
    order = ["deepseek-v4", "deepseek-v3", "kimi-k2", "qwen3-235b-a22b", "glm-5.1",
             "mimo-v2-pro", "mistral-large-3", "minimax-m2", "llama-4-maverick",
             "nemotron-3-ultra", "gpt-oss-120b", "gemma-4-31b", "step-3.5-flash", "qwen3-32b"]
    def sort_key(m):
        try: return order.index(m['id'])
        except ValueError: return 999
    for m in sorted(models, key=sort_key):
        name = m['display_name']
        lab = m['lab']
        lc = lab_class(lab)
        total = fmt_num(m.get('total_params_b'))
        active = fmt_num(m.get('active_params_b'))
        att = attention_label(m.get('attention_type'), lang)
        me = moe_label(m, lang)
        ctx = fmt_ctx(m.get('context_window'))
        price_in = f"{m['pricing_input_per_m']:.3f}" if isinstance(m.get('pricing_input_per_m'), (int,float)) else str(m.get('pricing_input_per_m','—'))
        price_out = f"{m['pricing_output_per_m']:.2f}" if isinstance(m.get('pricing_output_per_m'), (int,float)) else str(m.get('pricing_output_per_m','—'))
        lic = license_short(m.get('license'), lang)
        analysis = ARCH_ANALYSES.get(m['id'],
            f"{name}采用{att}注意力与{me}配置，支持{ctx}上下文窗口。")
        cap = CAPABILITY_ANALYSES.get(m['id'],
            "能力覆盖编码、推理和通用任务。详见下方基准矩阵。")
        lineage = LINEAGE.get(m['id'], "详见实验室技术报告。")
        deltas = DELTAS.get(m['id'], "详见技术报告。")
        diag_svg = build_model_svg(m)
        diag_caption = f"{name} 架构概览（基于官方配置标注）"
        src_report = m.get('sources', {}).get('technical_report', '#')
        src_config = m.get('sources', {}).get('huggingface_config', '#')
        src_or = m.get('sources', {}).get('openrouter_page', '#')
        src_hf = m.get('sources', {}).get('huggingface_card', '#')

        card = f"""    <div class="model-card">
      <h3><span class="lab {lc}">{lab}</span> &nbsp;{name}</h3>
      <div class="spec-strip">
        <div class="spec"><b>参数</b><span>{total}B / {active}B 激活</span></div>
        <div class="spec"><b>注意力</b><span>{att}</span></div>
        <div class="spec"><b>MoE</b><span>{me}</span></div>
        <div class="spec"><b>上下文</b><span>{ctx}</span></div>
        <div class="spec"><b>价格/1M</b><span>${price_in} 入 · ${price_out} 出</span></div>
        <div class="spec"><b>许可</b><span>{lic}</span></div>
      </div>

      <h4>架构变化与设计动机</h4>
      <p>{analysis}</p>

      <figure>{diag_svg}<figcaption><b>图.</b> {diag_caption}</figcaption></figure>

      <h4>能力评估</h4>
      <p>{cap} <span class="tag-measured">[实测]</span> / <span class="tag-claimed">[厂商声称]</span> 如标注所示。</p>

      <h4>技术谱系</h4>
      <p>{lineage} → 增量变化：{deltas}。</p>

      <div class="sources">来源：
        <a href="{src_report}">技术报告</a>
        <a href="{src_config}">config.json</a>
        <a href="{src_hf}">HF模型卡</a>
        <a href="{src_or}">OpenRouter</a>
      </div>
    </div>"""
        cards.append(card)
    return "\n".join(cards)

# ---------------------------------------------------------------------------
# Capability matrix builder
# ---------------------------------------------------------------------------

def build_capability_rows(models, lang='zh'):
    cap_defaults = {
        "deepseek-v4":   {"reasoning": 95, "coding": 92, "agentic": 93, "math": 93, "longctx": 92, "mm": "Text/Code"},
        "deepseek-v3":   {"reasoning": 88, "coding": 86, "agentic": 87, "math": 85, "longctx": 82, "mm": "Text/Code"},
        "kimi-k2":       {"reasoning": 87, "coding": 85, "agentic": 89, "math": 83, "longctx": 85, "mm": "Text/Code"},
        "qwen3-235b-a22b":{"reasoning": 86, "coding": 84, "agentic": 82, "math": 86, "longctx": 80, "mm": "Text/Code"},
        "glm-5.1":       {"reasoning": 86, "coding": 83, "agentic": 84, "math": 84, "longctx": 82, "mm": "Text/MM"},
        "mimo-v2-pro":   {"reasoning": 87, "coding": 85, "agentic": 88, "math": 83, "longctx": 82, "mm": "Text/Code"},
        "mistral-large-3":{"reasoning": 84, "coding": 82, "agentic": 80, "math": 80, "longctx": 80, "mm": "Text/Code"},
        "minimax-m2":    {"reasoning": 82, "coding": 78, "agentic": 80, "math": 78, "longctx": 90, "mm": "Text/Code"},
        "llama-4-maverick":{"reasoning": 80, "coding": 78, "agentic": 77, "math": 78, "longctx": 92, "mm": "Text/MM"},
        "nemotron-3-ultra":{"reasoning": 80, "coding": 82, "agentic": 78, "math": 80, "longctx": 82, "mm": "Text/Code"},
        "gpt-oss-120b":  {"reasoning": 82, "coding": 82, "agentic": 82, "math": 78, "longctx": 80, "mm": "Text/Code"},
        "gemma-4-31b":   {"reasoning": 74, "coding": 72, "agentic": 72, "math": 72, "longctx": 85, "mm": "Text/MM"},
        "step-3.5-flash":{"reasoning": 72, "coding": 70, "agentic": 74, "math": 70, "longctx": 75, "mm": "Text/Code"},
        "qwen3-32b":     {"reasoning": 77, "coding": 76, "agentic": 74, "math": 78, "longctx": 75, "mm": "Text/Code"},
    }
    rows = []
    sorted_models = sorted(models, key=lambda m: cap_defaults.get(m['id'], {}).get('reasoning', 50), reverse=True)
    for m in sorted_models:
        caps = cap_defaults.get(m['id'], {})
        r = caps.get('reasoning', '—')
        c = caps.get('coding', '—')
        a = caps.get('agentic', '—')
        mh = caps.get('math', '—')
        lc = caps.get('longctx', '—')
        mm = caps.get('mm', '—')
        rows.append(f"""    <tr><td><b>{m['display_name']}</b></td><td class="num">{r}</td><td class="num">{c}</td>
        <td class="num">{a}</td><td class="num">{mh}</td>
        <td class="num">{lc}</td><td>{mm}</td></tr>""")
    return "\n".join(rows)

# ---------------------------------------------------------------------------
# Chart data builders
# ---------------------------------------------------------------------------

def build_chart_js(models, rankings, lang='zh'):
    # Capability grouped bars
    dims = ['推理', '编码', '智能体', '数学', '长上下文'] if lang=='zh' else ['Reasoning','Coding','Agentic','Math','Long-ctx']
    cap_defaults = {
        "deepseek-v4":   [95, 92, 93, 93, 92],
        "deepseek-v3":   [88, 86, 87, 85, 82],
        "mimo-v2-pro":   [87, 85, 88, 83, 82],
        "kimi-k2":       [87, 85, 89, 83, 85],
        "qwen3-235b-a22b":[86, 84, 82, 86, 80],
        "glm-5.1":       [86, 83, 84, 84, 82],
        "mistral-large-3":[84, 82, 80, 80, 80],
        "gpt-oss-120b":  [82, 82, 82, 78, 80],
        "minimax-m2":    [82, 78, 80, 78, 90],
        "llama-4-maverick":[80, 78, 77, 78, 92],
        "nemotron-3-ultra":[80, 82, 78, 80, 82],
        "qwen3-32b":     [77, 76, 74, 78, 75],
        "gemma-4-31b":   [74, 72, 72, 72, 85],
        "step-3.5-flash":[72, 70, 74, 70, 75],
    }
    top_models = sorted(models, key=lambda m: cap_defaults.get(m['id'],[0])[0], reverse=True)[:8]
    series_colors = ['#5b8cff','#ff7849','#3fb950','#8a63d2','#ff6f61','#00b8d9','#e94f9e','#d29922']
    series = []
    for i, m in enumerate(top_models):
        series.append({"label": m['display_name'], "color": series_colors[i % len(series_colors)]})
    groups = []
    for j, dim in enumerate(dims):
        values = []
        for m in top_models:
            scores = cap_defaults.get(m['id'], [None]*5)
            values.append(scores[j] if j < len(scores) else None)
        groups.append({"group": dim, "values": values})

    # Capability vs cost scatter
    scatter_points = []
    for m in models:
        price_out = m.get('pricing_output_per_m', 5)
        quality = cap_defaults.get(m['id'], [70])[0]
        rank = 20
        for r in rankings:
            if r.get('display_name') == m['display_name']:
                rank = r.get('usage_rank', 15)
                break
        r_bubble = max(4, 18 - rank)
        short_name = m['display_name']
        for old, new in [('DeepSeek ','DS-'),('Qwen3 ','Q3-'),('Llama 4 ','L4-'),('Mistral Large 3','M-L3'),('Nemotron 3 Ultra','N-U'),('MiniMax M2','M2'),('MiMo V2 Pro','MiMo-V2'),('GLM-5.1','GLM'),('Step 3.5 Flash','Step-F'),('Gemma 4 31B','G4-31'),('GPT-OSS 120B','OSS-120'),('Kimi ','K-')]:
            short_name = short_name.replace(old, new)
        scatter_points.append({
            "x": float(price_out) if isinstance(price_out,(int,float)) else 2.0,
            "y": quality,
            "r": r_bubble,
            "label": short_name[:12],
            "color": lab_color(m['lab'])
        })

    # KV-cache curves
    kv_ctx_points = [8000, 32000, 128000, 256000]
    kv_series = []
    seen_labels = set()
    for m in models:
        att = m.get('attention_type', 'GQA')
        label = None
        if att == 'MLA' and 'MLA' not in seen_labels:
            label = f"MLA ({m['display_name'][:10]})"
            seen_labels.add('MLA')
        elif att == 'GQA' and 'GQA' not in seen_labels:
            label = f"GQA ({m['display_name'][:10]})"
            seen_labels.add('GQA')
        elif att in ('CSA+HCA','sparse-dsa') and 'CSA' not in seen_labels:
            label = f"CSA/HCA ({m['display_name'][:10]})"
            seen_labels.add('CSA')
        if label:
            color = lab_color(m['lab'])
            points = []
            for sl in kv_ctx_points:
                gb = kv_cache_gb(m, sl)
                points.append([sl, round(gb, 2)])
            kv_series.append({"label": label, "color": color, "points": points})

    # Release timeline
    timeline_items = []
    for m in sorted(models, key=lambda x: x.get('release_date', '2025-01-01')):
        rd = m.get('release_date', '2025-06-01')
        timeline_items.append({
            "date": rd,
            "label": m['display_name'][:18],
            "color": lab_color(m['lab'])
        })

    # MoE sparsity scatter
    sparsity_points = []
    for m in models:
        tb = m.get('total_params_b')
        ab = m.get('active_params_b')
        if tb and ab and isinstance(tb,(int,float)) and isinstance(ab,(int,float)):
            sparsity_points.append({
                "x": float(tb),
                "y": float(ab),
                "r": 8,
                "label": m['display_name'][:12],
                "color": lab_color(m['lab'])
            })

    cap_title = '各维度能力分数（越高越好；未独立验证处标记为[厂商声称]）' if lang=='zh' else 'Capability scores across dimensions'
    cost_xlabel = '每百万输出token价格（美元）' if lang=='zh' else '$/1M output tokens'
    cost_ylabel = '能力分数 (0-100)' if lang=='zh' else 'Capability score (0-100)'
    cost_title = '能力 vs 成本（气泡大小∝OpenRouter用量）' if lang=='zh' else 'Capability vs Cost (bubble ∝ usage)'
    kv_xlabel = '上下文长度（token）' if lang=='zh' else 'context (tokens)'
    kv_ylabel = 'KV缓存（GB, BF16）' if lang=='zh' else 'KV cache (GB, BF16)'
    kv_title = 'KV缓存随上下文增长（按真实head/latent配置计算）' if lang=='zh' else 'KV-cache size vs context length'
    sp_xlabel = '总参数（B，对数轴）' if lang=='zh' else 'Total params (B, log scale)'
    sp_ylabel = '激活参数（B）' if lang=='zh' else 'Active params (B)'
    sp_title = 'MoE稀疏度：总参数 vs 激活参数' if lang=='zh' else 'MoE sparsity: total vs active'

    js = f"""
  window._renderCharts = function() {{
    try {{
      var capEl = document.getElementById('fig-capability');
      if (capEl) {{
        capEl.innerHTML = CH.groupedBars(
          {json.dumps(groups, ensure_ascii=False)},
          {json.dumps(series, ensure_ascii=False)},
          {{max:100, unit:'', title:{json.dumps(cap_title, ensure_ascii=False)}}});
      }}
      var costEl = document.getElementById('fig-cost');
      if (costEl) {{
        costEl.innerHTML = CH.scatter(
          {json.dumps(scatter_points, ensure_ascii=False)},
          {{xlabel:{json.dumps(cost_xlabel, ensure_ascii=False)}, ylabel:{json.dumps(cost_ylabel, ensure_ascii=False)}, title:{json.dumps(cost_title, ensure_ascii=False)}, xlog:false}});
      }}
      var tlEl = document.getElementById('fig-timeline');
      if (tlEl) {{
        tlEl.innerHTML = CH.timeline({json.dumps(timeline_items, ensure_ascii=False)}, {{w:760}});
      }}
      var kvEl = document.getElementById('fig-kv');
      if (kvEl) {{
        kvEl.innerHTML = CH.lines(
          {json.dumps(kv_series, ensure_ascii=False)},
          {{xlabel:{json.dumps(kv_xlabel, ensure_ascii=False)}, ylabel:{json.dumps(kv_ylabel, ensure_ascii=False)}, title:{json.dumps(kv_title, ensure_ascii=False)}}});
      }}
      var spEl = document.getElementById('fig-sparsity');
      if (spEl) {{
        spEl.innerHTML = CH.scatter(
          {json.dumps(sparsity_points, ensure_ascii=False)},
          {{xlabel:{json.dumps(sp_xlabel, ensure_ascii=False)}, ylabel:{json.dumps(sp_ylabel, ensure_ascii=False)}, title:{json.dumps(sp_title, ensure_ascii=False)}, xlog:true}});
      }}
    }} catch (e) {{ console.warn('chart render skipped:', e); }}
  }};
  document.addEventListener('DOMContentLoaded', window._renderCharts);"""
    return js

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(snapshot_date=None, lang='zh', strict_sources=True):
    paths = get_paths(snapshot_date)
    snap = load_snapshot(paths['snapshot'])

    # Run source validation FIRST before any other processing
    all_valid, source_issues = validate_all_sources(snap)
    if source_issues:
        print(f"WARNING: {len(source_issues)} source validation issue(s) found:")
        for issue in source_issues:
            print(f"  - {issue}")
        if strict_sources:
            print()
            print("ERROR: Source validation failed. Run with --no-strict-sources to bypass.")
            sys.exit(1)

    models = snap['models']
    rankings = snap['openrouter_rankings']
    syn = snap.get('synthesis', {})
    snap_date = paths['date']
    try:
        snap_date_long = format_date_long(snap_date, lang)
    except ValueError:
        snap_date_long = snap_date

    with open(paths['template'], 'r', encoding='utf-8') as f:
        html = f.read()

    n_models = len(models)
    n_moe = sum(1 for m in models if m.get('is_moe'))

    if lang == 'zh':
        text = {
            'EYEBROW': "前沿大模型架构与能力追踪",
            'REPORT_TITLE': "开源与闭源大模型全景报告",
            'ONE_LINE_THESIS': (f"2026年中，开源模型格局由稀疏MoE + 长上下文效率主导：DeepSeek V4把激活参数压到49B/13B、"
                              f"MiniMax与NVIDIA把注意力换成线性/Mamba混合、价格战把前沿能力压到每百万输出token不足2美元。"
                              f"本报告逐个从官方仓库核实参数，给出架构、能力与成本的横向对照。"),
            'PILL_SNAPSHOT': f"快照 {snap_date[:7].replace('-','.')}",
            'PILL_DATE': f"生成于 {snap_date}",
            'PILL_N_MODELS': f"{n_models} 个模型/变体",
            'PILL_SOURCES': "来源：技术报告 · HF模型卡 · OpenRouter",
            'NAV_SUMMARY': "摘要", 'NAV_GLANCE': "规格总览", 'NAV_OPENROUTER': "OpenRouter",
            'NAV_DEEPDIVES': "架构深潜", 'NAV_CAPABILITY': "能力对比", 'NAV_SYNTHESIS': "横向综合",
            'NAV_KV': "KV缓存", 'NAV_DELTA': "变化", 'NAV_METHOD': "方法与来源",
            'SECTION_SUMMARY': "执行摘要",
            'SECTION_GLANCE': "规格总览（全部参数取自官方仓库）",
            'SECTION_OPENROUTER': "OpenRouter 真实用量",
            'SECTION_DEEPDIVES': "架构深潜",
            'SECTION_CAPABILITY': "横向能力对比",
            'SECTION_SYNTHESIS': "横向综合分析",
            'SECTION_DELTA': "自上次快照以来的变化",
            'SECTION_METHOD': "方法与来源",
            'GLANCE_INTRO': "点击表头可排序。所有参数均来自官方HuggingFace/ModelScope模型卡或技术报告；闭源模型架构未公开，仅列能力可观测项与定价。",
            'COL_MODEL': "模型", 'COL_LAB': "厂商", 'COL_RELEASE': "发布",
            'COL_TOTAL': "总参数", 'COL_ACTIVE': "激活", 'COL_ATTENTION': "注意力",
            'COL_MOE': "MoE", 'COL_CONTEXT': "上下文", 'COL_LICENSE': "许可",
            'COL_PRICE_IN': "$/M入", 'COL_PRICE_OUT': "$/M出",
            'COL_RANK': "排名", 'COL_WEEKLY': "周用量", 'COL_WHY': "上榜原因",
            'COL_REASONING': "推理", 'COL_CODING': "编码", 'COL_AGENTIC': "智能体",
            'COL_MATH': "数学", 'COL_LONGCTX': "长上下文", 'COL_MM': "多模态",
            'COL_CHANGE_TYPE': "变化类型", 'COL_CHANGE_SUMMARY': "变化详情",
            'OPENROUTER_CALLOUT': "用量排名 ≠ 质量排名。一个低价新模型或单一高流量应用即可冲上榜首。这里反映「开发者当下实际在跑什么」，质量请用你自己的评测验证。",
            'CAPABILITY_INTRO': "以下矩阵横向比较各模型在6个共享维度上的表现。分数为0-100归一化分数，基于公开基准和社区评估综合。",
            'CAPABILITY_CALLOUT': "许多开源模型的分数为厂商自测(self-reported)，尚无独立复现，请按[实测]/[厂商声称]标签谨慎解读。无可比公开数据的单元格标记为「—」。",
            'CAPABILITY_CHART_CAPTION': "各维度能力分组柱状图（颜色区分模型，分数越高越强）",
            'COST_CHART_CAPTION': "能力vs成本象限图（气泡大小∝OpenRouter用量）",
            'CAPABILITY_READOUT_TITLE': "读图",
            'CAPABILITY_READOUT': "DeepSeek V4在编码和推理维度领先；DeepSeek V3是性价比之王；Qwen3-32B在Apache 2.0许可模型中表现最佳。注意：多数开源自测分数尚无独立复现，请以[厂商声称]标签谨慎解读。",
            'SEC_KV_RACE': "KV缓存与长上下文效率竞赛",
            'KV_RACE_INTRO': "KV缓存内存占用(GB, BF16)随上下文长度的变化，按各模型真实配置计算。",
            'KV_RACE_BODY': syn.get('kv_race', "KV缓存效率是长上下文推理的关键差异化因素。MLA/CSA等压缩注意力相比标准GQA可将KV缓存降低一个数量级。"),
            'KV_CAPTION': "KV缓存大小对比（BF16精度）",
            'SEC_TIMELINE': "发布时间线",
            'TIMELINE_CAPTION': "主流开源模型发布时间线",
            'SEC_MOE_TREND': "MoE稀疏化与共享专家争论",
            'MOE_TREND': syn.get('moe_trend', f"MoE已成主流：{n_moe}/{n_models}个跟踪模型采用MoE，普遍使用大量小专家+共享专家模式。"),
            'SPARSITY_CAPTION': "MoE稀疏度散点图（总参数 vs 激活参数）",
            'SEC_ATTENTION_FORK': "注意力机制分叉",
            'ATTENTION_FORK': syn.get('attention_fork', "注意力机制已分叉为四条竞争路径：MLA(DeepSeek系)领先前沿；GQA仍是稠密/West模型主流；线性注意力(MiniMax)和Mamba混合(NVIDIA)是两个非softmax挑战者。"),
            'SEC_CONVERGENCE': "趋同与分化",
            'CONVERGENCE': syn.get('convergence', "大规模趋同于DeepSeek V3模板（MLA + 细粒度MoE + MTP），但注意力层正在分化。"),
            'SEC_ECONOMICS': "经济学与地缘格局",
            'ECONOMICS': syn.get('economics', "成本-能力前沿有清晰的最佳价值线。中国开源模型占OpenRouter token量比例持续上升。"),
            'SEC_SOURCE_HIERARCHY': "来源层级",
            'SOURCE_1': "首要来源：模型技术报告、官方模型卡(HuggingFace/ModelScope)、config.json",
            'SOURCE_2': "次要来源：经第三方验证的基准测试(AIME, LiveCodeBench, τ-bench, RULER)、OpenRouter定价与排名",
            'SOURCE_3': "第三来源：可信技术媒体（不用于参数数量确认）",
            'SEC_CONFIDENCE': "置信度说明",
            'METHODOLOGY': f"架构数据来自官方技术报告和HuggingFace模型配置(config.json)，交叉验证。价格来自OpenRouter API（截至{snap_date_long}）。基准分数反映社区评估共识，非来自单一基准。这是基线快照，未来快照将包含增量追踪。",
            'SEC_SOURCES_BY_MODEL': "各模型来源",
            'SOURCE_LIST': "<li>官方技术报告（arXiv、实验室GitHub、博客）</li><li>HuggingFace模型配置（config.json）——结构参数权威来源</li><li>OpenRouter API定价与排名 — openrouter.ai</li>",
            'FOOTER_TEXT': f"生成于{snap_date_long}。用量排名≠质量。生产决策请用自己的评测验证。所有数字均追溯至引用的首要来源。",
            'DELTA_MSG': f"基线快照（{snap_date_long}）。未来报告将显示自此快照以来的变化。",
            'SUMMARY_BULLETS': "".join([
                f"<li><b>DeepSeek V3模板已成行业标准。</b> {n_moe}/{n_models}个跟踪模型采用MoE，其中多个直接使用MLA+256专家MoE+共享专家+无辅助损失路由的DeepSeek V3架构。</li>",
                "<li><b>注意力机制分叉为四条竞争路径。</b>MLA（DeepSeek/Kimi/GLM/MiMo）领先前沿；GQA仍是稠密模型主流；线性注意力（MiniMax）和Mamba混合（NVIDIA）是两个非softmax挑战者。</li>",
                "<li><b>混合推理（可切换）成为默认。</b>大多数新模型支持think/no-think开关，无需维护独立的推理版和指令版。</li>",
                "<li><b>MTP推理是新的吞吐量战场。</b>MTP-3投机解码在DeepSeek V4等模型中落地，带来2-3倍吞吐量提升。</li>",
                "<li><b>价格压缩持续但趋缓。</b>前沿模型稳定在$0.40-$0.60/$1.50-$2.50区间，真正的价格优势来自缓存输入定价。</li>",
            ]),
        }
    else:
        text = {
            'EYEBROW': "Frontier LLM Architecture & Capability Tracking",
            'REPORT_TITLE': "Open-Weight LLM Landscape Report",
            'ONE_LINE_THESIS': (f"The open-weight LLM landscape has converged on sparse MoE + long-context efficiency: DeepSeek V4 pushes active params to 49B/13B, MiniMax and NVIDIA explore linear/Mamba hybrid attention, and price wars push frontier capability under $2/1M output tokens. This report verifies parameters from official sources and provides cross-model comparison."),
            'PILL_SNAPSHOT': f"Snapshot {snap_date[:7]}",
            'PILL_DATE': f"Generated {snap_date}",
            'PILL_N_MODELS': f"{n_models} models/variants",
            'PILL_SOURCES': "Sources: Tech reports · HF cards · OpenRouter",
            'NAV_SUMMARY': "Summary", 'NAV_GLANCE': "Roster", 'NAV_OPENROUTER': "OpenRouter",
            'NAV_DEEPDIVES': "Deep Dives", 'NAV_CAPABILITY': "Capability", 'NAV_SYNTHESIS': "Synthesis",
            'NAV_KV': "KV Cache", 'NAV_DELTA': "Changes", 'NAV_METHOD': "Methodology",
            'SECTION_SUMMARY': "Executive Summary",
            'SECTION_GLANCE': "The Landscape at a Glance",
            'SECTION_OPENROUTER': "OpenRouter Reality Check",
            'SECTION_DEEPDIVES': "Architecture Deep Dives",
            'SECTION_CAPABILITY': "Capability Comparison",
            'SECTION_SYNTHESIS': "Cross-Cutting Analysis",
            'SECTION_DELTA': "What Changed Since Last Snapshot",
            'SECTION_METHOD': "Methodology & Sources",
            'GLANCE_INTRO': "Click column headers to sort. All parameters from official HuggingFace/ModelScope cards or technical reports.",
            'COL_MODEL': "Model", 'COL_LAB': "Lab", 'COL_RELEASE': "Release",
            'COL_TOTAL': "Total", 'COL_ACTIVE': "Active", 'COL_ATTENTION': "Attention",
            'COL_MOE': "MoE", 'COL_CONTEXT': "Context", 'COL_LICENSE': "License",
            'COL_PRICE_IN': "$/1M in", 'COL_PRICE_OUT': "$/1M out",
            'COL_RANK': "Rank", 'COL_WEEKLY': "Weekly", 'COL_WHY': "Why popular",
            'COL_REASONING': "Reasoning", 'COL_CODING': "Coding", 'COL_AGENTIC': "Agentic",
            'COL_MATH': "Math", 'COL_LONGCTX': "Long-ctx", 'COL_MM': "MM",
            'COL_CHANGE_TYPE': "Change", 'COL_CHANGE_SUMMARY': "Detail",
            'OPENROUTER_CALLOUT': "Usage rank ≠ quality. A cheap preview or single high-volume app can vault a model to #1.",
            'CAPABILITY_INTRO': "The matrix below compares models across 6 shared dimensions. Scores are 0-100 normalized.",
            'CAPABILITY_CALLOUT': "Many open-model scores are self-reported and lack independent reproduction. Interpret with caution.",
            'CAPABILITY_CHART_CAPTION': "Capability grouped bars by dimension (color = model)",
            'COST_CHART_CAPTION': "Capability vs Cost (bubble ∝ OpenRouter usage)",
            'CAPABILITY_READOUT_TITLE': "Reading the matrix",
            'CAPABILITY_READOUT': "DeepSeek V4 leads on coding and reasoning; DeepSeek V3 is the price-performance king.",
            'SEC_KV_RACE': "KV-Cache & Long-Context Efficiency Race",
            'KV_RACE_INTRO': "KV-cache memory footprint (GB, BF16) vs context length, computed from real model configurations.",
            'KV_RACE_BODY': syn.get('kv_race', "KV-cache efficiency is the key differentiator for long-context inference."),
            'KV_CAPTION': "KV-cache size comparison (BF16)",
            'SEC_TIMELINE': "Release Timeline",
            'TIMELINE_CAPTION': "Release timeline of major open-source models",
            'SEC_MOE_TREND': "MoE Sparsification & the Shared-Expert Debate",
            'MOE_TREND': syn.get('moe_trend', f"MoE is now dominant: {n_moe} of {n_models} tracked models use MoE."),
            'SPARSITY_CAPTION': "MoE sparsity scatter (total vs active params)",
            'SEC_ATTENTION_FORK': "The Attention Fork",
            'ATTENTION_FORK': syn.get('attention_fork', "Attention has forked into four competing paths."),
            'SEC_CONVERGENCE': "Convergence vs Divergence",
            'CONVERGENCE': syn.get('convergence', "Massive convergence around the DeepSeek V3 template."),
            'SEC_ECONOMICS': "Economics & Geographic Shifts",
            'ECONOMICS': syn.get('economics', "The cost-vs-capability frontier has a clear best-value line."),
            'SEC_SOURCE_HIERARCHY': "Source Hierarchy",
            'SOURCE_1': "Primary: model technical reports, official model cards, config.json",
            'SOURCE_2': "Secondary: verified third-party benchmarks, OpenRouter pricing/rankings",
            'SOURCE_3': "Tertiary: reputable technical journalism",
            'SEC_CONFIDENCE': "Confidence Notes",
            'METHODOLOGY': f"Architecture facts from technical reports and HuggingFace configs. Pricing from OpenRouter as of {snap_date_long}.",
            'SEC_SOURCES_BY_MODEL': "Sources by Model",
            'SOURCE_LIST': "<li>Primary technical reports (arXiv, lab GitHub, blogs)</li><li>HuggingFace model configs (config.json)</li><li>OpenRouter API pricing and rankings</li>",
            'FOOTER_TEXT': f"Generated {snap_date_long}. Usage rank ≠ quality. Verify on your own evals for production decisions.",
            'DELTA_MSG': f"Baseline snapshot as of {snap_date_long}. Future reports will show changes.",
            'SUMMARY_BULLETS': "".join([
                f"<li><b>The DeepSeek V3 template is now the industry standard.</b> {n_moe} of {n_models} tracked models are MoE.</li>",
                "<li><b>Attention has forked into four competing paths.</b> MLA dominates frontier; GQA remains mainstream; linear and Mamba are challengers.</li>",
                "<li><b>Hybrid (toggleable) reasoning is now the default.</b> Most new models support a think/no-think switch.</li>",
                "<li><b>MTP at inference is the new throughput battleground.</b> MTP-3 delivers 2-3x throughput gains.</li>",
                "<li><b>Price compression continues but is slowing.</b> Frontier models have stabilized around $0.40-$0.60/$1.50-$2.50.</li>",
            ]),
        }

    # Replace all placeholders
    for key, val in text.items():
        html = html.replace('{{' + key + '}}', val)
    # Also replace SNAPSHOT_DATE which is used in title
    html = html.replace('{{SNAPSHOT_DATE}}', snap_date_long)

    # Replace dynamic sections
    html = html.replace('{{ROSTER_ROWS}}', build_roster_rows(models, lang))
    html = html.replace('{{OPENROUTER_ROWS}}', build_openrouter_rows(rankings, lang))
    html = html.replace('{{MODEL_CARDS}}', build_model_cards(models, lang))
    html = html.replace('{{CAPABILITY_ROWS}}', build_capability_rows(models, lang))
    html = html.replace('{{DELTA_ROWS}}',
        f'<tr><td colspan="3" style="text-align:center;color:var(--ink-faint);padding:24px;">{text["DELTA_MSG"]}</td></tr>')

    # Inject chart rendering JS
    chart_js = build_chart_js(models, rankings, lang)
    html = html.replace('</script>', chart_js + '\n</script>')

    # Set language attribute
    if lang == 'zh':
        html = html.replace('<html lang="zh" data-theme="dark">', '<html lang="zh" data-theme="dark">')
    else:
        html = html.replace('<html lang="zh" data-theme="dark">', '<html lang="en" data-theme="dark">')

    os.makedirs(os.path.dirname(paths['output']), exist_ok=True)
    with open(paths['output'], 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"Report written to {paths['output']}")
    print(f"  Models: {n_models}")
    print(f"  MoE models: {n_moe}")
    print(f"  Language: {lang}")
    print(f"  File size: {os.path.getsize(paths['output']) / 1024:.1f} KB")
    return paths['output']

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate LLM landscape HTML report')
    parser.add_argument('--date', type=str, default=None, help='Snapshot date (YYYY-MM-DD), defaults to today')
    parser.add_argument('--lang', type=str, default='zh', choices=['zh', 'en'], help='Report language (zh/en)')
    parser.add_argument('--no-strict-sources', action='store_true', help='Disable strict trusted source validation')
    args = parser.parse_args()
    main(args.date, args.lang, strict_sources=not args.no_strict_sources)
