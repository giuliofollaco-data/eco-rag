import os
import re
import json
import time
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from main import query_translation_chain, answer_chain
from vector import get_retrieved_context

# ── Palette & style ─────────────────────────────────────────────────────────────

STYLE = {
    "bg": "#FFFFFF",
    "grid": "#EEEEEE",
    "text": "#1A1A2E",
    "accent": "#2C5F8A",  # bleu principal
    "accent2": "#4A90C4",  # bleu secondaire
    "ok": "#2E7D32",  # vert (succès)
    "warn": "#C0392B",  # rouge (échec)
    "neutral": "#607D8B",  # gris bleu
    "bar_palette": [
        "#2C5F8A",
        "#4A90C4",
        "#76B0D4",
        "#A8D1E7",
        "#CFE9F5",
        "#E8F4FD",
        "#90A4AE",
    ],
}

plt.rcParams.update(
    {
        "figure.facecolor": STYLE["bg"],
        "axes.facecolor": STYLE["bg"],
        "axes.edgecolor": STYLE["neutral"],
        "axes.labelcolor": STYLE["text"],
        "axes.titlecolor": STYLE["text"],
        "axes.grid": True,
        "grid.color": STYLE["grid"],
        "grid.linewidth": 0.8,
        "xtick.color": STYLE["text"],
        "ytick.color": STYLE["text"],
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
    }
)

# ── Suite de tests ──────────────────────────────────────────────────────────────
# Chaque cas définit :
#   question          — question posée en français
#   expected_values   — valeurs clés que la réponse DOIT mentionner (exact_match)
#   expected_keywords — mots-clés thématiques attendus (keyword_score)
#   wrong_values      — valeurs/affirmations erronées dont l'absence est vérifiée

TEST_CASES = [
    {
        "id": "T01",
        "category": "A",
        "label": "Temp. 2011-2020 vs 1850-1900",
        "question": (
            "Quelle est l'augmentation de la température mondiale à la surface du globe "
            "observée sur la période 2011-2020 par rapport à 1850-1900 ?"
        ),
        "expected_values": ["1.1", "1,1"],
        "expected_keywords": ["température", "2011", "1850", "surface", "°C"],
        "wrong_values": ["1.5°C", "2°C", "0.5°C", "1.07"],
    },
    {
        "id": "T02",
        "category": "A",
        "label": "Émissions GES 2019",
        "question": (
            "Quel était le niveau des émissions mondiales nettes de gaz à effet de serre "
            "en 2019 selon le rapport ?"
        ),
        "expected_values": ["59", "GtCO2"],
        "expected_keywords": [
            "émissions",
            "2019",
            "gaz à effet de serre",
            "GES",
            "net",
        ],
        "wrong_values": ["40 Gt", "70 Gt", "100 Gt"],
    },
    {
        "id": "T03",
        "category": "A",
        "label": "Élévation du niveau de la mer 1901-2018",
        "question": (
            "De combien le niveau moyen mondial de la mer a-t-il augmenté entre 1901 et 2018 ?"
        ),
        "expected_values": ["0.20", "0,20", "20 cm"],
        "expected_keywords": ["mer", "niveau", "1901", "2018", "mètre", "cm"],
        "wrong_values": ["0.5 m", "1 mètre", "10 cm"],
    },
    {
        "id": "T04",
        "category": "A",
        "label": "Concentration CO₂ atmosphérique 2019",
        "question": (
            "Quelle était la concentration de CO2 dans l'atmosphère en 2019, "
            "et en quoi est-ce remarquable selon le GIEC ?"
        ),
        "expected_values": ["410", "ppm"],
        "expected_keywords": ["CO2", "concentration", "2019", "ppm", "sans précédent"],
        "wrong_values": ["350 ppm", "500 ppm", "280 ppm"],
    },
    {
        "id": "T05",
        "category": "B",
        "label": "Certitude — influence humaine sur extrêmes",
        "question": (
            "Est-il certain que les événements météorologiques extrêmes (vagues de chaleur, "
            "précipitations, cyclones) sont liés à l'influence humaine ? "
            "Utilise la terminologie du GIEC."
        ),
        "expected_values": [
            "très probable",
            "confiance élevée",
            "virtuellement certain",
            "pratiquement certain",
        ],
        "expected_keywords": [
            "extrêmes",
            "influence humaine",
            "vagues de chaleur",
            "cyclones",
            "probabilité",
        ],
        "wrong_values": ["aucun lien", "pas de preuve", "incertain"],
    },
    {
        "id": "T06",
        "category": "B",
        "label": "Conclusion — neutralité carbone",
        "question": (
            "Que faut-il atteindre en matière d'émissions de CO2 pour limiter "
            "le réchauffement climatique, selon la conclusion principale du rapport ?"
        ),
        "expected_values": [
            "neutralité carbone",
            "net zéro",
            "net-zéro",
            "zéro émission nette",
        ],
        "expected_keywords": [
            "CO2",
            "neutralité",
            "1.5",
            "2°C",
            "émissions",
            "limiter",
        ],
        "wrong_values": [],
    },
    {
        "id": "T07",
        "category": "B",
        "label": "Vulnérabilité — population exposée",
        "question": (
            "Combien de personnes vivent dans des contextes très vulnérables aux "
            "changements climatiques selon le rapport du GIEC 2023 ?"
        ),
        "expected_values": ["3.3", "3,3", "milliard", "billion"],
        "expected_keywords": [
            "vulnérable",
            "personnes",
            "population",
            "contexte",
            "risque",
        ],
        "wrong_values": [],
    },
    {
        "id": "T08",
        "category": "C",
        "label": "Limite d'adaptation — franchissement seuil",
        "question": (
            "Quels sont les exemples de limites dures à l'adaptation que le GIEC mentionne ?"
        ),
        "expected_values": [],
        "expected_keywords": [
            "limites",
            "adaptation",
            "dures",
            "franchissement",
            "seuil",
            "irréversible",
        ],
        "wrong_values": ["aucune limite", "adaptation illimitée"],
    },
    {
        "id": "T09",
        "category": "C",
        "label": "Financement climatique",
        "question": (
            "Quel écart existe-t-il entre les besoins de financement pour l'adaptation "
            "climatique et les flux financiers actuels dans les pays en développement ?"
        ),
        "expected_values": [],
        "expected_keywords": [
            "financement",
            "adaptation",
            "lacune",
            "écart",
            "pays en développement",
            "milliards",
        ],
        "wrong_values": [],
    },
    {
        "id": "T10",
        "category": "C",
        "label": "Synergie ODD",
        "question": (
            "Comment les actions climatiques peuvent-elles créer des synergies "
            "avec les objectifs de développement durable (ODD) ?"
        ),
        "expected_values": [],
        "expected_keywords": [
            "synergies",
            "ODD",
            "développement durable",
            "co-bénéfices",
            "santé",
            "énergie",
        ],
        "wrong_values": [],
    },
]

# ── Évaluation des métriques ────────────────────────────────────────────────────

JUDGE_PROMPT = """
Tu es un évaluateur expert du rapport GIEC AR6.
Note la réponse ci-dessous sur une échelle de 1 à 5 selon les critères suivants :

5 — Réponse complète, précise, bien sourcée, terminologie GIEC correcte.
4 — Bonne réponse avec quelques imprécisions mineures ou sources partiellement citées.
3 — Réponse partiellement correcte : éléments pertinents mais manque d'éléments clés ou imprécisions notables.
2 — Réponse superficielle, hors sujet partiel, ou contient des approximations importantes.
1 — Réponse incorrecte, inventée, ou hors sujet complet.

Question : {question}
Réponse à évaluer : {response}

Réponds UNIQUEMENT avec un entier entre 1 et 5. Aucun commentaire, aucune explication.
Score :"""


def _normalize(text: str) -> str:
    """Minuscule + suppression des accents pour comparaisons robustes."""
    import unicodedata

    text = text.lower()
    return "".join(
        c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn"
    )


def compute_exact_match(response: str, expected_values: list[str]) -> int:
    """1 si AU MOINS UNE valeur attendue est présente dans la réponse, sinon 0."""
    if not expected_values:
        return 1  # pas de valeur cible définie → non pénalisé
    resp_norm = _normalize(response)
    return int(any(_normalize(v) in resp_norm for v in expected_values))


def compute_keyword_score(response: str, expected_keywords: list[str]) -> float:
    """Fraction des mots-clés thématiques présents dans la réponse."""
    if not expected_keywords:
        return 1.0
    resp_norm = _normalize(response)
    hits = sum(1 for kw in expected_keywords if _normalize(kw) in resp_norm)
    return round(hits / len(expected_keywords), 2)


def compute_wrong_avoided(response: str, wrong_values: list[str]) -> int:
    """1 si AUCUNE valeur erronée n'est présente dans la réponse."""
    if not wrong_values:
        return 1
    resp_norm = _normalize(response)
    return int(all(_normalize(w) not in resp_norm for w in wrong_values))


def compute_spm_retrieved(context: str) -> int:
    """1 si le contexte contient au moins un extrait étiqueté SPM."""
    return int("SPM" in context)


def compute_llm_score(question: str, response: str, model) -> int:
    """LLM-as-judge : le modèle note la réponse de 1 à 5."""
    from langchain_core.prompts import ChatPromptTemplate

    prompt = ChatPromptTemplate.from_template(JUDGE_PROMPT)
    chain = prompt | model
    raw = chain.invoke({"question": question, "response": response}).strip()
    # Extraction robuste d'un entier 1-5
    match = re.search(r"[1-5]", raw)
    return int(match.group()) if match else 3


# ── Pipeline d'évaluation ───────────────────────────────────────────────────────


def run_backtest(test_cases: list[dict], output_dir: str) -> list[dict]:
    from langchain_ollama.llms import OllamaLLM

    model = OllamaLLM(model="llama3.2", temperature=0.0)

    results = []
    total = len(test_cases)

    for i, case in enumerate(test_cases, 1):
        print(f"\n{'─'*55}")
        print(f"  [{i}/{total}] {case['id']} — {case['label']}")
        print(f"{'─'*55}")

        question = case["question"]

        # ── Traduction de la question ──
        question_en = query_translation_chain.invoke({"question": question}).strip()
        print(f"  [EN] {question_en}")

        # ── Retrieval + mesure du temps de réponse ──
        t0 = time.time()
        context = get_retrieved_context(question_en)
        response = answer_chain.invoke({"context": context, "question": question})
        latency = round(time.time() - t0, 2)
        print(f"  ⏱  Latence : {latency}s")

        # ── Métriques ──
        exact = compute_exact_match(response, case.get("expected_values", []))
        kw_score = compute_keyword_score(response, case.get("expected_keywords", []))
        avoided = compute_wrong_avoided(response, case.get("wrong_values", []))
        spm = compute_spm_retrieved(context)
        llm_score = compute_llm_score(question, response, model)

        print(
            f"  exact_match={exact}  keyword={kw_score}  wrong_avoided={avoided}  "
            f"spm={spm}  llm_score={llm_score}"
        )

        results.append(
            {
                "id": case["id"],
                "category": case["category"],
                "label": case["label"],
                "question": question,
                "response": response,
                "exact_match": exact,
                "keyword_score": kw_score,
                "wrong_avoided": avoided,
                "spm_retrieved": spm,
                "llm_score": llm_score,
                "latency_s": latency,
            }
        )

    # Sauvegarde JSON
    json_path = os.path.join(output_dir, "results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n[✓] Résultats sauvegardés → {json_path}")

    return results


# ── Graphiques ──────────────────────────────────────────────────────────────────


def _save(fig: plt.Figure, path: str) -> None:  # type: ignore
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=STYLE["bg"])
    plt.close(fig)
    print(f"[✓] Graphique → {path}")


def _short_label(label: str, max_len: int = 22) -> str:
    return label if len(label) <= max_len else label[: max_len - 1] + "…"


def plot_llm_scores(results: list[dict], output_dir: str) -> None:
    labels = [r["id"] for r in results]
    scores = [r["llm_score"] for r in results]
    full_labels = [_short_label(r["label"]) for r in results]

    colors = [
        STYLE["ok"] if s >= 4 else (STYLE["accent"] if s == 3 else STYLE["warn"])
        for s in scores
    ]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(labels, scores, color=colors, width=0.6, zorder=2)

    ax.set_ylim(0, 5.5)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_xlabel("Question")
    ax.set_ylabel("Score LLM-as-judge (1–5)")
    ax.set_title("Score LLM-as-judge par question")
    ax.axhline(
        np.mean(scores),  # type: ignore
        color=STYLE["neutral"],
        linewidth=1.4,
        linestyle="--",
        label=f"Moyenne : {np.mean(scores):.2f}",
        zorder=3,
    )

    for bar, score, fl in zip(bars, scores, full_labels):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.1,
            str(score),
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            color=STYLE["text"],
        )
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            -0.45,
            fl,
            ha="center",
            va="top",
            fontsize=7.5,
            color=STYLE["neutral"],
            rotation=20,
        )

    patches = [
        mpatches.Patch(color=STYLE["ok"], label="Score ≥ 4"),
        mpatches.Patch(color=STYLE["accent"], label="Score = 3"),
        mpatches.Patch(color=STYLE["warn"], label="Score ≤ 2"),
    ]
    ax.legend(handles=patches + [ax.get_lines()[0]], loc="upper right", fontsize=9)
    ax.tick_params(axis="x", bottom=False)
    fig.tight_layout()
    _save(fig, os.path.join(output_dir, "01_llm_scores.png"))


def plot_latency(results: list[dict], output_dir: str) -> None:
    labels = [r["id"] for r in results]
    latencies = [r["latency_s"] for r in results]
    mean_lat = np.mean(latencies)

    colors = [
        (
            STYLE["ok"]
            if l < mean_lat * 0.8
            else (STYLE["warn"] if l > mean_lat * 1.3 else STYLE["accent2"])
        )
        for l in latencies
    ]

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(labels, latencies, color=colors, width=0.6, zorder=2)
    ax.axhline(
        mean_lat,  # type: ignore
        color=STYLE["neutral"],
        linewidth=1.4,
        linestyle="--",
        label=f"Moyenne : {mean_lat:.1f}s",
        zorder=3,
    )

    for x, lat in zip(labels, latencies):
        ax.text(
            x,
            lat + 0.3,
            f"{lat:.1f}s",
            ha="center",
            va="bottom",
            fontsize=9,
            color=STYLE["text"],
        )

    ax.set_xlabel("Question")
    ax.set_ylabel("Latence (secondes)")
    ax.set_title("Latence de réponse par question")
    ax.legend(fontsize=9)
    fig.tight_layout()
    _save(fig, os.path.join(output_dir, "02_latency.png"))


def plot_keyword_scores(results: list[dict], output_dir: str) -> None:
    labels = [r["id"] for r in results]
    scores = [r["keyword_score"] for r in results]
    full_labels = [_short_label(r["label"]) for r in results]

    fig, ax = plt.subplots(figsize=(10, 4.5))
    bars = ax.bar(labels, scores, color=STYLE["accent"], width=0.6, zorder=2)

    for bar, s, fl in zip(bars, scores, full_labels):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{s:.0%}",
            ha="center",
            va="bottom",
            fontsize=9,
            color=STYLE["text"],
        )
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            -0.06,
            fl,
            ha="center",
            va="top",
            fontsize=7.5,
            color=STYLE["neutral"],
            rotation=20,
        )

    ax.axhline(
        np.mean(scores),  # type: ignore
        color=STYLE["neutral"],
        linewidth=1.4,
        linestyle="--",
        label=f"Moyenne : {np.mean(scores):.0%}",
        zorder=3,
    )
    ax.set_ylim(0, 1.15)
    ax.set_xlabel("Question")
    ax.set_ylabel("Taux de correspondance")
    ax.set_title("Keyword score par question")
    ax.legend(fontsize=9)
    ax.tick_params(axis="x", bottom=False)
    fig.tight_layout()
    _save(fig, os.path.join(output_dir, "03_keyword_scores.png"))


def plot_exact_match(results: list[dict], output_dir: str) -> None:
    labels = [r["id"] for r in results]
    hits = [r["exact_match"] for r in results]
    full_labels = [_short_label(r["label"]) for r in results]

    colors = [STYLE["ok"] if h else STYLE["warn"] for h in hits]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(labels, hits, color=colors, width=0.55, zorder=2)

    for x, h, fl in zip(labels, hits, full_labels):
        ax.text(
            x,
            h + 0.03,
            "✓" if h else "✗",
            ha="center",
            va="bottom",
            fontsize=13,
            color=STYLE["ok"] if h else STYLE["warn"],
        )
        ax.text(
            x,
            -0.08,
            fl,
            ha="center",
            va="top",
            fontsize=7.5,
            color=STYLE["neutral"],
            rotation=20,
        )

    ok_total = sum(hits)
    ax.set_ylim(0, 1.4)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Absent (0)", "Présent (1)"])
    ax.set_xlabel("Question")
    ax.set_title(
        f"Exact match — valeur clé présente dans la réponse "
        f"({ok_total}/{len(hits)} réussites)"
    )
    patches = [
        mpatches.Patch(color=STYLE["ok"], label="Valeur trouvée"),
        mpatches.Patch(color=STYLE["warn"], label="Valeur absente"),
    ]
    ax.legend(handles=patches, fontsize=9)
    ax.tick_params(axis="x", bottom=False)
    fig.tight_layout()
    _save(fig, os.path.join(output_dir, "04_exact_match.png"))


def plot_radar(results: list[dict], output_dir: str) -> None:
    metrics = [
        "exact_match",
        "keyword_score",
        "wrong_avoided",
        "spm_retrieved",
        "llm_score",
    ]
    labels = [
        "Exact Match",
        "Keyword Score",
        "Wrong\nAvoided",
        "SPM\nRetrieved",
        "LLM Score",
    ]
    raw = [np.mean([r[m] for r in results]) for m in metrics]
    # Normalisation : llm_score est sur 5, les autres sur 1
    norm = [v / 5 if m == "llm_score" else v for v, m in zip(raw, metrics)]

    n = len(metrics)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]
    norm += norm[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"polar": True})
    ax.set_facecolor(STYLE["bg"])
    fig.patch.set_facecolor(STYLE["bg"])

    ax.plot(angles, norm, color=STYLE["accent"], linewidth=2.2, zorder=3)
    ax.fill(angles, norm, color=STYLE["accent2"], alpha=0.25, zorder=2)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10, color=STYLE["text"])
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(
        ["25%", "50%", "75%", "100%"], fontsize=8, color=STYLE["neutral"]
    )
    ax.set_ylim(0, 1)
    ax.grid(color=STYLE["grid"], linewidth=0.8)
    ax.spines["polar"].set_color(STYLE["neutral"])

    # Annotations des valeurs brutes
    for angle, nv, rv, m in zip(angles[:-1], norm[:-1], raw[:-1], metrics):
        label_str = f"{rv:.2f}" if m == "llm_score" else f"{rv:.0%}"
        ax.text(
            angle,
            nv + 0.07,  # type: ignore
            label_str,
            ha="center",
            va="center",
            fontsize=9,
            color=STYLE["text"],
            fontweight="bold",
        )

    ax.set_title(
        "Métriques moyennes (normalisées)",
        fontsize=13,
        fontweight="bold",
        pad=18,
        color=STYLE["text"],
    )
    fig.tight_layout()
    _save(fig, os.path.join(output_dir, "05_metrics_radar.png"))


def plot_category_summary(results: list[dict], output_dir: str) -> None:
    from collections import defaultdict

    categories = sorted({r["category"] for r in results})
    metrics = ["exact_match", "keyword_score", "wrong_avoided", "spm_retrieved"]
    metric_labels = ["Exact Match", "Keyword Score", "Wrong Avoided", "SPM Retrieved"]

    # Moyennes par catégorie
    data: dict[str, list[float]] = defaultdict(lambda: [0.0] * len(metrics))
    counts: dict[str, int] = defaultdict(int)
    for r in results:
        cat = r["category"]
        counts[cat] += 1
        for j, m in enumerate(metrics):
            data[cat][j] += r[m]
    for cat in categories:
        data[cat] = [v / counts[cat] for v in data[cat]]

    n_cats = len(categories)
    n_metr = len(metrics)
    x = np.arange(n_cats)
    w = 0.18
    offsets = np.linspace(-(n_metr - 1) * w / 2, (n_metr - 1) * w / 2, n_metr)

    fig, ax = plt.subplots(figsize=(9, 5))
    for j, (m_label, offset) in enumerate(zip(metric_labels, offsets)):
        vals = [data[cat][j] for cat in categories]
        bars = ax.bar(
            x + offset,
            vals,
            width=w,
            label=m_label,
            color=STYLE["bar_palette"][j],
            zorder=2,
        )
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.015,
                f"{v:.0%}",
                ha="center",
                va="bottom",
                fontsize=7.5,
                color=STYLE["text"],
            )

    ax.set_xticks(x)
    ax.set_xticklabels([f"Catégorie {c}\n(n={counts[c]})" for c in categories])
    ax.set_ylim(0, 1.2)
    ax.set_ylabel("Score moyen")
    ax.set_title("Synthèse des métriques par catégorie")
    ax.legend(fontsize=9, loc="upper right")
    fig.tight_layout()
    _save(fig, os.path.join(output_dir, "06_category_summary.png"))


# ── Récapitulatif terminal ──────────────────────────────────────────────────────


def print_summary(results: list[dict]) -> None:
    print("\n" + "═" * 55)
    print("  RÉCAPITULATIF DU BACKTEST")
    print("═" * 55)
    metrics = [
        "exact_match",
        "keyword_score",
        "wrong_avoided",
        "spm_retrieved",
        "llm_score",
    ]
    for m in metrics:
        values = [r[m] for r in results]
        mean = np.mean(values)
        label = f"{m:<20}"
        bar_len = int(mean / (5 if m == "llm_score" else 1) * 20)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        fmt = f"{mean:.2f}/5" if m == "llm_score" else f"{mean:.0%}"
        print(f"  {label}  {bar}  {fmt}")
    total_t = sum(r["latency_s"] for r in results)
    print(f"\n  Latence totale  : {total_t:.1f}s")
    print(f"  Latence moyenne : {total_t/len(results):.1f}s / question")
    print(f"  Questions       : {len(results)}")
    print("═" * 55)


# ── Point d'entrée ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("backtests", f"backtest_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 55)
    print(f"  BACKTEST RAG — GIEC AR6")
    print(f"  Dossier de sortie : {output_dir}")
    print("=" * 55)

    results = run_backtest(TEST_CASES, output_dir)

    print("\n[INFO] Génération des graphiques...")
    plot_llm_scores(results, output_dir)
    plot_latency(results, output_dir)
    plot_keyword_scores(results, output_dir)
    plot_exact_match(results, output_dir)
    plot_radar(results, output_dir)
    plot_category_summary(results, output_dir)

    print_summary(results)
    print(f"\n[✓] Tous les fichiers sont dans : {output_dir}/")
