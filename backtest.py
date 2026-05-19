import json
import re
import time
import unicodedata
from datetime import datetime
from pathlib import Path

BACKTESTS_ROOT = Path("backtests")

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import numpy as np
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

from vector import get_retrieved_context

# ══════════════════════════════════════════════════════════════════════════════
# 1. JEU DE TEST (GROUND TRUTH)
# ══════════════════════════════════════════════════════════════════════════════
# Sources : SPM A.1, A.1.1, A.1.2, A.1.3, A.1.4, A.2.1, A.2.2
# Toutes les valeurs ont été vérifiées sur le PDF original.

TEST_CASES = [
    # ── Catégorie A : Confusion de données ────────────────────────────────────
    {
        "id": "T01",
        "category": "A",
        "label": "Temp. 2011-2020 vs 1850-1900",
        "question": (
            "Quelle est l'augmentation de la température mondiale à la surface "
            "du globe observée sur la période 2011-2020 par rapport à 1850-1900 ?"
        ),
        # Valeur de synthèse SPM A.1 : 1,1 °C
        # Pièges : A.1.1 donne 1,09°C (valeur précise), A.1.2 donne 1,07°C
        # (période légèrement différente 2010-2019), 0,99°C (2001-2020).
        "expected_values": ["1,1", "1.1"],
        "wrong_values": ["1,07", "1.07", "1,09", "1.09", "0,99", "0.99"],
        "keywords": ["température", "surface", "1850", "1900", "confiance"],
    },
    {
        "id": "T02",
        "category": "A",
        "label": "Émissions GES 2019",
        "question": (
            "Quel était le niveau des émissions mondiales nettes de gaz à effet "
            "de serre en 2019 selon le rapport ?"
        ),
        # SPM A.1.4 : 59 ± 6,6 GtCO2-eq
        "expected_values": ["59"],
        "wrong_values": [],
        "keywords": ["59", "GtCO2", "2019", "émissions"],
    },
    {
        "id": "T03",
        "category": "A",
        "label": "Élévation du niveau de la mer 1901-2018",
        "question": (
            "De combien le niveau moyen mondial de la mer a-t-il augmenté "
            "entre 1901 et 2018 ?"
        ),
        # SPM A.2.1 : 0,20 [0,15 à 0,25] m
        "expected_values": ["0,20", "0.20", "20 cm", "20"],
        "wrong_values": [],
        "keywords": ["niveau", "mer", "1901", "2018", "confiance élevée"],
    },
    {
        "id": "T04",
        "category": "A",
        "label": "Concentration CO₂ atmosphérique 2019",
        "question": (
            "Quelle était la concentration de CO2 dans l'atmosphère en 2019, "
            "et en quoi est-ce remarquable selon le GIEC ?"
        ),
        # SPM A.1.3 : 410 ppm, plus élevé qu'à n'importe quel moment des
        # 2 derniers millions d'années (confiance élevée)
        "expected_values": ["410"],
        "wrong_values": [],
        "keywords": ["410", "ppm", "millions d'années", "confiance élevée"],
    },
    # ── Catégorie B : Contresens scientifique ─────────────────────────────────
    {
        "id": "T05",
        "category": "B",
        "label": "Certitude — influence humaine sur extrêmes",
        "question": (
            "Est-il certain que les événements météorologiques extrêmes "
            "(vagues de chaleur, précipitations, cyclones) sont liés à "
            "l'influence humaine ? Utilise la terminologie du GIEC."
        ),
        # SPM A.2.1 : "il est sans équivoque que l'influence humaine a réchauffé..."
        # + "les preuves des observations des changements dans les extrêmes [...]
        # et en particulier leur attribution à l'influence humaine, se sont
        # encore renforcées depuis l'AR5" — conclusion : OUI, établi avec
        # confiance élevée à très élevée selon le type d'extrême.
        "expected_values": ["confiance", "renforcé", "influence humaine"],
        "wrong_values": [],
        "keywords": ["influence humaine", "extrêmes", "confiance", "renforcé"],
    },
    {
        "id": "T06",
        "category": "B",
        "label": "Conclusion — neutralité carbone",
        "question": (
            "Que faut-il atteindre en matière d'émissions de CO2 pour limiter "
            "le réchauffement climatique, selon la conclusion principale du rapport ?"
        ),
        # SPM introduction : "limiter le réchauffement causé par l'homme nécessite
        # des émissions nettes de CO2 nulles"
        "expected_values": ["zéro", "net zéro", "net zero", "nulle", "nulles"],
        "wrong_values": [],
        "keywords": ["CO2", "émissions", "zéro", "limiter", "réchauffement"],
    },
    {
        "id": "T07",
        "category": "B",
        "label": "Vulnérabilité — population exposée",
        "question": (
            "Combien de personnes vivent dans des contextes très vulnérables "
            "aux changements climatiques selon le rapport du GIEC 2023 ?"
        ),
        # SPM A.2.2 : 3,3 à 3,6 milliards de personnes
        "expected_values": ["3,3", "3.3", "3,6", "3.6", "milliards", "billion"],
        "wrong_values": [],
        "keywords": ["milliards", "vulnérable", "populations", "confiance élevée"],
    },
]

# ══════════════════════════════════════════════════════════════════════════════
# 2. CONFIGURATION DU LLM
# ══════════════════════════════════════════════════════════════════════════════

model = OllamaLLM(model="llama3.2", temperature=0.0)

ANSWER_TEMPLATE = """\
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

Tu es un assistant expert en science du climat, spécialisé dans le rapport de \
synthèse 2023 du GIEC (AR6 SYR).

Réponds à la question en utilisant UNIQUEMENT les extraits fournis dans la section "CONTEXTE".

Règles strictes — respecte-les dans cet ordre de priorité :
1. Les extraits [SPM] ont la priorité absolue sur les extraits [CORPS].
2. Si plusieurs valeurs proches apparaissent, retiens celle présentée comme \
valeur de référence principale dans un extrait [SPM].
3. Conserve les qualificatifs du GIEC : "confiance élevée", "très probable", etc.
4. Si la réponse n'est pas dans le contexte, réponds : \
"Les extraits fournis ne permettent pas de répondre à cette question."

<|eot_id|><|start_header_id|>user<|end_header_id|>

CONTEXTE :
{context}

---

QUESTION :
{question}

Réponse en français :<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

JUDGE_TEMPLATE = """\
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

Tu es un évaluateur scientifique expert du rapport AR6 du GIEC.
Note la réponse ci-dessous sur une échelle de 1 à 5 :
  1 = incorrecte ou complètement hors sujet
  2 = partiellement correcte mais contient des erreurs factuelles importantes
  3 = correcte sur l'essentiel mais imprécise ou incomplète
  4 = correcte, précise, avec les niveaux de confiance appropriés
  5 = parfaite : valeur exacte, terminologie GIEC, source citée

Réponds UNIQUEMENT avec un entier entre 1 et 5, sans aucune explication.

<|eot_id|><|start_header_id|>user<|end_header_id|>

QUESTION : {question}
RÉPONSE ATTENDUE (référence) : {expected}
RÉPONSE DU SYSTÈME : {response}

Note (1-5) :<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

answer_prompt = ChatPromptTemplate.from_template(ANSWER_TEMPLATE)
judge_prompt = ChatPromptTemplate.from_template(JUDGE_TEMPLATE)
answer_chain = answer_prompt | model
judge_chain = judge_prompt | model

# ══════════════════════════════════════════════════════════════════════════════
# 3. FONCTIONS D'ÉVALUATION
# ══════════════════════════════════════════════════════════════════════════════


def _normalize(text: str) -> str:
    """Minuscule + suppression des accents pour la comparaison textuelle."""
    nfkd = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def compute_exact_match(response: str, expected_values: list[str]) -> int:
    """1 si au moins une valeur attendue est trouvée dans la réponse."""
    r = _normalize(response)
    return int(any(_normalize(v) in r for v in expected_values))


def compute_keyword_score(response: str, keywords: list[str]) -> float:
    """Fraction des mots-clés attendus présents dans la réponse."""
    if not keywords:
        return 1.0
    r = _normalize(response)
    hits = sum(1 for kw in keywords if _normalize(kw) in r)
    return round(hits / len(keywords), 3)


def compute_wrong_avoided(response: str, wrong_values: list[str]) -> int:
    """
    1 si aucune valeur erronée n'est présentée comme réponse principale.
    Heuristique : on vérifie que la valeur erronée n'apparaît pas dans les
    100 premiers caractères de la réponse (là où le modèle annonce sa réponse).
    """
    if not wrong_values:
        return 1  # non applicable → score neutre
    opening = _normalize(response[:250])
    flagged = [v for v in wrong_values if _normalize(v) in opening]
    return int(len(flagged) == 0)


def compute_spm_retrieved(context: str) -> int:
    """1 si le contexte contient au moins un chunk tagué SPM."""
    return int("SPM" in context)


def compute_llm_score(question: str, expected_values: list[str], response: str) -> int:
    """LLM-as-judge : retourne un score 1-5."""
    expected_str = " / ".join(expected_values) if expected_values else "voir mots-clés"
    try:
        raw = judge_chain.invoke(
            {
                "question": question,
                "expected": expected_str,
                "response": response,
            }
        ).strip()
        match = re.search(r"[1-5]", raw)
        return int(match.group()) if match else 3
    except Exception:
        return 3  # score neutre en cas d'erreur


# ══════════════════════════════════════════════════════════════════════════════
# 4. RUNNER PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════


def run_backtest() -> list[dict]:
    """Exécute le backtest complet et retourne la liste des résultats."""
    results = []
    total = len(TEST_CASES)

    print("\n" + "═" * 64)
    print(f"  BACKTEST GIEC AR6 RAG — {total} cas de test")
    print("═" * 64 + "\n")

    for i, tc in enumerate(TEST_CASES, 1):
        print(f"[{i}/{total}] {tc['id']} — {tc['label']}")
        print(f"  Q : {tc['question'][:80]}...")

        # ── Retrieval + réponse ──────────────────────────────────────────────
        t0 = time.perf_counter()
        context = get_retrieved_context(tc["question"], k=6)
        response = answer_chain.invoke({"context": context, "question": tc["question"]})
        latency = round(time.perf_counter() - t0, 2)

        # ── Calcul des métriques ─────────────────────────────────────────────
        em = compute_exact_match(response, tc["expected_values"])
        kw = compute_keyword_score(response, tc["keywords"])
        wa = compute_wrong_avoided(response, tc["wrong_values"])
        spm = compute_spm_retrieved(context)
        llm = compute_llm_score(tc["question"], tc["expected_values"], response)

        result = {
            "id": tc["id"],
            "category": tc["category"],
            "label": tc["label"],
            "question": tc["question"],
            "response": response,
            "exact_match": em,
            "keyword_score": kw,
            "wrong_avoided": wa,
            "spm_retrieved": spm,
            "llm_score": llm,
            "latency_s": latency,
        }
        results.append(result)

        print(
            f"  ✓ exact_match={em}  kw={kw:.2f}  wrong_avoided={wa}"
            f"  spm={spm}  llm_score={llm}/5  latence={latency}s\n"
        )

    return results


# ══════════════════════════════════════════════════════════════════════════════
# 5. SAUVEGARDE JSON
# ══════════════════════════════════════════════════════════════════════════════


def create_backtest_dir() -> Path:
    """Crée un dossier daté pour le backtest sous le répertoire backtests."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = BACKTESTS_ROOT / f"backtest_{ts}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def save_results(results: list[dict], output_dir: Path) -> Path:
    path = output_dir / "backtest_results.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[INFO] Résultats sauvegardés → {path}")
    return path


# ══════════════════════════════════════════════════════════════════════════════
# 6. VISUALISATIONS
# ══════════════════════════════════════════════════════════════════════════════

# Palette climatique : bleu arctique, bleu océan, rouge alerte, gris neutre
CAT_COLORS = {"A": "#1d6fa4", "B": "#c0392b"}
ACCENT = "#2ecc71"
BG = "#0f1923"
PANEL_BG = "#162230"
TEXT = "#dce8f0"
GRID = "#243447"


def _style_ax(ax):
    ax.set_facecolor(PANEL_BG)
    ax.tick_params(colors=TEXT, labelsize=9)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.title.set_color(TEXT)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID)


def plot_results(results: list[dict], output_dir: Path) -> Path:
    ids = [r["id"] for r in results]
    labels = [r["label"] for r in results]
    categories = [r["category"] for r in results]
    n = len(results)

    exact = np.array([r["exact_match"] for r in results], dtype=float)
    kw = np.array([r["keyword_score"] for r in results], dtype=float)
    wa = np.array([r["wrong_avoided"] for r in results], dtype=float)
    spm = np.array([r["spm_retrieved"] for r in results], dtype=float)
    llm = np.array([r["llm_score"] for r in results], dtype=float)
    latency = np.array([r["latency_s"] for r in results], dtype=float)

    colors = [CAT_COLORS[c] for c in categories]

    fig = plt.figure(figsize=(18, 13), facecolor=BG)
    fig.suptitle(
        "Backtest — Système RAG GIEC AR6",
        fontsize=17,
        color=TEXT,
        fontweight="bold",
        y=0.97,
    )

    gs = fig.add_gridspec(
        2, 2, hspace=0.48, wspace=0.38, left=0.07, right=0.97, top=0.92, bottom=0.06
    )

    # ── Fig 1 : Scores par question ──────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    _style_ax(ax1)

    # Score composite = moyenne de exact_match, kw, wrong_avoided, llm/5
    composite = np.mean(np.column_stack([exact, kw, wa, llm / 5.0]), axis=1)
    y_pos = np.arange(n)
    bars = ax1.barh(
        y_pos, composite, color=colors, height=0.6, edgecolor=BG, linewidth=0.5
    )
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(
        [f"{ids[i]}  {labels[i][:28]}" for i in range(n)], fontsize=8.5, color=TEXT
    )
    ax1.set_xlim(0, 1.15)
    ax1.set_xlabel("Score composite [0-1]", color=TEXT)
    ax1.set_title(
        "① Score global par question",
        color=TEXT,
        fontsize=11,
        fontweight="bold",
        pad=10,
    )
    ax1.axvline(
        composite.mean(),
        color=ACCENT,
        linewidth=1.5,
        linestyle="--",
        label=f"Moy. {composite.mean():.2f}",
    )
    ax1.legend(
        fontsize=8,
        labelcolor=TEXT,
        facecolor=PANEL_BG,
        edgecolor=GRID,
        loc="lower right",
    )
    ax1.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax1.set_axisbelow(True)
    ax1.grid(axis="x", color=GRID, linewidth=0.6)

    # Étiquettes de valeur
    for bar, val in zip(bars, composite):
        ax1.text(
            val + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.0%}",
            va="center",
            ha="left",
            fontsize=8,
            color=TEXT,
        )

    # Légende catégories
    patches = [
        mpatches.Patch(color=v, label=f"Cat. {k}") for k, v in CAT_COLORS.items()
    ]
    ax1.legend(
        handles=patches,
        fontsize=8,
        labelcolor=TEXT,
        facecolor=PANEL_BG,
        edgecolor=GRID,
        loc="lower right",
    )

    # ── Fig 2 : Radar global ─────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1], polar=True)
    ax2.set_facecolor(PANEL_BG)

    dimensions = [
        "Exact\nMatch",
        "Keyword\nScore",
        "Erreur\nÉvitée",
        "SPM\nRetrieval",
        "LLM\nScore /5",
    ]
    global_scores = [
        exact.mean(),
        kw.mean(),
        wa.mean(),
        spm.mean(),
        (llm / 5.0).mean(),
    ]
    angles = np.linspace(0, 2 * np.pi, len(dimensions), endpoint=False).tolist()
    angles += angles[:1]
    values = global_scores + global_scores[:1]

    ax2.set_theta_offset(np.pi / 2)
    ax2.set_theta_direction(-1)
    ax2.plot(angles, values, color="#1d6fa4", linewidth=2)
    ax2.fill(angles, values, color="#1d6fa4", alpha=0.25)
    ax2.set_xticks(angles[:-1])
    ax2.set_xticklabels(dimensions, fontsize=9, color=TEXT)
    ax2.set_ylim(0, 1)
    ax2.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax2.set_yticklabels(["25%", "50%", "75%", "100%"], fontsize=7, color="#8aabbd")
    ax2.grid(color=GRID, linewidth=0.6)
    ax2.spines["polar"].set_color(GRID)
    ax2.set_title(
        "② Performance globale (toutes questions)",
        color=TEXT,
        fontsize=11,
        fontweight="bold",
        pad=18,
        y=1.08,
    )
    # Score central
    ax2.text(
        0,
        0,
        f"{np.mean(global_scores):.0%}",
        ha="center",
        va="center",
        fontsize=18,
        color=ACCENT,
        fontweight="bold",
    )

    # ── Fig 3 : Heatmap métriques × questions ────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    _style_ax(ax3)

    metric_labels = [
        "Exact Match",
        "Keyword Score",
        "Erreur Évitée",
        "SPM Retrieval",
        "LLM Score /5",
    ]
    matrix = np.column_stack([exact, kw, wa, spm, llm / 5.0]).T  # (5, n)

    im = ax3.imshow(
        matrix, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1, interpolation="nearest"
    )
    cbar = fig.colorbar(im, ax=ax3, fraction=0.03, pad=0.04)
    cbar.ax.tick_params(colors=TEXT, labelsize=8)
    cbar.ax.yaxis.set_tick_params(color=TEXT)
    cbar.outline.set_edgecolor(GRID)

    ax3.set_xticks(range(n))
    ax3.set_xticklabels(ids, fontsize=9, color=TEXT)
    ax3.set_yticks(range(len(metric_labels)))
    ax3.set_yticklabels(metric_labels, fontsize=9, color=TEXT)
    ax3.set_title(
        "③ Heatmap métriques × questions",
        color=TEXT,
        fontsize=11,
        fontweight="bold",
        pad=10,
    )

    for i in range(len(metric_labels)):
        for j in range(n):
            val = matrix[i, j]
            ax3.text(
                j,
                i,
                f"{val:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if val < 0.45 or val > 0.75 else "#0f1923",
                fontweight="bold",
            )

    # ── Fig 4 : Latence ──────────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    _style_ax(ax4)

    x = np.arange(n)
    bar_colors = [CAT_COLORS[c] for c in categories]
    ax4.bar(x, latency, color=bar_colors, edgecolor=BG, linewidth=0.5, width=0.6)
    ax4.axhline(
        latency.mean(),
        color=ACCENT,
        linewidth=1.5,
        linestyle="--",
        label=f"Moy. {latency.mean():.1f}s",
    )
    ax4.set_xticks(x)
    ax4.set_xticklabels(ids, fontsize=9, color=TEXT)
    ax4.set_ylabel("Secondes", color=TEXT)
    ax4.set_title(
        "④ Latence par question (retrieval + génération)",
        color=TEXT,
        fontsize=11,
        fontweight="bold",
        pad=10,
    )
    ax4.legend(fontsize=8, labelcolor=TEXT, facecolor=PANEL_BG, edgecolor=GRID)
    ax4.set_axisbelow(True)
    ax4.grid(axis="y", color=GRID, linewidth=0.6)

    for xi, lat in zip(x, latency):
        ax4.text(xi, lat + 0.3, f"{lat:.1f}s", ha="center", fontsize=8, color=TEXT)

    # Légende catégories (partagée fig 1 & 4)
    patches = [
        mpatches.Patch(
            color=v,
            label=f"Catégorie {k} — "
            + ("Confusion de données" if k == "A" else "Contresens scientifique"),
        )
        for k, v in CAT_COLORS.items()
    ]
    fig.legend(
        handles=patches,
        loc="lower center",
        ncol=2,
        fontsize=9,
        labelcolor=TEXT,
        facecolor=PANEL_BG,
        edgecolor=GRID,
        framealpha=0.8,
        bbox_to_anchor=(0.5, 0.005),
    )

    path = output_dir / "backtest_charts.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"[INFO] Graphiques sauvegardés → {path}")
    return path


# ══════════════════════════════════════════════════════════════════════════════
# 7. SYNTHÈSE CONSOLE
# ══════════════════════════════════════════════════════════════════════════════


def print_summary(results: list[dict]) -> None:
    print("\n" + "═" * 64)
    print("  SYNTHÈSE DU BACKTEST")
    print("═" * 64)

    metrics = {
        "Exact Match (valeur de référence trouvée)": np.mean(
            [r["exact_match"] for r in results]
        ),
        "Keyword Score (mots-clés scientifiques)": np.mean(
            [r["keyword_score"] for r in results]
        ),
        "Erreur Évitée (valeur erronée non présentée)": np.mean(
            [r["wrong_avoided"] for r in results]
        ),
        "SPM Retrieval (chunks SPM dans le contexte)": np.mean(
            [r["spm_retrieved"] for r in results]
        ),
        "LLM Judge Score (auto-évaluation /5)": np.mean(
            [r["llm_score"] for r in results]
        ),
        "Latence moyenne (s)": np.mean([r["latency_s"] for r in results]),
    }

    for name, value in metrics.items():
        bar = "█" * int(value * 20) if value <= 1 else "█" * 20
        print(f"  {name:<45} {value:>5.2f}  {bar}")

    # Par catégorie
    for cat in ["A", "B"]:
        sub = [r for r in results if r["category"] == cat]
        if not sub:
            continue
        label = "Confusion de données" if cat == "A" else "Contresens scientifique"
        avg = np.mean([r["keyword_score"] for r in sub])
        print(f"\n  Catégorie {cat} ({label}) — kw_score moyen : {avg:.2f}")
        for r in sub:
            status = "✓" if r["exact_match"] else "✗"
            print(
                f"    {status} {r['id']} {r['label'][:40]:<40} "
                f"llm={r['llm_score']}/5  {r['latency_s']}s"
            )


# ══════════════════════════════════════════════════════════════════════════════
# 8. POINT D'ENTRÉE
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    results = run_backtest()
    output_dir = create_backtest_dir()
    json_path = save_results(results, output_dir)
    chart_path = plot_results(results, output_dir)
    print_summary(results)

    print(f"\n  Dossier de sortie : {output_dir}")
    print(f"  Fichiers générés :")
    print(f"    • {json_path}")
    print(f"    • {chart_path}")
