import json
import os
import unicodedata
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from main import query_translation_chain
from vector import get_retrieved_context


def normalize(text: str) -> str:
    """Minuscule et suppression des accents pour des comparaisons robustes."""
    text = text.lower()
    return "".join(
        c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn"
    )


def main():
    console = Console()
    console.print(
        Panel("[bold green]Evaluation du RAG (KCR, KD)[/bold green]", expand=False)
    )

    eval_json_path = "eval.json"
    if not os.path.exists(eval_json_path):
        console.print(
            f"[bold red]Erreur : Le fichier {eval_json_path} est introuvable.[/bold red]"
        )
        return

    with open(eval_json_path, "r", encoding="utf-8") as f:
        cases = json.load(f)

    table = Table(
        title="Resultats de l'evaluation KCR & KD",
        show_header=True,
        header_style="bold blue",
    )
    table.add_column("ID", style="dim", width=4)
    table.add_column("Question", width=35)
    table.add_column("Mots-cles trouves", style="green", width=25)
    table.add_column("Mots-cles manquants", style="red", width=20)
    table.add_column("KCR", justify="right", style="bold", width=8)
    table.add_column("KD", justify="right", style="bold magenta", width=8)

    total_kcr = 0.0
    total_kd = 0.0

    for case in cases:
        qid = case.get("id", "?")
        question = case.get("question", "")
        keywords = case.get("keywords_chunking", [])

        console.print(
            f'\n[bold cyan]* [ID {qid}] Evaluation de la question :[/bold cyan] [italic]"{question}"[/italic]'
        )

        # 1. Traduction de la question
        console.print("  [dim]Traduction de la question...[/dim]")
        question_en = query_translation_chain.invoke({"question": question}).strip()
        console.print(f"  [dim]-> Traduction : {question_en}[/dim]")

        # 2. Récupération du contexte via le RAG
        console.print("  [dim]Recherche du contexte dans le rapport...[/dim]")
        context = get_retrieved_context(question_en)

        # 3. Calcul du KCR (Taux de Couverture des Mots-clés) dans le contexte
        ctx_norm = normalize(context)
        found_kws = []
        missing_kws = []

        for kw in keywords:
            if normalize(kw) in ctx_norm:
                found_kws.append(kw)
            else:
                missing_kws.append(kw)

        kcr = len(found_kws) / len(keywords) if keywords else 1.0
        total_kcr += kcr

        # 4. Calcul du KD (Densité des Mots-clés) dans le contexte
        # L = longueur (en nombre de mots) du bloc contexte
        words = ctx_norm.split()
        L = len(words)

        sum_kf = 0
        for kw in found_kws:
            kw_norm = normalize(kw)
            f_j = ctx_norm.count(kw_norm)  # fréquence du mot-clé j dans le contexte
            k_j = len(kw.split())  # nombre de mots constituant le mot-clé j
            sum_kf += k_j * f_j

        kd = (sum_kf / L) if L > 0 else 0.0
        total_kd += kd

        # Affichage du statut dans la console
        console.print(
            f"  [bold green]V {len(found_kws)}[/bold green] trouves, "
            f"[bold red]X {len(missing_kws)}[/bold red] manquants | "
            f"[bold yellow]KCR: {kcr:.1%}[/bold yellow] | "
            f"[bold magenta]KD: {kd:.2%}[/bold magenta]"
        )

        table.add_row(
            qid,
            question,
            ", ".join(found_kws),
            ", ".join(missing_kws),
            f"{kcr:.1%}",
            f"{kd:.2%}",
        )

    avg_kcr = total_kcr / len(cases) if cases else 0.0
    avg_kd = total_kd / len(cases) if cases else 0.0

    console.print("\n")
    console.print(table)

    color_kcr = "green" if avg_kcr >= 0.8 else ("yellow" if avg_kcr >= 0.5 else "red")
    color_kd = "green" if avg_kd >= 0.04 else ("yellow" if avg_kd >= 0.01 else "red")

    console.print(
        Panel(
            f"[bold {color_kcr}]Taux de Couverture des Mots-cles (KCR) Moyen : {avg_kcr:.2%}[/bold {color_kcr}]\n"
            f"[bold {color_kd}]Densite des Mots-cles (KD) Moyenne : {avg_kd:.2%}[/bold {color_kd}]",
            title="[bold]Score de l'evaluation globale[/bold]",
            expand=False,
        )
    )


if __name__ == "__main__":
    main()
