import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy import stats
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

# Carpeta donde se guardarán los gráficos de intervalos
RESULTS_DIR = os.path.join("img", "intervalos")


def save_interval_plot(fig, filename_prefix: str) -> str:
    """
    Guarda una figura de matplotlib en img/intervalos con un nombre único.
    Retorna la ruta del archivo guardado.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{filename_prefix}_{timestamp}.png"
    path = os.path.join(RESULTS_DIR, filename)

    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f" Gráfico guardado. Revisa: {path}")
    return path


# 1. IDENTIFICAR INTERVALOS DE ESTRÉS

def identificar_intervalos_estres(df: pd.DataFrame, min_duracion: int = 10):
    """
    Identifica intervalos de tiempo donde se presenta estrés en el dataframe.
    min_duracion está en número de muestras.
    """
    if "stress" not in df.columns:
        raise ValueError("El DataFrame debe contener la columna 'stress'")

    # Identificar cambios en el estado de estrés
    cambios = df["stress"].diff()
    inicios_estres = cambios[cambios == 1].index.tolist()
    fin_estres = cambios[cambios == -1].index.tolist()

    # Si el primer dato es estrés
    if len(df) > 0 and df.iloc[0]["stress"] == 1:
        inicios_estres.insert(0, 0)

    # Si termina en estrés
    if len(df) > 0 and df.iloc[-1]["stress"] == 1:
        fin_estres.append(len(df) - 1)

    # Crear intervalos
    intervalos = []
    for inicio, fin in zip(inicios_estres, fin_estres):
        duracion = fin - inicio
        if duracion >= min_duracion:
            intervalos.append((inicio, fin))

    return intervalos


# 2. VISUALIZAR UN INTERVALO DE ESTRÉS (SEÑALES + ANÁLISIS)

def visualizar_intervalo_unico(
    df: pd.DataFrame,
    intervalo_idx: int | None = None,
    intervalo_especifico: tuple[int, int] | None = None,
    ventana_contexto: int = 50,
):
    """
    Visualiza un intervalo de estrés concreto y realiza análisis estadístico.
    También guarda el gráfico en img/intervalos.
    """
    # Selección de intervalo
    if intervalo_especifico is not None:
        inicio, fin = intervalo_especifico
        intervalo_idx = 0
    else:
        intervalos = identificar_intervalos_estres(df, min_duracion=5)

        if len(intervalos) == 0:
            print("No se encontraron intervalos de estrés significativos.")
            return None

        if intervalo_idx is None:
            intervalo_idx = 0
            print(f"Usando el primer intervalo (índice {intervalo_idx})")

        if intervalo_idx >= len(intervalos):
            print(f"Índice {intervalo_idx} fuera de rango. Hay {len(intervalos)} intervalos.")
            return None

        inicio, fin = intervalos[intervalo_idx]

    print(f"\n{'=' * 70}")
    print(f"ANÁLISIS DETALLADO DEL INTERVALO {intervalo_idx + 1}")
    print(f"{'=' * 70}")
    print(f"Intervalo: muestras {inicio} a {fin} (duración: {fin - inicio} muestras)")

    # Contexto alrededor del intervalo
    inicio_contexto = max(0, inicio - ventana_contexto)
    fin_contexto = min(len(df), fin + ventana_contexto)
    intervalo_completo_df = df.iloc[inicio_contexto:fin_contexto].copy()

    tiempo_relativo = np.arange(len(intervalo_completo_df)) - (inicio - inicio_contexto)

    # ── FIGURA PRINCIPAL ────────────────────────────────────────
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 1, hspace=0.4)

    # 1) Acelerómetro
    ax1 = fig.add_subplot(gs[0])
    for i, col in enumerate(["acc_x", "acc_y", "acc_z"]):
        if col in intervalo_completo_df.columns:
            ax1.plot(
                tiempo_relativo,
                intervalo_completo_df[col],
                label=col.upper(),
                color=plt.cm.Set1(i),
                alpha=0.8,
                linewidth=1.5,
            )

    ax1.axvspan(0, fin - inicio, alpha=0.2, color="red", label="Estrés (stress=1)")
    ax1.axvline(x=0, color="red", linestyle="--", alpha=0.5)
    ax1.axvline(x=fin - inicio, color="red", linestyle="--", alpha=0.5)

    ax1.set_xlabel("Muestras relativas al inicio del estrés")
    ax1.set_ylabel("Aceleración")
    ax1.set_title("Comportamiento del Acelerómetro")
    ax1.legend(loc="upper right", ncol=4)
    ax1.grid(True, alpha=0.3)

    # 2) BVP y EDA
    ax2 = fig.add_subplot(gs[1])
    if "bvp" in intervalo_completo_df.columns:
        ax2.plot(
            tiempo_relativo,
            intervalo_completo_df["bvp"],
            label="BVP",
            color="purple",
            linewidth=2,
        )

    ax2b = ax2.twinx()
    if "eda" in intervalo_completo_df.columns:
        ax2b.plot(
            tiempo_relativo,
            intervalo_completo_df["eda"],
            label="EDA",
            color="green",
            linewidth=2,
            alpha=0.7,
        )

    ax2.axvspan(0, fin - inicio, alpha=0.2, color="red")
    ax2.set_xlabel("Muestras relativas al inicio del estrés")
    ax2.set_ylabel("BVP", color="purple")
    ax2b.set_ylabel("EDA", color="green")
    ax2.set_title("Volumen de Pulso Sanguíneo (BVP) y Actividad Electrodermal (EDA)")

    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2b.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    ax2.grid(True, alpha=0.3)

    # 3) Temperatura + stress
    ax3 = fig.add_subplot(gs[2])
    if "temp" in intervalo_completo_df.columns:
        ax3.plot(
            tiempo_relativo,
            intervalo_completo_df["temp"],
            label="TEMP",
            color="orange",
            linewidth=2,
        )

    if "stress" in intervalo_completo_df.columns:
        # Normalizar stress para visualizarlo
        stress_scaled = intervalo_completo_df["stress"] * (
            intervalo_completo_df["temp"].max() - intervalo_completo_df["temp"].min()
        ) * 0.3 + intervalo_completo_df["temp"].min()
        ax3.plot(
            tiempo_relativo,
            stress_scaled,
            label="STRESS (escalado)",
            color="red",
            linestyle="--",
            alpha=0.7,
        )

    ax3.axvspan(0, fin - inicio, alpha=0.2, color="red")
    ax3.set_xlabel("Muestras relativas al inicio del estrés")
    ax3.set_ylabel("Temperatura (y stress escalado)")
    ax3.set_title("Temperatura de la Piel y Evento de Estrés")
    ax3.legend(loc="upper right")
    ax3.grid(True, alpha=0.3)

    fig.suptitle(
        f"Intervalo de Estrés: muestras {inicio}–{fin} "
        f"(duración: {fin - inicio} muestras)",
        fontsize=16,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0.03, 1, 0.92])

    #  Guardar figura en vez de mostrarla
    save_interval_plot(fig, f"intervalo_{inicio}_{fin}")

    # ── ANÁLISIS ESTADÍSTICO ─────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("ANÁLISIS ESTADÍSTICO DETALLADO")
    print(f"{'=' * 70}")

    periodo_antes = intervalo_completo_df[tiempo_relativo < 0]
    periodo_durante = intervalo_completo_df[
        (tiempo_relativo >= 0) & (tiempo_relativo <= (fin - inicio))
    ]
    periodo_despues = intervalo_completo_df[tiempo_relativo > (fin - inicio)]

    periodos = [periodo_antes, periodo_durante, periodo_despues]
    nombres_periodos = ["Antes", "Durante", "Después"]

    print(f"\nDuración de cada período:")
    print(f"  Antes del estrés: {len(periodo_antes)} muestras")
    print(f"  Durante el estrés: {len(periodo_durante)} muestras")
    print(f"  Después del estrés: {len(periodo_despues)} muestras")

    print(f"\n{'=' * 70}")
    print("ESTADÍSTICAS DESCRIPTIVAS POR PERÍODO:")
    print(f"{'=' * 70}")

    params_comparacion = ["acc_x", "acc_y", "acc_z", "bvp", "eda", "temp"]

    for param in params_comparacion:
        if param in intervalo_completo_df.columns:
            print(f"\n{param.upper()}:")
            print(f"{'Período':<10} {'Media':<10} {'Std':<10} {'Min':<10} {'Max':<10}")
            print("-" * 50)

            for periodo, nombre in zip(periodos, nombres_periodos):
                if len(periodo) > 0:
                    datos = periodo[param].dropna()
                    if len(datos) > 0:
                        print(
                            f"{nombre:<10} {datos.mean():<10.2f} {datos.std():<10.2f} "
                            f"{datos.min():<10.2f} {datos.max():<10.2f}"
                        )

    print(f"\n{'=' * 70}")
    print("TESTS ESTADÍSTICOS (Durante vs Antes):")
    print(f"{'=' * 70}")

    for param in params_comparacion:
        if param in intervalo_completo_df.columns:
            datos_antes = periodo_antes[param].dropna()
            datos_durante = periodo_durante[param].dropna()

            if len(datos_antes) > 1 and len(datos_durante) > 1:
                t_stat, p_value = stats.ttest_ind(
                    datos_durante, datos_antes, equal_var=False
                )

                n1, n2 = len(datos_durante), len(datos_antes)
                sd_pooled = np.sqrt(
                    ((n1 - 1) * datos_durante.std() ** 2 +
                     (n2 - 1) * datos_antes.std() ** 2) / (n1 + n2 - 2)
                )
                cohens_d = (datos_durante.mean() - datos_antes.mean()) / sd_pooled

                if p_value < 0.001:
                    significancia = "*** (p < 0.001)"
                elif p_value < 0.01:
                    significancia = "** (p < 0.01)"
                elif p_value < 0.05:
                    significancia = "* (p < 0.05)"
                else:
                    significancia = "ns (no significativo)"

                if abs(cohens_d) < 0.2:
                    efecto = "Muy pequeño"
                elif abs(cohens_d) < 0.5:
                    efecto = "Pequeño"
                elif abs(cohens_d) < 0.8:
                    efecto = "Mediano"
                else:
                    efecto = "Grande"

                print(f"\n{param.upper()}:")
                print(f"  t = {t_stat:.3f}, p = {p_value:.4f} {significancia}")
                print(f"  Cohen's d = {cohens_d:.3f} ({efecto})")
                print(
                    f"  Cambio: {datos_durante.mean() - datos_antes.mean():+.2f} "
                    f"({(datos_durante.mean()/datos_antes.mean()-1)*100:+.1f}%)"
                )

    # Correlaciones durante el estrés
    if len(periodo_durante) > 10:
        print(f"\n{'=' * 70}")
        print("CORRELACIONES DURANTE EL ESTRÉS:")
        print(f"{'=' * 70}")

        params_corr = [p for p in params_comparacion if p in periodo_durante.columns]
        if len(params_corr) >= 2:
            corr_matrix = periodo_durante[params_corr].corr()

            print("\nMatriz de correlación:")
            print(" " * 10 + "".join([f"{p:>10}" for p in params_corr]))
            for i, p1 in enumerate(params_corr):
                row = f"{p1:10}"
                for j, p2 in enumerate(params_corr):
                    corr_val = corr_matrix.iloc[i, j]
                    row += f"{corr_val:10.2f}"
                print(row)

    return {
        "inicio": inicio,
        "fin": fin,
        "duracion": fin - inicio,
        "periodo_antes": periodo_antes,
        "periodo_durante": periodo_durante,
        "periodo_despues": periodo_despues,
    }


# 3. IMPORTANCIA RELATIVA (COHEN'S d) POR PARÁMETRO

def graficar_importancia_intervalo_unico(
    df: pd.DataFrame,
    inicio: int,
    fin: int,
    ventana_contexto: int = 50,
):
    """
    Calcula tamaños de efecto (Cohen's d) entre 'Antes' y 'Durante' para cada parámetro
    y los grafica como barras horizontales. Guarda el gráfico en img/intervalos.
    """
    inicio_contexto = max(0, inicio - ventana_contexto)
    fin_contexto = min(len(df), fin + ventana_contexto)
    intervalo_completo_df = df.iloc[inicio_contexto:fin_contexto].copy()

    tiempo_relativo = np.arange(len(intervalo_completo_df)) - (inicio - inicio_contexto)

    periodo_antes = intervalo_completo_df[tiempo_relativo < 0]
    periodo_durante = intervalo_completo_df[
        (tiempo_relativo >= 0) & (tiempo_relativo <= (fin - inicio))
    ]

    params_comparacion = ["acc_x", "acc_y", "acc_z", "bvp", "eda", "temp"]
    diferencias = []
    nombres = []

    for param in params_comparacion:
        if param in intervalo_completo_df.columns:
            datos_durante = periodo_durante[param].dropna()
            datos_antes = periodo_antes[param].dropna()

            if len(datos_durante) > 0 and len(datos_antes) > 0:
                mean_diff = datos_durante.mean() - datos_antes.mean()
                std_pooled = np.sqrt(
                    (datos_durante.std() ** 2 + datos_antes.std() ** 2) / 2
                )

                if std_pooled > 0:
                    d = abs(mean_diff / std_pooled)
                else:
                    d = 0

                diferencias.append(d)
                nombres.append(param.upper())

    fig, ax = plt.subplots(figsize=(12, 6))

    if diferencias:
        indices_ordenados = np.argsort(diferencias)[::-1]
        diferencias_ordenadas = [diferencias[i] for i in indices_ordenados]
        nombres_ordenados = [nombres[i] for i in indices_ordenados]

        colores = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(diferencias)))

        bars = ax.barh(nombres_ordenados, diferencias_ordenadas, color=colores, alpha=0.8)

        for i, (bar, d_val) in enumerate(zip(bars, diferencias_ordenadas)):
            ax.text(
                bar.get_width() * 1.02,
                bar.get_y() + bar.get_height() / 2,
                f"d = {d_val:.3f}",
                va="center",
                fontsize=10,
                fontweight="bold",
            )

            if d_val >= 0.8:
                efecto = "GRANDE"
                color_texto = "darkred"
            elif d_val >= 0.5:
                efecto = "MEDIANO"
                color_texto = "darkorange"
            elif d_val >= 0.2:
                efecto = "PEQUEÑO"
                color_texto = "darkgreen"
            else:
                efecto = "MUY PEQUEÑO"
                color_texto = "gray"

            ax.text(
                0.02,
                bar.get_y() + bar.get_height() / 2,
                efecto,
                va="center",
                fontsize=9,
                color=color_texto,
                fontweight="bold",
            )

        ax.set_xlabel("Tamaño del Efecto (Cohen's d)", fontsize=12, fontweight="bold")
        ax.set_title(
            "Importancia Relativa de Parámetros en el Intervalo de Estrés\n"
            "(Cambio estandarizado: Durante vs Antes)",
            fontsize=14,
            fontweight="bold",
            pad=15,
        )

        for d_val, label, color in [
            (0.2, "Pequeño", "green"),
            (0.5, "Mediano", "orange"),
            (0.8, "Grande", "red"),
        ]:
            ax.axvline(x=d_val, color=color, linestyle="--", alpha=0.5)
            ax.text(
                d_val + 0.05,
                len(diferencias) - 0.5,
                label,
                color=color,
                fontsize=9,
                fontweight="bold",
            )

        ax.grid(True, alpha=0.3, axis="x")

    #  Guardar gráfico en vez de mostrarlo
    save_interval_plot(fig, f"importancia_intervalo_{inicio}_{fin}")

    if diferencias:
        print(f"\n{'=' * 70}")
        print("RESUMEN DE IMPORTANCIA PARA ESTE INTERVALO:")
        print(f"{'=' * 70}")

        for nombre, d_val in zip(nombres_ordenados, diferencias_ordenadas):
            if d_val >= 0.8:
                importancia = "MUY IMPORTANTE"
            elif d_val >= 0.5:
                importancia = "IMPORTANTE"
            elif d_val >= 0.2:
                importancia = "MODERADAMENTE IMPORTANTE"
            else:
                importancia = "POCO IMPORTANTE"

            print(f"{nombre}: d = {d_val:.3f} → {importancia}")


# 4. FUNCIÓN PRINCIPAL PARA LLAMAR DESDE project.py


def analizar_principales_intervalos(
    df: pd.DataFrame,
    n_intervalos: int = 3,
    min_duracion: int = 5,
    ventana_contexto: int = 50,
):
    """
    Analiza los n_intervalos de estrés más largos:
      - imprime resumen,
      - grafica señales del intervalo,
      - grafica importancia relativa (Cohen's d),
      - guarda todos los gráficos en img/intervalos.
    """
    print("=" * 70)
    print(f"ANÁLISIS DE LOS {n_intervalos} INTERVALOS DE ESTRÉS MÁS LARGOS")
    print("=" * 70)

    if df is None or len(df) == 0:
        print("ERROR: DataFrame vacío o None.")
        return

    intervalos = identificar_intervalos_estres(df, min_duracion=min_duracion)

    if len(intervalos) == 0:
        print("No se encontraron intervalos de estrés.")
        return

    print(f"\nSe encontraron {len(intervalos)} intervalos de estrés:")

    intervalos_ordenados = sorted(intervalos, key=lambda x: x[1] - x[0], reverse=True)

    for i, (inicio, fin) in enumerate(intervalos_ordenados[:10]):
        duracion = fin - inicio
        if i < n_intervalos:
            print(
                f"  Intervalo {i + 1} (LARGO): muestras {inicio}-{fin} "
                f"(duración: {duracion} muestras)"
            )
        else:
            print(
                f"  Intervalo {i + 1}: muestras {inicio}-{fin} "
                f"(duración: {duracion} muestras)"
            )

    if len(intervalos) > 10:
        print(f"  ... y {len(intervalos) - 10} intervalos más")

    if len(intervalos_ordenados) < n_intervalos:
        print(f"\nNota: Solo hay {len(intervalos_ordenados)} intervalos disponibles.")
        intervalos_seleccionados = intervalos_ordenados
    else:
        intervalos_seleccionados = intervalos_ordenados[:n_intervalos]

    for idx, (inicio, fin) in enumerate(intervalos_seleccionados):
        print(f"\n{'=' * 70}")
        print(f"INTERVALO {idx + 1} de {len(intervalos_seleccionados)}")
        print(f"{'=' * 70}")

        visualizar_intervalo_unico(
            df,
            intervalo_especifico=(inicio, fin),
            ventana_contexto=ventana_contexto,
        )
        graficar_importancia_intervalo_unico(
            df,
            inicio=inicio,
            fin=fin,
            ventana_contexto=ventana_contexto,
        )

    print(f"\n Gráficos de intervalos guardados en '{RESULTS_DIR}'.")
