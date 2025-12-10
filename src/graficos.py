# Creacion de Graficos

# 1. Comparacion de Rendimiento entre modelos usando df reduced
import os
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier


import numpy as np

#Guardar gràficos en ruta para visualizaciòn
RESULTS_DIR = "./img/modelos"


def _ensure_results_dir():
    os.makedirs(RESULTS_DIR, exist_ok=True)


def save_plot(fig, filename_prefix: str = "grafico") -> str:
    """
    Guarda una figura matplotlib en img con un nombre único.
    Retorna la ruta del archivo guardado.
    """
    _ensure_results_dir()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{filename_prefix}_{timestamp}.png"
    path = os.path.join(RESULTS_DIR, filename)

    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return path

def plot_metrics_df_reduced() -> str:

    model_names = ['Logistic Regression', 'Decision Tree', 'KNN', 'Naive Bayes', 'Random Forest']
    accuracy = [0.8882, 0.9896, 0.9512, 0.8700, 0.9846]
    precision = [0.6202, 0.9552, 0.8023, 0.3893, 0.9702]
    recall = [0.0698, 0.9542, 0.7632, 0.2315, 0.8931]
    f1 = [0.1255, 0.9547, 0.7823, 0.2903, 0.9301]

    x = np.arange(len(model_names))
    width = 0.2

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - 1.5*width, accuracy, width, label='Accuracy', color='skyblue')
    bars2 = ax.bar(x - 0.5*width, precision, width, label='Precision', color='lightgreen')
    bars3 = ax.bar(x + 0.5*width, recall, width, label='Recall', color='salmon')
    bars4 = ax.bar(x + 1.5*width, f1, width, label='F1-Score', color='gold')

    ax.set_xlabel('Modelos', fontsize=12)
    ax.set_ylabel('Puntuación', fontsize=12)
    ax.set_title('Comparación de Rendimiento entre Modelos (df_reduced)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.legend(loc='lower right')
    ax.grid(axis='y', alpha=0.3)

    # Añadir valores en las barras
    for bars in [bars1, bars2, bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    path = save_plot(fig, filename_prefix="metricas_df_reduced")
    return path


#2. Grafico de AUC ROC, evaluacion de los modelos


def plot_roc_models_df_reduced(
    X_train,
    X_test,
    y_train,
    y_test,
    use_scaled: bool = True,
) -> str:
    """
    Entrena los modelos clásicos sobre df_reduced (X_train, X_test, y_train, y_test),
    calcula las curvas ROC y guarda la figura en PNG.

    Asume que X_train / X_test ya vienen escalados si use_scaled=True.
    Devuelve la ruta del archivo PNG.
    """

    # Mismos modelos que usabas
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Naive Bayes": GaussianNB(),
        "Random Forest": RandomForestClassifier(
            n_estimators=10, max_depth=10, random_state=42
        ),
    }

    models_roc = {}
    print("Entrenando modelos y calculando curvas ROC...")

    # Entrenar cada modelo y calcular ROC
    for model_name, model in models.items():
        print(f"  Procesando {model_name}...")

        model.fit(X_train, y_train)

        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        else:
            # Por seguridad, aunque en tu caso todos tienen predict_proba
            scores = model.decision_function(X_test)
            y_prob = (scores - scores.min()) / (scores.max() - scores.min())

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)

        models_roc[model_name] = {
            "fpr": fpr,
            "tpr": tpr,
            "roc_auc": roc_auc,
            "y_prob": y_prob,
        }

        print(f"    AUC para {model_name}: {roc_auc:.4f}")

    # Crear la figura ROC
    fig, ax = plt.subplots(figsize=(12, 10))

    # Línea aleatoria
    ax.plot(
        [0, 1],
        [0, 1],
        "k--",
        linewidth=2,
        label="Clasificador Aleatorio (AUC = 0.50)",
    )

    colors = {
        "Logistic Regression": "blue",
        "Decision Tree": "green",
        "KNN": "red",
        "Naive Bayes": "purple",
        "Random Forest": "orange",
    }

    # Graficar cada curva
    for model_name, roc_data in models_roc.items():
        ax.plot(
            roc_data["fpr"],
            roc_data["tpr"],
            color=colors.get(model_name, "gray"),
            linewidth=3,
            label=f'{model_name} (AUC = {roc_data["roc_auc"]:.4f})',
        )

    ax.set_xlabel("Tasa de Falsos Positivos (FPR)", fontsize=14)
    ax.set_ylabel("Tasa de Verdaderos Positivos (TPR)", fontsize=14)
    ax.set_title(
        "Curvas ROC - Comparación de Todos los Modelos (df_reduced)",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(loc="lower right", fontsize=12, frameon=True, fancybox=True, shadow=True)
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])

    ax.text(
        0.6,
        0.05,
        f"Número de muestras: {len(X_test):,}",
        transform=ax.transAxes,
        fontsize=10,
        bbox=dict(
            boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8
        ),
    )

    # Mejor modelo
    best_model_name, best_roc_data = max(
        models_roc.items(), key=lambda x: x[1]["roc_auc"]
    )
    ax.fill_between(
        best_roc_data["fpr"],
        best_roc_data["tpr"],
        alpha=0.2,
        color=colors.get(best_model_name, "blue"),
        label=f"Área bajo curva de {best_model_name}",
    )

    plt.tight_layout()

    path = save_plot(fig, filename_prefix="roc_df_reduced")
    return path
