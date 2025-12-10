import os
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore', category=UserWarning)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from xgboost import XGBClassifier

# Verificar disponibilidad de TabNet
TABNET_AVAILABLE = True
try:
    from pytorch_tabnet.tab_model import TabNetClassifier
    import torch
except ImportError:
    TABNET_AVAILABLE = False
    TabNetClassifier = None
    torch = None
    print("ADVERTENCIA: TabNet no esta instalado. Instala con: pip install pytorch-tabnet torch")

# Carpeta donde se guardaran los graficos de comparacion
RESULTS_DIR = os.path.join("img", "deep")


def _ensure_results_dir():
    os.makedirs(RESULTS_DIR, exist_ok=True)


def save_deep_plot(fig, filename_prefix: str = "deep_models") -> str:
    """
    Guarda una figura matplotlib en img/deep con un nombre unico.
    Retorna la ruta del archivo guardado.
    """
    _ensure_results_dir()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{filename_prefix}_{timestamp}.png"
    path = os.path.join(RESULTS_DIR, filename)

    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Grafico Deep Learning guardado. Revisa: {path}")
    return path


def evaluate_model(model, model_name, X_train, X_test, y_train, y_test):
    """
    Entrena y evalua un modelo tradicional de ML.
    Devuelve un diccionario con metricas y tiempos.
    """
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time

    start_time = time.time()
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    predict_time = time.time() - start_time

    results = {
        "Modelo": model_name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1-Score": f1_score(y_test, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None,
        "Train Time (s)": train_time,
        "Predict Time (s)": predict_time,
        "y_pred": y_pred,
        "y_pred_proba": y_pred_proba,
        "model_obj": model,
    }
    return results


def run_deep_experiments(
    X_train,
    X_test,
    y_train,
    y_test,
    save_plots: bool = True,
):
    """
    Ejecuta el estudio comparativo:
    - Modelos tradicionales: Random Forest, XGBoost
    - Modelo Deep Learning: TabNet
    Usando los datos YA preprocesados (X_train, X_test, y_train, y_test).

    Devuelve:
        results_df (DataFrame con metricas)
    """

    # Aseguramos numpy arrays (por si vienen como DataFrame/Series)
    X_train = np.asarray(X_train)
    X_test = np.asarray(X_test)
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)

    print("\n" + "=" * 80)
    print("ENTRENANDO MODELOS TRADICIONALES (Random Forest, XGBoost)")
    print("=" * 80)

    traditional_models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=50,
            max_depth=10,
            min_samples_split=20,
            min_samples_leaf=10,
            max_features='sqrt',
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        ),

        "XGBoost": XGBClassifier(
            n_estimators=50,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.3,
            reg_lambda=1.0,
            scale_pos_weight=7.78,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss',
        ),
    }

    traditional_results = []
    for name, model in traditional_models.items():
        print(f"\nEvaluando {name}...")
        results = evaluate_model(model, name, X_train, X_test, y_train, y_test)
        traditional_results.append(results)

    # Modelo Deep: TabNet
    print("\n" + "=" * 80)
    print("ENTRENANDO MODELO DEEP LEARNING: TabNet")
    print("=" * 80)

    # Verificar disponibilidad de TabNet
    if not TABNET_AVAILABLE:
        print("ADVERTENCIA: TabNet no esta instalado. Solo se usaron modelos tradicionales.")
        print("Instala con: pip install pytorch-tabnet torch")
        
        all_results = traditional_results
        results_df = pd.DataFrame(
            [
                {
                    k: v
                    for k, v in r.items()
                    if k not in ["y_pred", "y_pred_proba", "model_obj"]
                }
                for r in all_results
            ]
        )
        
        print("\n" + "=" * 80)
        print("COMPARACION DE MODELOS - CLASIFICACION DE ESTRES (WESAD)")
        print("=" * 80)
        print(results_df.to_string(index=False))
        
        return results_df

    # Determinar device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Usando device: {device}")

    # Crear modelo TabNet
    tabnet_model = TabNetClassifier(
        n_d=64,
        n_a=64,
        n_steps=5,
        gamma=1.5,
        lambda_sparse=1e-3,
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=2e-2),
        mask_type="entmax",
        scheduler_params={"step_size": 10, "gamma": 0.9},
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
        verbose=0,
        device_name=device
    )

    start_time = time.time()
    tabnet_model.fit(
        X_train=X_train,
        y_train=y_train,
        eval_set=[(X_test, y_test)],
        eval_name=["test"],
        eval_metric=["auc"],
        max_epochs=50,
        patience=10,
        batch_size=1024,
        virtual_batch_size=128,
        num_workers=0,
        drop_last=False,
    )
    tabnet_train_time = time.time() - start_time

    start_time = time.time()
    y_pred_tabnet = tabnet_model.predict(X_test)
    y_pred_proba_tabnet = tabnet_model.predict_proba(X_test)[:, 1]
    tabnet_predict_time = time.time() - start_time

    tabnet_results = {
        "Modelo": "TabNet (Deep Learning)",
        "Accuracy": accuracy_score(y_test, y_pred_tabnet),
        "Precision": precision_score(y_test, y_pred_tabnet, zero_division=0),
        "Recall": recall_score(y_test, y_pred_tabnet, zero_division=0),
        "F1-Score": f1_score(y_test, y_pred_tabnet, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, y_pred_proba_tabnet),
        "Train Time (s)": tabnet_train_time,
        "Predict Time (s)": tabnet_predict_time,
        "y_pred": y_pred_tabnet,
        "y_pred_proba": y_pred_proba_tabnet,
        "model_obj": tabnet_model,
    }

    # COMPARAR RESULTADOS
    all_results = traditional_results + [tabnet_results]
    results_df = pd.DataFrame(
        [
            {
                k: v
                for k, v in r.items()
                if k not in ["y_pred", "y_pred_proba", "model_obj"]
            }
            for r in all_results
        ]
    )

    print("\n" + "=" * 80)
    print("COMPARACION DE MODELOS - CLASIFICACION DE ESTRES (WESAD)")
    print("=" * 80)
    print(results_df.to_string(index=False))

    print("\n" + "=" * 80)
    print("ANALISIS COMPARATIVO")
    print("=" * 80)

    for metric in ["Accuracy", "F1-Score", "ROC-AUC"]:
        if metric in results_df.columns and results_df[metric].notna().any():
            best_model_row = results_df.loc[results_df[metric].idxmax()]
            print(f"Mejor {metric}: {best_model_row['Modelo']} ({best_model_row[metric]:.4f})")

    print(
        f"\nModelo mas rapido en entrenamiento: "
        f"{results_df.loc[results_df['Train Time (s)'].idxmin()]['Modelo']}"
    )
    print(
        f"Modelo mas rapido en prediccion: "
        f"{results_df.loc[results_df['Predict Time (s)'].idxmin()]['Modelo']}"
    )

    # VISUALIZACION (4 subplots)
    if save_plots:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        metrics_to_plot = ["Accuracy", "F1-Score", "ROC-AUC"]
        x = np.arange(len(metrics_to_plot))
        width = 0.25

        # 1) Comparacion de metricas
        for i, (_, row) in enumerate(results_df.iterrows()):
            values = [
                row[m] if (m in results_df.columns and pd.notna(row[m])) else 0
                for m in metrics_to_plot
            ]
            axes[0, 0].bar(x + i * width - width, values, width, label=row["Modelo"])

        axes[0, 0].set_ylabel("Score")
        axes[0, 0].set_title("Comparacion de Metricas por Modelo")
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(metrics_to_plot)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 2) Tiempos de entrenamiento
        axes[0, 1].bar(
            results_df["Modelo"],
            results_df["Train Time (s)"],
            color="skyblue",
        )
        axes[0, 1].set_ylabel("Segundos")
        axes[0, 1].set_title("Tiempo de Entrenamiento")
        axes[0, 1].tick_params(axis="x", rotation=45)
        axes[0, 1].grid(True, alpha=0.3)

        # 3) Tiempos de prediccion
        axes[1, 0].bar(
            results_df["Modelo"],
            results_df["Predict Time (s)"],
            color="lightcoral",
        )
        axes[1, 0].set_ylabel("Segundos")
        axes[1, 0].set_title("Tiempo de Prediccion")
        axes[1, 0].tick_params(axis="x", rotation=45)
        axes[1, 0].grid(True, alpha=0.3)

        # 4) Matriz de confusion del mejor modelo (por F1-Score)
        best_model_idx = results_df["F1-Score"].idxmax()
        best_model_name = results_df.loc[best_model_idx, "Modelo"]

        # Recuperar predicciones del mejor modelo
        best_result = next(r for r in all_results if r["Modelo"] == best_model_name)
        y_pred_best = best_result["y_pred"]

        cm = confusion_matrix(y_test, y_pred_best)
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["No Estres", "Estres"],
        )
        disp.plot(ax=axes[1, 1], cmap="Blues")
        axes[1, 1].set_title(f"Matriz de Confusion - {best_model_name}")

        plt.tight_layout()

        # Guardar en img/deep
        save_deep_plot(fig, filename_prefix="comparacion_deep_models")

    return results_df