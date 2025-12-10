# CARGA Y CACHÈ DE DATOS

import pickle
import numpy as np
import pandas as pd
from scipy.signal import resample
import kagglehub
import os

PROCESSED_DIR = "data/processed"
FULL_CACHE = os.path.join(PROCESSED_DIR, "df_full.parquet")
REDUCED_CACHE = os.path.join(PROCESSED_DIR, "df_reduced.parquet")


# La misma lista de sujetos que usas en tu script
subjects = ["S2", "S3", "S4", "S5", "S6", "S7",
            "S8", "S9", "S10", "S11", "S13",
            "S14", "S15", "S16", "S17"]


def load_subject(subject):
    """
    MISMA lógica que tu función original: descarga el .pkl,
    re-muestrea a 700 Hz y devuelve un DataFrame por sujeto.
    """
    print(f"Procesando {subject} ...")

    path = kagglehub.dataset_download(
        "orvile/wesad-wearable-stress-affect-detection-dataset",
        f"WESAD/{subject}/{subject}.pkl"
    )

    with open(path, "rb") as f:
        data = pickle.load(f, encoding="latin1")

    # Señales del reloj
    acc  = np.array(data["signal"]["wrist"]["ACC"])            # Nx3 @32Hz
    bvp  = np.array(data["signal"]["wrist"]["BVP"]).squeeze()  # Nx1 @64Hz
    eda  = np.array(data["signal"]["wrist"]["EDA"]).squeeze()  # Nx1 @4Hz
    temp = np.array(data["signal"]["wrist"]["TEMP"]).squeeze() # Nx1 @4Hz

    labels = np.array(data["label"])  # 700Hz
    L = len(labels)

    # Re-sample todas las señales a 700Hz (misma lógica que ya tienes)
    acc_rs  = resample(acc,  L)
    bvp_rs  = resample(bvp,  L)
    eda_rs  = resample(eda,  L)
    temp_rs = resample(temp, L)

    # Solo estrés/no estrés
    stress_bin = np.where(labels == 2, 1, 0)

    df = pd.DataFrame({
        "subject": subject,
        "acc_x": acc_rs[:, 0],
        "acc_y": acc_rs[:, 1],
        "acc_z": acc_rs[:, 2],
        "bvp":   bvp_rs,
        "eda":   eda_rs,
        "temp":  temp_rs,
        "stress": stress_bin
    })

    return df


def window_reduce(df, hz=700, window_seconds=1):
    """
    Reduce el dataset a ventanas de tamaño fijo (en segundos):
    - Hace la media de las columnas numéricas por ventana
    - Usa max(stress) como etiqueta de la ventana (si hubo algún 1 → 1)
    - Conserva 'subject' (asigna el más frecuente en la ventana)
    """
    window = hz * window_seconds

    # Ajustar longitud para que sea múltiplo exacto de window
    total = len(df)
    usable = (total // window) * window
    df = df.iloc[:usable].copy()

    # Índice de grupo por ventana
    group_index = df.index // window

    # Columnas numéricas (features + stress)
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c != "stress"]

    # 1) Features numéricas: promedio por ventana
    features_windowed = df[feature_cols].groupby(group_index).mean()

    # 2) subject por ventana: el más frecuente o el primero
    subject_windowed = df["subject"].groupby(group_index).agg(
        lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0]
    )

    # 3) stress por ventana: 1 si en algún sample hubo estrés
    stress_windowed = df["stress"].groupby(group_index).max()

    # 4) Construir df_windowed final
    df_windowed = features_windowed.copy()
    df_windowed["subject"] = subject_windowed
    df_windowed["stress"] = stress_windowed

    # Orden de columnas (opcional)
    cols_order = ["subject"] + feature_cols + ["stress"]
    df_windowed = df_windowed[cols_order]

    return df_windowed



def load_all_subjects(subject_list=None):
    """
    MISMO bucle que ya tienes en proyect.py pero envuelto en función.
    Carga todos los sujetos, los concatena y devuelve df_full.
    """
    if subject_list is None:
        subject_list = subjects

    all_dfs = []

    for sbj in subject_list:
        try:
            df = load_subject(sbj)
            all_dfs.append(df)
        except Exception as e:
            print(f"Error en {sbj}: {e}")

    if not all_dfs:
        raise RuntimeError("No se pudo cargar ningún sujeto")

    df_full = pd.concat(all_dfs, ignore_index=True)

    print("\nDataset combinado listo!")
    print(df_full.shape)
    print(df_full["stress"].value_counts())

    return df_full


def build_full_and_reduced(subject_list=None, hz=700, window_seconds=1):
    """
    Carga datasets procesados desde cache si existen.
    Si no existen, procesa todo desde cero, guarda cache y devuelve los datos.
    """
    # 1. Si el cache ya existe, cargarlo rápido
    if os.path.exists(FULL_CACHE) and os.path.exists(REDUCED_CACHE):
        print(" Cargando dataset procesado desde cache…")
        df_full = pd.read_parquet(FULL_CACHE)
        df_reduced = pd.read_parquet(REDUCED_CACHE)
        print(" Cache cargado exitosamente.")
        return df_full, df_reduced

    # 2. Si no existe cache → construir normalmente
    print(" Cache no encontrado. Procesando dataset completo (esto tarda)…")
    df_full = load_all_subjects(subject_list=subject_list)
    df_reduced = window_reduce(df_full, hz=hz, window_seconds=window_seconds)

    # 3. Guardar cache
    print(" Guardando cache procesado…")
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    df_full.to_parquet(FULL_CACHE)
    df_reduced.to_parquet(REDUCED_CACHE)
    print(" Cache guardado exitosamente.")

    return df_full, df_reduced
