import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List, Dict, Any


def preprocess_reduced_by_subject(
    df_reduced: pd.DataFrame,
    test_subjects: List[str] = None,
) -> Tuple:
    """
    Preprocesamiento SIN data leakage: split → impute → scale
    
    Args:
        df_reduced: DataFrame con columnas ['subject', 'stress', features...]
        test_subjects: Lista de sujetos para test (default: ["S16", "S17"])
    
    Returns:
        tuple: (df_reduced_imputed, X_train, X_test, y_train, y_test,
                scaler, X_train_scaled, X_test_scaled, imputer)
    """
    if test_subjects is None:
        test_subjects = ["S16", "S17"]

    print("="*70)
    print("PREPROCESAMIENTO SIN DATA LEAKAGE")
    print("="*70)

    # PASO 1: SPLIT POR SUBJECT PRIMERO
    print("\n[1/4] Split por sujeto...")
    
    if "subject" not in df_reduced.columns:
        raise ValueError(
            "df_reduced no tiene columna 'subject'. "
            "Verifica que window_reduce() en wesad_data.py la conserve."
        )

    # Máscaras para split
    test_mask = df_reduced["subject"].isin(test_subjects)
    train_mask = ~test_mask

    df_train = df_reduced[train_mask].copy()
    df_test = df_reduced[test_mask].copy()

    print(f"  Test subjects: {test_subjects}")
    print(f"  Train subjects: {sorted(df_train['subject'].unique().tolist())}")
    print(f"  Train: {df_train.shape[0]:,} muestras")
    print(f"  Test:  {df_test.shape[0]:,} muestras")
    
    # Distribución de clases
    print(f"  Train stress: {dict(df_train['stress'].value_counts())}")
    print(f"  Test stress:  {dict(df_test['stress'].value_counts())}")

    
    # PASO 2: IMPUTACIÓN - FIT con TRAIN, TRANSFORM en TEST
    print("\n[2/4] Imputación...")
    
    # Identificar columnas
    numeric_cols = df_train.select_dtypes(include="number").columns.tolist()
    other_cols = [c for c in df_train.columns if c not in numeric_cols]
    
    # Separar
    df_train_num = df_train[numeric_cols]
    df_train_other = df_train[other_cols]
    df_test_num = df_test[numeric_cols]
    df_test_other = df_test[other_cols]
    
    # Verificar nulos
    train_nulls = df_train_num.isnull().sum().sum()
    test_nulls = df_test_num.isnull().sum().sum()
    
    print(f"  Nulos en train: {train_nulls}")
    print(f"  Nulos en test:  {test_nulls}")
    
    imputer = None
    
    if train_nulls > 0 or test_nulls > 0:
        imputer = SimpleImputer(strategy="mean")
        
        #  FIT solo con TRAIN
        df_train_num_imputed = pd.DataFrame(
            imputer.fit_transform(df_train_num),
            columns=numeric_cols,
            index=df_train_num.index,
        )
        
        #  TRANSFORM en TEST
        df_test_num_imputed = pd.DataFrame(
            imputer.transform(df_test_num),
            columns=numeric_cols,
            index=df_test_num.index,
        )
        
        print("  ✓ Imputación completada (fit: train, transform: test)")
    else:
        df_train_num_imputed = df_train_num.copy()
        df_test_num_imputed = df_test_num.copy()
        print("  ✓ No hay nulos")

    # Reconstituir
    df_train_imputed = pd.concat([df_train_other, df_train_num_imputed], axis=1)
    df_train_imputed = df_train_imputed[df_train.columns]
    
    df_test_imputed = pd.concat([df_test_other, df_test_num_imputed], axis=1)
    df_test_imputed = df_test_imputed[df_test.columns]

    # PASO 3: Preparar X, y
    
    print("\n[3/4] Preparando features y target...")
    
    X_train = df_train_imputed.drop(["stress", "subject"], axis=1)
    y_train = df_train_imputed["stress"]
    
    X_test = df_test_imputed.drop(["stress", "subject"], axis=1)
    y_test = df_test_imputed["stress"]
    
    print(f"  X_train: {X_train.shape}")
    print(f"  X_test:  {X_test.shape}")
    print(f"  Features: {list(X_train.columns)}")

    # PASO 4: ESCALADO - FIT con TRAIN
    print("\n[4/4] Escalado...")
    
    scaler = StandardScaler()
    
    #  FIT solo con TRAIN
    X_train_scaled = scaler.fit_transform(X_train)
    #  TRANSFORM en TEST
    X_test_scaled = scaler.transform(X_test)
    
    print(f"  ✓ StandardScaler aplicado (fit: train, transform: test)")
    print(f"  X_train_scaled: {X_train_scaled.shape}")
    print(f"  X_test_scaled:  {X_test_scaled.shape}")

    print("\n" + "="*70)
    print("✓ PREPROCESAMIENTO COMPLETADO")
    print("="*70 + "\n")
    
    # Combinar para retornar
    df_reduced_imputed = pd.concat([df_train_imputed, df_test_imputed])
    df_reduced_imputed = df_reduced_imputed.sort_index()

    return (
        df_reduced_imputed,
        X_train, 
        X_test, 
        y_train, 
        y_test,
        scaler, 
        X_train_scaled, 
        X_test_scaled,
        imputer
    )


def preprocess_for_cross_validation(
    df_reduced: pd.DataFrame,
    n_splits: int = 5
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler, SimpleImputer]:
    """
    Preprocesamiento para validación cruzada por sujeto.
    NO hace split, solo prepara los datos para usar con GroupKFold.
    
    Args:
        df_reduced: DataFrame completo con todos los sujetos
        n_splits: Número de folds para GroupKFold
    
    Returns:
        tuple: (X_scaled, y, groups, scaler, imputer)
    """
    print("="*70)
    print("PREPROCESAMIENTO PARA VALIDACIÓN CRUZADA")
    print("="*70)
    
    if "subject" not in df_reduced.columns:
        raise ValueError("df_reduced debe tener columna 'subject'")
    
    print(f"\n[1/3] Preparando datos...")
    print(f"  Total muestras: {len(df_reduced):,}")
    print(f"  Sujetos únicos: {df_reduced['subject'].nunique()}")
    print(f"  Splits a usar: {n_splits}")
    
    # PASO 1: Imputación (si es necesario)

    print(f"\n[2/3] Imputación...")
    
    numeric_cols = df_reduced.select_dtypes(include="number").columns.tolist()
    other_cols = [c for c in df_reduced.columns if c not in numeric_cols]
    
    df_num = df_reduced[numeric_cols]
    df_other = df_reduced[other_cols]
    
    nulls = df_num.isnull().sum().sum()
    
    imputer = None
    
    if nulls > 0:
        print(f"  Nulos encontrados: {nulls}")
        imputer = SimpleImputer(strategy="mean")
        df_num_imputed = pd.DataFrame(
            imputer.fit_transform(df_num),
            columns=numeric_cols,
            index=df_num.index,
        )
        print("  Imputación aplicada")
    else:
        df_num_imputed = df_num.copy()
        print("   No hay nulos")
    
    df_imputed = pd.concat([df_other, df_num_imputed], axis=1)
    df_imputed = df_imputed[df_reduced.columns]
    
    
    # PASO 2: Separar X, y, groups
    print(f"\n[3/3] Preparando X, y, groups...")
    
    X = df_imputed.drop(["stress", "subject"], axis=1)
    y = df_imputed["stress"]
    groups = df_imputed["subject"]
    
    print(f"  X: {X.shape}")
    print(f"  y: {y.shape}")
    print(f"  groups: {len(groups.unique())} sujetos únicos")
    
   
    # PASO 3: Escalado (se aplicará dentro de cada fold)
    # IMPORTANTE: Para validación cruzada, el escalado se hace 
    # DENTRO de cada fold, no aquí
    scaler = StandardScaler()
    
    print("\n" + "="*70)
    print(" DATOS PREPARADOS PARA VALIDACIÓN CRUZADA")
    print("="*70)
    print("\nNOTA: El escalado se aplicará dentro de cada fold")
    print("para evitar data leakage entre folds.\n")
    
    return X.values, y.values, groups.values, scaler, imputer


def verify_no_leakage(df_reduced: pd.DataFrame, test_subjects: List[str]) -> bool:
    """Verifica que train y test están completamente separados."""
    if "subject" not in df_reduced.columns:
        print(" No se puede verificar: falta columna 'subject'")
        return False
    
    all_subjects = set(df_reduced["subject"].unique())
    test_set = set(test_subjects)
    train_set = all_subjects - test_set
    overlap = train_set & test_set
    
    print(f"\nVerificación de Data Leakage:")
    print(f"  Todos: {sorted(all_subjects)}")
    print(f"  Train: {sorted(train_set)}")
    print(f"  Test:  {sorted(test_set)}")
    print(f"  Overlap: {overlap}")
    
    if len(overlap) == 0:
        print("  SIN LEAKAGE")
        return True
    else:
        print(f"   LEAKAGE: {overlap}")
        return False