import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report, 
    confusion_matrix,
    f1_score, 
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score
)
from typing import Dict, Any, Tuple
import time


# MODELOS OPTIMIZADOS: ANTI-SOBREAJUSTE + MANEJO DE DESBALANCEO

def get_models_anti_overfit(quick_mode: bool = False) -> Dict[str, Any]:
    """
    Modelos optimizados para:
    1. Evitar sobreajuste (regularización agresiva)
    2. Manejar desbalanceo de clases (class_weight='balanced')
    
    Para tu dataset:
    - 88% clase 0, 12% clase 1 (ratio 7.78:1)
    - 75K train samples, 6 features
    - Sujetos S16 y S17 para test
    
    Args:
        quick_mode: Si True, solo 2 modelos rápidos para testing
    
    Returns:
        Dict con modelos configurados
    """
    
    if quick_mode:
        models = {
            'Logistic Regression L2': LogisticRegression(
                C=0.1,  # ← Menos regularización para evitar underfit
                penalty='l2',
                max_iter=2000,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            ),
            
            'Random Forest Simple': RandomForestClassifier(
                n_estimators=50,
                max_depth=3,  # ← MUY conservador para evitar overfit
                min_samples_split=100,
                min_samples_leaf=50,
                max_features=2,  # Solo 2 de 6 features por árbol
                class_weight='balanced',
                random_state=42,
                n_jobs=-1,
                max_samples=0.6  # Solo 60% de datos por árbol
            )
        }
    else:
        models = {
            # REGRESIÓN LOGÍSTICA - Varias configuraciones
            # Estos suelen ser los mejores para datos desbalanceados
            
            'Logistic Regression L2 (C=1.0)': LogisticRegression(
                C=1.0,  # Sin regularización (baseline)
                penalty='l2',
                max_iter=2000,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1,
                solver='lbfgs'
            ),
            
            'Logistic Regression L2 (C=0.1)': LogisticRegression(
                C=0.1,  # Regularización moderada
                penalty='l2',
                max_iter=2000,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1,
                solver='lbfgs'
            ),
            
            'Logistic Regression L2 (C=0.01)': LogisticRegression(
                C=0.01,  # Regularización fuerte
                penalty='l2',
                max_iter=2000,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1,
                solver='lbfgs'
            ),
            
            'Logistic Regression L1 (C=0.1)': LogisticRegression(
                C=0.1,
                penalty='l1',  # Hace selección de features
                max_iter=2000,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1,
                solver='saga'
            ),
            
            # RANDOM FOREST - Configuraciones conservadoras
            # Árboles más simples para evitar sobreajuste
            'Random Forest (depth=3)': RandomForestClassifier(
                n_estimators=50,
                max_depth=3,  # Muy conservador
                min_samples_split=100,
                min_samples_leaf=50,
                max_features='sqrt',
                class_weight='balanced',
                random_state=42,
                n_jobs=-1,
                max_samples=0.7  # Usar 70% de datos por árbol
            ),
            
            'Random Forest (depth=5)': RandomForestClassifier(
                n_estimators=50,
                max_depth=5,  # Moderado
                min_samples_split=50,
                min_samples_leaf=25,
                max_features='sqrt',
                class_weight='balanced',
                random_state=42,
                n_jobs=-1,
                max_samples=0.8
            ),
            
            'Random Forest (depth=7)': RandomForestClassifier(
                n_estimators=50,
                max_depth=7,  # Menos restrictivo
                min_samples_split=30,
                min_samples_leaf=15,
                max_features='sqrt',
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            ),
            
            # XGBOOST - Con scale_pos_weight para desbalanceo
            'XGBoost (Conservative)': XGBClassifier(
                n_estimators=50,
                max_depth=2,  # Muy conservador
                learning_rate=0.01,  # Aprende despacio
                subsample=0.7,
                colsample_bytree=0.7,
                reg_alpha=0.5,
                reg_lambda=2.0,
                min_child_weight=10,
                scale_pos_weight=7.78,  # ← Ratio 66734/8583
                random_state=42,
                use_label_encoder=False,
                eval_metric='logloss',
                n_jobs=-1
            ),
            
            'XGBoost (Moderate)': XGBClassifier(
                n_estimators=50,
                max_depth=3,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.3,
                reg_lambda=1.0,
                min_child_weight=5,
                scale_pos_weight=7.78,
                random_state=42,
                use_label_encoder=False,
                eval_metric='logloss',
                n_jobs=-1
            ),
            
            'XGBoost (Balanced)': XGBClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=0.5,
                scale_pos_weight=7.78,
                random_state=42,
                use_label_encoder=False,
                eval_metric='logloss',
                n_jobs=-1
            ),
            
            # DECISION TREE - Muy simple
            'Decision Tree (depth=4)': DecisionTreeClassifier(
                max_depth=4,
                min_samples_split=100,
                min_samples_leaf=50,
                max_features='sqrt',
                class_weight='balanced',
                random_state=42
            ),
            
            # SVM - Con class_weight
            
            'SVM Linear (C=0.1)': SVC(
                kernel='linear',
                C=0.1,
                class_weight='balanced',
                random_state=42,
                probability=True,
                max_iter=2000
            ),
            
            'SVM RBF (C=1.0)': SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                class_weight='balanced',
                random_state=42,
                probability=True,
                max_iter=2000
            ),
            
            # NAIVE BAYES -
            'Naive Bayes': GaussianNB()
        }
    
    return models


def get_models_requirements_anti_overfit() -> Dict[str, bool]:
    """
    Indica qué modelos REQUIEREN datos escalados.
    
    Returns:
        Dict con nombre_modelo: requiere_escalado
    """
    return {
        # Quick mode models
        'Logistic Regression L2': True,  # ← CRÍTICO: Necesita escalado
        'Random Forest Simple': False,
        
        # Full mode models - Logistic Regression - NECESITA escalado
        'Logistic Regression L2 (C=1.0)': True,
        'Logistic Regression L2 (C=0.1)': True,
        'Logistic Regression L2 (C=0.01)': True,
        'Logistic Regression L1 (C=0.1)': True,
        
        # Tree-based - NO necesita escalado
        'Random Forest (depth=3)': False,
        'Random Forest (depth=5)': False,
        'Random Forest (depth=7)': False,
        'XGBoost (Conservative)': False,
        'XGBoost (Moderate)': False,
        'XGBoost (Balanced)': False,
        'Decision Tree (depth=4)': False,
        
        # SVM y Naive Bayes - NECESITAN escalado
        'SVM Linear (C=0.1)': True,
        'SVM RBF (C=1.0)': True,
        'Naive Bayes': True
    }


# FUNCIONES DE ENTRENAMIENTO Y EVALUACIÓN

def train_model(
    model, 
    model_name: str,
    X_train: np.ndarray, 
    y_train: np.ndarray,
    verbose: bool = True
) -> Tuple[Any, float]:
    """
    Entrena un modelo y mide el tiempo.
    
    Args:
        model: Instancia del modelo
        model_name: Nombre del modelo
        X_train: Features de entrenamiento
        y_train: Target de entrenamiento
        verbose: Si mostrar información
    
    Returns:
        Tupla (modelo_entrenado, tiempo_entrenamiento)
    """
    if verbose:
        print(f"\n Entrenando {model_name}...")
    
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    if verbose:
        print(f"    Completado en {train_time:.2f}s")
    
    return model, train_time


def evaluate_model(
    model,
    model_name: str,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    train_time: float = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Evalúa un modelo con métricas completas.
    
    Args:
        model: Modelo entrenado
        model_name: Nombre del modelo
        X_train: Features de entrenamiento
        X_test: Features de test
        y_train: Target de entrenamiento
        y_test: Target de test
        train_time: Tiempo de entrenamiento
        verbose: Si mostrar información
    
    Returns:
        Dict con todas las métricas
    """
    if verbose:
        print(f"\n  Evaluando {model_name}...")
        print("-"*60)
    
    # Predicciones
    start_time = time.time()
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    predict_time = time.time() - start_time
    
    # Métricas básicas
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    
    train_f1 = f1_score(y_train, y_train_pred, zero_division=0)
    test_f1 = f1_score(y_test, y_test_pred, zero_division=0)
    
    test_precision = precision_score(y_test, y_test_pred, zero_division=0)
    test_recall = recall_score(y_test, y_test_pred, zero_division=0)
    
    # ROC-AUC
    test_roc_auc = None
    if hasattr(model, "predict_proba"):
        try:
            y_test_proba = model.predict_proba(X_test)[:, 1]
            test_roc_auc = roc_auc_score(y_test, y_test_proba)
        except:
            pass
    
    # Gap de sobreajuste
    gap = train_f1 - test_f1
    
    # Estado del sobreajuste
    if gap > 0.10:
    
        status_text = "ALTO"
    elif gap > 0.05:

        status_text = "MODERADO"
    elif gap < -0.05:

        status_text = "UNDERFIT"
    else:
        status_text = "BUENO"
    
    if verbose:
        print(f"   Train Acc:  {train_acc:.4f}")
        print(f"   Test Acc:   {test_acc:.4f}")
        print(f"   Train F1:   {train_f1:.4f}")
        print(f"   Test F1:    {test_f1:.4f}")
        print(f"   Gap:        {gap:+.4f}  {status_text}")
        print(f"   Precision:  {test_precision:.4f}")
        print(f"   Recall:     {test_recall:.4f}")
        if test_roc_auc:
            print(f"   ROC-AUC:    {test_roc_auc:.4f}")
    
    return {
        'Modelo': model_name,
        'Train Acc': train_acc,
        'Test Acc': test_acc,
        'Train F1': train_f1,
        'Test F1': test_f1,
        'Gap': gap,
        'Precision': test_precision,
        'Recall': test_recall,
        'ROC-AUC': test_roc_auc,
        'Train Time (s)': train_time,
        'Predict Time (s)': predict_time,
        'Status': status_text, 

    }


def train_and_evaluate_models_anti_overfit(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    X_train_scaled: np.ndarray = None,
    X_test_scaled: np.ndarray = None,
    quick_mode: bool = False,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Entrena y evalúa todos los modelos con configuración optimizada.
    
    Args:
        X_train: Features de entrenamiento (sin escalar)
        X_test: Features de test (sin escalar)
        y_train: Target de entrenamiento
        y_test: Target de test
        X_train_scaled: Features de entrenamiento escaladas
        X_test_scaled: Features de test escaladas
        quick_mode: Si True, solo 2 modelos rápidos
        verbose: Si mostrar información detallada
    
    Returns:
        DataFrame con resultados ordenados por Test F1
    """
    if verbose:
        print("\n" + "="*70)
        print(" ENTRENAMIENTO - MODELOS OPTIMIZADOS")
        print("="*70)
        print(f"   Modo: {'RÁPIDO (2 modelos)' if quick_mode else 'COMPLETO (14 modelos)'}")
        print(f"   Train samples: {len(y_train):,}")
        print(f"   Test samples:  {len(y_test):,}")
        print(f"   Features:      {X_train.shape[1]}")
        print("="*70)
    
    models = get_models_anti_overfit(quick_mode=quick_mode)
    requirements = get_models_requirements_anti_overfit()
    
    results = []
    
    for model_name, model in models.items():
        if verbose:
            print(f"\n{'='*70}")
            print(f" MODELO: {model_name}")
            print(f"{'='*70}")
        
        # Verificar si necesita escalado
        needs_scaling = requirements.get(model_name, False)
        
        if needs_scaling:
            if X_train_scaled is None or X_test_scaled is None:
                if verbose:
                    print(f"    {model_name} requiere datos escalados")
                    print(f"     Saltando modelo...")
                continue
            X_tr = X_train_scaled
            X_te = X_test_scaled
            if verbose:
                print("   Usando datos ESCALADOS")
        else:
            X_tr = X_train
            X_te = X_test
            if verbose:
                print("    Usando datos SIN ESCALAR")
        
        try:
            # Entrenar
            trained_model, train_time = train_model(
                model, model_name, X_tr, y_train, verbose=verbose
            )
            
            # Evaluar
            metrics = evaluate_model(
                trained_model, model_name, 
                X_tr, X_te, y_train, y_test,
                train_time=train_time,
                verbose=verbose
            )
            
            results.append(metrics)
            
        except Exception as e:
            if verbose:
                print(f"    ERROR en {model_name}: {str(e)}")
            continue
    
    # Crear DataFrame con resultados
    results_df = pd.DataFrame(results)
    
    # Ordenar por Test F1 (descendente)
    results_df = results_df.sort_values('Test F1', ascending=False)
    
    return results_df


def print_results_summary(results_df: pd.DataFrame):
    """
    Imprime resumen bonito de los resultados.
    
    Args:
        results_df: DataFrame con resultados de modelos
    """
    print("\n" + "="*70)
    print(" RESUMEN DE RESULTADOS")
    print("="*70)
    
    # Seleccionar columnas importantes
    cols = ['Modelo', 'Test F1', 'Test Acc', 'Gap', 'Precision', 'Recall', 'ROC-AUC', 'Status']
    summary = results_df[cols].copy()
    
    print(summary.to_string(index=False))
    
    # Mejor modelo
    print("\n" + "="*70)
    if len(results_df) > 0:
        best_model = results_df.iloc[0]
        print(f" MEJOR MODELO: {best_model['Modelo']}")
        print(f"   Test F1:     {best_model['Test F1']:.4f}")
        print(f"   Test Acc:    {best_model['Test Acc']:.4f}")
        print(f"   Gap:         {best_model['Gap']:+.4f}")
        print(f"   Precision:   {best_model['Precision']:.4f}")
        print(f"   Recall:      {best_model['Recall']:.4f}")
        if best_model['ROC-AUC']:
            print(f"   ROC-AUC:     {best_model['ROC-AUC']:.4f}")
        print(f"   Status:      {best_model['Status']}")
    print("="*70)
    
    # Estadísticas de sobreajuste
    print("\n ESTADÍSTICAS DE SOBREAJUSTE:")
    good = len(results_df[results_df['Gap'].abs() <= 0.05])
    moderate = len(results_df[(results_df['Gap'] > 0.05) & (results_df['Gap'] <= 0.10)])
    high = len(results_df[results_df['Gap'] > 0.10])
    underfit = len(results_df[results_df['Gap'] < -0.05])
    
    print(f"    Bueno (|Gap| ≤ 0.05):   {good} modelos")
    print(f"    Moderado (0.05-0.10):    {moderate} modelos")
    print(f"    Alto (Gap > 0.10):       {high} modelos")
    if underfit > 0:
        print(f"    Underfit (Gap < -0.05):  {underfit} modelos")
    print("="*70 + "\n")




def get_best_models(results_df: pd.DataFrame, n: int = 3) -> pd.DataFrame:
    """
    Retorna los N mejores modelos balanceando F1 y Gap.
    
    Args:
        results_df: DataFrame con resultados
        n: Número de modelos a retornar
    
    Returns:
        DataFrame con los mejores modelos
    """
    # Filtrar modelos con buen balance
    good_models = results_df[
        (results_df['Gap'].abs() <= 0.10) &  # Gap razonable
        (results_df['Test F1'] > 0.20)  # F1 mínimo aceptable
    ].copy()
    
    if len(good_models) == 0:
        print(" No hay modelos con Gap < 0.10 y F1 > 0.20")
        return results_df.head(n)
    
    # Ordenar por Test F1
    best = good_models.sort_values('Test F1', ascending=False).head(n)
    
    return best


def train_and_evaluate_cv(
    df_reduced: pd.DataFrame,
    models: Dict[str, Any] = None,
    n_splits: int = 5,
    scoring: str = 'f1',
    verbose: bool = True
) -> pd.DataFrame:
    """Validación cruzada para modelos anti-overfit"""
    from sklearn.model_selection import GroupKFold
    from pre_processing import preprocess_for_cross_validation
    
    # Preparar datos
    X, y, groups, _, _ = preprocess_for_cross_validation(df_reduced, n_splits)
    
    if models is None:
        models = get_models_anti_overfit()
    
    requirements = get_models_requirements_anti_overfit()
    gkf = GroupKFold(n_splits=n_splits)
    
    all_results = []
    
    for model_name, model in models.items():
        if verbose:
            print(f"\nModelo: {model_name}")
        
        fold_scores = []
        fold_train_scores = []
        
        for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups), 1):
            X_train_fold = X[train_idx]
            X_test_fold = X[test_idx]
            y_train_fold = y[train_idx]
            y_test_fold = y[test_idx]
            
            # Escalar si necesario
            if requirements.get(model_name, False):
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
                X_train_fold = scaler.fit_transform(X_train_fold)
                X_test_fold = scaler.transform(X_test_fold)
            
            # Entrenar y evaluar
            model.fit(X_train_fold, y_train_fold)
            
            y_train_pred = model.predict(X_train_fold)
            y_test_pred = model.predict(X_test_fold)
            
            train_score = f1_score(y_train_fold, y_train_pred)
            test_score = f1_score(y_test_fold, y_test_pred)
            
            fold_scores.append(test_score)
            fold_train_scores.append(train_score)
        
        mean_test = np.mean(fold_scores)
        mean_train = np.mean(fold_train_scores)
        gap = mean_train - mean_test
        
        all_results.append({
            'Modelo': model_name,
            'Test F1': mean_test,
            'Std': np.std(fold_scores),
            'Train F1': mean_train,
            'Gap': gap,
            'Status': '✅' if gap < 0.05 else 'X'
        })
    
    return pd.DataFrame(all_results).sort_values('Test F1', ascending=False)