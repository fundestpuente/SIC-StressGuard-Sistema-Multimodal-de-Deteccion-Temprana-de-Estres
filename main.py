from wesad_data import build_full_and_reduced
from src.pre_processing import preprocess_reduced_by_subject, verify_no_leakage
from src.modelado_regularizado import train_and_evaluate_models_anti_overfit, print_results_summary
from src.graficos import plot_metrics_df_reduced, plot_roc_models_df_reduced
from src.intervalos import analizar_principales_intervalos
from Deep_Learning.deep_models import run_deep_experiments

#Sujetos para test
TEST_SUBJECTS = ["S16", "S17"]

# Modo rápido (para pruebas) - usa menos modelos y menos iteraciones
QUICK_MODE = False  # Cambiar a True para pruebas rápidas
USE_CROSS_VALIDATION = False

print("\n" + "="*70)
print("WESAD - DETECCIÓN DE ESTRÉS")
print("="*70)


print("\n[PASO 1/6] Cargando datos...")
df_full, df_reduced = build_full_and_reduced()

print(f"\n  Dataset completo: {df_full.shape}")
print(f"  Dataset reducido: {df_reduced.shape}")
print(f"  Sujetos: {sorted(df_reduced['subject'].unique())}")
print(f"  Distribución stress: {dict(df_reduced['stress'].value_counts())}")

print("\n[PASO 2/6] Preprocesamiento...")

# Verificar que no hay data leakage
verify_no_leakage(df_reduced, TEST_SUBJECTS)

# Preprocesar
(
    df_reduced_imputed,
    X_train, X_test, y_train, y_test,
    scaler, X_train_scaled, X_test_scaled,
    imputer
) = preprocess_reduced_by_subject(
    df_reduced,
    test_subjects=TEST_SUBJECTS
)

print("\n[PASO 3/6] Entrenamiento y evaluación de modelos...")

# Modo 1: Evaluación Simple
if not USE_CROSS_VALIDATION:
    results_df = train_and_evaluate_models_anti_overfit( 
        X_train=X_train.values,
        X_test=X_test.values,
        y_train=y_train.values,
        y_test=y_test.values,
        X_train_scaled=X_train_scaled,
        X_test_scaled=X_test_scaled,
        quick_mode=QUICK_MODE,
        verbose=True
    )
    
    # Mostrar resumen mejorado
    print_results_summary(results_df)
    

    
    
# Modo 2: Validación Cruzada
else:
    # Importar los nuevos modelos
    from src.modelado_regularizado import (
        get_models_anti_overfit,
        get_models_requirements_anti_overfit
    )
    
    # Obtener modelos anti-sobreajuste
    models_anti_overfit = get_models_anti_overfit(quick_mode=QUICK_MODE)
    
   
    
    
print("\n[PASO 4/6] Gràfica de datos...")
metrics_path = plot_metrics_df_reduced()
print("Gráfico de métricas guardado, revisar en img/results")
print(f"Ruta completa: {metrics_path}")

# 2) Gráfico ROC usando los datos preprocesados
#    Puedes elegir si pasas X_train/X_test escalados o sin escalar.
roc_path = plot_roc_models_df_reduced(
    X_train_scaled,   # o X_train
    X_test_scaled,    # o X_test
    y_train,
    y_test,
    use_scaled=True
)
print("Gráfico ROC guardado, revisar en img/modelos")
print(f"Ruta completa: {roc_path}")

print("\n[PASO 5/6] Anàlisis de intervalos...")
analizar_principales_intervalos(df_reduced_imputed, n_intervalos=3, min_duracion=5)

print("\n[PASO 6/6] Deep Learning...")

results_df_deep = run_deep_experiments(
    X_train_scaled,
    X_test_scaled,
    y_train,
    y_test,
    save_plots=True,
)

print("\nResultados Deep / Tradicional:")
print(results_df_deep)