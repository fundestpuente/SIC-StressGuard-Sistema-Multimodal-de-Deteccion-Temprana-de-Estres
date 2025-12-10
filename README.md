# 📌 StressGuard: Sistema Multimodal de Detección Temprana de Estrés mediante Fusión de Sensores Wearables e Inteligencia Artificial.
Curso: Samsung Innovation Campus – Módulo de Python (Ecuador 2025)
Seccion: EC03
Grupo: 01
Carpeta: /EC03/SIC-STRESSGUARD-DETECCION-DE-ESTRES-EN-PERSONAS-MEDIANTE-ANALISIS-DE-VARIABILIDAD-CARDIACA-HRV-Y-ML

Integrantes del Grupo
- Kevin Perez
- Alejandro Obando
- Danna Ayala
- Valentina Cañizares
- Daniela Mata

Descripción del Proyecto
    El estrés crónico es considerado por la Organización Mundial de la Salud (OMS) como una "epidemia de salud mundial del siglo XXI". El proyecto desarrolla un sistema algorítmico de clasificación automática para identificar estados de estrés agudo a partir de datos fisiológicos objetivos recolectados por dispositivos wearables. Utiliza señales multimodales del dataset WESAD (EDA, ECG, EMG, Temperatura) y compara algoritmos como Random Forest, KNN y Decision Tree para detectar patrones de estrés, sentando las bases para una futura aplicación móvil de monitoreo en tiempo real e intervención preventiva.

Instrucciones de Instalación y Ejecución
- Requisitos
- Python 3.9+ (recomendado)
- Git
- Pasos
- Clonar el repositorio (o asegurarse de estar en la carpeta del proyecto):

git clone https://github.com/fundestpuente/SIC-StressGuard-Deteccion-de-Estres-en-personas-mediante-analisis-de-Variabilidad-Cardiaca-HRV-y-ML.git
cd '.\SIC-StressGuard-Deteccion-de-Estres-en-personas-mediante-analisis-de-Variabilidad-Cardiaca-HRV-y-ML\'
Abrir carpeta SIC-StressGuard-Deteccion-de-Estres-en-personas-mediante-analisis-de-Variabilidad-Cardiaca-HRV-y-ML

Ejecutar la aplicación: a. Abrir archivo StressGuard.ipynb. b. Se puede ejecutar por celda de forma ordenada. c. Se puede ejecutar todas las celdas con el boton en la parte superior del IDE que dice 'Run All'.

Herramientas Implementadas
- Lenguaje: Python 3.12
- Librerías principales: pandas, numpy, scipy, kagglehub
- Otras herramientas: Visual Studio Code, GitHub



## Split por Sujeto
Es una estrategia común en ML cuando trabajas con datos de personas:
Leave-Subject-Out (LSO)

Dejas algunos sujetos completamente fuera del entrenamiento
Simula predecir en personas nuevas que el modelo nunca vio
Más realista que mezclar datos del mismo sujeto en train y test

## Ventajas:
✅ Evita data leakage - Ningún dato del sujeto S16/S17 contamina el train
✅ Generalización real - Prueba si el modelo funciona con personas nuevas
✅ Más conservador - Test accuracy será más realista (puede ser menor)


## Modo de uso:
 Puede seleccionar los sujetos de su preferencia
TEST_SUBJECTS = ["S16", "S17"]

## PASOS PARA EJECUTAR
1.- En terminal ejecutar: "pip install -r requirements.txt"
2.- Ejecutar main.py 


