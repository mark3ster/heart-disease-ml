# Predicción de enfermedad cardíaca con Machine Learning

Trabajo final de **Aprendizaje Automático II** (UDIT – University of Design, Innovation and Technology).

Clasificación de enfermedad cardíaca con el dataset [Heart Disease (UCI)](https://archive.ics.uci.edu/dataset/45/heart+disease). El proyecto amplía un trabajo previo de Aprendizaje Automático I incorporando **KNN** y **PCA**.

## Contenido

| Archivo | Descripción |
|---|---|
| `heart_disease_analysis.ipynb` | Notebook con todo el código: preprocesamiento, EDA, modelos, PCA y KNN |
| `docs/informe_final.pdf` | Memoria del trabajo (resultados, discusión y conclusiones) |
| `requirements.txt` | Dependencias |

## Metodología

1. **Preprocesamiento:** eliminación de nulos, `MinMaxScaler` para variables numéricas y `OneHotEncoder` para categóricas (`cp`, `restecg`, `slope`, `thal`, `ca`).
2. **EDA:** distribución de la variable objetivo, histogramas, boxplots y matriz de correlación.
3. **Modelos** (pipelines de scikit-learn, split 70/30 estratificado, `random_state=42`):
   - Regresión Logística
   - Árbol de Decisión
   - Random Forest (con `GridSearchCV`)
   - SVM (kernel RBF)
   - **KNN** (con `GridSearchCV` sobre `k`, ponderación y métrica de distancia)
4. **PCA:** reducción conservando el 95 % de la varianza + Regresión Logística, y proyección a 2 componentes para visualización.
5. **Métricas:** accuracy, precision, recall y F1 (ponderadas).

## Resultados principales

| Modelo | Precision | Recall | F1-score |
|---|---|---|---|
| Regresión Logística | 0.529 | 0.600 | 0.557 |
| KNN | 0.467 | 0.567 | 0.507 |
| KNN optimizado | 0.460 | 0.567 | 0.494 |

- La Regresión Logística fue el mejor modelo global.
- Optimizar KNN no aportó mejora significativa.
- PCA redujo dimensionalidad sin una pérdida notable de rendimiento, a costa de interpretabilidad.
- Conclusión: en este problema, los modelos simples y bien ajustados funcionan mejor.

## Cómo ejecutarlo

```bash
git clone https://github.com/<tu-usuario>/heart-disease-ml.git
cd heart-disease-ml
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook heart_disease_analysis.ipynb
```

El dataset se descarga automáticamente con `ucimlrepo`, así que no hay que añadir datos al repositorio.

## Autor

- Marcos Jiménez González

