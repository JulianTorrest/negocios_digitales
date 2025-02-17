import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# Generar datos dummies
np.random.seed(42)
dates = pd.date_range(start="2023-01-01", periods=100000, freq='H')
status = np.random.choice(['Abierto', 'En proceso', 'Cerrado'], size=100000, p=[0.2, 0.3, 0.5])
times_to_resolve = np.random.exponential(scale=48, size=100000)  # Horas para resolver
priorities = np.random.choice(['Alta', 'Media', 'Baja'], size=100000, p=[0.3, 0.5, 0.2])
users = np.random.randint(1, 500, size=100000)

# Crear DataFrame
df = pd.DataFrame({
    'Fecha': dates,
    'Estado': status,
    'Tiempo_Resolucion': times_to_resolve,
    'Prioridad': priorities,
    'Usuario': users
})

# Filtrar usuarios específicos
df = df[~df['Usuario'].isin([400, 401])]

# Ajustar tiempos de resolución por prioridad
def ajustar_tiempo_prioridad(row):
    if row['Prioridad'] == 'Alta':
        return np.random.uniform(30, 60)  # Entre 30 y 60 minutos
    elif row['Prioridad'] == 'Media':
        return np.random.uniform(120, 180)  # Entre 2 y 3 horas
    else:
        return np.random.uniform(180, 360)  # Más de 3 horas

df['Tiempo_Resolucion'] = df.apply(ajustar_tiempo_prioridad, axis=1)

# Cálculo de KPIs
num_incidentes = len(df)
tiempo_prom_resolucion = df['Tiempo_Resolucion'].mean()
estado_counts = df['Estado'].value_counts()
incidentes_por_prioridad = df['Prioridad'].value_counts()
promedio_resolucion_por_prioridad = df.groupby('Prioridad')['Tiempo_Resolucion'].mean()
incidentes_por_usuario = df.groupby('Usuario').size()
max_incidentes_usuario = incidentes_por_usuario.max()
usuario_mas_incidentes = incidentes_por_usuario.idxmax()
min_incidentes_usuario = incidentes_por_usuario.min()
usuario_menos_incidentes = incidentes_por_usuario.idxmin()

# KPIs adicionales
incidentes_por_hora = df.groupby(df['Fecha'].dt.hour).size()
dias_semana = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']
tendencia_semanal = df.groupby(df['Fecha'].dt.dayofweek).size()
tendencia_semanal.index = dias_semana
tiempos_resolucion_por_dia = df.groupby(df['Fecha'].dt.dayofweek)['Tiempo_Resolucion'].mean()
tiempos_resolucion_por_dia.index = dias_semana
incidentes_por_estado_y_prioridad = df.groupby(['Estado', 'Prioridad']).size().unstack().fillna(0)
porcentaje_resolucion_sla = (df[df['Tiempo_Resolucion'] <= 48].shape[0] / num_incidentes) * 100
reapertura_tasa = df[df['Estado'] == 'Abierto'].shape[0] / num_incidentes * 100

# Interfaz Streamlit
st.title("Análisis de Soporte a Productos Financieros")

st.subheader("Indicadores Clave de Rendimiento (KPIs)")
st.metric(label="Número Total de Incidentes", value=num_incidentes)
st.metric(label="Tiempo Promedio de Resolución (horas)", value=round(tiempo_prom_resolucion, 2))
st.metric(label="Máximo de Incidentes por Usuario", value=max_incidentes_usuario)
st.metric(label="Usuario con Más Incidentes", value=usuario_mas_incidentes)
st.metric(label="Mínimo de Incidentes por Usuario", value=min_incidentes_usuario)
st.metric(label="Usuario con Menos Incidentes", value=usuario_menos_incidentes)
st.metric(label="Porcentaje de Resolución dentro de SLA", value=round(porcentaje_resolucion_sla, 2))
st.metric(label="Tasa de Reapertura de Incidentes", value=round(reapertura_tasa, 2))

# Gráfico de distribución de estados
total_estado_fig = px.pie(values=estado_counts.values, names=estado_counts.index, title="Distribución de Estados de Incidentes")
st.plotly_chart(total_estado_fig)

# Histograma de tiempos de resolución
st.subheader("Distribución de tiempos de resolución")
fig = px.histogram(df, x="Tiempo_Resolucion", nbins=50, title="Histograma de Tiempos de Resolución")
st.plotly_chart(fig)

# Evolución de reportes por estado
df['Fecha'] = pd.to_datetime(df['Fecha'])
df_estado = df.groupby([df['Fecha'].dt.date, 'Estado']).size().unstack().fillna(0)
st.subheader("Evolución de reportes por estado")
st.line_chart(df_estado)

# Distribución por prioridad
st.subheader("Distribución de Incidentes por Prioridad")
fig_prioridad = px.bar(x=incidentes_por_prioridad.index, y=incidentes_por_prioridad.values, title="Incidentes por Prioridad")
st.plotly_chart(fig_prioridad)

# Promedio de resolución por prioridad
st.subheader("Tiempo Promedio de Resolución por Prioridad")
fig_tiempo_prioridad = px.bar(x=promedio_resolucion_por_prioridad.index, y=promedio_resolucion_por_prioridad.values, title="Tiempo Promedio de Resolución por Prioridad")
st.plotly_chart(fig_tiempo_prioridad)

# Mapa de calor de incidentes
st.subheader("Mapa de Calor de Reportes por Día y Hora")
df['Hora'] = df['Fecha'].dt.hour
df['Dia'] = df['Fecha'].dt.dayofweek
heatmap_data = df.pivot_table(index='Dia', columns='Hora', values='Tiempo_Resolucion', aggfunc='count').fillna(0)
fig_heatmap, ax = plt.subplots(figsize=(12, 6))
sns.heatmap(heatmap_data, cmap="YlGnBu", ax=ax)
st.pyplot(fig_heatmap)

#Predicción del Tiempo de Resolución

# Resumen de tiempos de resolución
st.subheader("Resumen de Tiempos de Resolución")
st.dataframe(df[['Estado', 'Tiempo_Resolucion']].groupby('Estado').describe())

# Preparar datos para el modelo de Machine Learning
df_ml = df[['Estado', 'Prioridad', 'Usuario', 'Tiempo_Resolucion']].dropna()
df_ml['Estado'] = df_ml['Estado'].map({'Abierto': 0, 'En proceso': 1, 'Cerrado': 2})
df_ml['Prioridad'] = df_ml['Prioridad'].map({'Alta': 0, 'Media': 1, 'Baja': 2})

X = df_ml[['Estado', 'Prioridad', 'Usuario']]
y = df_ml['Tiempo_Resolucion']

# Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear el modelo
model = LinearRegression()

# Entrenar el modelo
model.fit(X_train, y_train)

# Realizar predicciones
y_pred = model.predict(X_test)

# Evaluar el modelo
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Mostrar métricas
st.subheader("Evaluación del Modelo de Machine Learning Regresión Lineal")
st.write(f"Error Absoluto Medio (MAE): {mae:.2f}")
st.write(f"Coeficiente de Determinación (R²): {r2:.2f}")

# Predicción de tiempos de resolución para un nuevo conjunto de datos
nuevo_dato = pd.DataFrame({'Estado': [1], 'Prioridad': [0], 'Usuario': [10]})
prediccion = model.predict(nuevo_dato)

st.write(f"Predicción de Tiempo de Resolución para el incidente con Estado 'En proceso', Prioridad 'Alta' y Usuario 10: {prediccion[0]:.2f} horas")

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Modelo de Random Forest para clasificación de Estado
# Convertir columnas categóricas a variables numéricas
df_classification = pd.get_dummies(df, columns=['Prioridad', 'Usuario'], drop_first=False)

# Definir las variables de entrada (X) y salida (y)
X_classification = df_classification[['Prioridad_Media', 'Prioridad_Alta', 'Usuario_1', 'Usuario_2']]  # Asegúrate de usar las columnas correctas
y_classification = df_classification['Estado']

# Dividir en conjuntos de entrenamiento y prueba
X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(X_classification, y_classification, test_size=0.2, random_state=42)

# Entrenar el modelo de Random Forest
model_class = RandomForestClassifier(n_estimators=100, random_state=42)
model_class.fit(X_train_class, y_train_class)

# Realizar predicciones
y_pred_class = model_class.predict(X_test_class)

# Evaluar el modelo
accuracy = accuracy_score(y_test_class, y_pred_class)
st.write(f"Precisión del modelo de clasificación: {accuracy:.2f}")

from sklearn.metrics import classification_report, confusion_matrix

# Evaluar el modelo con métricas adicionales
st.write("Reporte de clasificación:")
st.text(classification_report(y_test_class, y_pred_class))

st.write("Matriz de confusión:")
cm = confusion_matrix(y_test_class, y_pred_class)
st.write(cm)

# Definir el espacio de búsqueda de hiperparámetros
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

## Riesgo Operacional
# Tasa de reapertura vs tiempo de resolución
df_reabiertos = df[df['Estado'] == 'Abierto']
fig_scatter = px.scatter(df_reabiertos, x='Tiempo_Resolucion', y=df_reabiertos.index, title="Relación entre Tiempo de Resolución y Reapertura")
st.plotly_chart(fig_scatter)

# Riesgo por prioridad
fig_box = px.box(df, x="Prioridad", y="Tiempo_Resolucion", color="Prioridad", title="Tiempo de Resolución por Prioridad")
st.plotly_chart(fig_box)

# Tendencia semanal y estacionalidad
fig_tendencia = px.line(x=dias_semana, y=tiempos_resolucion_por_dia.values, title="Tendencia de Tiempos de Resolución por Día de la Semana")
st.plotly_chart(fig_tendencia)

# Modelos de Riesgo Operacional con Machine Learning Predicción de Tiempo de Resolución (Regresión)
# Convertir variables categóricas en numéricas
df_ml = df.copy()
df_ml['Dia_Semana'] = df_ml['Fecha'].dt.dayofweek
df_ml = pd.get_dummies(df_ml, columns=['Prioridad', 'Estado'], drop_first=True)

# Variables de entrada y salida
X = df_ml[['Dia_Semana', 'Usuario'] + list(df_ml.columns[df_ml.columns.str.startswith('Prioridad_')]) + list(df_ml.columns[df_ml.columns.str.startswith('Estado_')])]
y = df_ml['Tiempo_Resolucion']

# División de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modelo de regresión lineal
modelo = LinearRegression()
modelo.fit(X_train, y_train)
y_pred = modelo.predict(X_test)

# Evaluación del modelo
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Mostrar resultados en Streamlit
st.subheader("Modelo de Predicción de Tiempo de Resolución")
st.write(f"Error Absoluto Medio (MAE): {mae:.2f} horas")
st.write(f"Coeficiente de Determinación (R²): {r2:.2f}")


# Detección de Riesgo Operacional (Clasificación)
# Crear variable de clasificación
df_ml['Riesgo_Alto'] = (df_ml['Tiempo_Resolucion'] > 180).astype(int)

# Variables de entrada y salida
X = df_ml[['Dia_Semana', 'Usuario'] + list(df_ml.columns[df_ml.columns.str.startswith('Prioridad_')]) + list(df_ml.columns[df_ml.columns.str.startswith('Estado_')])]
y = df_ml['Riesgo_Alto']

# División de datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modelo de clasificación
from sklearn.ensemble import RandomForestClassifier

modelo_clf = RandomForestClassifier(n_estimators=100, random_state=42)
modelo_clf.fit(X_train, y_train)

# Predicción y evaluación
y_pred_clf = modelo_clf.predict(X_test)
from sklearn.metrics import accuracy_score

st.subheader("Clasificación de Riesgo Operacional")
st.write(f"Precisión del Modelo: {accuracy_score(y_test, y_pred_clf) * 100:.2f}%")

# Visualización de Riesgo
fig_heatmap = px.imshow(incidentes_por_estado_y_prioridad, labels=dict(x="Prioridad", y="Estado", color="Cantidad"),
                        title="Incidentes por Estado y Prioridad")
st.plotly_chart(fig_heatmap)





