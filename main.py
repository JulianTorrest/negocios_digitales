import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

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
promedio_resolucion_por_prioridad = df.groupby([df['Fecha'].dt.date, 'Prioridad'])['Tiempo_Resolucion'].mean().unstack()
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
# Asegúrate de que 'promedio_resolucion_por_prioridad' sea un DataFrame
promedio_resolucion_por_prioridad = df.groupby([df['Fecha'].dt.date, 'Prioridad'])['Tiempo_Resolucion'].mean().unstack().reset_index()

# Ajusta los nombres de las columnas y organiza los datos
fig_tiempo_prioridad = px.bar(promedio_resolucion_por_prioridad, x='Fecha', y=promedio_resolucion_por_prioridad.columns[1:], 
                              title="Tiempo Promedio de Resolución por Prioridad")

st.plotly_chart(fig_tiempo_prioridad)

# Promedio de resolución por prioridad
# Convertir promedio_resolucion_por_prioridad a DataFrame
promedio_resolucion_por_prioridad = df.groupby([df['Fecha'].dt.date, 'Prioridad'])['Tiempo_Resolucion'].mean().unstack().reset_index()

# Asegurarte de que el DataFrame tiene las columnas correctas
fig_tiempo_prioridad = px.bar(promedio_resolucion_por_prioridad, 
                              x='Fecha', 
                              y=promedio_resolucion_por_prioridad.columns[1:], 
                              title="Tiempo Promedio de Resolución por Prioridad")

# Mostrar el gráfico
st.plotly_chart(fig_tiempo_prioridad)


# Distribución de incidentes por hora del día
st.subheader("Distribución de Incidentes por Hora del Día")
fig_hora = px.bar(x=incidentes_por_hora.index, y=incidentes_por_hora.values, title="Incidentes Reportados por Hora")
st.plotly_chart(fig_hora)

# Tendencia semanal de incidentes
st.subheader("Tendencia Semanal de Incidentes")
fig_semanal = px.line(x=dias_semana, y=tendencia_semanal.values, title="Incidentes por Día de la Semana")
st.plotly_chart(fig_semanal)

# Tiempo de resolución promedio por día de la semana
st.subheader("Tiempo Promedio de Resolución por Día de la Semana")
fig_res_dia = px.bar(x=dias_semana, y=tiempos_resolucion_por_dia.values, title="Tiempo Promedio de Resolución por Día")
st.plotly_chart(fig_res_dia)

# Mapa de calor de incidentes
st.subheader("Mapa de Calor de Reportes por Día y Hora")
df['Hora'] = df['Fecha'].dt.hour
df['Dia'] = df['Fecha'].dt.dayofweek
heatmap_data = df.pivot_table(index='Dia', columns='Hora', values='Tiempo_Resolucion', aggfunc='count').fillna(0)
fig_heatmap, ax = plt.subplots(figsize=(12, 6))
sns.heatmap(heatmap_data, cmap="YlGnBu", ax=ax)
st.pyplot(fig_heatmap)

# Resumen de tiempos de resolución
st.subheader("Resumen de Tiempos de Resolución")
st.dataframe(df[['Estado', 'Tiempo_Resolucion']].groupby('Estado').describe())
