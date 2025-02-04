import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# Generar datos dummies
np.random.seed(42)
dates = pd.date_range(start="2023-01-01", periods=100000, freq='H')
status = np.random.choice(['Abierto', 'En proceso', 'Cerrado'], size=100000, p=[0.2, 0.3, 0.5])
times_to_resolve = np.random.exponential(scale=48, size=100000)  # Horas para resolver

# Crear DataFrame
df = pd.DataFrame({
    'Fecha': dates,
    'Estado': status,
    'Tiempo_Resolucion': times_to_resolve
})

# Interfaz Streamlit
st.title("Análisis de Soporte a Productos Financieros")

# Sección de diagnóstico inicial
st.header("1. Diagnóstico Inicial")
st.markdown("**Pasos iniciales para diagnóstico:**")
st.write("1. Verificar credenciales y permisos del usuario.")
st.write("2. Confirmar si la plataforma está operativa o hay fallos generales.")
st.write("3. Consultar logs de acceso y errores.")
st.write("4. Contactar al área técnica en caso de problemas con la infraestructura.")

st.markdown("**Preguntas clave para el cliente:**")
st.write("- ¿Cuál es el mensaje de error exacto?")
st.write("- ¿Desde cuándo ocurre el problema?")
st.write("- ¿Ha probado con otro navegador o dispositivo?")
st.write("- ¿Puede acceder a otros servicios de la Fiduciaria?")

# Sección de manejo del cliente
st.header("2. Gestionar Expectativas del Cliente")
st.markdown("**Respuesta para manejar la frustración:**")
st.write("Estimado cliente, entendemos su preocupación y estamos priorizando su caso. Vamos a diagnosticar la causa lo antes posible.")

st.markdown("**Construcción de confianza:**")
st.write("Le mantendremos informado sobre los avances cada 30 minutos hasta la solución del problema.")

# Sección de resolución del problema
st.header("3. Resolución del Problema y Trabajo en Equipo")
st.markdown("**Acciones en caso de permisos incorrectos:**")
st.write("- Revisar la configuración de roles y accesos.")
st.write("- Validar logs para detectar cambios recientes.")
st.write("- Coordinar con el equipo de soporte técnico para ajustes inmediatos.")

st.markdown("**Prevención de futuros incidentes:**")
st.write("- Implementar alertas tempranas para fallos de acceso.")
st.write("- Automatizar revisiones de permisos.")
st.write("- Ofrecer entrenamiento a los clientes sobre el uso de la plataforma.")

# Sección de reporte de tiempos de respuesta
st.header("4. Reporte y Análisis de Datos")
st.markdown("**Indicadores clave:**")
st.write("- Promedio de tiempo de resolución.")
st.write("- Distribución de estados de PQRS.")
st.write("- Tendencias en la cantidad de reportes a lo largo del tiempo.")

# Visualización de datos
st.subheader("Distribución de tiempos de resolución")
fig = px.histogram(df, x="Tiempo_Resolucion", nbins=50, title="Histograma de Tiempos de Resolución")
st.plotly_chart(fig)

st.subheader("Evolución de reportes por estado")
df['Fecha'] = pd.to_datetime(df['Fecha'])
df_estado = df.groupby([df['Fecha'].dt.date, 'Estado']).size().unstack().fillna(0)
st.line_chart(df_estado)

st.subheader("Resumen de Tiempos de Resolución")
st.write(df[['Estado', 'Tiempo_Resolucion']].groupby('Estado').describe())

st.write("### Conclusión:")
st.write("Estos análisis permiten optimizar los tiempos de respuesta y mejorar la experiencia del cliente.")
