import streamlit as st
import json
import email
from email import policy
from email.parser import BytesParser
from openai import OpenAI

# Configuración de la interfaz
st.set_page_config(
    page_title="Gestor LOPDP - Triaje de Reclamos",
    page_icon="⚖️",
    layout="wide"
)

# Configuración de API Key (Fallback entre Secrets de Streamlit e Input manual)
api_key_secrets = st.secrets.get("OPENAI_API_KEY", "")

with st.sidebar:
    st.header("⚙️ Configuración")
    api_key_input = st.text_input(
        "OpenAI API Key (Opcional)", 
        type="password", 
        help="Si no ingresas una clave, la app utilizará la clave compartida por defecto."
    )
    st.markdown("---")
    st.markdown("**Módulo:** Fundamentos de IA")
    st.markdown("**Proyecto:** Triaje de Reclamos LOPDP")

openai_api_key = api_key_input.strip() if api_key_input.strip() else api_key_secrets

st.title("⚖️ Sistema Inteligente de Triaje de Reclamos LOPDP")
st.caption("Procesamiento y clasificación automática de hilos de correo (.eml) bajo la LOPDP")

# Prompt del Sistema
SYSTEM_PROMPT = """
Eres un especialista legal y de TI en la Ley Orgánica de Protección de Datos Personales (LOPDP).
Tu trabajo es analizar un HILO DE CORREOS ELECTRÓNICOS y clasificar el reclamo en EXACTAMENTE UNA de las siguientes 3 categorías:

CATEGORÍAS Y DEFINICIONES CONCEPTUALES:

1. "Contacto de tercero"
   - Definición: Ocurre cuando el canal de contacto (teléfono, email, WhatsApp) utilizado pertenece a una persona distinta al titular de la obligación/deuda (un familiar, compañero, o tercero sin relación) que solicita el cese de gestión hacia su persona.

2. "Ejercicio de derechos por no titular"
   - Definición: Ocurre cuando un tercero (abogado, apoderado o familiar) solicita ejercitar un derecho ARCOP/LOPDP (acceso, rectificación, cancelación, oposición) A NOMBRE Y EN REPRESENTACIÓN del titular, pero la solicitud requiere validación de representación legal.

3. "Derivación - No es cliente"
   - Definición: Ocurre cuando el reclamante manifiesta no tener relación contractual/comercial con la entidad, o cuando la nota interna del hilo especifica que la cartera pertenece a otra institución (cartera cedida o gestionada para un tercero).

REGLAS DE SALIDA:
- Analiza la intención principal expresada en el texto.
- Devuelve la respuesta ÚNICAMENTE en el objeto JSON solicitado sin texto adicional.

FORMATO JSON:
{
  "categoria": "<Contacto de tercero | Ejercicio de derechos por no titular | Derivación - No es cliente>",
  "certeza_porcentaje": <Número entero de 0 a 100>,
  "titular_afectado": "<Nombre del cliente o titular>",
  "identificacion": "<Cédula/CI/RUC detectado o 'No detectado'>",
  "remitente_original": "<Nombre del remitente que origina la queja>",
  "canales_contacto": "<Teléfono / Email mencionados>",
  "resumen_hilo": "<Resumen ejecutivo de 2 líneas>",
  "recomendacion_analista": "<Acción técnica e inmediata para el operador humano>"
}
"""


# Función auxiliar para extraer el texto plano de un archivo .eml
def extraer_texto_eml(eml_bytes):
    msg = BytesParser(policy=policy.default).parsebytes(eml_bytes)
    
    asunto = msg.get('subject', 'Sin asunto')
    remitente = msg.get('from', 'Desconocido')
    destinatario = msg.get('to', 'Desconocido')
    
    cuerpo = ""
    if msg.is_multipart():
        for part in msg.walk():
            content_type = part.get_content_type()
            content_disposition = str(part.get('Content-Disposition'))
            if content_type == 'text/plain' and 'attachment' not in content_disposition:
                cuerpo += part.get_payload(decode=True).decode(part.get_content_charset('utf-8'), errors='ignore')
    else:
        cuerpo = msg.get_payload(decode=True).decode(msg.get_content_charset('utf-8'), errors='ignore')
        
    texto_completo = f"De: {remitente}\nPara: {destinatario}\nAsunto: {asunto}\n\n{cuerpo}"
    return texto_completo

# Selección de modo de entrada mediante pestañas
tab1, tab2 = st.tabs(["📂 Subir Archivo .eml", "📝 Pegar Texto del Correo"])

hilo_correo = ""

with tab1:
    uploaded_file = st.file_uploader("Cargue el archivo del correo (.eml)", type=["eml"])
    if uploaded_file is not None:
        try:
            bytes_data = uploaded_file.getvalue()
            hilo_correo = extraer_texto_eml(bytes_data)
            st.success(f"Archivo **{uploaded_file.name}** cargado correctamente.")
            with st.expander("Ver contenido extraído del .eml"):
                st.text(hilo_correo[:1000] + "..." if len(hilo_correo) > 1000 else hilo_correo)
        except Exception as e:
            st.error(f"Error al procesar el archivo .eml: {e}")

with tab2:
    texto_manual = st.text_area(
        "Pegue el hilo de correo completo aquí:",
        height=200,
        placeholder="De: ...\nEnviado: ...\nAsunto: ..."
    )
    if texto_manual.strip():
        hilo_correo = texto_manual

# Botón de Procesamiento
if st.button("🔍 Procesar y Clasificar Hilo"):
    if not hilo_correo.strip():
        st.warning("⚠️ Por favor sube un archivo .eml o pega el texto de un correo antes de procesar.")
    elif not openai_api_key:
        st.error("🔑 No se detectó una API Key válida. Por favor ingresa una en el menú lateral.")
    else:
        try:
            client = OpenAI(api_key=openai_api_key)
            
            with st.spinner("Analizando contenido del archivo .eml y extrayendo metadatos..."):
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": hilo_correo}
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.1
                )
                
                res_json = json.loads(response.choices[0].message.content)

            st.success("✅ Hilo procesado y clasificado exitosamente")
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("### 📊 Clasificación LOPDP")
                st.metric("Categoría Asignada", res_json.get("categoria"))
                
                porcentaje = res_json.get("certeza_porcentaje", 0)
                st.progress(porcentaje / 100, text=f"Certeza de Clasificación: {porcentaje}%")
                
                st.markdown("### 📝 Resumen Operativo")
                st.info(res_json.get("resumen_hilo"))

            with col2:
                st.markdown("### 📋 Metadatos Extraídos del Hilo")
                st.write(f"• **Titular Afectado:** {res_json.get('titular_afectado')}")
                st.write(f"• **Identificación / Cédula:** `{res_json.get('identificacion')}`")
                st.write(f"• **Cliente Relacionado:** {res_json.get('cliente_relacionado')}")
                st.write(f"• **Remitente Original / Abogado:** {res_json.get('remitente_original')}")
                st.write(f"• **Canales Autorizados:** `{res_json.get('canales_contacto')}`")
                st.write(f"• **Contacto Reportado:** `{res_json.get('contacto_reportado')}`")
                st.write(f"• **Empresa Gestion:** `{res_json.get('empresa_contacto')}`")

            st.markdown("---")
            st.markdown("### 💡 Acción Sugerida para el Operador (Human-in-the-Loop)")
            st.warning(res_json.get("recomendacion_analista"))

        except Exception as e:
            st.error(f"❌ Error al procesar la solicitud con OpenAI: {str(e)}")




            
               
       
            
