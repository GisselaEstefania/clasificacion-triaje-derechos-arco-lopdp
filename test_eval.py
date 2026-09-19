import os
import json
import pandas as pd
from openai import OpenAI
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    classification_report, 
    confusion_matrix
)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# -----------------------------------------------------------------------------
# CAPA 2: System Prompt con Guardraíl de Descarte y 4 Categorías
# -----------------------------------------------------------------------------
SYSTEM_PROMPT = """
Eres un especialista legal y de TI en la Ley Orgánica de Protección de Datos Personales (LOPDP).
Tu trabajo es analizar un HILO DE CORREOS ELECTRÓNICOS y clasificar el reclamo en EXACTAMENTE UNA de las siguientes 4 categorías:

CATEGORÍAS Y DEFINICIONES CONCEPTUALES:

1. "Contacto de tercero"
   - Definición: Ocurre cuando el canal de contacto (teléfono, email, WhatsApp) utilizado pertenece a una persona distinta al titular de la obligación/deuda (un familiar, compañero, o tercero sin relación) que solicita el cese de gestión hacia su persona.

2. "Ejercicio de derechos por no titular"
   - Definición: Ocurre cuando un tercero (abogado, apoderado o familiar) solicita ejercitar un derecho ARCOP/LOPDP (acceso, rectificación, cancelación, oposición) A NOMBRE Y EN REPRESENTACIÓN del titular, pero la solicitud requiere validación de representación legal.

3. "Derivación - No es cliente"
   - Definición: Ocurre cuando el reclamante manifiesta no tener relación contractual/comercial con la entidad, o cuando la nota interna del hilo especifica que la cartera pertenece a otra institución (cartera cedida o gestionada para un tercero).

4. "Información Insuficiente / Saludo"
   - Definición: Ocurre cuando el texto es únicamente un saludo, una prueba, una frase corta sin contexto o NO contiene información suficiente ni explícita sobre un reclamo, cobro o tratamiento de datos LOPDP.

REGLAS MANDATORIAS:
- NUNCA asumas ni infieras una categoría de riesgo (como Contacto de Tercero) si el texto no contiene un reclamo o hecho explícito.
- Devuelve la respuesta ÚNICAMENTE en el objeto JSON solicitado sin texto adicional.

FORMATO JSON:
{
  "categoria": "<Contacto de tercero | Ejercicio de derechos por no titular | Derivación - No es cliente | Información Insuficiente / Saludo>",
  "certeza_porcentaje": <Número entero de 0 a 100>,
  "titular_afectado": "<Nombre del cliente o titular o 'No detectado'>",
  "identificacion": "<Cédula/CI/RUC detectado o 'No detectado'>",
  "remitente_original": "<Nombre del remitente que origina la queja o 'No detectado'>",
  "canales_contacto": "<Teléfono / Email mencionados o 'No detectado'>",
  "resumen_hilo": "<Resumen ejecutivo de 2 líneas>",
  "recomendacion_analista": "<Acción técnica e inmediata para el operador humano>"
}
"""

CATEGORIAS_OFICIALES = [
    "Contacto de tercero",
    "Ejercicio de derechos por no titular",
    "Derivación - No es cliente",
    "Información Insuficiente / Saludo"
]

# -----------------------------------------------------------------------------
# Dataset de Evaluación Expandido (Incluye Casos Triviales/Borde)
# -----------------------------------------------------------------------------
DATASET_EVALUACION_NUEVO = [
    # Categoría 1: Contacto de tercero
    {
        "hilo": "De: perez_family@gmail.com\nAsunto: Molestias por llamada\n\nSres. Banco, dejen de llamar al celular de mi esposa a preguntar por mi primo. Ella no debe nada.",
        "categoria_esperada": "Contacto de tercero"
    },
    {
        "hilo": "De: recepcion@mifirma.com\nAsunto: Teléfono corporativo\n\nLlaman a la línea fija de la empresa a pedir con un ex empleado. Este número es corporativo, remuevan el registro.",
        "categoria_esperada": "Contacto de tercero"
    },
    {
        "hilo": "De: vecino@yahoo.com\nAsunto: Mensajes equivocados\n\nMe llegan SMS de cobro para un señor de apellido Ramos. Yo compré esta línea telefónica hace un mes.",
        "categoria_esperada": "Contacto de tercero"
    },
    {
        "hilo": "De: garante_no@hotmail.com\nAsunto: Intimación de cobro\n\nEscriben a mi correo a exigirme el pago de un amigo. Yo nunca firmé como garante ni acepté notificaciones.",
        "categoria_esperada": "Contacto de tercero"
    },
    {
        "hilo": "De: contacto_ec@gmail.com\nAsunto: Retiro de datos\n\nPor favor eliminen mi WhatsApp de su sistema. Me contactaron buscando a una persona que no habita en este domicilio.",
        "categoria_esperada": "Contacto de tercero"
    },

    # Categoría 2: Ejercicio de derechos por no titular
    {
        "hilo": "De: consorcio_legal@estudio.ec\nAsunto: Solicitud de Oposición LOPDP\n\nComo defensa técnica del Ing. Marco Silva, solicito la suspensión del tratamiento de sus datos personales.",
        "categoria_esperada": "Ejercicio de derechos por no titular"
    },
    {
        "hilo": "De: tutor_legal@mail.com\nAsunto: Baja de base de datos\n\nSolicito formalmente la eliminación de los datos de mi representado menor de edad de sus registros comerciales.",
        "categoria_esperada": "Ejercicio de derechos por no titular"
    },
    {
        "hilo": "De: familiar_perez@gmail.com\nAsunto: Corrección de dirección\n\nMi abuelo ya no vive ahí. Como su familiar a cargo pido actualizar su domicilio en su sistema.",
        "categoria_esperada": "Ejercicio de derechos por no titular"
    },
    {
        "hilo": "De: apoderado_ec@outlook.com\nAsunto: Acceso a información LOPDP\n\nAdjunto carta de mi cliente donde me faculta a requerir la copia del historial de datos que conservan de su persona.",
        "categoria_esperada": "Ejercicio de derechos por no titular"
    },
    {
        "hilo": "De: abg_mendoza@juridico.com\nAsunto: Reclamo LOPDP\n\nA nombre del Sr. Roberto Gómez (CI 1700000000), exijo la cancelación inmediata de sus datos de marcación.",
        "categoria_esperada": "Ejercicio de derechos por no titular"
    },

    # Categoría 3: Derivación - No es cliente
    {
        "hilo": "De: usuario_indignado@gmail.com\nAsunto: Error de cobro\n\nMe están notificando por un crédito automotriz que jamás contraté con ustedes. Verifiquen su base.",
        "categoria_esperada": "Derivación - No es cliente"
    },
    {
        "hilo": "De: delegadopdp@banco.com\nPara: cobranzas@ext.com\nAsunto: RE: Inconsistencia\n\nRevisado en core: La persona no registra operaciones activas ni pasivas en la institución. Derivar caso.",
        "categoria_esperada": "Derivación - No es cliente"
    },
    {
        "hilo": "De: reclamos@empresa.com\nAsunto: Cartera castigada de tercero\n\nLa obligación reportada en el Buró corresponde a la Cooperativa XYZ, no a su entidad. Solicito canalizar a donde corresponda.",
        "categoria_esperada": "Derivación - No es cliente"
    },
    {
        "hilo": "De: ciudadano123@yahoo.com\nAsunto: Sin contrato registrado\n\nJamás he abierto una cuenta ni solicitado tarjeta en su banco. Favor reorientar la gestión de cobro.",
        "categoria_esperada": "Derivación - No es cliente"
    },
    {
        "hilo": "De: soporte@banco.com\nAsunto: Cartera cedida\n\nEstimados, la cuenta reportada pertenece a la cartera comprada a Banco Pasado. Proceder con el redireccionamiento.",
        "categoria_esperada": "Derivación - No es cliente"
    },

    # Categoría 4: Información Insuficiente / Saludo (Casos Borde / Triviales)
    {
        "hilo": "hola",
        "categoria_esperada": "Información Insuficiente / Saludo"
    },
    {
        "hilo": "Buenos días estimada ayuda por favor",
        "categoria_esperada": "Información Insuficiente / Saludo"
    },
    {
        "hilo": "De: prueba@test.com\nAsunto: Consulta\n\nHola, quisiera saber información de su horario de atención.",
        "categoria_esperada": "Información Insuficiente / Saludo"
    }
]

# -----------------------------------------------------------------------------
# Función para Evaluar Caso Individual (Capa 1 en Python + Capa 2 con OpenAI)
# -----------------------------------------------------------------------------
def clasificar_hilo(client, texto_hilo):
    # Capa 1: Validación local en Python (Longitud / Cantidad de Palabras)
    texto_limpio = texto_hilo.strip().lower()
    palabras = texto_limpio.split()
    
    if len(texto_limpio) < 15 or len(palabras) < 3:
        return "Información Insuficiente / Saludo"

    # Capa 2: Evaluación con GPT-3.5-Turbo vía API
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": texto_hilo}
        ],
        response_format={"type": "json_object"},
        temperature=0.1
    )
    
    res_json = json.loads(response.choices[0].message.content)
    return res_json.get("categoria", "Error / No detectado")

# -----------------------------------------------------------------------------
# Ejecución del Script de Evaluación y Generación de Métricas
# -----------------------------------------------------------------------------
def ejecutar_evaluacion_masiva():
    if not OPENAI_API_KEY or OPENAI_API_KEY == "sk-proj-tu-key-aqui":
        print("❌ Error: Configure la variable de entorno OPENAI_API_KEY antes de ejecutar.")
        return

    client = OpenAI(api_key=OPENAI_API_KEY)
    total = len(DATASET_EVALUACION_NUEVO)
    
    y_true = []
    y_pred = []

    print(f"🚀 Iniciando evaluación de {total} casos sintéticos con GPT-3.5-Turbo...\n")

    for i, item in enumerate(DATASET_EVALUACION_NUEVO, start=1):
        hilo = item["hilo"]
        esperado = item["categoria_esperada"]

        try:
            obtenido = clasificar_hilo(client, hilo)
            y_true.append(esperado)
            y_pred.append(obtenido)

            es_correcto = (obtenido == esperado)
            status = "✅ OK" if es_correcto else "❌ FALLO"

            # Resumir el hilo impreso en consola si es muy largo
            hilo_print = hilo.replace('\n', ' ')
            if len(hilo_print) > 60:
                hilo_print = hilo_print[:57] + "..."

            print(f"Caso {i:02d}/{total} [{status}] -> Input: '{hilo_print}'")
            print(f"   • Esperado: {esperado}")
            print(f"   • Obtenido: {obtenido}")
            print("-" * 65)

        except Exception as e:
            print(f"Caso {i:02d}/{total} [❌ ERROR API]: {e}")

    # -------------------------------------------------------------------------
    # CÁLCULO E IMPRESIÓN DE MÉTRICAS CON SCIKIT-LEARN Y PANDAS
    # -------------------------------------------------------------------------
    acc = accuracy_score(y_true, y_pred)
    prec_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)

    print("\n" + "=" * 65)
    print("📊 METRICAS GLOBALES DEL MODELO DE CLASIFICACIÓN LOPDP")
    print("=" * 65)
    print(f" • Total de Casos Evaluados: {total}")
    print(f" • Accuracy Global:         {acc * 100:.2f}%")
    print(f" • Precision Promedio:       {prec_macro * 100:.2f}%")
    print(f" • Recall Promedio:          {rec_macro * 100:.2f}%")
    print(f" • F1-Score Promedio:        {f1_macro * 100:.2f}%\n")

    # Reporte Detallado por Categoría
    print("=" * 65)
    print("📋 REPORTE DETALLADO DE DESEMPEÑO POR CATEGORÍA REGULADA")
    print("=" * 65)
    reporte_dict = classification_report(
        y_true, 
        y_pred, 
        labels=CATEGORIAS_OFICIALES, 
        output_dict=True, 
        zero_division=0
    )
    df_reporte = pd.DataFrame(reporte_dict).transpose()
    print(df_reporte.round(2).to_string())

    # Matriz de Confusión
    print("\n" + "=" * 65)
    print("MATRIZ DE CONFUSIÓN (FILAS: REAL | COLUMNAS: PREDICCIÓN)")
    print("=" * 65)
    cm = confusion_matrix(y_true, y_pred, labels=CATEGORIAS_OFICIALES)
    df_cm = pd.DataFrame(cm, index=CATEGORIAS_OFICIALES, columns=CATEGORIAS_OFICIALES)
    print(df_cm.to_string())
    print("=" * 65)

if __name__ == "__main__":
    ejecutar_evaluacion_masiva()
