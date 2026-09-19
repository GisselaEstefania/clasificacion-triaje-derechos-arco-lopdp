import os
import json
from openai import OpenAI


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

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

# Dataset de evaluación con Hilos de Correo Sintéticos Expandidos
DATASET_EVALUACION_NUEVO = [
    # Categoría 1: Contacto de tercero (Redacción variada)
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

    # Categoría 2: Ejercicio de derechos por no titular (Redacción variada)
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

    # Categoría 3: Derivación - No es cliente (Redacción variada)
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
    }
]

def ejecutar_evaluacion_masiva():
    if not OPENAI_API_KEY or OPENAI_API_KEY == "sk-proj-tu-key-aqui":
        print("❌ Error: Configure una API Key válida de OpenAI antes de ejecutar.")
        return

    client = OpenAI(api_key=OPENAI_API_KEY)
    print(f"🚀 Iniciando evaluación de {len(DATASET_EVALUACION_NUEVO)} casos con GPT-3.5...\n")

    aciertos = 0
    total = len(DATASET_EVALUACION_NUEVO)

    for i, item in enumerate(DATASET_EVALUACION_NUEVO, start=1):
        hilo = item["hilo"]
        esperado = item["categoria_esperada"]

        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": hilo}
                ],
                response_format={"type": "json_object"},
                temperature=0.1
            )

            res_json = json.loads(response.choices[0].message.content)
            obtenido = res_json.get("categoria", "Error / No detectado")

            es_correcto = (obtenido == esperado)
            if es_correcto:
                aciertos += 1
                status = "✅ OK"
            else:
                status = "❌ FALLO"

            print(f"Caso {i:02d}/{total} [{status}]")
            print(f"   • Esperado: {esperado}")
            print(f"   • Obtención: {obtenido}")
            print("-" * 50)

        except Exception as e:
            print(f"Caso {i:02d}/{total} [❌ ERROR API]: {e}")

    # Muestreo de métrica final
    precision = (aciertos / total) * 100
    print("\n" + "=" * 50)
    print(f"📊 RESUMEN DE LA EVALUACIÓN:")
    print(f"   • Casos Totales: {total}")
    print(f"   • Casos Correctos: {aciertos}")
    print(f"   • Precisión Global (Accuracy): {precision:.2f}%")
    print("=" * 50)

if __name__ == "__main__":
    ejecutar_evaluacion_masiva()
