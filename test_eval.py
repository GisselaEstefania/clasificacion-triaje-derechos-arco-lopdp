import os
import json
from openai import OpenAI


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

SYSTEM_PROMPT = """
Eres un especialista legal y de TI en la Ley Orgánica de Protección de Datos Personales (LOPDP).
Tu trabajo es leer un HILO COMPLETO DE CORREOS ELECTRÓNICOS (que contiene notas internas, avisos de confidencialidad y el reclamo original) y realizar un análisis estructurado.

Debes categorizar el reclamo en UNA de las siguientes 3 clases exactas:
1. "Contacto de tercero" (Reclamos por llamadas/mensajes a familiares, referencias o no titulares).
2. "Ejercicio de derechos por no titular" (Solicitudes presentadas por abogados,personas que no son clientes y solo el contacto esta asociado, apoderados o terceros sin acreditar debidamente el poder legal del titular).
3. "Derivación - Es cliente cedente" (Casos donde la persona es cliente de una cartera que pertenece a una institución cedente y aqui solo se gestiona).

Devuelve la respuesta ÚNICAMENTE en este objeto JSON:
{
  "categoria": "<Nombre exacto de la categoría>",
  "contacto_reportado": "<dato que se solicita suspender su uso>", 
  "certeza_porcentaje": <Número entero de 0 a 100>,
  "titular_afectado": "<Nombre del titular>",
  "cliente_relacionado":<Puede se el mismo titular si es que es nuestro cliente, u otro si el cliente esta atado al contacto reportado>",
  "identificacion": "<Cédula/CI/RUC detectado o 'No detectado'>",
  "remitente_original": "<Nombre del abogado o persona que origina la queja>",
  "canales_contacto": "<Teléfono / Email autorizados>",
  "empresa_contacto": "<Usuario o empresa que envió el Email o llamo a hacer la cobranza>",
  "resumen_hilo": "<Breve resumen de 2 líneas de lo que solicita la DPO o el cliente>",
  "recomendacion_analista": "<Acción técnica e inmediata para el operador humano>"
}
"""

# Dataset de evaluación con Hilos de Correo Sintéticos Expandidos
DATASET_EVALUACION = [
    # Categoría 1: Contacto de tercero
    {
        "hilo": "De: delegadopdp@banco.com\nPara: gestion@cobranzas.com\nAsunto: RE: Queja LOPDP\n\nEstimados, trasladar a lista no gestionable por llamada a tercero.\n\n---\nDe: hermano@mail.com\nAsunto: Dejen de llamar a mi celular\n\nMe están llamando a mi teléfono a cobrar la deuda de un hermano. Yo no he autorizado el uso de mi número ni soy garante.",
        "categoria_esperada": "Contacto de tercero"
    },
    {
        "hilo": "De: delegadopdp@banco.com\nPara: ti@cobranzas.com\nAsunto: RE: Reclamo Nro 123\n\nFavor depurar de la base.\n\n---\nDe: usuario@gmail.com\nAsunto: Eliminación de datos\n\nRecibo correos y mensajes de texto sobre una cuenta de un tercero que no conozco. Solicito eliminen mi teléfono de su base de datos.",
        "categoria_esperada": "Contacto de tercero"
    },
    {
        "hilo": "De: esposo@hotmail.com\nPara: contacto@banco.com\nAsunto: Violación de privacidad\n\nEstán contactando a mi esposa para ubicarme. Ella jamás dio su consentimiento para que la llamen. Exijo la eliminación inmediata de su número.",
        "categoria_esperada": "Contacto de tercero"
    },
    {
        "hilo": "De: companero@empresa.com\nPara: reclamos@cobranzas.com\nAsunto: Cese de llamadas\n\nMe llaman insistentemente a consultar por un compañero de trabajo. No soy el titular de la deuda ni referencia, dejen de llamar a la oficina.",
        "categoria_esperada": "Contacto de tercero"
    },
    {
        "hilo": "De: familiar@gmail.com\nPara: servicioalcliente@banco.com\nAsunto: Contacto por WhatsApp sin autorización\n\nEstimados, me escribieron por WhatsApp a pedir referencias de un familiar. No he otorgado autorización de mis datos personales.",
        "categoria_esperada": "Contacto de tercero"
    },

    # Categoría 2: Ejercicio de derechos por no titular
    {
        "hilo": "De: madre@mail.com\nPara: delegadopdp@banco.com\nAsunto: Petición LOPDP\n\nBuenos días, solicito la eliminación de los datos personales y récord crediticio de mi hijo. Adjunto su número de cédula 1712345678.",
        "categoria_esperada": "Ejercicio de derechos por no titular"
    },
    {
        "hilo": "De: estudio@abogados.com\nPara: dpo@banco.com\nAsunto: Solicitud de Acceso - LOPDP\n\nSoy el abogado de la Sra. María Pérez (Cédula 0987654321) y pido el acceso a su historial de datos bajo la LOPDP sin adjuntar poder especial aún.",
        "categoria_esperada": "Ejercicio de derechos por no titular"
    },
    {
        "hilo": "De: hija@outlook.com\nPara: servcliente@banco.com\nAsunto: Actualización de datos de mi madre\n\nEscribo a nombre de mi madre para solicitar la rectificación de su dirección y correo electrónico registrados en su sistema.",
        "categoria_esperada": "Ejercicio de derechos por no titular"
    },
    {
        "hilo": "De: apoderado@juridico.ec\nPara: delegadopdp@banco.com\nAsunto: Eliminación de datos\n\nRequiero que borren la información comercial de mi representado, el Sr. Juan Gómez, cédula 1102345678.",
        "categoria_esperada": "Ejercicio de derechos por no titular"
    },
    {
        "hilo": "De: primo@gmail.com\nPara: dpo@banco.com\nAsunto: Oposición de tratamiento\n\nVengo en representación de un familiar para ingresar una solicitud de oposición al tratamiento de sus datos personales.",
        "categoria_esperada": "Ejercicio de derechos por no titular"
    },

    # Categoría 3: Derivación - No es cliente
    {
        "hilo": "De: ciudadano@yahoo.com\nPara: cobro@banco.com\nAsunto: Cobro no identificado\n\nEstimados, me cobran un valor de una tarjeta de crédito, pero yo nunca he sido cliente de su institución financiera.",
        "categoria_esperada": "Derivación - No es cliente"
    },
    {
        "hilo": "De: delegadopdp@banco.com\nPara: operaciones@cobranzas.com\nAsunto: RE: Cartera cedida\n\nVerificar origen. La cartera corresponde a Banco X y no a nuestra cartera propia.\n\n---\nDe: usuario@mail.com\nAsunto: Notificación de cobro\n\nRecibí una notificación, sin embargo la obligación pertenece al Banco X.",
        "categoria_esperada": "Derivación - No es cliente"
    },
    {
        "hilo": "De: cliente@gmail.com\nPara: estadosdecuenta@banco.com\nAsunto: Error en envío\n\nReviso mi casillero y me llega estado de cuenta de un producto que no contraté con ustedes, derivar al banco emisor correspondiente.",
        "categoria_esperada": "Derivación - No es cliente"
    },
    {
        "hilo": "De: afectado@outlook.com\nPara: reclamos@banco.com\nAsunto: Cartera castigada\n\nMe indican que estoy en cartera castigada pero la obligación es de otra cooperativa. Solicito derivar mi caso a la entidad dueña de la cartera.",
        "categoria_esperada": "Derivación - No es cliente"
    },
    {
        "hilo": "De: no_cliente@hotmail.com\nPara: dpo@banco.com\nAsunto: Sin relación comercial\n\nNo tengo ninguna relación comercial registrada con ustedes. Favor verificar si la gestión le pertenece a un tercero contratante.",
        "categoria_esperada": "Derivación - No es cliente"
    }
]

def ejecutar_evaluacion_masiva():
    if not OPENAI_API_KEY or OPENAI_API_KEY == "sk-proj-tu-key-aqui":
        print("❌ Error: Configure una API Key válida de OpenAI antes de ejecutar.")
        return

    client = OpenAI(api_key=OPENAI_API_KEY)
    print(f"🚀 Iniciando evaluación de {len(DATASET_EVALUACION)} casos con GPT-3.5...\n")

    aciertos = 0
    total = len(DATASET_EVALUACION)

    for i, item in enumerate(DATASET_EVALUACION, start=1):
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