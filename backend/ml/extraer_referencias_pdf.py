import fitz  # PyMuPDF
import os
import json

# Ruta donde se guardarán las referencias
REFERENCIAS_PATH = "backend/ml/data/referencias.json"

# Conceptos que se buscarán en los textos de los PDFs
CLAVES_CONCEPTUALES = [
    "regla del producto",
    "regla del cociente",
    "integral por partes",
    "sustitución trigonométrica",
    "límite lateral",
    "límite indeterminado",
    "continuidad",
    "teorema del sándwich",
    "derivada implícita",
    "regla de la cadena",
    "fracciones parciales",
    "cambio de variable",
    "función continua",
    "teorema de bolzano"
]

def extraer_conceptos_de_pdf(pdf_path):
    """Extrae los conceptos clave del contenido de un PDF."""
    referencias = {}
    nombre_archivo = os.path.basename(pdf_path)

    try:
        with fitz.open(pdf_path) as doc:
            for num_pagina, pagina in enumerate(doc, start=1):
                texto = pagina.get_text().lower()
                for concepto in CLAVES_CONCEPTUALES:
                    if concepto in texto:
                        if concepto not in referencias:
                            referencias[concepto] = []
                        referencia = f"📄 {nombre_archivo} - página {num_pagina}"
                        if referencia not in referencias[concepto]:
                            referencias[concepto].append(referencia)
        return referencias
    except Exception as e:
        print(f"❌ Error leyendo PDF: {e}")
        return {}

def actualizar_referencias_json(nuevas_refs):
    """Agrega nuevas referencias al archivo JSON, sin sobrescribir las existentes."""
    if os.path.exists(REFERENCIAS_PATH):
        with open(REFERENCIAS_PATH, "r", encoding="utf-8") as f:
            existentes = json.load(f)
    else:
        existentes = {}

    for concepto, refs in nuevas_refs.items():
        if concepto not in existentes:
            existentes[concepto] = []
        for ref in refs:
            if ref not in existentes[concepto]:
                existentes[concepto].append(ref)

    with open(REFERENCIAS_PATH, "w", encoding="utf-8") as f:
        json.dump(existentes, f, indent=2, ensure_ascii=False)

    print(f"✅ Referencias actualizadas en {REFERENCIAS_PATH}")

if __name__ == "__main__":
    pdf_path = input("🔍 Ingresa la ruta del PDF a procesar: ")
    if not os.path.isfile(pdf_path):
        print("❌ Archivo no encontrado.")
    else:
        nuevas_referencias = extraer_conceptos_de_pdf(pdf_path)
        if nuevas_referencias:
            actualizar_referencias_json(nuevas_referencias)
        else:
            print("⚠️ No se encontraron conceptos clave en el PDF.")
