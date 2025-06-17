import os
import json
import fitz  # PyMuPDF

# Ruta del dataset JSON
DATASET_PATH = "backend/ml/data/synthetic_math_dataset.json"

def extraer_texto_pdf(pdf_path):
    texto = ""
    try:
        with fitz.open(pdf_path) as doc:
            for page in doc:
                texto += page.get_text()
        return texto.strip()
    except Exception as e:
        print(f"❌ Error leyendo PDF: {e}")
        return ""

def convertir_pdf_a_entrada(pdf_path):
    texto = extraer_texto_pdf(pdf_path)
    if not texto:
        return []

    lineas = texto.split("\n")
    ejemplos = []
    for linea in lineas:
        linea = linea.strip()
        if len(linea) > 0:
            ejemplos.append({"input": linea, "output": ""})
    return ejemplos

def agregar_a_dataset(pdf_path: str) -> str:
    try:
        nuevos_ejemplos = convertir_pdf_a_entrada(pdf_path)

        if not nuevos_ejemplos:
            return "⚠️ El PDF no contiene texto útil."

        if os.path.exists(DATASET_PATH):
            with open(DATASET_PATH, "r", encoding="utf-8") as f:
                dataset_actual = json.load(f)
        else:
            dataset_actual = []

        dataset_actual.extend(nuevos_ejemplos)
        dataset_actual = list({json.dumps(e, sort_keys=True): e for e in dataset_actual}.values())  # Eliminar duplicados

        with open(DATASET_PATH, "w", encoding="utf-8") as f:
            json.dump(dataset_actual, f, indent=2, ensure_ascii=False)

        return f"✅ Se agregaron {len(nuevos_ejemplos)} nuevos ejemplos al dataset. Total: {len(dataset_actual)}"

    except Exception as e:
        return f"❌ Error al procesar el archivo PDF: {e}"

