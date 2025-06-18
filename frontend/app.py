import gradio as gr
from backend.nlp_processor import NLPProcessor
from backend.derivador import DerivadorIA
from backend.limite import LimiteIA
from backend.integrador import IntegradorIA
from backend.ml.dataset_builder import agregar_a_dataset
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application
from collections import Counter
import PIL.Image
import csv
import os
from datetime import datetime
import json
import pandas as pd

class MathAIChatbot:
    def __init__(self):
        self.nlp = NLPProcessor()
        self.solvers = {
            "derivada": DerivadorIA(),
            "limite": LimiteIA(),
            "integral": IntegradorIA()
        }
        self.history = []
        self.usage_counter = Counter()
        self.last_plot_image = None
        self.historial_csv = "historial_consultas.csv"
        self._crear_csv_si_no_existe()
        self.referencias_path = "backend/ml/data/referencias.json"

    def _crear_csv_si_no_existe(self):
        if not os.path.exists(self.historial_csv):
            with open(self.historial_csv, mode='w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow(["Fecha", "Expresión", "Tipo de operación", "Resultado"])

    def _guardar_en_historial_csv(self, expr: str, tipo: str, resultado: str):
        with open(self.historial_csv, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            fecha = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            writer.writerow([fecha, expr, tipo, resultado])

    def preprocess_expression(self, expr: str) -> str:
        expr = expr.replace("^", "**")
        expr = expr.replace(")(", ")*(")
        return expr

    def is_valid_expression(self, expr: str) -> bool:
        try:
            transformations = standard_transformations + (implicit_multiplication_application,)
            parse_expr(expr, transformations=transformations)
            return True
        except Exception:
            return False

    def get_top_queries_markdown(self):
        try:
            df = pd.read_csv('historial_consultas.csv')
            top_expresiones = df['Expresión'].value_counts().head(5)
            markdown = "### Ejercicios más consultados:\n"
            for idx, (expresion, count) in enumerate(top_expresiones.items(), 1):
                markdown += f"{idx}. {expresion} ({count} consultas)\n"
            return markdown
        except Exception as e:
            return "No hay datos de ejercicios consultados aún."

    def buscar_referencias(self, pasos_texto: str) -> str:
        if not os.path.exists(self.referencias_path):
            return ""
        with open(self.referencias_path, "r", encoding="utf-8") as f:
            referencias = json.load(f)

        resultado = []
        for clave in referencias:
            if clave.lower() in pasos_texto.lower():
                refs = referencias[clave]
                resultado.append(f"🔎 **Referencia para '{clave}':**\n" + "\n".join(f"- {r}" for r in refs))

        return "\n\n".join(resultado)

    def solve_problem(self, text: str, history=None):
        if history is None:
            history = []
        self.last_plot_image = None
        try:
            operation_details = self.nlp.get_operation_details(text)
            op_type = operation_details["operation"]
            expr = operation_details["expression"]
            expr = self.preprocess_expression(expr)
            params = operation_details.get("params", {})

             # --- PRINTS DE DEPURACIÓN ---
            print("Texto original recibido:", text)
            print("Expresión después de NLPProcessor:", operation_details["expression"])
            print("Expresión después de preprocess_expression:", expr)
            print("Parámetros detectados:", params)

            if op_type not in self.solvers:
                raise ValueError("No se pudo clasificar la operación solicitada. Intenta usar palabras clave como 'deriva', 'límite' o 'integra'.")

            if not self.is_valid_expression(expr):
                error_msg = (
                    f"⚠️ **La expresión ingresada no es válida:** `{expr}`\n\n"
                    "🔧 Asegúrate de usar sintaxis correcta, por ejemplo:\n"
                    "- Usa `**` en lugar de `^` para potencias\n"
                    "- Usa `*` para multiplicar: `x * sin(x)`\n"
                    "- Evita paréntesis mal cerrados o símbolos extraños\n\n"
                    "📌 Ejemplo válido: `x**2 + sin(x)`"
                )
                history += [
                    {"role": "user", "content": text},
                    {"role": "assistant", "content": error_msg}
                ]
                return history

            solver = self.solvers[op_type]
            if op_type == "derivada":
                result, steps = solver.resolver(expr)
                self.last_plot_image = solver.graficar(expr)
            elif op_type == "limite":
                variable = params.get("variable", "x")
                punto = params.get("punto", "oo")
                result, steps = solver.resolver(expr, variable=variable, punto=punto)
                self.last_plot_image = solver.graficar_limite(expr, punto=punto)
            elif op_type == "integral":
                result, steps = solver.resolver(expr)
                self.last_plot_image = solver.graficar_integral(expr)

            self.usage_counter[expr] += 1
            self._guardar_en_historial_csv(expr, op_type, str(result))

            referencias = self.buscar_referencias(steps)

            response = (
                f"🔎 **Problema clasificado como**: {op_type.upper()}\n\n"
                f"📝 **Expresión**: `{expr}`\n\n"
                f"📚 **Pasos de solución**:\n{steps}\n\n"
                f"{referencias}\n\n"
                f"✅ **Resultado final**: `{result}`"
            )

            self.history.append({"input": text, "output": response, "type": op_type})
            history += [
                {"role": "user", "content": text},
                {"role": "assistant", "content": response}
            ]
            return history

        except Exception as e:
            history += [
                {"role": "user", "content": text},
                {"role": "assistant", "content": f"❌ Error: {str(e)}"}
            ]
            return history

    def launch_interface(self):
        with gr.Blocks(title="IA de Matemáticas", theme=gr.themes.Soft()) as app:
            gr.Markdown("# 🧮 Asistente de Cálculo Inteligente")
            gr.Image(value="frontend/assets/logo.png", width=200, show_label=False)

            with gr.Row():
                chatbot = gr.Chatbot(label="Conversación", height=500, type="messages")
                with gr.Column():
                    input_text = gr.Textbox(label="Escribe tu problema", placeholder="Ej: Deriva x^2 + sin(x)")
                    submit_btn = gr.Button("Resolver", variant="primary")
                    clear_btn = gr.Button("Limpiar")

            with gr.Row():
                top_queries_output = gr.Markdown(value=self.get_top_queries_markdown())

            image_output = gr.Image(label="📈 Gráfica generada")

            with gr.Row():
                file_input = gr.File(label="📁 Sube PDF con ejercicios", file_types=[".pdf"])
                upload_btn = gr.Button("Agregar al dataset")

            with gr.Accordion("📚 Ver bibliografía recomendada", open=False):
                gr.Markdown("""
                ### Bibliografía recomendada:
                - **Cálculo de una variable**, James Stewart, 8va Edición.
                - **Matemáticas II**, Granville.
                - **Compendios de clase (PDFs cargados)**
                - **Videos de Khan Academy sobre derivadas e integrales**
                """)

            examples = gr.Examples(
                examples=[
                    "Calcula la derivada de x^3 + 2x",
                    "Encuentra el límite de sin(x)/x cuando x tiende a 0",
                    "Integra e^x * cos(x)"
                ],
                inputs=input_text
            )

            def wrapped_solver(text, chat):
                updated_chat = self.solve_problem(text, chat)
                top_queries_output.value = self.get_top_queries_markdown()
                return updated_chat, self.last_plot_image

            def wrapped_file_upload(file):
                return agregar_a_dataset(file.name)

            submit_btn.click(fn=wrapped_solver, inputs=[input_text, chatbot], outputs=[chatbot, image_output])
            clear_btn.click(fn=lambda: ([], None), inputs=None, outputs=[chatbot, image_output], queue=False)
            input_text.submit(fn=wrapped_solver, inputs=[input_text, chatbot], outputs=[chatbot, image_output])

            upload_btn.click(fn=wrapped_file_upload, inputs=file_input, outputs=None)

        return app

if __name__ == "__main__":
    chatbot = MathAIChatbot()
    app = chatbot.launch_interface()
    app.launch(server_port=8502, share=True)