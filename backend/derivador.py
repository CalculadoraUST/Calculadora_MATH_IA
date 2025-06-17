from IPython.display import display
import sympy as sp
from sympy.parsing.sympy_parser import (parse_expr, standard_transformations,
                                       implicit_multiplication_application)
from datetime import datetime
from typing import Tuple, List, Dict, Any, Union
import numpy as np
import matplotlib.pyplot as plt
import io
from PIL import Image

SymPyObject = Union[sp.Expr, sp.Symbol, sp.Number]

class DerivadorIA:
    def __init__(self):
        self.historial: List[Dict[str, Any]] = []
        self.variable: sp.Symbol = sp.symbols('x')

    def _parsear_expresion(self, expresion: str) -> sp.Expr:
        transformations = standard_transformations + (implicit_multiplication_application,)
        try:
            return sp.parse_latex(expresion)
        except Exception as e_latex:
            try:
                return parse_expr(expresion, transformations=transformations)
            except Exception as e_standard:
                raise ValueError(f"No se pudo interpretar la expresión matemática '{expresion}'. "
                               f"(Error LaTeX: {e_latex}) (Error estándar: {e_standard})")

    def _identificar_componentes(self, expr: sp.Expr) -> str:
        componentes = []
        if expr.is_Add:
            componentes.append("Suma algebraica de términos")
        if expr.is_Mul:
            componentes.append("Producto de factores")
        if expr.is_Pow:
            componentes.append("Potenciación")
        if any(expr.has(func) for func in [sp.sin, sp.cos, sp.tan, sp.asin, sp.acos, sp.atan]):
            componentes.append("Funciones trigonométricas")
        if any(expr.has(func) for func in [sp.sinh, sp.cosh, sp.tanh, sp.asinh, sp.acosh, sp.atanh]):
            componentes.append("Funciones hiperbólicas")
        if expr.has(sp.exp):
            componentes.append("Función exponencial")
        if expr.has(sp.log):
            componentes.append("Función logarítmica")
        if expr.has(sp.sqrt):
            componentes.append("Raíz cuadrada")
        if expr.has(sp.Abs):
            componentes.append("Valor absoluto")
        return ", ".join(componentes) if componentes else "Expresión simple o atómica"

    def _aplicar_reglas(self, expr: sp.Expr, orden: int) -> str:
        reglas = []
        x = self.variable
        if expr.is_Add:
            reglas.append("Regla de la suma: (f + g)' = f' + g'")
        if expr.is_Mul and len(expr.args) > 1:
            reglas.append("Regla del producto: (f·g)' = f'·g + f·g'")
        if expr.is_Function and any(arg.has(x) for arg in expr.args) and not (len(expr.args) == 1 and expr.args[0] == x):
            reglas.append("Regla de la cadena: f(g(x))' = f'(g(x))*g'(x)")
        if expr.has(sp.sin): 
            reglas.append("Derivada de sen(u): d/dx sin(u) = cos(u) * u'")
        if expr.has(sp.cos): 
            reglas.append("Derivada de cos(u): d/dx cos(u) = -sin(u) * u'")
        if expr.has(sp.exp): 
            reglas.append("Derivada de e^u: d/dx e^u = e^u * u'")
        if expr.has(sp.log): 
            reglas.append("Derivada de ln(u): d/dx ln(u) = 1/u * u'")
        reglas_unicas = sorted(list(set(reglas)))
        if reglas_unicas:
            return "Reglas de derivación aplicables:\n" + "\n".join(f"- {r}" for r in reglas_unicas)
        else:
            return "Regla de potencias o derivada fundamental aplicada."

    def _generar_pasos_intermedios(self, expr, variable):
        pasos = []
        x = variable if isinstance(variable, sp.Symbol) else sp.symbols(variable)
        # Regla del producto
        if expr.is_Mul:
            factores = expr.args
            pasos.append("Aplicando la regla del producto:")
            for i, f in enumerate(factores):
                otros = [factores[j] for j in range(len(factores)) if j != i]
                pasos.append(f"d/d{variable}({sp.sstr(f)} * {sp.sstr(sp.Mul(*otros))}) = d/d{variable}({sp.sstr(f)}) * {sp.sstr(sp.Mul(*otros))} + {sp.sstr(f)} * d/d{variable}({sp.sstr(sp.Mul(*otros))})")
            pasos.append("Se suman las derivadas de cada factor multiplicado por el producto de los demás.")
        # Regla de la cadena para funciones compuestas
        elif expr.is_Function and expr.args:
            pasos.append("Aplicando la regla de la cadena:")
            pasos.append("Si y = f(u) y u = g(x), entonces dy/dx = f'(u) * g'(x)")
        # Regla de la potencia
        elif expr.is_Pow and expr.base.has(x) and not expr.exp.has(x):
            pasos.append("Aplicando la regla de la potencia:")
            pasos.append("d/dx(x^n) = n * x^(n-1)")
        # Derivada de funciones trigonométricas
        elif expr == sp.sin(x):
            pasos.append("Aplicando la derivada de sin(x):")
            pasos.append("d/dx(sin(x)) = cos(x)")
        elif expr == sp.cos(x):
            pasos.append("Aplicando la derivada de cos(x):")
            pasos.append("d/dx(cos(x)) = -sin(x)")
        # Si no hay caso especial, mensaje genérico
        if not pasos:
            pasos.append("SymPy ha calculado la derivada. La generación de pasos intermedios detallados para esta forma de expresión no está implementada en este momento.")
        return pasos

    def resolver(self, expresion: str, orden: int = 1, variable: str = None) -> Tuple[str, str]:
        try:
            if variable:
                self.variable = sp.symbols(variable)
            expr = self._parsear_expresion(expresion)
            pasos = [
                f"Ejercicio: Derivar {sp.sstr(expr)} respecto a {self.variable}",
                f"Paso 1: Identificación de la función: {self._identificar_componentes(expr)}",
                f"Paso 2: {self._aplicar_reglas(expr, orden)}"
            ]
            if orden == 1:
                pasos.extend(self._generar_pasos_intermedios(expr, self.variable))
            derivada = sp.diff(expr, self.variable, orden)
            derivada_simplificada = sp.simplify(derivada)
            if derivada_simplificada.has(sp.sin, sp.cos, sp.tan):
                derivada_simplificada = sp.trigsimp(derivada_simplificada)
            derivada_simplificada_str = sp.sstr(derivada_simplificada)

            # Resultado final en LaTeX
            resultado_final_latex = f"$$\\frac{{d}}{{d{sp.latex(self.variable)}}}\\left({sp.latex(expr)}\\right) = {sp.latex(derivada_simplificada)}$$"
            pasos.append("Resultado final (fórmula):")
            pasos.append(resultado_final_latex)
            pasos.append(f"Fecha y hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            self._guardar_historial(expresion, derivada_simplificada, orden)
            return derivada_simplificada_str, "\n".join(pasos)
        except Exception as e:
            return "", f"❌ Error interno: {str(e)}"

    def _guardar_historial(self, expresion: str, resultado: sp.Expr, orden: int):
        self.historial.append({
            'fecha': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'expresion': expresion,
            'resultado': str(resultado),
            'orden': orden,
            'variable': str(self.variable)
        })

    def mostrar_historial(self, formato: str = 'markdown'):
        if not self.historial:
            return "El historial está vacío."
        if formato == 'markdown':
            tabla = [
                "| Fecha | Expresión | Resultado | Orden | Variable |",
                "|-------|-----------|-----------|-------|----------|"
            ]
            for item in self.historial:
                tabla.append(
                    f"| {item['fecha']} | `{item['expresion']}` "
                    f"| `{item['resultado']}` | {item['orden']}° "
                    f"| {item['variable']} |"
                )
            return "\n".join(tabla)
        else:
            return str(self.historial)

    def graficar(self, expr_str: str, rango: Tuple[float, float] = (-5, 5)) -> Image.Image:
        try:
            expr = self._parsear_expresion(expr_str)
            x = self.variable
            f = sp.lambdify(x, expr, 'numpy')
            df = sp.lambdify(x, sp.diff(expr, x), 'numpy')
            x_vals = np.linspace(rango[0], rango[1], 400)
            y_vals = f(x_vals)
            dy_vals = df(x_vals)
            plt.figure(figsize=(10, 6))
            plt.plot(x_vals, y_vals, label=f"f({x})")
            plt.plot(x_vals, dy_vals, label=f"f'({x})", linestyle='--')
            plt.legend()
            plt.grid(True)
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            plt.close()
            buf.seek(0)
            return Image.open(buf)
        except Exception as e:
            print(f"Error al graficar: {e}")
            return None

    def limpiar_historial(self):
        self.historial = []
        print("Historial limpiado.")