import sympy as sp
from sympy.parsing.sympy_parser import (parse_expr, standard_transformations, 
                                       implicit_multiplication_application)
from datetime import datetime
from typing import Tuple, List, Dict, Optional, Union
import numpy as np
import matplotlib.pyplot as plt
import io
from PIL import Image
import re

class LimiteIA:
    def __init__(self):
        self.historial: List[Dict] = []
        self.variable_symbol: sp.Symbol = sp.symbols('x')
        self.variable_str: str = 'x'

    def resolver(self, expresion: str, variable: str = 'x', 
                punto: str = 'oo', direccion: str = '+') -> Tuple[str, str]:
        self.variable_str = variable
        self.variable_symbol = sp.symbols(variable)
        x = self.variable_symbol

        try:
            if not expresion:
                raise ValueError("La expresión no puede estar vacía.")
            try:
                expr = self._parsear_expresion(expresion)
                if x not in expr.free_symbols and str(x) in expresion:
                    raise ValueError(f"La variable '{variable}' no fue reconocida en la expresión.")
                elif x not in expr.free_symbols and str(x) not in expresion:
                    return str(expr), f"Límite de una constante o expresión sin la variable: el límite es {str(expr)}."
            except Exception as e:
                raise ValueError(f"Error al parsear la expresión: {e}")

            try:
                punto_sym = self._parsear_punto_limite(punto)
            except Exception as e:
                raise ValueError(f"Error al parsear el punto del límite '{punto}': {e}")

            direccion = self._normalizar_direccion(direccion)
            if direccion not in ['+', '-', '+-', '']:
                raise ValueError(f"Dirección de límite inválida '{direccion}'. Use '+', '-', '+-', o ''.")

            # 2. Generar explicación inicial
            pasos = [
                f"Ejercicio: Calcular el límite de {sp.sstr(expr)} cuando {variable} tiende a {str(punto_sym)}",
                f"Paso 1: Identificación del tipo de límite y la función: {self._identificar_tipo(expr, punto_sym, direccion)}",
                f"Paso 2: Evaluación inicial y técnicas aplicadas: {self._aplicar_tecnicas(expr, x, punto_sym, direccion)}"
            ]
            
            # 3. Calcular el límite (y verificar existencia para dos lados)
            limite_existe = True
            if direccion == '+-':
                try:
                    lim_der = sp.limit(expr, x, punto_sym, dir='+')
                    lim_izq = sp.limit(expr, x, punto_sym, dir='-')
                    if lim_der != lim_izq:
                        limite_existe = False
                        resultado = f"No existe (Límites laterales diferentes: {lim_der} ≠ {lim_izq})"
                        resultado_simplificado = resultado
                except Exception as e:
                    limite_existe = False
                    resultado = f"Error al calcular límites laterales: {e}"
                    resultado_simplificado = resultado
            else:
                try:
                    resultado = sp.limit(expr, x, punto_sym, dir=direccion if direccion != '' else 'real')
                    resultado_simplificado = sp.simplify(resultado) if not isinstance(resultado, str) else resultado
                except Exception as e:
                    limite_existe = False
                    resultado = f"Error al calcular el límite: {e}"
                    resultado_simplificado = resultado

            # 4. Análisis de Indeterminaciones y pasos intermedios
            pasos_intermedios = self._analizar_indeterminacion(expr, x, punto_sym, resultado)
            pasos_intermedios.extend(self._generar_pasos_intermedios_limite(expr, x, punto_sym, resultado))
            pasos.extend(pasos_intermedios)

            # 5. Construcción de la explicación final
            if limite_existe and not isinstance(resultado_simplificado, str):
                resultado_final_latex = f"$$\\lim_{{{variable} \\to {sp.latex(punto_sym)}}} {sp.latex(expr)} = {sp.latex(resultado_simplificado)}$$"
                pasos.append("Resultado final (fórmula):")
                pasos.append(resultado_final_latex)
                final_result_str = str(resultado_simplificado)
            else:
                pasos.append("Resultado final:")
                final_result_str = str(resultado)
                pasos.append(final_result_str)

            pasos.append(f"Fecha y hora: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

            self._guardar_historial(expresion, final_result_str, punto, direccion, limite_existe)
            return final_result_str, "\n".join(pasos)

        except ValueError as ve:
            error_msg = f"❌ Error de entrada: {str(ve)}"
            return "", error_msg
        except Exception as e:
            error_msg = f"❌ Error interno al calcular límite: {str(e)}"
            return "", error_msg

    def _parsear_expresion(self, expresion: str) -> sp.Expr:
        transformations = standard_transformations + (implicit_multiplication_application,)
        try:
            return sp.parse_latex(expresion)
        except Exception:
            try:
                return parse_expr(expresion, transformations=transformations, evaluate=False)
            except Exception:
                return parse_expr(expresion, transformations=standard_transformations, evaluate=False)

    def _parsear_punto_limite(self, punto_str: str) -> sp.Expr:
        punto_str_lower = punto_str.lower().strip()
        if punto_str_lower in ['inf', 'infinity', 'oo']:
            return sp.oo
        elif punto_str_lower in ['-inf', '-infinity', '-oo']:
            return -sp.oo
        else:
            try:
                return sp.sympify(punto_str)
            except Exception:
                transformations = standard_transformations + (implicit_multiplication_application,)
                return parse_expr(punto_str, transformations=transformations)

    def _normalizar_direccion(self, direccion_str: str) -> str:
        direccion_str_lower = direccion_str.lower().strip()
        if direccion_str_lower in ['right', 'derecha', '+']:
            return '+'
        elif direccion_str_lower in ['left', 'izquierda', '-']:
            return '-'
        elif direccion_str_lower in ['both', 'doslados', '+-']:
            return '+-'
        elif direccion_str_lower == '':
            return ''
        else:
            return direccion_str

    def _identificar_tipo(self, expr: sp.Expr, punto: sp.Expr, direccion: str) -> str:
        componentes = ["Función:"]
        if expr.is_polynomial(self.variable_symbol):
            componentes.append("Polinómica")
        elif expr.is_rational_function(self.variable_symbol):
            componentes.append("Racional")
        elif expr.has(sp.sin, sp.cos, sp.tan, sp.asin, sp.acos, sp.atan):
            componentes.append("Trigonométrica")
        elif expr.has(sp.exp):
            componentes.append("Exponencial")
        elif expr.has(sp.log):
            componentes.append("Logarítmica")
        elif expr.has(sp.sqrt):
            componentes.append("Con raíz")
        elif expr.is_Pow and expr.base.has(self.variable_symbol):
            componentes.append("Potencia")
        elif expr.is_Add or expr.is_Mul:
            if len(componentes) == 1:
                componentes.append("Compuesta (combinación de funciones)")
        if punto == sp.oo:
            componentes.append(f"Límite cuando {self.variable_str} tiende a +infinito.")
        elif punto == -sp.oo:
            componentes.append(f"Límite cuando {self.variable_str} tiende a -infinito.")
        elif punto.is_number:
            componentes.append(f"Límite cuando {self.variable_str} tiende a un punto finito ({punto}).")
        else:
            componentes.append(f"Límite cuando {self.variable_str} tiende a un punto simbólico ({punto}).")
        if direccion == '+':
            componentes.append("Dirección: por la derecha.")
        elif direccion == '-':
            componentes.append("Dirección: por la izquierda.")
        elif direccion == '+-':
            componentes.append("Dirección: límites laterales.")
        else:
            componentes.append("Dirección: bilateral (por ambos lados).")
        return " | ".join(componentes)

    def _aplicar_tecnicas(self, expr: sp.Expr, x: sp.Symbol, 
                        punto: sp.Expr, direccion: str) -> str:
        tecnicas = []
        try:
            eval_at_point = expr.subs(x, punto)
            if not eval_at_point.is_finite and not eval_at_point.is_infinite and not eval_at_point.has(sp.nan):
                tecnicas.append("Sustitución directa: Si la función es continua en el punto, el límite es f(punto).")
            elif eval_at_point.is_finite or eval_at_point.is_infinite:
                tecnicas.append("Evaluación directa: Se intenta sustituir el punto en la expresión.")
        except Exception:
            tecnicas.append("Evaluación en el punto: Se intenta sustituir el valor del punto.")
        if expr.is_rational_function(x) and (punto.is_number or punto in [sp.oo, -sp.oo]):
            tecnicas.append("Para funciones racionales: comparar grados del numerador y denominador (para límites en infinito) o factorización/L'Hôpital (para indeterminaciones finitas).")
        if expr.has(sp.sin, sp.cos, sp.tan, sp.asin, sp.acos, sp.atan) and punto.is_number:
            tecnicas.append("Para funciones trigonométricas: usar identidades trigonométricas o límites notables.")
        if expr.has(sp.exp, sp.log):
            tecnicas.append("Para funciones exponenciales/logarítmicas: considerar propiedades de logaritmos/exponentes, cambio de base, o L'Hôpital.")
        if punto in [sp.oo, -sp.oo]:
            if expr.is_polynomial(x):
                tecnicas.append(f"Para polinomios en infinito: el límite está determinado por el término de mayor grado de {x}.")
            elif expr.is_rational_function(x):
                tecnicas.append(f"Para funciones racionales en infinito: comparar el grado más alto de {x} en el numerador y denominador.")
            tecnicas.append("Comparación de crecimientos de funciones comunes (polinómicas, exponenciales, logarítmicas).")
        if expr.has(sp.Abs):
            tecnicas.append("Para valor absoluto: considerar los límites laterales para analizar el comportamiento cerca del punto crítico.")
        tecnicas.append("Regla de L'Hôpital: aplicable a indeterminaciones del tipo 0/0 o infinito/infinito. Consiste en derivar el numerador y denominador y tomar el límite de la nueva fracción.")
        tecnicas.append("Cambio de variable: útil para simplificar la expresión o el punto del límite.")
        tecnicas.append("Expansión en series de Taylor/Maclaurin: puede ser útil para evaluar límites en un punto finito, especialmente cuando hay indeterminaciones.")
        tecnicas.append("Límites notables: reconocer y aplicar límites estándar conocidos (ej: lim x→0 sin(x)/x = 1).")
        return "; ".join(tecnicas)

    def _analizar_indeterminacion(self, expr: sp.Expr, x: sp.Symbol, punto_sym: sp.Expr, resultado_sympy: sp.Expr) -> List[str]:
        pasos_ind = ["Análisis de indeterminaciones (si aplica):"]
        try:
            eval_subs = expr.subs(x, punto_sym)
            if eval_subs.has(sp.nan, sp.zoo) or not eval_subs.is_finite and not eval_subs.is_infinite:
                pasos_ind.append("Se detecta una forma indeterminada en la evaluación directa.")
                num, den = expr.as_numer_denom()
                try:
                    num_val = num.subs(x, punto_sym)
                    den_val = den.subs(x, punto_sym)
                    if (num_val == 0 and den_val == 0) or (num_val.is_infinite and den_val.is_infinite):
                        pasos_ind.append(f"Forma indeterminada: {num_val}/{den_val}. Considerar la Regla de L'Hôpital.")
                        if not isinstance(resultado_sympy, str) and (resultado_sympy.is_finite or resultado_sympy.is_infinite):
                            pasos_ind.append("SymPy aplicó técnicas (posiblemente L'Hôpital u otras) internamente para encontrar el resultado.")
                    elif num_val == 0 and den_val.is_infinite:
                        pasos_ind.append(f"Forma indeterminada: 0 * infinito. Puede requerir reescribir la expresión (ej: a 0/0 o infinito/infinito) para aplicar L'Hôpital o usar otras simplificaciones.")
                    elif num_val.is_infinite and den_val == 0:
                        pasos_ind.append(f"Forma indeterminada: infinito/0 o -infinito/0. Esto suele resultar en ±infinito. Analizar el signo del denominador cerca del punto.")
                    else:
                        pasos_ind.append("Indeterminación detectada, pero el tipo específico requeriría un análisis más profundo.")
                except Exception:
                    pasos_ind.append("No se pudo evaluar el numerador/denominador en el punto para identificar la forma indeterminada exacta.")
        except Exception:
            pasos_ind.append("No se pudo realizar un análisis detallado de la indeterminación.")
        if len(pasos_ind) == 1 and not isinstance(resultado_sympy, str) and (resultado_sympy.is_finite or resultado_sympy.is_infinite):
            pasos_ind.append("La evaluación directa parece no producir una forma indeterminada principal.")
        return pasos_ind if len(pasos_ind) > 1 else []

    def _generar_pasos_intermedios_limite(self, expr: sp.Expr, x: sp.Symbol, punto: sp.Expr, resultado: sp.Expr) -> list:
        pasos = []
        try:
            valor_sust = expr.subs(x, punto)
            if valor_sust.is_finite:
                pasos.append(f"Sustitución directa: Al evaluar en el punto, se obtiene {str(valor_sust)}.")
            elif valor_sust.has(sp.nan, sp.zoo):
                pasos.append("Sustitución directa: Se obtiene una forma indeterminada.")
        except Exception:
            pasos.append("Sustitución directa: No se pudo evaluar la expresión en el punto.")
        num, den = expr.as_numer_denom()
        try:
            num_val = num.subs(x, punto)
            den_val = den.subs(x, punto)
            if num_val == 0 and den_val == 0:
                pasos.append("Forma indeterminada: 0/0, se puede aplicar la Regla de L'Hôpital o simplificar.")
            elif num_val.is_infinite and den_val.is_infinite:
                pasos.append("Forma indeterminada: infinito/infinito, se puede aplicar la Regla de L'Hôpital o simplificar.")
        except Exception:
            pass
        if expr.has(sp.sin, sp.cos, sp.tan) and punto == 0:
            if expr == sp.sin(x)/x:
                pasos.append("Límite notable: lim x→0 sin(x)/x = 1.")
        if not pasos:
            pasos.append("SymPy ha calculado el límite. La generación de pasos intermedios detallados para esta forma de expresión es compleja y no está implementada genéricamente en este momento.")
        return pasos

    def _guardar_historial(self, expresion: str, resultado_str: str,
                          punto_str: str, direccion_str: str, existe: bool):
        self.historial.append({
            'fecha': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'expresion': expresion,
            'resultado': resultado_str,
            'punto': punto_str,
            'direccion': direccion_str,
            'existe': existe,
            'variable': self.variable_str
        })

    def mostrar_historial(self, formato: str = 'markdown') -> str:
        if not self.historial:
            return "El historial está vacío."
        if formato == 'markdown':
            return self._historial_markdown()
        else:
            return str(self.historial)

    def _historial_markdown(self) -> str:
        tabla = [
            "| Fecha | Expresión | Límite | Punto | Dirección | ¿Existe? |",
            "|-------|-----------|--------|-------|-----------|----------|"
        ]
        for item in self.historial:
            expr_display = item['expresion'].replace('|', '\\|')
            res_display = item['resultado'].replace('|', '\\|')
            tabla.append(
                f"| {item['fecha']} | `{expr_display}` "
                f"| `{res_display}` | {item['punto']} "
                f"| {item['direccion']} | {'Sí' if item['existe'] else 'No'} |"
            )
        return "\n".join(tabla)

    def graficar_limite(self, expr_str: str,
                    punto: str = 'oo',
                    rango: Tuple[float, float] = (-5, 5)) -> Optional[Image.Image]:
        try:
            expr = self._parsear_expresion(expr_str)
            variables = list(expr.free_symbols)
            if len(variables) != 1:
                raise ValueError("La gráfica solo puede generarse para funciones con una sola variable.")
            x = variables[0]
            self.variable_str = str(x)
            f = sp.lambdify(x, expr, 'numpy')
            punto_sym = self._parsear_punto_limite(punto)
            if punto_sym == sp.oo:
                plot_range = (10, 100) if rango[1] <= 10 else (max(10, rango[0]), rango[1])
                x_vals = np.linspace(plot_range[0], plot_range[1], 400)
            elif punto_sym == -sp.oo:
                plot_range = (-100, -10) if rango[0] >= -10 else (rango[0], min(-10, rango[1]))
                x_vals = np.linspace(plot_range[0], plot_range[1], 400)
            else:
                try:
                    p = float(sp.N(punto_sym))
                    delta = min(abs(rango[1] - p), abs(p - rango[0])) / 2.0
                    if delta == 0:
                        delta = (rango[1] - rango[0]) / 10.0
                    plot_start = max(rango[0], p - delta)
                    plot_end = min(rango[1], p + delta)
                    if plot_end - plot_start < (rango[1] - rango[0]) / 100:
                        mid = (plot_start + plot_end) / 2
                        plot_start = max(rango[0], mid - (rango[1] - rango[0]) / 50)
                        plot_end = min(rango[1], mid + (rango[1] - rango[0]) / 50)
                    if plot_start >= plot_end:
                        plot_start = rango[0]
                        plot_end = rango[1]
                    x_vals = np.linspace(plot_start, plot_end, 400)
                except Exception:
                    x_vals = np.linspace(rango[0], rango[1], 400)
                    p = None
            try:
                if punto_sym.is_number and 'p' in locals() and p is not None:
                    if not expr.is_polynomial(x) and not expr.is_Piecewise:
                        tolerance = (x_vals[-1] - x_vals[0]) / 1000
                        mask = np.abs(x_vals - p) > tolerance
                        x_vals_plot = x_vals[mask]
                        y_vals = f(x_vals_plot)
                    else:
                        x_vals_plot = x_vals
                        y_vals = f(x_vals_plot)
                else:
                    x_vals_plot = x_vals
                    y_vals = f(x_vals_plot)
            except Exception as eval_error:
                with np.errstate(all='ignore'):
                    y_vals = f(x_vals)
                    y_vals[np.isinf(y_vals) | np.isnan(y_vals)] = np.nan
                x_vals_plot = x_vals
            plt.figure(figsize=(10, 6))
            plt.plot(x_vals_plot, y_vals, label=f"f({x})")
            if punto_sym.is_number and 'p' in locals() and p is not None:
                if plot_start <= p <= plot_end:
                    plt.axvline(x=p, color='r', linestyle='--', alpha=0.5, label=f"{self.variable_str} → {str(punto_sym)}")
            plt.title(f"Comportamiento de f({self.variable_str}) cerca de {self.variable_str} → {str(punto_sym)}")
            plt.xlabel(self.variable_str)
            plt.ylabel(f"f({self.variable_str})")
            plt.grid(True)
            plt.legend()
            finite_y_vals = y_vals[np.isfinite(y_vals)]
            if len(finite_y_vals) > 1:
                y_min, y_max = np.min(finite_y_vals), np.max(finite_y_vals)
                y_range = y_max - y_min
                if y_range > 0:
                    plt.ylim(y_min - y_range * 0.1, y_max + y_range * 0.1)
                else:
                    plt.ylim(y_min - 1, y_max + 1)
            elif len(finite_y_vals) == 1:
                plt.ylim(finite_y_vals[0] - 1, finite_y_vals[0] + 1)
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            plt.close()
            buf.seek(0)
            return Image.open(buf)
        except Exception as e:
            print(f"Error al graficar: {str(e)}")
            plt.close('all')
            return None

    def limpiar_historial(self):
        self.historial = []
        print("Historial de límites limpiado.")