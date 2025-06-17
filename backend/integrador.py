import sympy as sp
from sympy.parsing.sympy_parser import (parse_expr, standard_transformations,
                                       implicit_multiplication_application,
                                       convert_xor)
from datetime import datetime
from typing import Tuple, List, Dict, Optional
import numpy as np
import matplotlib.pyplot as plt
import io
from PIL import Image
import re

class IntegradorIA:
    """
    Clase para calcular integrales simbólicas (indefinidas y definidas)
    y generar explicaciones paso a paso en texto plano.
    También incluye funcionalidades para historial y graficación.
    """
    def __init__(self):
        self.historial: List[Dict] = []
        self.variable = 'x'  # Variable predeterminada como string

    def resolver(self, expresion: str, variable: str = 'x',
            limites: Optional[Tuple[str, str]] = None) -> Tuple[Optional[str], str]:
        self.variable = variable
        x = sp.symbols(variable)

        try:
            # 1. Parsear la expresión
            expr = self._parsear_expresion(expresion, x)

            # 2. Generar explicación inicial
            pasos = [
                f"Ejercicio: Integrar {sp.sstr(expr)} respecto a {variable}",
                f"Paso 1: Identificación del integrando: {self._identificar_componentes(expr)}",
                f"Paso 2: Métodos de integración aplicables: {self._aplicar_metodos(expr, x)}"
            ]

            # 3. Calcular la integral
            if limites:
                a, b = sp.sympify(limites[0]), sp.sympify(limites[1])
                if a.has(x) or b.has(x):
                    raise ValueError(f"Los límites de integración no deben contener la variable '{variable}'.")
                resultado_simbolico = sp.integrate(expr, (x, a, b))
                tipo = "definida"
                evaluacion_str = f"Evaluado de {a} a {b}"
                try:
                    resultado_num = float(resultado_simbolico.evalf())
                    evaluacion_str += f" ≈ {resultado_num:.6f}"
                except (TypeError, ValueError):
                    resultado_num = None
            else:
                resultado_simbolico = sp.integrate(expr, x)
                tipo = "indefinida"
                evaluacion_str = "+ Constante de integración C"
                resultado_num = None

            # 4. Simplificar el resultado simbólico
            resultado_simplificado = sp.simplify(resultado_simbolico)
            resultado_simplificado = resultado_simplificado.subs(sp.log('e'), 1).subs({'e': sp.E})
            if resultado_simplificado.has(sp.sin, sp.cos, sp.tan):
                resultado_simplificado = sp.trigsimp(resultado_simplificado)
            resultado_simplificado_str = sp.sstr(resultado_simplificado)

            # 5. Pasos intermedios automáticos (texto plano)
            pasos_intermedios = self._generar_pasos_intermedios(expr, x, resultado_simbolico, tipo)
            pasos.extend(pasos_intermedios)

            # 6. Resultado final (ahora en LaTeX)
            if limites:
                resultado_final_latex = f"$$\\int_{{{sp.latex(a)}}}^{{{sp.latex(b)}}} {sp.latex(expr)} \\,d{variable} = {sp.latex(resultado_simplificado)}$$"
                pasos.append("Resultado final (fórmula):")
                pasos.append(resultado_final_latex)
            else:
                resultado_final_latex = f"$$\\int {sp.latex(expr)} \\,d{variable} = {sp.latex(resultado_simplificado)} + C$$"
                pasos.append("Resultado final (fórmula):")
                pasos.append(resultado_final_latex)

            pasos.append(f"Fecha y hora: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

            # 7. Guardar en el historial
            self._guardar_historial(expresion, resultado_simbolico, tipo, limites, self.variable)

            # 8. Retornar resultado y explicación
            return resultado_simplificado_str, "\n".join(pasos)

        except (sp.SympifyError, ValueError, TypeError) as e:
            error_msg = f"❌ Error al procesar la expresión: {str(e)}"
            return None, error_msg
        except Exception as e:
            error_msg = f"❌ Ha ocurrido un error inesperado: {str(e)}"
            return None, error_msg

    def _parsear_expresion(self, expresion: str, variable: sp.Symbol) -> sp.Expr:
        expresion_limpia = expresion.strip()
        expresion_limpia = expresion_limpia.replace('^', '**')
        expresion_limpia = re.sub(r'd/d\w+', '', expresion_limpia)
        expresion_limpia = re.sub(r'\\?int\s*', '', expresion_limpia)
        expresion_limpia = re.sub(r'\,?d\w+', '', expresion_limpia)
        transformations = standard_transformations + (implicit_multiplication_application, convert_xor)
        try:
            local_dict_latex = {str(variable): variable}
            try:
                return sp.parse_latex(expresion_limpia, local_dict=local_dict_latex)
            except Exception:
                pass
            local_dict_parse = {str(variable): variable}
            return parse_expr(expresion_limpia, transformations=transformations, local_dict=local_dict_parse, evaluate=True)
        except Exception as e:
            raise sp.SympifyError(f"No se pudo convertir la cadena '{expresion}' a una expresión matemática. Verifique la sintaxis. Error: {e}")

    def _identificar_componentes(self, expr: sp.Expr) -> str:
        componentes = []
        if expr.is_Add:
            componentes.append("Suma/Resta de términos")
        if expr.is_Mul:
            componentes.append("Producto de factores")
        if expr.is_Pow:
            if expr.base.has(self.variable) and not expr.exp.has(self.variable):
                componentes.append("Expresión potencial (f(x)^n)")
            elif not expr.base.has(self.variable) and expr.exp.has(self.variable):
                componentes.append("Expresión exponencial (a^{f(x)})")
            elif expr.base.has(self.variable) and expr.exp.has(self.variable):
                componentes.append("Función Potencia-Exponencial (f(x)^{g(x)})")
            else:
                componentes.append("Constante elevada a Potencia")
        if expr.has(sp.sin):
            componentes.append("Funciones trigonométricas (seno)")
        if expr.has(sp.cos):
            componentes.append("Funciones trigonométricas (coseno)")
        if expr.has(sp.tan):
            componentes.append("Funciones trigonométricas (tangente)")
        if expr.has(sp.asin, sp.acos, sp.atan):
            componentes.append("Funciones trigonométricas inversas")
        if expr.has(sp.sinh, sp.cosh, sp.tanh):
            componentes.append("Funciones hiperbólicas")
        if expr.has(sp.exp):
            if not (expr.is_Pow and expr.base == sp.E):
                componentes.append("Función exponencial (e^{f(x)})")
        if expr.has(sp.log):
            componentes.append("Función logarítmica (ln o log)")
        if expr.has(sp.sqrt):
            componentes.append("Función raíz cuadrada")
        if expr.is_rational_function(self.variable):
            componentes.append("Función racional (fracción de polinomios)")
        componentes = sorted(list(set(componentes)))
        return ", ".join(componentes) if componentes else "Integrando básico (constante o variable simple)"

    def _aplicar_metodos(self, expr: sp.Expr, x: sp.Symbol) -> str:
        metodos = []
        if expr.is_Add:
            metodos.append("Linealidad de la integral: ∫(f(x) ± g(x)) dx = ∫f(x) dx ± ∫g(x) dx")
            metodos.append("Integral de constante por función: ∫c·f(x) dx = c·∫f(x) dx")
        if expr.is_Mul:
            metodos.append("Considerar sustitución (cambio de variable) si la expresión contiene una función y su derivada.")
            metodos.append("Considerar integración por partes: ∫u dv = uv - ∫v du. Útil para productos como x·sin(x), x·e^x, ln(x), etc.")
        if expr.is_Pow:
            base, exp = expr.base, expr.exp
            if exp == -1 and base.is_polynomial(x):
                metodos.append("Integral de 1/u: ∫1/u du = ln|u| + C")
            elif not exp.has(x) and base.has(x):
                metodos.append("Regla de la potencia: ∫x^n dx = x^(n+1)/(n+1) + C (para n ≠ -1)")
            elif exp.has(x) and not base.has(x):
                metodos.append("Integral exponencial general: ∫a^x dx = a^x/ln(a) + C (para a > 0, a ≠ 1)")
                metodos.append("Integral de e^x: ∫e^x dx = e^x + C")
        if expr.has(sp.sin, sp.cos, sp.tan, sp.sec, sp.csc, sp.cot):
            metodos.append("Integrales trigonométricas básicas: ∫sin(x) dx = -cos(x) + C, ∫cos(x) dx = sin(x) + C, etc.")
            metodos.append("Considerar identidades trigonométricas para simplificar antes de integrar.")
        if expr.has(sp.exp):
            if expr.is_Pow and expr.base == sp.E:
                pass
            else:
                metodos.append("Integral de e^{f(x)}: A menudo requiere sustitución si f'(x) está presente.")
        if expr.has(sp.log):
            metodos.append("Integral de ln(x): ∫ln(x) dx = x·ln(x) - x + C (a menudo por partes).")
            metodos.append("Integral de log_b(x): Convertir a base natural: log_b(x) = ln(x)/ln(b).")
        if expr.is_rational_function(self.variable):
            metodos.append("Fracciones parciales: Descomponer P(x)/Q(x) en fracciones más simples si el grado del numerador es menor que el del denominador. Si no, realizar división polinómica primero.")
        if expr.has(sp.sqrt):
            metodos.append("Considerar sustitución trigonométrica para expresiones con sqrt(a^2 ± x^2) o sqrt(x^2 ± a^2).")
            metodos.append("Regla de la potencia para raíces: sqrt(x) = x^(1/2), aplicar regla de la potencia.")
        if expr.has(sp.asin, sp.acos, sp.atan):
            metodos.append("Integrales de funciones trigonométricas inversas (directas o por partes).")
        metodos = sorted(list(set(metodos)))
        if metodos:
            return "; ".join(metodos)
        else:
            return "Método directo (regla de potencias, etc.) o combinación no identificada de técnicas. SymPy resolverá directamente."

    def _generar_pasos_intermedios(self, expr: sp.Expr, x: sp.Symbol, resultado: sp.Expr, tipo: str) -> List[str]:
        pasos = []
        # Suma/Resta de términos
        if expr.is_Add:
            pasos.append("Aplicando la linealidad de la integral:")
            suma = " + ".join([f"∫ {sp.sstr(arg)} d{x}" for arg in expr.args])
            pasos.append(f"∫ ({sp.sstr(expr)}) d{x} = {suma}")
            pasos.append("Se integra cada término por separado:")
            for arg in expr.args:
                try:
                    integral_arg = sp.integrate(arg, x)
                    pasos.append(f"∫ {sp.sstr(arg)} d{x} = {sp.sstr(integral_arg)}")
                except:
                    pasos.append(f"∫ {sp.sstr(arg)} d{x} = [cálculo]")
            pasos.append(f"Sumando los resultados: {sp.sstr(resultado)}" + (" + C" if tipo == "indefinida" else ""))
        # Producto de funciones (caso simple)
        elif expr.is_Mul:
            pasos.append("Producto de funciones:")
            factores = expr.args
            pasos.append("Se recomienda considerar integración por partes o sustitución si corresponde.")
            pasos.append("Factores:")
            for f in factores:
                pasos.append(f"- {sp.sstr(f)}")
        # Potencia simple
        elif expr.is_Pow and expr.base.has(x) and not expr.exp.has(x) and expr.exp != -1:
            base = expr.base
            exp = expr.exp
            if base == x:
                pasos.append("Aplicando la regla de la potencia:")
                pasos.append(f"∫ x^n dx = x^(n+1)/(n+1) + C con n={exp}")
                pasos.append(f"∫ {sp.sstr(expr)} d{x} = {sp.sstr(x)}^{exp+1}/({exp+1}) + C")
            else:
                pasos.append("Potencia de función:")
                pasos.append("Para ∫ [u(x)]^n dx, considerar sustitución si u'(x) está presente.")
        # Integral de 1/x
        elif expr.is_Pow and expr.base.has(x) and expr.exp == -1:
            base = expr.base
            pasos.append("Aplicando la regla de integración de 1/u:")
            pasos.append(f"∫ 1/{sp.sstr(base)} d{x} = ln|{sp.sstr(base)}| + C")
        # Trigonométricas básicas
        elif expr == sp.sin(x):
            pasos.append("Integral básica de seno:")
            pasos.append("∫ sin(x) dx = -cos(x) + C")
        elif expr == sp.cos(x):
            pasos.append("Integral básica de coseno:")
            pasos.append("∫ cos(x) dx = sin(x) + C")
        # Exponencial simple
        elif expr == sp.exp(x):
            pasos.append("Integral básica de exponencial:")
            pasos.append("∫ e^x dx = e^x + C")
        # Logarítmica simple
        elif expr == sp.log(x):
            pasos.append("Integral básica de logaritmo natural:")
            pasos.append("∫ ln(x) dx = x·ln(x) - x + C")
        # Fracción racional simple
        elif expr.is_rational_function(x):
            pasos.append("Fracción racional:")
            pasos.append("Se recomienda descomponer en fracciones parciales si es posible.")
        # Si no se generaron pasos específicos
        if not pasos and str(resultado) != str(expr):
            pasos.append("SymPy ha calculado la integral. La generación de pasos intermedios detallados para esta forma de expresión es compleja y no está implementada genéricamente en este momento.")
            pasos.append(f"Resultado obtenido: {sp.sstr(resultado)}" + (" + C" if tipo == "indefinida" else ""))
        elif not pasos:
            pasos.append("La integral de esta expresión es directa o se obtuvo mediante métodos no desglosados aquí.")
        return pasos

    def _guardar_historial(self, expresion: str, resultado: sp.Expr,
                          tipo: str, limites: Optional[Tuple[str, str]], variable: str):
        self.historial.append({
            'fecha': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'expresion': expresion,
            'resultado': str(resultado),
            'tipo': tipo,
            'limites': limites,
            'variable': variable
        })

    def limpiar_historial(self):
        self.historial = []
        print("Historial de integración limpiado.")

    def graficar_integral(self, expr_str: str, rango: Tuple[float, float] = (-5, 5)) -> Optional[Image.Image]:
        try:
            x = sp.symbols(self.variable)
            expr = self._parsear_expresion(expr_str, x)
            f = sp.lambdify(x, expr, 'numpy')
            integral_expr = sp.integrate(expr, x)
            F = sp.lambdify(x, integral_expr, 'numpy')
            x_vals = np.linspace(rango[0], rango[1], 400)
            y_vals = f(x_vals)
            y_int_vals = F(x_vals)
            plt.figure(figsize=(10, 6))
            plt.plot(x_vals, y_vals, label=f"f({self.variable})")
            plt.plot(x_vals, y_int_vals, label=f"Integral indefinida", linestyle='--')
            plt.legend()
            plt.grid(True)
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            plt.close()
            buf.seek(0)
            return Image.open(buf)
        except Exception as e:
            print(f"Error al graficar la integral: {e}")
            return None