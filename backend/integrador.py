import sympy as sp
from sympy.parsing.sympy_parser import (parse_expr,standard_transformations, 
                                       implicit_multiplication_application)
from datetime import datetime
from typing import Tuple, List, Dict, Optional
import numpy as np
import matplotlib.pyplot as plt
import io
from PIL import Image

class IntegradorIA:
    def __init__(self):
        self.historial: List[Dict] = []
        self.variable = 'x'  # Variable predeterminada
        
    def resolver(self, expresion: str, variable: str = 'x', 
                limites: Optional[Tuple[str, str]] = None) -> Tuple[str, str]:
        """
        Calcula la integral y genera explicación detallada en Markdown.
        
        Args:
            expresion: Expresión matemática como string.
            variable: Variable de integración (default: 'x').
            limites: Tupla (a, b) para integral definida (opcional).
            
        Returns:
            Tuple: (resultado, explicación_markdown)
        """
        try:
            self.variable = variable
            x = sp.symbols(variable)
            expr = self._parsear_expresion(expresion)
            
            # Generar explicación paso a paso
            pasos = [
                f"## 🔍 Cálculo de la integral: ${sp.latex(expr)}\\ d{variable}$" + 
                (f" desde ${limites[0]}$ hasta ${limites[1]}$" if limites else ""),
                "### 📚 **Paso 1: Identificación de componentes**",
                self._identificar_componentes(expr),
                "### 🛠 **Paso 2: Métodos de integración aplicados**",
                self._aplicar_metodos(expr, x),
            ]
            
            # Calcular la integral
            if limites:
                a, b = sp.sympify(limites[0]), sp.sympify(limites[1])
                resultado = sp.integrate(expr, (x, a, b))
                tipo = "definida"
                evaluacion = f"Evaluado en {variable}={limites[0]} a {variable}={limites[1]}"
                resultado_num = float(resultado.evalf())
            else:
                resultado = sp.integrate(expr, x)
                tipo = "indefinida"
                evaluacion = f"+ Constante de integración C"
                resultado_num = None
                
            # Simplificar el resultado
            resultado_simplificado = sp.simplify(resultado)
            
            # Construcción de la explicación final
            pasos.extend([
                "### ✅ **Resultado final**",
                self._formatear_resultado(expr, resultado_simplificado, tipo, variable),
                "---",
                f"*🔎 Simplificación:* ${sp.latex(resultado)} \\Rightarrow {sp.latex(resultado_simplificado)}$",
                f"*📅 Operación registrada el {datetime.now().strftime('%Y-%m-%d %H:%M')}*",
                f"*{evaluacion}*" + (f" ≈ {resultado_num:.4f}" if resultado_num is not None else "")
            ])
            
            self._guardar_historial(expresion, resultado_simplificado, tipo, limites)
            return str(resultado_simplificado), "\n".join(pasos)
            
        except Exception as e:
            error_msg = f"❌ **Error al integrar**: {str(e)}"
            return "", error_msg

    def _parsear_expresion(self, expresion: str) -> sp.Expr:
        """Convierte string a expresión SymPy con transformaciones avanzadas."""
        try:
            # Intentar parsear como LaTeX
            return sp.parse_latex(expresion)
        except:
            # Usar parser estándar con transformaciones
            transformations = standard_transformations + (implicit_multiplication_application,)
            return parse_expr(expresion, transformations=transformations)

    def _identificar_componentes(self, expr: sp.Expr) -> str:
        """Analiza y describe la estructura del integrando."""
        componentes = []
        
        if expr.is_Add:
            componentes.append("**Suma de términos** (integral por partes)")
        if expr.is_Mul:
            componentes.append("**Producto de factores** (considerar integración por partes)")
        if expr.is_Pow:
            componentes.append("**Expresión potencial**")
        if expr.has(sp.sin, sp.cos, sp.tan):
            componentes.append("**Funciones trigonométricas**")
        if expr.has(sp.exp):
            componentes.append("**Función exponencial**")
        if expr.has(sp.log):
            componentes.append("**Función logarítmica**")
        if expr.is_rational_function():
            componentes.append("**Función racional** (considerar fracciones parciales)")
            
        return ", ".join(componentes) if componentes else "Integrando básico"

    def _aplicar_metodos(self, expr: sp.Expr, x: sp.Symbol) -> str:
        """Explica los métodos de integración aplicables con mayor detalle."""
        metodos = []
        
        # 1. Métodos básicos
        if expr.is_polynomial() and not expr.has(sp.log):
            metodos.append(
                "**Regla de la potencia**:\n"
                "$$\int x^n dx = \\begin{cases}\n"
                "\\frac{x^{n+1}}{n+1} + C & \\text{si } n \\neq -1 \\\\\n"
                "\\ln|x| + C & \\text{si } n = -1\n"
                "\\end{cases}$$"
            )
        
        # 2. Funciones exponenciales
        if expr.has(sp.exp):
            metodos.append(
                "**Integral exponencial**:\n"
                "$$\int e^x dx = e^x + C$$\n"
                "$$\int a^x dx = \\frac{a^x}{\ln a} + C$$"
            )
        
        # 3. Funciones trigonométricas
        trig_funcs = {
            sp.sin: "$\int \sin(x) dx = -\cos(x) + C$",
            sp.cos: "$\int \cos(x) dx = \sin(x) + C$",
            sp.tan: "$\int \tan(x) dx = -\ln|\cos(x)| + C$",
            sp.sec: "$\int \sec(x) dx = \ln|\sec(x) + \tan(x)| + C$",
            sp.csc: "$\int \csc(x) dx = -\ln|\csc(x) + \cot(x)| + C$",
            sp.cot: "$\int \cot(x) dx = \ln|\sin(x)| + C$"
        }
        
        for func, formula in trig_funcs.items():
            if expr.has(func):
                metodos.append(f"**Integral trigonométrica**:\n{formula}")
        
        # 4. Sustituciones especiales
        if any(expr.has(f) for f in [sp.sin, sp.cos, sp.tan]):
            metodos.append(
                "**Sustitución trigonométrica**:\n"
                "- $\sqrt{a^2-x^2}$ → $x = a\sin\theta$\n"
                "- $\sqrt{a^2+x^2}$ → $x = a\tan\theta$\n"
                "- $\sqrt{x^2-a^2}$ → $x = a\sec\theta$"
            )
        
        # 5. Integración por partes
        if expr.is_Mul and len(expr.args) == 2:
            metodos.append(
                "**Integración por partes**:\n"
                "$$\int u dv = uv - \int v du$$\n"
                "**Estrategia LIATE**: Logarítmicas, Inversas, Algebraicas, Trigonométricas, Exponenciales"
            )
        
        # 6. Fracciones parciales
        if expr.is_rational_function():
            metodos.append(
                "**Fracciones parciales**:\n"
                "1. Factorizar denominador\n"
                "2. Descomponer en fracciones simples\n"
                "3. Integrar cada término por separado"
            )
        
        # 7. Sustituciones especiales
        if expr.has(sp.sqrt):
            args = expr.atoms(sp.sqrt)
            for arg in args:
                if x in arg.free_symbols:
                    inner = arg.args[0]
                    if inner.is_Pow and inner.exp == 2:
                        metodos.append(
                            f"**Sustitución para raíces cuadradas**:\n"
                            f"Para $\sqrt{{{sp.latex(inner)}}}$, usar sustitución trigonométrica"
                        )
                    else:
                        metodos.append(
                            f"**Sustitución simple para $\sqrt{{{sp.latex(inner)}}}$**:\n"
                            f"Intentar $u = {sp.latex(inner)}$"
                        )
        
        # 8. Funciones hiperbólicas
        if any(expr.has(f) for f in [sp.sinh, sp.cosh, sp.tanh]):
            metodos.append(
                "**Integrales hiperbólicas**:\n"
                "$\int \sinh(x) dx = \cosh(x) + C$\n"
                "$\int \cosh(x) dx = \sinh(x) + C$\n"
                "$\int \tanh(x) dx = \ln(\cosh(x)) + C$"
            )
        
        # 9. Integrales especiales
        if expr.has(sp.log):
            metodos.append(
                "**Integral de logaritmos**:\n"
                "$\int \ln(x) dx = x\ln(x) - x + C$\n"
                "$\int \log_a(x) dx = \\frac{x\ln(x) - x}{\ln a} + C$"
            )
        
        # 10. Métodos combinados
        if len(metodos) > 1:
            metodos.append(
                "**Combinación de métodos**:\n"
                "Puede requerir aplicar varios métodos secuencialmente"
            )
        
        # Formateo de salida
        if metodos:
            header = "### 🧮 **Métodos de integración aplicables:**\n"
            return header + "\n".join(f"- {m}" for m in metodos)
        else:
            return "**Método directo** o combinación no identificada de técnicas"

    def _formatear_resultado(self, expr: sp.Expr, resultado: sp.Expr, 
                           tipo: str, variable: str) -> str:
        """Formatea el resultado según el tipo de integral."""
        if tipo == "definida":
            return f"$$\\int_{{{self._limites[0]}}}^{{{self._limites[1]}}} {sp.latex(expr)}\\,d{variable} = {sp.latex(resultado)}$$"
        else:
            return f"$$\\int {sp.latex(expr)}\\,d{variable} = {sp.latex(resultado)} + C$$"

    def _guardar_historial(self, expresion: str, resultado: sp.Expr, 
                          tipo: str, limites: Optional[Tuple[str, str]]):
        """Registra la operación en el historial."""
        self.historial.append({
            'fecha': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'expresion': expresion,
            'resultado': str(resultado),
            'latex_resultado': sp.latex(resultado),
            'tipo': tipo,
            'limites': limites if limites else None,
            'variable': self.variable
        })

    def mostrar_historial(self, formato: str = 'markdown') -> str:
        """Devuelve el historial en diferentes formatos."""
        if not self.historial:
            return "📜 El historial está vacío."
            
        if formato == 'markdown':
            return self._historial_markdown()
        elif formato == 'latex':
            return self._historial_latex()
        else:
            return str(self.historial)

    def _historial_markdown(self) -> str:
        """Formatea el historial como tabla Markdown."""
        tabla = [
            "| Fecha | Expresión | Resultado | Tipo | Límites | Variable |",
            "|-------|-----------|-----------|------|---------|----------|"
        ]
        for item in self.historial:
            limites = f"{item['limites'][0]} a {item['limites'][1]}" if item['limites'] else "Indefinida"
            tabla.append(
                f"| {item['fecha']} | `{item['expresion']}` "
                f"| `{item['resultado']}` | {item['tipo']} "
                f"| {limites} | {item['variable']} |"
            )
        return "\n".join(tabla)

    def _historial_latex(self) -> str:
        """Formatea el historial para LaTeX."""
        items = []
        for item in self.historial:
            if item['limites']:
                integral = (
                    f"\\int_{{{item['limites'][0]}}}^{{{item['limites'][1]}}} "
                    f"{sp.latex(sp.sympify(item['expresion']))}\\,d{item['variable']}"
                )
            else:
                integral = f"\\int {sp.latex(sp.sympify(item['expresion']))}\\,d{item['variable']}"
            
            items.append(
                f"\\item {item['fecha']}: "
                f"${integral} = {item['latex_resultado']}$"
                f" ({item['tipo'].capitalize()})"
            )
        return "\\begin{itemize}\n" + "\n".join(items) + "\n\\end{itemize}"

    def graficar_integral(self, expr_str: str,
                      limites: Optional[Tuple[float, float]] = None,
                      rango: Tuple[float, float] = (-5, 5)) -> Optional[Image.Image]:
        """
        Grafica la función y su integral. Devuelve una imagen PIL.Image.
        """
        try:
            expr = self._parsear_expresion(expr_str)

            # Validar variables
            variables = list(expr.free_symbols)
            if len(variables) != 1:
                raise ValueError("La gráfica solo puede generarse para funciones con una sola variable.")

            x = variables[0]
            self.variable = str(x)
            f = sp.lambdify(x, expr, 'numpy')

            x_vals = np.linspace(rango[0], rango[1], 400)
            y_vals = f(x_vals)

            # Integral indefinida
            integral = sp.integrate(expr, x)
            F = sp.lambdify(x, integral, 'numpy')
            int_vals = F(x_vals)

            plt.figure(figsize=(12, 6))

            # Subplot original
            plt.subplot(1, 2, 1)
            plt.plot(x_vals, y_vals, label=f"f({x}) = {sp.latex(expr)}")
            plt.title("Función original")
            plt.grid(True)
            plt.legend()

            # Subplot integral
            plt.subplot(1, 2, 2)
            plt.plot(x_vals, int_vals, label=f"F({x}) = {sp.latex(integral)} + C")
            if limites:
                a, b = limites
                mask = (x_vals >= a) & (x_vals <= b)
                plt.fill_between(x_vals[mask], int_vals[mask], alpha=0.3)
            plt.title("Integral")
            plt.grid(True)
            plt.legend()

            buf = io.BytesIO()
            plt.tight_layout()
            plt.savefig(buf, format='png')
            plt.close()
            buf.seek(0)
            return Image.open(buf)

        except Exception as e:
            print(f"❌ Error al graficar integral: {str(e)}")
            return None


    def limpiar_historial(self):
        """Reinicia el historial de operaciones."""
        self.historial = []