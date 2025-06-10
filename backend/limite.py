import sympy as sp
from sympy.parsing.sympy_parser import (parse_expr, standard_transformations, 
                                       implicit_multiplication_application)
from datetime import datetime
from typing import Tuple, List, Dict, Optional
import numpy as np
import matplotlib.pyplot as plt
import io
from PIL import Image

class LimiteIA:
    def __init__(self):
        self.historial: List[Dict] = []
        self.variable = 'x'  # Variable predeterminada
        
    def resolver(self, expresion: str, variable: str = 'x', 
                punto: str = 'oo', direccion: str = '+') -> Tuple[str, str]:
        """
        Calcula el límite y genera explicación detallada en Markdown.
        
        Args:
            expresion: Expresión matemática como string.
            variable: Variable del límite (default: 'x').
            punto: Punto hacia donde tiende (ej: '0', 'oo', 'pi/2').
            direccion: '+' (derecha), '-' (izquierda) o '+-' (ambos).
            
        Returns:
            Tuple: (resultado, explicación_markdown)
        """
        try:
            self.variable = variable
            x = sp.symbols(variable)
            expr = self._parsear_expresion(expresion)
            punto_sym = sp.sympify(punto)
            
            # Generar explicación paso a paso
            pasos = [
                f"## 🔍 Cálculo del límite: $\\lim_{{{variable} \\to {punto}^{{{direccion}}}}} {sp.latex(expr)}$",
                "### 📚 **Paso 1: Identificación del tipo de límite**",
                self._identificar_tipo(expr, punto_sym, direccion),
                "### 🛠 **Paso 2: Técnicas aplicadas**",
                self._aplicar_tecnicas(expr, x, punto_sym, direccion),
            ]
            
            # Calcular el límite
            if direccion in ['+', '-']:
                resultado = sp.limit(expr, x, punto_sym, dir=direccion)
            else:
                resultado = sp.limit(expr, x, punto_sym)
                
            # Verificar existencia del límite
            limite_existe = True
            try:
                if direccion == '+-':
                    lim_der = sp.limit(expr, x, punto_sym, dir='+')
                    lim_izq = sp.limit(expr, x, punto_sym, dir='-')
                    if lim_der != lim_izq:
                        limite_existe = False
                        resultado = f"∄ (Límites laterales diferentes: {lim_der} ≠ {lim_izq})"
            except:
                limite_existe = False
                resultado = "∄ (No existe el límite)"
            
            # Construcción de la explicación final
            pasos.extend([
                "### ✅ **Resultado final**",
                f"$$\\lim_{{{variable} \\to {punto}^{{{direccion}}}}} {sp.latex(expr)} = {sp.latex(resultado) if limite_existe else resultado}$$",
                "---",
                f"*📅 Operación registrada el {datetime.now().strftime('%Y-%m-%d %H:%M')}*"
            ])
            
            self._guardar_historial(expresion, resultado, punto, direccion, limite_existe)
            return str(resultado), "\n".join(pasos)
            
        except Exception as e:
            error_msg = f"❌ **Error al calcular el límite**: {str(e)}"
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

    def _identificar_tipo(self, expr: sp.Expr, punto: sp.Expr, direccion: str) -> str:
        """Analiza y describe el tipo de límite."""
        componentes = []
        
        # Límites notables
        if expr.has(sp.sin, sp.cos) and punto == 0:
            componentes.append("**Límite trigonométrico notable**")
        if expr.has(sp.exp) and (punto == sp.oo or punto == -sp.oo):
            componentes.append("**Límite exponencial**")
            
        # Comportamiento en el punto
        try:
            eval_punto = expr.subs(sp.symbols(self.variable), punto)
            if eval_punto.has(sp.zoo, sp.nan):
                componentes.append("**Indeterminación** (requiere análisis especial)")
            else:
                componentes.append("Evaluación directa posible")
        except:
            componentes.append("Requiere técnicas avanzadas")
            
        # Direccionalidad
        if direccion in ['+', '-']:
            componentes.append(f"Límite lateral ({'derecha' if direccion == '+' else 'izquierda'})")
            
        return ", ".join(componentes)

    def _identificar_tipo(self, expr: sp.Expr, punto: sp.Expr, direccion: str) -> str:
        """Analiza y describe el tipo de límite con mayor detalle."""
        componentes = []
        x = sp.symbols(self.variable)
        
        # 1. Tipos especiales de límites
        if expr.has(sp.sin, sp.cos) and punto == 0:
            if expr.has(sp.sin(x)/x) or expr.has(x/sp.sin(x)):
                componentes.append("**Límite trigonométrico fundamental**: sin(x)/x → 1 cuando x→0")
            else:
                componentes.append("**Límite trigonométrico notable**")
        
        if expr.has(sp.exp):
            if punto == sp.oo:
                componentes.append("**Límite exponencial en ∞**: eˣ → ∞")
            elif punto == -sp.oo:
                componentes.append("**Límite exponencial en -∞**: eˣ → 0")
            elif punto == 0 and expr.has((sp.exp(x)-1)/x):
                componentes.append("**Límite exponencial fundamental**: (eˣ-1)/x → 1 cuando x→0")
        
        # 2. Comportamiento en el punto
        try:
            eval_punto = expr.subs(x, punto)
            if eval_punto.has(sp.zoo, sp.nan):
                if expr.is_rational_function():
                    num, den = expr.as_numer_denom()
                    num_val = num.subs(x, punto)
                    den_val = den.subs(x, punto)
                    if num_val == 0 and den_val == 0:
                        componentes.append("**Indeterminación 0/0**")
                    elif num_val.is_infinite and den_val.is_infinite:
                        componentes.append("**Indeterminación ∞/∞**")
                    elif (num_val.is_nonzero and den_val == 0) or (num_val.is_infinite and den_val.is_finite):
                        componentes.append("**Comportamiento asintótico** (puede ser ∞, -∞ o no existir)")
                else:
                    componentes.append("**Indeterminación** (requiere análisis especial)")
            else:
                componentes.append("**Evaluación directa posible**")
        except:
            componentes.append("**Requiere técnicas avanzadas**")
        
        # 3. Direccionalidad y continuidad
        if direccion in ['+', '-']:
            componente_dir = {
                '+': "**Límite por la derecha**",
                '-': "**Límite por la izquierda**"
            }
            componentes.append(componente_dir[direccion])
            
            # Verificar continuidad
            try:
                lim = sp.limit(expr, x, punto, dir=direccion)
                val = expr.subs(x, punto)
                if lim == val:
                    componentes.append(f"**Función continua** en {punto} por {direccion}")
            except:
                pass
        
        # 4. Comportamiento asintótico
        if punto in [sp.oo, -sp.oo]:
            componentes.append("**Comportamiento en el infinito**")
        elif expr.has(sp.log(x)) and punto == 0:
            componentes.append("**Comportamiento logarítmico en 0** (ln(x) → -∞)")
        
        return "### 📊 **Tipo de límite:**\n" + "\n".join(f"- {c}" for c in componentes)

    def _aplicar_tecnicas(self, expr: sp.Expr, x: sp.Symbol, 
                        punto: sp.Expr, direccion: str) -> str:
        """Explica las técnicas para resolver el límite con más detalle."""
        tecnicas = []
        
        # 1. Evaluación directa
        try:
            eval_punto = expr.subs(x, punto)
            if not eval_punto.has(sp.zoo, sp.nan):
                tecnicas.append(
                    f"**Evaluación directa**:\n"
                    f"Sustituir ${sp.latex(x)} = {sp.latex(punto)}$\n"
                    f"Resultado: ${sp.latex(eval_punto)}$"
                )
                return "### 🛠 **Técnicas aplicadas:**\n" + "\n".join(f"- {t}" for t in tecnicas)
        except:
            pass
            
        # 2. Indeterminaciones
        if expr.is_rational_function():
            num, den = expr.as_numer_denom()
            num_val = num.subs(x, punto)
            den_val = den.subs(x, punto)
            
            if num_val == 0 and den_val == 0:
                tecnicas.append("**Indeterminación 0/0**:")
                tecnicas.append("  - **Factorización**: Buscar términos comunes")
                tecnicas.append("  - **Regla de L'Hôpital**: Derivar numerador y denominador")
                tecnicas.append("  - **Expansión en series de Taylor**: Para funciones trascendentes")
                
            elif num_val.is_infinite and den_val.is_infinite:
                tecnicas.append("**Indeterminación ∞/∞**:")
                tecnicas.append("  - **Dividir por la mayor potencia**: Para funciones racionales en ∞")
                tecnicas.append("  - **Regla de L'Hôpital**: Derivar numerador y denominador")
                tecnicas.append("  - **Comparación de infinitos**: Analizar órdenes de crecimiento")
        
        # 3. Técnicas trigonométricas
        if expr.has(sp.sin, sp.cos, sp.tan):
            tecnicas.append("**Técnicas trigonométricas**:")
            tecnicas.append("  - **Identidades trigonométricas**: sin²x + cos²x = 1, etc.")
            tecnicas.append("  - **Límites notables**:")
            tecnicas.append("    - $\lim_{x→0}\\frac{\\sin x}{x} = 1$")
            tecnicas.append("    - $\lim_{x→0}\\frac{1-\\cos x}{x} = 0$")
            tecnicas.append("  - **Sustitución trigonométrica**: Para expresiones con √(a²±x²)")
        
        # 4. Límites en infinito
        if punto in [sp.oo, -sp.oo]:
            tecnicas.append("**Técnicas para límites en ∞**:")
            tecnicas.append("  - **Comparación de términos dominantes**:")
            tecnicas.append("    - Exponencial > Polinomial > Logarítmico")
            tecnicas.append("  - **Factorización del término dominante**:")
            tecnicas.append("  - **Uso de órdenes de crecimiento**:")
            tecnicas.append("    - $e^x > x^n > \ln x$ cuando x→∞")
        
        # 5. Técnicas exponenciales y logarítmicas
        if expr.has(sp.exp):
            tecnicas.append("**Técnicas exponenciales**:")
            tecnicas.append("  - $\lim_{x→0}\\frac{e^x-1}{x} = 1$")
            tecnicas.append("  - Para formas 1^∞: Usar $\\lim f(x)^{g(x)} = e^{\\lim (f(x)-1)g(x)}$")
        
        if expr.has(sp.log):
            tecnicas.append("**Técnicas logarítmicas**:")
            tecnicas.append("  - $\lim_{x→0^+} \ln x = -∞$")
            tecnicas.append("  - $\lim_{x→∞} \\frac{\ln x}{x^n} = 0$ (n > 0)")
        
        # 6. Sustitución de variables
        tecnicas.append("**Técnicas generales**:")
        tecnicas.append("  - **Sustitución de variable**: y = 1/x para límites en ∞")
        tecnicas.append("  - **Expansión en series**: Para funciones complejas cerca del punto")
        
        return "### 🛠 **Técnicas aplicables:**\n" + "\n".join(f"- {t}" for t in tecnicas) if tecnicas else "**Análisis directo** o combinación de técnicas"

    def _guardar_historial(self, expresion: str, resultado: sp.Expr, 
                          punto: str, direccion: str, existe: bool):
        """Registra la operación en el historial."""
        self.historial.append({
            'fecha': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'expresion': expresion,
            'resultado': str(resultado),
            'latex_resultado': sp.latex(resultado) if existe else resultado,
            'punto': punto,
            'direccion': direccion,
            'existe': existe,
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
            "| Fecha | Expresión | Límite | Punto | Dirección | ¿Existe? |",
            "|-------|-----------|--------|-------|-----------|----------|"
        ]
        for item in self.historial:
            tabla.append(
                f"| {item['fecha']} | `{item['expresion']}` "
                f"| `{item['resultado']}` | {item['punto']} "
                f"| {item['direccion']} | {'Sí' if item['existe'] else 'No'} |"
            )
        return "\n".join(tabla)

    def _historial_latex(self) -> str:
        """Formatea el historial para LaTeX."""
        items = []
        for item in self.historial:
            items.append(
                f"\\item {item['fecha']}: "
                f"$\\lim_{{{item['variable']} \\to {item['punto']}^{{{item['direccion']}}}}} "
                f"{sp.latex(sp.sympify(item['expresion']))} = "
                f"{item['latex_resultado']}$"
                f" ({'Existe' if item['existe'] else 'No existe'})"
            )
            
        return "\\begin{itemize}\n" + "\n".join(items) + "\n\\end{itemize}"

    def graficar_limite(self, expr_str: str,
                    punto: str = 'oo',
                    rango: Tuple[float, float] = (-5, 5)) -> Optional[Image.Image]:
        """
        Grafica el comportamiento de la función cerca del punto del límite.
        """
        try:
            expr = self._parsear_expresion(expr_str)

            # Validar una sola variable
            variables = list(expr.free_symbols)
            if len(variables) != 1:
                raise ValueError("La gráfica solo puede generarse para funciones con una sola variable.")

            x = variables[0]
            self.variable = str(x)
            f = sp.lambdify(x, expr, 'numpy')

            # Rango según el punto
            if punto == 'oo':
                x_vals = np.linspace(10, 100, 400)
            elif punto == '-oo':
                x_vals = np.linspace(-100, -10, 400)
            else:
                p = float(sp.sympify(punto).evalf())
                radio = min(abs(rango[1] - p), abs(p - rango[0])) / 2
                x_vals = np.linspace(p - radio, p + radio, 400)

            y_vals = f(x_vals)

            plt.figure(figsize=(10, 6))
            plt.plot(x_vals, y_vals, label=f"f({x}) = {sp.latex(expr)}")
            if punto not in ['oo', '-oo']:
                p = float(sp.sympify(punto).evalf())
                plt.axvline(x=p, color='r', linestyle='--', alpha=0.5, label=f"{x} → {punto}")

            plt.title(f"Comportamiento cerca de {x} → {punto}")
            plt.grid(True)
            plt.legend()

            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            plt.close()
            buf.seek(0)
            return Image.open(buf)

        except Exception as e:
            print(f"❌ Error al graficar límite: {str(e)}")
            return None


    def limpiar_historial(self):
        """Reinicia el historial de operaciones."""
        self.historial = []