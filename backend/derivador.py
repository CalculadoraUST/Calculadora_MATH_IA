import sympy as sp
from sympy.parsing.sympy_parser import (parse_expr,standard_transformations, 
                                       implicit_multiplication_application)
from datetime import datetime
from typing import Tuple, List, Dict
import numpy as np
import matplotlib.pyplot as plt
import io
from PIL import Image

class DerivadorIA:
    def __init__(self):
        self.historial: List[Dict] = []
        self.variable = sp.symbols('x')  # Símbolo SymPy por defecto
        
    def resolver(self, expresion: str, orden: int = 1, variable: str = None) -> Tuple[str, str]:
        """
        Calcula la derivada y genera una explicación detallada en Markdown.
        
        Args:
            expresion (str): Expresión matemática como string.
            orden (int): Orden de derivación (default: 1).
            variable (str): Variable de derivación (opcional).
            
        Returns:
            Tuple[str, str]: (resultado_simplificado, explicación_markdown)
        """
        try:
            # Validación de entrada
            if not expresion or orden < 1:
                raise ValueError("Expresión vacía u orden inválido")
                
            # Configurar variable
            if variable:
                self.variable = sp.symbols(variable)
                
            # Parsear expresión
            expr = self._parsear_expresion(expresion)
            
            # Generar explicación paso a paso
            pasos = [
                f"## 🔍 Derivada de orden {orden}: {sp.latex(expr)}",
                "### 📚 **Paso 1: Identificar componentes**",
                self._identificar_componentes(expr),
                "### 🛠 **Paso 2: Aplicar reglas de derivación**",
                self._aplicar_reglas(expr, orden),
            ]
            
            # Calcular derivada
            derivada = sp.diff(expr, self.variable, orden)
            derivada_simplificada = sp.simplify(derivada)
            
            # Construir explicación final
            pasos.extend([
                "### ✅ **Resultado final**",
                f"$$\\frac{{d^{orden}}}{{d{sp.latex(self.variable)}^{orden}}}\\left({sp.latex(expr)}\\right) = {sp.latex(derivada_simplificada)}$$",
                "---",
                f"*🔎 Simplificación:* ${sp.latex(derivada)} \\Rightarrow {sp.latex(derivada_simplificada)}$",
                f"*📅 Operación registrada el {datetime.now().strftime('%Y-%m-%d %H:%M')}*"
            ])
            
            self._guardar_historial(expresion, derivada_simplificada, orden)
            return str(derivada_simplificada), "\n".join(pasos)
            
        except Exception as e:
            error_msg = f"❌ **Error al derivar**: {str(e)}"
            return "", error_msg

    def _parsear_expresion(self, expresion: str) -> sp.Expr:
        """Convierte string a expresión SymPy con soporte avanzado."""
        try:
            # Intentar parsear como LaTeX primero
            return sp.parse_latex(expresion)
        except:
            # Fallback a parser estándar con transformaciones
            transformations = standard_transformations + (implicit_multiplication_application,)
            return parse_expr(expresion, transformations=transformations)

    def _identificar_componentes(self, expr: sp.Expr) -> str:
        """Analiza y describe la estructura de la expresión."""
        componentes = []
        
        if expr.is_Add:
            componentes.append("**Suma algebraica** de términos")
        if expr.is_Mul:
            componentes.append("**Producto** de factores")
        if expr.is_Pow:
            componentes.append("**Potenciación**")
        if expr.has(sp.sin, sp.cos, sp.tan):
            componentes.append("**Funciones trigonométricas**")
        if expr.has(sp.exp):
            componentes.append("**Función exponencial**")
        if expr.has(sp.log):
            componentes.append("**Función logarítmica**")
            
        return ", ".join(componentes) if componentes else "Expresión atómica"

    def _aplicar_reglas(self, expr: sp.Expr, orden: int) -> str:
        """Genera explicación detallada de las reglas de derivación aplicables."""
        reglas = []
        x = self.variable
        
        # 1. Reglas básicas
        if expr.is_Add:
            reglas.append("**Regla de la suma**: (f + g)' = f' + g'")
            
        if expr.is_Mul and len(expr.args) > 1:
            reglas.append("**Regla del producto**: (fg)' = f'g + fg'")
            
        if any(arg.has(x) for arg in expr.args if expr.is_Function):
            reglas.append("**Regla de la cadena**: f(g(x))' = f'(g(x))·g'(x)")
        
        # 2. Reglas para funciones específicas
        if expr.has(sp.sin):
            reglas.append("**Derivada trigonométrica**: d/dx sin(x) = cos(x)")
        if expr.has(sp.cos):
            reglas.append("**Derivada trigonométrica**: d/dx cos(x) = -sin(x)")
        if expr.has(sp.tan):
            reglas.append("**Derivada trigonométrica**: d/dx tan(x) = sec²(x)")
        if expr.has(sp.sec):
            reglas.append("**Derivada trigonométrica**: d/dx sec(x) = sec(x)tan(x)")
        if expr.has(sp.csc):
            reglas.append("**Derivada trigonométrica**: d/dx csc(x) = -csc(x)cot(x)")
        if expr.has(sp.cot):
            reglas.append("**Derivada trigonométrica**: d/dx cot(x) = -csc²(x)")
        
        # 3. Funciones exponenciales y logarítmicas
        if expr.has(sp.exp):
            reglas.append("**Derivada exponencial**: d/dx eˣ = eˣ")
        if expr.has(sp.log):
            reglas.append("**Derivada logarítmica**: d/dx ln(x) = 1/x")
        if any(base != sp.E for base in expr.atoms(sp.Pow) if isinstance(base, sp.Number)):
            reglas.append("**Derivada exponencial general**: d/dx aˣ = aˣ·ln(a)")
        
        # 4. Regla del cociente (cuando detecta división)
        if expr.is_Pow and expr.exp == -1:
            reglas.append("**Regla del cociente**: (f/g)' = (f'g - fg')/g²")
        elif expr.is_Mul and any(arg.is_Pow and arg.exp == -1 for arg in expr.args):
            reglas.append("**Regla del cociente implícita** (convertida a producto)")
        
        # 5. Funciones hiperbólicas
        if expr.has(sp.sinh):
            reglas.append("**Derivada hiperbólica**: d/dx sinh(x) = cosh(x)")
        if expr.has(sp.cosh):
            reglas.append("**Derivada hiperbólica**: d/dx cosh(x) = sinh(x)")
        if expr.has(sp.tanh):
            reglas.append("**Derivada hiperbólica**: d/dx tanh(x) = sech²(x)")
        
        # 6. Funciones especiales
        if expr.has(sp.sqrt):
            reglas.append("**Derivada de raíz cuadrada**: d/dx √x = 1/(2√x)")
        if expr.has(sp.Abs):
            reglas.append("**Derivada de valor absoluto**: d/dx |x| = x/|x| (para x ≠ 0)")
        
        # 7. Regla de la potencia generalizada
        if expr.is_Pow:
            base, exp = expr.base, expr.exp
            if base.has(x) and exp.has(x):
                reglas.append("**Regla de la potencia-exponencial**: Usar diferenciación logarítmica")
            elif exp.has(x):
                reglas.append("**Derivada exponencial general**: d/dx a^f(x) = a^f(x)·ln(a)·f'(x)")
            elif base.has(x):
                reglas.append("**Regla de la potencia**: d/dx xⁿ = n·xⁿ⁻¹")
        
        # 8. Funciones inversas
        if expr.has(sp.asin):
            reglas.append("**Derivada arcsin**: d/dx arcsin(x) = 1/√(1-x²)")
        if expr.has(sp.acos):
            reglas.append("**Derivada arccos**: d/dx arccos(x) = -1/√(1-x²)")
        if expr.has(sp.atan):
            reglas.append("**Derivada arctan**: d/dx arctan(x) = 1/(1+x²)")
        
        # Formateo de salida
        if reglas:
            header = "### 📝 **Reglas de derivación aplicables:**\n"
            return header + "\n".join(f"- {r}" for r in reglas)
        else:
            return "**Derivada directa** (regla de potencias o combinación no identificada)"

    def _guardar_historial(self, expresion: str, resultado: sp.Expr, orden: int):
        """Registra la operación en el historial."""
        self.historial.append({
            'fecha': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'expresion': expresion,
            'resultado': str(resultado),
            'latex_resultado': sp.latex(resultado),
            'orden': orden,
            'variable': str(self.variable)
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

    def _historial_latex(self) -> str:
        """Formatea el historial para LaTeX."""
        items = []
        for item in self.historial:
            items.append(
                f"\\item {item['fecha']}: "
                f"${sp.latex(sp.sympify(item['expresion']))}$ → "
                f"${item['latex_resultado']}$ (Orden {item['orden']})"
            )
        return "\\begin{itemize}\n" + "\n".join(items) + "\n\\end{itemize}"

    def graficar(self, expr_str: str, rango: Tuple[float, float] = (-5, 5)) -> Image.Image:
        """
        Grafica la función y su derivada, retorna la imagen como objeto PIL.Image.
        """
        try:
            expr = self._parsear_expresion(expr_str)

            # Validar que hay solo una variable simbólica
            variables = list(expr.free_symbols)
            if len(variables) != 1:
                raise ValueError("La gráfica solo puede generarse para funciones con una sola variable.")

            x = variables[0]
            f = sp.lambdify(x, expr, 'numpy')

            x_vals = np.linspace(rango[0], rango[1], 400)
            y_vals = f(x_vals)

            # Derivada
            derivada = sp.diff(expr, x)
            df = sp.lambdify(x, derivada, 'numpy')
            dy_vals = df(x_vals)

            # Crear la figura
            plt.figure(figsize=(10, 6))
            plt.plot(x_vals, y_vals, label=f"f({x}) = {sp.latex(expr)}")
            plt.plot(x_vals, dy_vals, label=f"f'({x}) = {sp.latex(derivada)}", linestyle='--')
            plt.title("Función y su derivada")
            plt.xlabel(str(x))
            plt.ylabel("y")
            plt.grid(True)
            plt.legend()

            # Guardar en imagen
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            plt.close()
            buf.seek(0)
            return Image.open(buf)

        except Exception as e:
            print(f"❌ Error al graficar: {str(e)}")
            return None

    def limpiar_historial(self):
        """Reinicia el historial de operaciones."""
        self.historial = []