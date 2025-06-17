import re
import sympy as sp # Importar sympy para verificar si el parseo es válido
from sympy.parsing.sympy_parser import (parse_expr, standard_transformations,
                                       implicit_multiplication_application)
from typing import Dict, Any, Tuple, Optional

class NLPProcessor:
    def __init__(self):
        # Lista de funciones matemáticas conocidas para protección
        self.funciones_conocidas = [
            "sin", "cos", "tan", "exp", "log", "ln", # ln es común para logaritmo natural
            "sec", "csc", "cot",
            "sinh", "cosh", "tanh", "sech", "csch", "coth",
            "asin", "acos", "atan", # Inversas trigonométricas
            "sqrt", "Abs" # Raíz cuadrada y valor absoluto
        ]
        # Palabras clave para identificar operaciones
        self.derivada_keywords = ["deriva", "derivada", "diferencia", "derivative", "d/dx"]
        self.integral_keywords = ["integra", "integral", "∫", "area bajo", "antiderivada", "primitiva", "integrar", "find the integral", "antiderivative", "primitive"]
        self.limite_keywords = ["limite", "límite", "tiende", "lim", "cuando"]

        # Palabras de "ruido" que se pueden eliminar para simplificar el texto
        self.palabras_ruido = [
            "calcula", "calcular", "encuentra", "encuentre", "hallar", "halla",
            "la", "el", "de", "del", "a", "una", "un", "por", "resultado", "es", "para", "con", "respecto", "de",
            ":", "=", ";", ",", "en", "punto" # Signos comunes como ruido
        ]

    def limpiar_expresion(self, expr: str) -> str:
        # 1. Une funciones separadas por espacio: "cos (x)" -> "cos(x)"
        for func in self.funciones_conocidas:
            expr = re.sub(rf"\b{func}\s+\(", f"{func}(", expr)
        # 2. Protege funciones con argumentos: cos(x) -> #cos#(x)
        for func in self.funciones_conocidas:
            expr = re.sub(rf"\b{func}\(([^()]*)\)", rf"#{func}#(\1)", expr)
        # 3. Inserta multiplicaciones implícitas SOLO entre número y letra, o paréntesis y letra/número
        expr = re.sub(r"(\d)([a-zA-Z])", r"\1*\2", expr)
        expr = re.sub(r"([a-zA-Z)])\s*([\(])", r"\1*\2", expr)
        expr = re.sub(r"([\)])\s*([a-zA-Z])", r"\1*\2", expr)
        expr = re.sub(r"(\d)\s*([\(])", r"\1*\2", expr)
        # NO uses expr = re.sub(r"([a-zA-Z])([a-zA-Z])", r"\1*\2", expr)
        # 4. Restaura funciones protegidas
        for func in self.funciones_conocidas:
            expr = re.sub(rf"#{func}#\(([^()]*)\)", rf"{func}(\1)", expr)
        # 5. Limpieza final
        expr = re.sub(r"[^\w*^+/().=\-\[\]]", " ", expr)
        expr = re.sub(r"\s+", " ", expr).strip()
        expr = expr.replace("=", "")
        expr = re.sub(r"\bd[a-zA-Z]\b", "", expr).strip()
        return expr

    def get_operation_details(self, text: str) -> Dict[str, Any]:
        """
        Procesa una string de lenguaje natural para identificar la operación matemática
        solicitada y extraer la expresión y parámetros relevantes.

        Args:
            text (str): La string de entrada del usuario.

        Returns:
            Dict[str, Any]: Un diccionario con 'operation' (derivada, integral,
                            limite, desconocido), 'expression' (string limpia)
                            y 'params' (dict con parámetros como variable, orden, límites, punto).
        """
        text_lower = text.lower()
        op_type = "desconocido"
        params: Dict[str, Any] = {}
        expresion_raw = text_lower # Usaremos una copia para procesar

        # --- 1. Identificar el tipo de operación ---
        if any(p in text_lower for p in self.derivada_keywords):
            op_type = "derivada"
        elif any(p in text_lower for p in self.limite_keywords):
            op_type = "limite"
        elif any(p in text_lower for p in self.integral_keywords):
            op_type = "integral"
        # Si no se identifica una operación clara, intentamos procesar el resto como una expresión simple.

        # --- 2. Extraer Parámetros Específicos (antes de eliminar ruido si es posible) ---
        # Es más fácil encontrar patrones de parámetros en el texto original.

        if op_type == "derivada":
            # Intentar extraer orden (e.g., "segunda derivada", "derivada de orden 3")
            orden_match = re.search(r"(?:de\s+)?orden\s+(\d+)", text_lower)
            if orden_match:
                params["orden"] = int(orden_match.group(1))
            else:
                # Buscar ordinales comunes si no se especifica "orden"
                if re.search(r"\bprimer(?:a)?\s+derivada\b", text_lower):
                    params["orden"] = 1
                elif re.search(r"\bsegund(?:a)?\s+derivada\b", text_lower):
                    params["orden"] = 2
                elif re.search(r"\btercer(?:a)?\s+derivada\b", text_lower):
                    params["orden"] = 3
                else:
                    params["orden"] = 1 # Orden por defecto si no se encuentra nada

            # Intentar extraer variable (e.g., "con respecto a y")
            variable_match = re.search(r"(?:con\s+respecto\s+a|d\/d)([a-zA-Z])\b", text_lower)
            if variable_match:
                params["variable"] = variable_match.group(1)
            else:
                params["variable"] = 'x' # Variable por defecto

        elif op_type == "integral":
            # Intentar extraer límites para integral definida (e.g., "de 0 a 1", "desde -inf hasta inf")
            limites_match = re.search(r"(?:de|desde)\s+([\w\-\.]+)\s+(?:a|hasta)\s+([\w\-\.]+)", text_lower)
            if limites_match:
                params["limites"] = (limites_match.group(1), limites_match.group(2))
            else:
                params["limites"] = None # Integral indefinida

            # Intentar extraer variable (e.g., "integral de ... dx", "con respecto a y")
            variable_match = re.search(r"d([a-zA-Z])\b", text_lower) # Busca el dx en la notación de integral
            if variable_match:
                params["variable"] = variable_match.group(1)
            else:
                # Fallback a buscar "con respecto a" si no encuentra dx
                variable_match = re.search(r"(?:con\s+respecto\s+a)([a-zA-Z])\b", text_lower)
                if variable_match:
                    params["variable"] = variable_match.group(1)
                else:
                    params["variable"] = 'x' # Variable por defecto

        elif op_type == "limite":
            # Intentar extraer variable, punto y dirección (e.g., "limite de x cuando x tiende a 0 por la derecha")
            try:
                # Buscar patrón "variable tiende a punto"
                match_tiende = re.search(r"([a-zA-Z])\s+tiende\s+a\s+([^\s]+)", text_lower)
                if match_tiende:
                    params["variable"] = match_tiende.group(1)
                    punto_raw = match_tiende.group(2).strip()

                    # Buscar dirección (+ o -) inmediatamente después del punto o en el texto
                    direccion = "+" # Dirección por defecto
                    if punto_raw.endswith('+'):
                        params["punto"] = punto_raw[:-1]
                        params["direccion"] = "+"
                    elif punto_raw.endswith('-'):
                        params["punto"] = punto_raw[:-1]
                        params["direccion"] = "-"
                    else:
                        params["punto"] = punto_raw
                        # Intentar detectar dirección en el texto si no está adjunta al número
                        if re.search(r"\bpor\s+la\s+derecha\b", text_lower):
                            params["direccion"] = "+"
                        elif re.search(r"\bpor\s+la\s+izquierda\b", text_lower):
                            params["direccion"] = "-"
                        elif re.search(r"\bpor\s+ambos\s+lados\b", text_lower):
                            params["direccion"] = "+-" # Indicador para verificar límites laterales
                        else:
                            params["direccion"] = "+" # Dirección por defecto si no se especifica nada más allá del punto

                elif re.search(r"\blim\s*\(?\s*([a-zA-Z])\s*->\s*([^\s\)]+)\)?", text_lower):
                    # Alternativa de notación lim(x -> 0)
                    match_arrow = re.search(r"\blim\s*\(?\s*([a-zA-Z])\s*->\s*([^\s\)]+)\)?", text_lower)
                    if match_arrow:
                        params["variable"] = match_arrow.group(1)
                        params["punto"] = match_arrow.group(2).strip().replace('inf', 'oo') # Usar oo para infinito en SymPy
                        params["direccion"] = "+" # Por defecto si no se especifica

                else:
                    # Si no se encuentra patrón claro, usar valores por defecto o marcar como desconocido
                    params["variable"] = 'x'
                    params["punto"] = 'oo' # Límite por defecto al infinito si no se especifica punto
                    params["direccion"] = '+'

            except Exception:
                # Si falla la extracción compleja, usar defaults
                params["variable"] = 'x'
                params["punto"] = 'oo'
                params["direccion"] = '+'

        # --- 3. Limpiar el texto para aislar la expresión ---
        # Eliminar palabras clave de operación y ruido
        for p in self.derivada_keywords + self.integral_keywords + self.limite_keywords + self.palabras_ruido:
            expresion_raw = re.sub(rf"\b{p}\b", " ", expresion_raw)

        # Eliminar partes de parámetros ya extraídos (ej: "cuando x tiende a 0")
        if "variable" in params and "punto" in params:
            expresion_raw = re.sub(rf"\b{params['variable']}\s+tiende\s+a\s+{re.escape(params['punto'])}\b", " ", expresion_raw)
            expresion_raw = re.sub(rf"\b{params['variable']}\s*->\s*{re.escape(params['punto'])}\b", " ", expresion_raw)
            expresion_raw = expresion_raw.replace(params['punto'] + '+', params['punto']).replace(params['punto'] + '-', params['punto'])

        if "limites" in params and params["limites"] is not None:
            l_inf, l_sup = params["limites"]
            expresion_raw = re.sub(rf"\b(?:de|desde)\s+{re.escape(l_inf)}\s+(?:a|hasta)\s+{re.escape(l_sup)}\b", " ", expresion_raw)

        # --- 4. Preparar la string para el parsing simbólico ---
        expr = self.limpiar_expresion(expresion_raw.strip())

        # Validación con SymPy (igual que antes)
        try:
            test_var = sp.symbols(params.get('variable', 'x'))
            parse_expr(expr, transformations=standard_transformations + (implicit_multiplication_application,),
                       evaluate=False, local_dict={test_var.name: test_var})
            is_parsable = True
        except Exception:
            is_parsable = False

        if not expr and op_type != "desconocido":
            print(f"Advertencia: Operación '{op_type}' detectada, pero no se pudo extraer una expresión válida de '{text}'.")
            op_type = "desconocido"

        return {
            "operation": op_type,
            "expression": expr,
            "params": params,
            "is_parsable": is_parsable
        }