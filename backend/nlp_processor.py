from typing import Dict
import re

class NLPProcessor:
    def __init__(self):
        pass

    def get_operation_details(self, text: str) -> dict:
        text_lower = text.lower()

        # Palabras clave para integrales ampliadas
        integral_keywords = [
            "integra", "integral", "∫", "area bajo", "antiderivada", "primitiva", "integrar",
            "encuentra la integral", "find the integral", "antiderivative", "primitive"
        ]

        # Clasificación
        if any(p in text_lower for p in ["deriva", "derivada", "diferencia", "derivative"]):
            op_type = "derivada"
        elif any(p in text_lower for p in ["limite", "límite", "tiende", "lim", "cuando"]):
            op_type = "limite"
        elif any(p in text_lower for p in integral_keywords):
            op_type = "integral"
        else:
            op_type = "desconocido"

        # Palabras a eliminar
        palabras_a_eliminar = [
            "calcula", "calcular", "encuentra", "encuentre", "hallar", "halla",
            "la", "el", "de", "del", "cuando", "tiende", "a", "una", "un", "por",
            "limite", "límite", "deriva", "derivada", "integra", "integral", "resultado", ":", "=", "es"
        ]
        expr = text_lower
        for palabra in palabras_a_eliminar:
            expr = re.sub(rf"\b{palabra}\b", "", expr)

        expr = expr.strip()

        # PROTEGER FUNCIONES MATEMÁTICAS (con o sin paréntesis y con espacios)
        funciones = [
            "sin", "cos", "tan", "exp", "log", "sec", "csc", "cot",
            "sinh", "cosh", "tanh", "sech", "csch", "coth"
        ]

        for func in funciones:
            # Protege func(x), func (x), funcx, func x
            expr = re.sub(rf"{func}\s*\(\s*([^)]+?)\s*\)", f"#{func}#(\\1)", expr)  # func(x) o func (x)
            expr = re.sub(rf"{func}\s+([a-zA-Z0-9]+)", f"#{func}#(\\1)", expr)      # func x
            expr = re.sub(rf"{func}([a-zA-Z0-9]+)", f"#{func}#(\\1)", expr)         # funcx
            expr = re.sub(rf"{func}\b", f"#{func}#", expr)                          # func solo

        # Insertar * donde corresponde (2x -> 2*x, xy -> x*y)
        expr = re.sub(r"(\d)([a-zA-Z])", r"\1*\2", expr)
        expr = re.sub(r"([a-zA-Z])([a-zA-Z])", r"\1*\2", expr)

        # Restaurar funciones matemáticas
        for func in funciones:
            expr = re.sub(rf"#{func}#\(([^)]+)\)", f"{func}(\\1)", expr)
            expr = expr.replace(f"#{func}#", func)

        # Eliminar caracteres no válidos y limpiar espacios
        expr = re.sub(r"[^\w*^+/().=\-]", " ", expr)
        expr = re.sub(r"\s+", " ", expr).strip()

        # Eliminar residuos como "x 0" al final (para límites)
        expr = re.sub(r"\b[a-zA-Z]\s+\d+\b", "", expr).strip()

        # Parámetros para límite (igual que antes)
        params = {}
        if op_type == "limite":
            try:
                if "cuando" in text_lower and "tiende a" in text_lower:
                    variable = text_lower.split("cuando")[-1].split("tiende")[0].strip()
                    punto = text_lower.split("tiende a")[-1].split()[0].strip()
                    params["variable"] = variable
                    params["punto"] = punto
            except Exception:
                pass

        return {
            "operation": op_type,
            "expression": expr,
            "params": params
        }