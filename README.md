# Práctica  de árboles de decisión (ML)

# Isais Emanuel Estrella Marquez 

# Materia: Programacion Logica
# Docente: MARIA DE LOURDES AGUILLON RUIZ

## Descripción del Trabajo
Este trabajo implementa un clasificador de árbol de decisión para el dataset de vinos de scikit-learn, que contiene 178 muestras de vino con 13 características químicas cada una, clasificadas en 3 tipos diferentes.

# Resultados Obtenidos

## Precisión del Modelo
- **Árbol con max_depth=2**: 86.11%
- **Árbol con max_depth=4**: 94.44% 
- **Árbol sin límite de profundidad**: 94.44%

## Conclusiones

1. **El dataset es IDEAL para árboles de decisión** - Las características químicas permiten crear reglas efectivas
2. **Máxima eficiencia en max_depth=4** - Mayor profundidad no mejora la precisión
3. **Reglas interpretables** - El árbol con profundidad 2 muestra reglas simples y comprensibles
4. **Alta precisión general** - 94.44% es excelente para un modelo interpretable

## Características más importantes
El árbol identificó como características clave:
- **color_intensity** (primera división)
- **flavanoids** 

- **proline**
