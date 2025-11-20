# Importar librerías necesarias
from sklearn.datasets import load_wine
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

# Cargar el dataset del vino
wine = load_wine()
X, y = wine.data, wine.target

# Dividir los datos en entrenamiento y prueba (80% entrenamiento, 20% prueba)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("=== CLASIFICACIÓN CON ÁRBOL DE DECISIÓN - WINE DATASET ===\n")
print(f"Tamaño del dataset completo: {X.shape[0]} muestras")
print(f"Tamaño conjunto entrenamiento: {X_train.shape[0]} muestras")
print(f"Tamaño conjunto prueba: {X_test.shape[0]} muestras")
print(f"Número de características: {X.shape[1]}")
print(f"Clases: {wine.target_names}\n")

# Actividad 1: Árbol con profundidad limitada (max_depth=2)
print("--- ÁRBOL CON PROFUNDIDAD LIMITADA (max_depth=2) ---")
tree_limited = DecisionTreeClassifier(max_depth=2, random_state=42)
tree_limited.fit(X_train, y_train)

# Exportar y visualizar reglas
rules_limited = export_text(tree_limited, feature_names=wine.feature_names)
print("Reglas del árbol (max_depth=2):")
print(rules_limited)

# Precisión
accuracy_limited = tree_limited.score(X_test, y_test)
print(f"Precisión en datos de prueba (max_depth=2): {accuracy_limited:.4f}\n")

# Actividad 2: Probar con diferente profundidad (max_depth=4)
print("--- ÁRBOL CON PROFUNDIDAD MEDIA (max_depth=4) ---")
tree_medium = DecisionTreeClassifier(max_depth=4, random_state=42)
tree_medium.fit(X_train, y_train)

accuracy_medium = tree_medium.score(X_test, y_test)
print(f"Precisión en datos de prueba (max_depth=4): {accuracy_medium:.4f}\n")

# Actividad 3: Árbol sin limitar profundidad
print("--- ÁRBOL SIN LIMITAR PROFUNDIDAD (max_depth=None) ---")
tree_full = DecisionTreeClassifier(max_depth=None, random_state=42)
tree_full.fit(X_train, y_train)

rules_full = export_text(tree_full, feature_names=wine.feature_names)
print("Primeras reglas del árbol completo:")
# Mostrar solo las primeras reglas para no saturar la salida
lines = rules_full.split('\n')
for i in range(min(15, len(lines))):
    print(lines[i])
print("... (árbol muy grande para mostrar completo)\n")

accuracy_full = tree_full.score(X_test, y_test)
print(f"Precisión en datos de prueba (max_depth=None): {accuracy_full:.4f}\n")

# Resumen comparativo
print("=== RESUMEN DE RESULTADOS ===")
print(f"Precisión max_depth=2: {accuracy_limited:.4f}")
print(f"Precisión max_depth=4: {accuracy_medium:.4f}")
print(f"Precisión max_depth=None: {accuracy_full:.4f}")

# Información adicional del dataset
print("\n=== INFORMACIÓN DEL DATASET ===")
print("Características disponibles:")
for i, feature in enumerate(wine.feature_names):
    print(f"{i+1}. {feature}")
    