import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext

# Librerías de Scikit-Learn
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression, Perceptron, Ridge, Lasso, RidgeClassifier
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier 
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.pipeline import make_pipeline
from sklearn import metrics

# --- NUEVA LIBRERÍA DE IA ---
import tensorflow as tf
# ----------------------------

class MLApp:
    def __init__(self, root):
        self.root = root
        self.root.title("App Armando Cruz Guerrero ")
        self.root.geometry("1150x900")
        self.root.configure(bg="#212121")

        self.df = None
        self.col_x_list = []
        self.col_y = ""
        
        self.last_model = None
        self.last_X_test = None
        self.last_y_test = None
        self.is_regression = True
        self.tf_model = None # Para guardar el modelo de TensorFlow
        self.last_class_labels = None
        
        # --- VARIABLES PARA MEMORIA DE TENSORFLOW ---
        self.tf_trained_x_cols = []
        self.tf_trained_y_col = ""
        self.tf_scaler = None
        self.tf_es_clasificacion = False
        self.tf_clases = None
        # --------------------------------------------

        self.setup_ui()

    def setup_ui(self):
        style = ttk.Style()
        style.theme_use('clam')
        
        style.configure('TextoNegro.TButton', foreground='black')
        style.map('TextoNegro.TButton', foreground=[('disabled', 'black'), ('!disabled', 'black')])
        
        # Sidebar con scroll para que quepan todos los modelos
        sidebar_container = tk.Frame(self.root, width=350, bg="#2d2d2d")
        sidebar_container.pack(side="left", fill="y")
        sidebar_container.pack_propagate(False)

        sidebar_canvas = tk.Canvas(sidebar_container, bg="#2d2d2d", highlightthickness=0)
        sidebar_scrollbar = ttk.Scrollbar(sidebar_container, orient="vertical", command=sidebar_canvas.yview)
        sidebar_canvas.configure(yscrollcommand=sidebar_scrollbar.set)

        sidebar_scrollbar.pack(side="right", fill="y")
        sidebar_canvas.pack(side="left", fill="both", expand=True)

        sidebar = tk.Frame(sidebar_canvas, bg="#2d2d2d", padx=20, pady=20)
        sidebar_window = sidebar_canvas.create_window((0, 0), window=sidebar, anchor="nw")

        def actualizar_scroll(event):
            sidebar_canvas.configure(scrollregion=sidebar_canvas.bbox("all"))

        def ajustar_ancho(event):
            sidebar_canvas.itemconfig(sidebar_window, width=event.width)

        sidebar.bind("<Configure>", actualizar_scroll)
        sidebar_canvas.bind("<Configure>", ajustar_ancho)
        sidebar_canvas.bind_all("<MouseWheel>", lambda event: sidebar_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units"))

        tk.Label(sidebar, text="1. CARGAR ARCHIVO", fg="#00e5ff", bg="#2d2d2d", font=("Arial", 10, "bold")).pack(pady=(0,10))
        ttk.Button(sidebar, text="Seleccionar CSV", command=self.cargar_csv).pack(fill="x")
        
        self.lbl_info = tk.Label(sidebar, text="Archivo no cargado", fg="#aaaaaa", bg="#2d2d2d", font=("Arial", 8))
        self.lbl_info.pack(pady=5)

        tk.Label(sidebar, text="2. COLUMNAS ENTRADA (X)", fg="#00e5ff", bg="#2d2d2d", font=("Arial", 10, "bold")).pack(pady=(10,5))
        self.list_x = tk.Listbox(sidebar, selectmode='multiple', height=6, bg="#3d3d3d", fg="white", 
                                 borderwidth=0, highlightthickness=1, exportselection=0, font=("Arial", 10))
        self.list_x.pack(fill="x")

        tk.Label(sidebar, text="3. COLUMNA OBJETIVO (Y)", fg="#00e5ff", bg="#2d2d2d", font=("Arial", 10, "bold")).pack(pady=(10,5))
        self.list_y = tk.Listbox(sidebar, selectmode='single', height=3, bg="#3d3d3d", fg="white", 
                                 borderwidth=0, highlightthickness=1, exportselection=0, font=("Arial", 10))
        self.list_y.pack(fill="x")

        tk.Label(sidebar, text="4. MODELOS DE PREDICCIÓN", fg="#ffea00", bg="#2d2d2d", font=("Arial", 10, "bold")).pack(pady=(15,5))
        ttk.Button(sidebar, text="Árbol & Random Forest", command=self.ejecutar_arbol_y_bosque).pack(fill="x", pady=2)
        ttk.Button(sidebar, text="Regresión Lineal", command=self.ejecutar_regresion_lineal).pack(fill="x", pady=2)
        ttk.Button(sidebar, text="Regresión Polinomial", command=self.ejecutar_regresion_polinomial).pack(fill="x", pady=2)
        ttk.Button(sidebar, text="Ridge & Lasso", command=self.ejecutar_ridge_lasso).pack(fill="x", pady=2)
        ttk.Button(sidebar, text="SVR", command=self.ejecutar_svr).pack(fill="x", pady=2)
        ttk.Button(sidebar, text="KNN Regressor", command=self.ejecutar_knn_regresor).pack(fill="x", pady=2)

        tk.Label(sidebar, text="5. MODELOS DE CLASIFICACIÓN", fg="#ffea00", bg="#2d2d2d", font=("Arial", 10, "bold")).pack(pady=(15,5))
        ttk.Button(sidebar, text="Regresión Logística", command=self.ejecutar_regresion_logistica).pack(fill="x", pady=2)
        ttk.Button(sidebar, text="SVM", command=self.ejecutar_svm).pack(fill="x", pady=2)
        ttk.Button(sidebar, text="Clasificador KNN", command=self.ejecutar_knn).pack(fill="x", pady=2)
        ttk.Button(sidebar, text="Árbol Clasificador", command=self.ejecutar_arbol_clasificador).pack(fill="x", pady=2)
        ttk.Button(sidebar, text="Random Forest Clasificador", command=self.ejecutar_random_forest_clasificador).pack(fill="x", pady=2)
        ttk.Button(sidebar, text="Extra Trees", command=self.ejecutar_extra_trees).pack(fill="x", pady=2)
        ttk.Button(sidebar, text="Clasificador Ridge", command=self.ejecutar_ridge_classifier).pack(fill="x", pady=2)
        ttk.Button(sidebar, text="Análisis Discriminante (QDA)", command=self.ejecutar_qda).pack(fill="x", pady=2)
        ttk.Button(sidebar, text="Naive Bayes", command=self.ejecutar_naive_bayes).pack(fill="x", pady=2)
        ttk.Button(sidebar, text="Gradient Boosting", command=self.ejecutar_gradient_boosting).pack(fill="x", pady=2)
        ttk.Button(sidebar, text="AdaBoost", command=self.ejecutar_adaboost).pack(fill="x", pady=2)
        ttk.Button(sidebar, text="Perceptrón (Normal)", command=self.ejecutar_perceptron).pack(fill="x", pady=2)
        ttk.Button(sidebar, text="Perceptrón Multicapa (NN)", command=self.ejecutar_red_neuronal).pack(fill="x", pady=2)

        # --- SECCIÓN TENSORFLOW ---
        tk.Label(sidebar, text="6. IA AVANZADA (TENSORFLOW)", fg="#ff5722", bg="#2d2d2d", font=("Arial", 10, "bold")).pack(pady=(15,5))
        ttk.Button(sidebar, text="Entrenar Red Neuronal IA", command=self.ejecutar_tensorflow).pack(fill="x", pady=2)
        # --------------------------

        tk.Frame(sidebar, height=1, bg="#555555").pack(fill="x", pady=10)
        self.btn_metricas = ttk.Button(sidebar, text="Ver Métricas Detalladas", command=self.mostrar_metricas_detalladas, state="disabled", style="TextoNegro.TButton")
        self.btn_metricas.pack(fill="x", pady=2)

        # Main frame / Consola
        main_frame = tk.Frame(self.root, bg="#212121", padx=20, pady=20)
        main_frame.pack(side="right", expand=True, fill="both")

        tk.Label(main_frame, text="LOG DE OPERACIONES Y RESULTADOS", fg="#00e5ff", bg="#212121", font=("Consolas", 12, "bold")).pack(anchor="w")
        self.console = scrolledtext.ScrolledText(main_frame, bg="#121212", fg="#00ff41", font=("Consolas", 11), borderwidth=0)
        self.console.pack(expand=True, fill="both", pady=10)

    def log(self, msg):
        self.console.insert(tk.END, msg + "\n")
        self.console.see(tk.END)

    def cargar_csv(self):
        path = filedialog.askopenfilename(filetypes=[("CSV", "*.csv")])
        if path:
            try:
                self.df = pd.read_csv(path).dropna()
                self.lbl_info.config(text=f"Archivo: {path.split('/')[-1]}", fg="#00ff41")
                self.list_x.delete(0, tk.END)
                self.list_y.delete(0, tk.END)
                for col in self.df.columns:
                    self.list_x.insert(tk.END, col)
                    self.list_y.insert(tk.END, col)
                self.log(f"--- Datos cargados correctamente: {len(self.df)} filas ---")
            except Exception as e:
                messagebox.showerror("Error", f"No se pudo leer el archivo: {e}")

    def obtener_datos_divididos(self):
        if self.df is None:
            messagebox.showwarning("Aviso", "Primero carga un archivo CSV")
            return None
        sel_x = self.list_x.curselection()
        sel_y = self.list_y.curselection()
        if not sel_x or not sel_y:
            messagebox.showwarning("Aviso", "Debes seleccionar X e Y")
            return None
        
        self.col_x_list = [self.list_x.get(i) for i in sel_x]
        self.col_y = self.list_y.get(sel_y[0])
        
        try:
            X = self.df[self.col_x_list].values
            y = self.df[self.col_y].values
            return train_test_split(X, y, test_size=0.2, random_state=42)
        except Exception as e:
            messagebox.showerror("Error", f"Error procesando los datos: {e}")
            return None

    def pedir_valores_prediccion(self):
        ventana = tk.Toplevel(self.root)
        ventana.title("Entrada de Datos para Predicción")
        ventana.geometry("400x350")
        ventana.configure(bg="#2d2d2d")
        ventana.grab_set()

        entries = []
        for col in self.col_x_list:
            frame = tk.Frame(ventana, bg="#2d2d2d", pady=5)
            frame.pack(fill="x", padx=20)
            tk.Label(frame, text=f"{col}:", fg="white", bg="#2d2d2d", width=20, anchor="w").pack(side="left")
            e = ttk.Entry(frame)
            e.pack(side="right", expand=True, fill="x")
            entries.append(e)

        res = {"status": False, "data": []}

        def confirmar():
            try:
                res["data"] = [float(ent.get()) for ent in entries]
                res["status"] = True
                ventana.destroy()
            except ValueError:
                messagebox.showerror("Error", "Ingresa números válidos")

        ttk.Button(ventana, text="Confirmar Predicción", command=confirmar).pack(pady=20)
        self.root.wait_window(ventana)
        return res

    def mostrar_probabilidades_clases(self, clases, probabilidades):
        self.log("Probabilidades por clase:")
        pares = sorted(zip(clases, probabilidades), key=lambda item: item[1], reverse=True)
        for clase, probabilidad in pares:
            self.log(f" - {clase}: {probabilidad * 100:.2f}%")

    def mostrar_prediccion_clasificacion(self, modelo, dato_nuevo, nombre_modelo):
        prediccion = modelo.predict(dato_nuevo)[0]
        self.log(f"Predicción {nombre_modelo}: {prediccion}")

        if hasattr(modelo, "predict_proba"):
            probabilidades = modelo.predict_proba(dato_nuevo)[0]
            self.mostrar_probabilidades_clases(modelo.classes_, probabilidades)
        else:
            self.log("Este modelo no calcula probabilidades por clase.")

    def guardar_estado_metricas(self, modelo, X_test, y_test, es_regresion, class_labels=None):
        self.last_model = modelo
        self.last_X_test = X_test
        self.last_y_test = y_test
        self.is_regression = es_regresion
        self.last_class_labels = class_labels
        self.btn_metricas.config(state="normal")

    def mostrar_metricas_detalladas(self):
        if self.last_model is None: return
        self.log("\n" + "="*40)
        self.log("MÉTRICAS DETALLADAS DEL ÚLTIMO MODELO")
        
        if self.is_regression:
            if isinstance(self.last_model, tf.keras.Model):
                y_pred = self.last_model.predict(self.last_X_test).flatten()
            else:
                y_pred = self.last_model.predict(self.last_X_test)
            self.log(f"Error Absoluto Medio (MAE): {metrics.mean_absolute_error(self.last_y_test, y_pred):.4f}")
            self.log(f"Error Cuadrático Medio (MSE): {metrics.mean_squared_error(self.last_y_test, y_pred):.4f}")
        else:
            if isinstance(self.last_model, tf.keras.Model):
                probabilidades = self.last_model.predict(self.last_X_test)
                y_pred = np.argmax(probabilidades, axis=1)
            else:
                y_pred = self.last_model.predict(self.last_X_test)
            if self.last_class_labels is not None:
                labels = np.arange(len(self.last_class_labels))
                self.log("\nMatriz de Confusión:\n" + str(metrics.confusion_matrix(self.last_y_test, y_pred, labels=labels)))
                self.log("\nReporte de Clasificación:\n" + metrics.classification_report(
                    self.last_y_test,
                    y_pred,
                    labels=labels,
                    target_names=self.last_class_labels,
                    zero_division=0
                ))
            else:
                self.log("\nMatriz de Confusión:\n" + str(metrics.confusion_matrix(self.last_y_test, y_pred)))
                self.log("\nReporte de Clasificación:\n" + metrics.classification_report(self.last_y_test, y_pred, zero_division=0))
        self.log("="*40)

    # --- MODELO TENSORFLOW IA ---
    def ejecutar_tensorflow(self):
        datos = self.obtener_datos_divididos()
        if not datos: return
        X_train, X_test, y_train, y_test = datos

        # 1. VERIFICAR SI PODEMOS REUTILIZAR EL MODELO YA ENTRENADO
        if (self.tf_model is not None and 
            self.col_x_list == self.tf_trained_x_cols and 
            self.col_y == self.tf_trained_y_col):
            
            self.log("\n[!] Dataset y columnas sin cambios. Usando la Red Neuronal ya entrenada...")
            self._realizar_prediccion_tf()
            return

        # 2. SI HUBO CAMBIOS, PROCEDEMOS A ENTRENAR
        self.log("\n Iniciando Entrenamiento con TensorFlow (IA)...")
        
        try:
            X_train = X_train.astype(float)
            X_test = X_test.astype(float)

            es_clasificacion = (
                self.df[self.col_y].dtype == "object"
                or not np.issubdtype(np.array(y_train).dtype, np.number)
            )

            # Guardar el estado actual para futuras comparaciones
            self.tf_trained_x_cols = self.col_x_list.copy()
            self.tf_trained_y_col = self.col_y
            self.tf_es_clasificacion = es_clasificacion

            if es_clasificacion:
                clases = np.unique(np.concatenate([y_train, y_test]))
                self.tf_clases = clases # Guardar clases en memoria
                mapa_clases = {clase: i for i, clase in enumerate(clases)}
                y_train_cod = np.array([mapa_clases[clase] for clase in y_train])
                y_test_cod = np.array([mapa_clases[clase] for clase in y_test])

                self.tf_scaler = StandardScaler() # Guardar el scaler en memoria
                X_train_escalado = self.tf_scaler.fit_transform(X_train)
                X_test_escalado = self.tf_scaler.transform(X_test)

                modelo = tf.keras.Sequential([
                    tf.keras.layers.Input(shape=(len(self.col_x_list),)),
                    tf.keras.layers.Dense(16, activation='relu'),
                    tf.keras.layers.Dense(8, activation='relu'),
                    tf.keras.layers.Dense(len(clases), activation='softmax')
                ])

                modelo.compile(
                    optimizer=tf.keras.optimizers.Adam(0.01),
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy']
                )

                self.log(f" Modo clasificación detectado. Clases: {', '.join(clases)}")
                self.log(" Entrenando... (esto puede tardar unos segundos)")
                modelo.fit(X_train_escalado, y_train_cod, epochs=150, verbose=False)

                perdida, exactitud = modelo.evaluate(X_test_escalado, y_test_cod, verbose=False)
                self.log(" ¡Entrenamiento completado exitosamente!")
                self.log(f" Exactitud TensorFlow: {exactitud * 100:.2f}%")

                self.tf_model = modelo
                self.guardar_estado_metricas(modelo, X_test_escalado, y_test_cod, False, clases)

                # Llamar a la predicción
                self._realizar_prediccion_tf()

            else:
                y_train = y_train.astype(float)
                y_test = y_test.astype(float)

                capa = tf.keras.layers.Dense(units=1, input_shape=[len(self.col_x_list)])
                modelo = tf.keras.Sequential([capa])

                modelo.compile(
                    optimizer=tf.keras.optimizers.Adam(0.1),
                    loss='mean_squared_error'
                )

                self.log(" Modo regresión detectado.")
                self.log(" Entrenando... (esto puede tardar unos segundos)")
                modelo.fit(X_train, y_train, epochs=500, verbose=False)
                
                self.log(" ¡Entrenamiento completado exitosamente!")
                
                self.tf_model = modelo
                self.guardar_estado_metricas(modelo, X_test, y_test, True)

                pesos = capa.get_weights()
                self.log(f"\n El modelo ha aprendido que el peso (factor) es: {pesos[0][0][0]:.2f}")

                # Llamar a la predicción
                self._realizar_prediccion_tf()

        except Exception as e:
            messagebox.showerror("Error IA", f"Fallo en TensorFlow: {e}")

    # --- FUNCIÓN EXCLUSIVA PARA PREDECIR CON EL MODELO EN MEMORIA ---
    def _realizar_prediccion_tf(self):
        inputs = self.pedir_valores_prediccion()
        if inputs["status"]:
            if self.tf_es_clasificacion:
                # Usar el scaler que guardamos durante el entrenamiento
                dato_nuevo = self.tf_scaler.transform(np.array([inputs["data"]], dtype=float))
                probabilidades = self.tf_model.predict(dato_nuevo, verbose=False)[0]
                indice = int(np.argmax(probabilidades))
                
                self.log(f"\n-> RESULTADO IA TENSORFLOW:")
                self.log(f"Para los valores {inputs['data']}, la IA predice {self.tf_trained_y_col} = {self.tf_clases[indice]}")
                self.mostrar_probabilidades_clases(self.tf_clases, probabilidades)
            else:
                dato_nuevo = np.array([inputs["data"]])
                prediccion = self.tf_model.predict(dato_nuevo, verbose=False)
                
                self.log(f"\n-> RESULTADO IA TENSORFLOW:")
                self.log(f"Para los valores {inputs['data']}, la IA predice {self.tf_trained_y_col} = {prediccion[0][0]:.4f}")

    # --- MODELOS DE REGRESIÓN SKLEARN ---

    def ejecutar_arbol_y_bosque(self):
        datos = self.obtener_datos_divididos()
        if not datos: return
        X_train, X_test, y_train, y_test = datos
        try:
            arbol = DecisionTreeRegressor(max_depth=3).fit(X_train, y_train)
            bosque = RandomForestRegressor(n_estimators=100).fit(X_train, y_train)
            self.log(f"\nPrecisión Árbol (R²): {metrics.r2_score(y_test, arbol.predict(X_test)):.4f}")
            self.log(f"Precisión Forest (R²): {metrics.r2_score(y_test, bosque.predict(X_test)):.4f}")
            self.guardar_estado_metricas(bosque, X_test, y_test, True)
        except Exception as e: messagebox.showerror("Error", str(e)); return

        inputs = self.pedir_valores_prediccion()
        if inputs["status"]:
            dn = np.array([inputs["data"]])
            self.log(f"Predicción Árbol: {arbol.predict(dn)[0]:.2f} | Forest: {bosque.predict(dn)[0]:.2f}")

    def ejecutar_regresion_lineal(self):
        datos = self.obtener_datos_divididos()
        if not datos: return
        X_train, X_test, y_train, y_test = datos
        try:
            modelo = LinearRegression().fit(X_train, y_train)
            self.log(f"\nPrecisión Lineal (R²): {metrics.r2_score(y_test, modelo.predict(X_test)):.4f}")
            self.guardar_estado_metricas(modelo, X_test, y_test, True)
        except Exception as e: messagebox.showerror("Error", str(e)); return

        inputs = self.pedir_valores_prediccion()
        if inputs["status"]:
            dn = np.array([inputs["data"]])
            self.log(f"Predicción Lineal: {modelo.predict(dn)[0]:.2f}")

    def ejecutar_regresion_polinomial(self):
        datos = self.obtener_datos_divididos()
        if not datos: return
        X_train, X_test, y_train, y_test = datos
        try:
            trans = PolynomialFeatures(degree=2)
            X_tr_c = trans.fit_transform(X_train)
            X_ts_c = trans.transform(X_test)
            modelo = LinearRegression().fit(X_tr_c, y_train)
            self.log(f"\nPrecisión Polinomial (R²): {metrics.r2_score(y_test, modelo.predict(X_ts_c)):.4f}")
            self.guardar_estado_metricas(modelo, X_ts_c, y_test, True)
        except Exception as e: messagebox.showerror("Error", str(e)); return

        inputs = self.pedir_valores_prediccion()
        if inputs["status"]:
            dn = trans.transform(np.array([inputs["data"]]))
            self.log(f"Predicción Polinomial: {modelo.predict(dn)[0]:.2f}")

    def ejecutar_ridge_lasso(self):
        datos = self.obtener_datos_divididos()
        if not datos: return
        X_train, X_test, y_train, y_test = datos
        try:
            y_train = y_train.astype(float)
            y_test = y_test.astype(float)
            ridge = Ridge(alpha=1.0).fit(X_train, y_train)
            lasso = Lasso(alpha=0.1, max_iter=10000).fit(X_train, y_train)
            self.log("\n--- RESULTADO DEL ENTRENAMIENTO (RIDGE & LASSO) ---")
            self.log(f"Precisión Ridge (R²): {metrics.r2_score(y_test, ridge.predict(X_test)):.4f}")
            self.log(f"Precisión Lasso (R²): {metrics.r2_score(y_test, lasso.predict(X_test)):.4f}")
            self.guardar_estado_metricas(ridge, X_test, y_test, True)
        except Exception as e:
            messagebox.showerror("Error", f"Error al entrenar Ridge/Lasso: {e}")
            return

        inputs = self.pedir_valores_prediccion()
        if inputs["status"]:
            dn = np.array([inputs["data"]])
            self.log("\n-> PREDICCIÓN CON NUEVOS DATOS:")
            self.log(f"Predicción Ridge: {ridge.predict(dn)[0]:.2f}")
            self.log(f"Predicción Lasso: {lasso.predict(dn)[0]:.2f}")

    def ejecutar_svr(self):
        datos = self.obtener_datos_divididos()
        if not datos: return
        X_train, X_test, y_train, y_test = datos
        try:
            y_train = y_train.astype(float)
            y_test = y_test.astype(float)
            modelo = make_pipeline(StandardScaler(), SVR(kernel='rbf', C=100, gamma='scale', epsilon=0.1))
            modelo.fit(X_train, y_train)
            self.log("\n--- RESULTADO DEL ENTRENAMIENTO (SVR) ---")
            self.log(f"Precisión SVR (R²): {metrics.r2_score(y_test, modelo.predict(X_test)):.4f}")
            self.guardar_estado_metricas(modelo, X_test, y_test, True)
        except Exception as e:
            messagebox.showerror("Error", f"Error al entrenar SVR: {e}")
            return

        inputs = self.pedir_valores_prediccion()
        if inputs["status"]:
            dn = np.array([inputs["data"]])
            self.log("\n-> PREDICCIÓN CON NUEVOS DATOS:")
            self.log(f"Predicción SVR: {modelo.predict(dn)[0]:.2f}")

    def ejecutar_knn_regresor(self):
        datos = self.obtener_datos_divididos()
        if not datos: return
        X_train, X_test, y_train, y_test = datos
        try:
            y_train = y_train.astype(float)
            y_test = y_test.astype(float)
            modelo = KNeighborsRegressor(n_neighbors=5).fit(X_train, y_train)
            self.log("\n--- RESULTADO DEL ENTRENAMIENTO (KNN REGRESSOR) ---")
            self.log(f"Precisión KNN Regressor (R²): {metrics.r2_score(y_test, modelo.predict(X_test)):.4f}")
            self.guardar_estado_metricas(modelo, X_test, y_test, True)
        except Exception as e:
            messagebox.showerror("Error", f"Error al entrenar KNN Regressor: {e}")
            return

        inputs = self.pedir_valores_prediccion()
        if inputs["status"]:
            dn = np.array([inputs["data"]])
            self.log("\n-> PREDICCIÓN CON NUEVOS DATOS:")
            self.log(f"Predicción KNN Regressor: {modelo.predict(dn)[0]:.2f}")

    # --- MODELOS DE CLASIFICACIÓN ---

    def ejecutar_regresion_logistica(self):
        datos = self.obtener_datos_divididos()
        if not datos: return
        X_train, X_test, y_train, y_test = datos
        try:
            modelo = LogisticRegression(max_iter=1000).fit(X_train, y_train)
            self.log(f"\nExactitud Logística: {metrics.accuracy_score(y_test, modelo.predict(X_test))*100:.2f}%")
            self.guardar_estado_metricas(modelo, X_test, y_test, False)
        except Exception as e: messagebox.showerror("Error", str(e)); return

        inputs = self.pedir_valores_prediccion()
        if inputs["status"]:
            self.mostrar_prediccion_clasificacion(modelo, np.array([inputs["data"]]), "Logística")

    def ejecutar_svm(self):
        datos = self.obtener_datos_divididos()
        if not datos: return
        X_train, X_test, y_train, y_test = datos
        try:
            modelo = SVC(probability=True, kernel='rbf').fit(X_train, y_train)
            self.log(f"\nExactitud SVM: {metrics.accuracy_score(y_test, modelo.predict(X_test))*100:.2f}%")
            self.guardar_estado_metricas(modelo, X_test, y_test, False)
        except Exception as e: messagebox.showerror("Error", str(e)); return
        inputs = self.pedir_valores_prediccion()
        if inputs["status"]:
            self.mostrar_prediccion_clasificacion(modelo, np.array([inputs["data"]]), "SVM")

    def ejecutar_knn(self):
        datos = self.obtener_datos_divididos()
        if not datos: return
        X_train, X_test, y_train, y_test = datos
        try:
            modelo = KNeighborsClassifier(n_neighbors=5).fit(X_train, y_train)
            self.log(f"\nExactitud KNN: {metrics.accuracy_score(y_test, modelo.predict(X_test))*100:.2f}%")
            self.guardar_estado_metricas(modelo, X_test, y_test, False)
        except Exception as e: messagebox.showerror("Error", str(e)); return
        inputs = self.pedir_valores_prediccion()
        if inputs["status"]:
            self.mostrar_prediccion_clasificacion(modelo, np.array([inputs["data"]]), "KNN")

    def ejecutar_arbol_clasificador(self):
        datos = self.obtener_datos_divididos()
        if not datos: return
        X_train, X_test, y_train, y_test = datos
        try:
            modelo = DecisionTreeClassifier(max_depth=4, random_state=42).fit(X_train, y_train)
            self.log("\n--- RESULTADO DEL ENTRENAMIENTO (ÁRBOL CLASIFICADOR) ---")
            self.log(f"Exactitud Árbol Clasificador: {metrics.accuracy_score(y_test, modelo.predict(X_test))*100:.2f}%")
            self.guardar_estado_metricas(modelo, X_test, y_test, False)
        except Exception as e:
            messagebox.showerror("Error", f"Error al entrenar Árbol Clasificador: {e}")
            return

        inputs = self.pedir_valores_prediccion()
        if inputs["status"]:
            self.mostrar_prediccion_clasificacion(modelo, np.array([inputs["data"]]), "Árbol Clasificador")

    def ejecutar_random_forest_clasificador(self):
        datos = self.obtener_datos_divididos()
        if not datos: return
        X_train, X_test, y_train, y_test = datos
        try:
            modelo = RandomForestClassifier(n_estimators=150, random_state=42).fit(X_train, y_train)
            self.log("\n--- RESULTADO DEL ENTRENAMIENTO (RANDOM FOREST CLASIFICADOR) ---")
            self.log(f"Exactitud Random Forest Clasificador: {metrics.accuracy_score(y_test, modelo.predict(X_test))*100:.2f}%")
            self.guardar_estado_metricas(modelo, X_test, y_test, False)
        except Exception as e:
            messagebox.showerror("Error", f"Error al entrenar Random Forest Clasificador: {e}")
            return

        inputs = self.pedir_valores_prediccion()
        if inputs["status"]:
            self.mostrar_prediccion_clasificacion(modelo, np.array([inputs["data"]]), "Random Forest Clasificador")

    # --- NUEVOS MODELOS DE CLASIFICACIÓN AGREGADOS ---

    def ejecutar_extra_trees(self):
        datos = self.obtener_datos_divididos()
        if not datos: return
        X_train, X_test, y_train, y_test = datos
        try:
            modelo = ExtraTreesClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)
            self.log("\n--- RESULTADO DEL ENTRENAMIENTO (EXTRA TREES) ---")
            self.log(f"Exactitud Extra Trees: {metrics.accuracy_score(y_test, modelo.predict(X_test))*100:.2f}%")
            self.guardar_estado_metricas(modelo, X_test, y_test, False)
        except Exception as e:
            messagebox.showerror("Error", f"Error al entrenar Extra Trees: {e}")
            return

        inputs = self.pedir_valores_prediccion()
        if inputs["status"]:
            self.mostrar_prediccion_clasificacion(modelo, np.array([inputs["data"]]), "Extra Trees")

    def ejecutar_ridge_classifier(self):
        datos = self.obtener_datos_divididos()
        if not datos: return
        X_train, X_test, y_train, y_test = datos
        try:
            modelo = RidgeClassifier().fit(X_train, y_train)
            self.log("\n--- RESULTADO DEL ENTRENAMIENTO (RIDGE CLASSIFIER) ---")
            self.log(f"Exactitud Clasificador Ridge: {metrics.accuracy_score(y_test, modelo.predict(X_test))*100:.2f}%")
            self.guardar_estado_metricas(modelo, X_test, y_test, False)
        except Exception as e:
            messagebox.showerror("Error", f"Error al entrenar Ridge Classifier: {e}")
            return

        inputs = self.pedir_valores_prediccion()
        if inputs["status"]:
            self.mostrar_prediccion_clasificacion(modelo, np.array([inputs["data"]]), "Clasificador Ridge")

    def ejecutar_qda(self):
        datos = self.obtener_datos_divididos()
        if not datos: return
        X_train, X_test, y_train, y_test = datos
        try:
            modelo = QuadraticDiscriminantAnalysis().fit(X_train, y_train)
            self.log("\n--- RESULTADO DEL ENTRENAMIENTO (QDA) ---")
            self.log(f"Exactitud QDA: {metrics.accuracy_score(y_test, modelo.predict(X_test))*100:.2f}%")
            self.guardar_estado_metricas(modelo, X_test, y_test, False)
        except Exception as e:
            messagebox.showerror("Error", f"Error al entrenar QDA: {e}")
            return

        inputs = self.pedir_valores_prediccion()
        if inputs["status"]:
            self.mostrar_prediccion_clasificacion(modelo, np.array([inputs["data"]]), "Análisis Discriminante (QDA)")

    # --------------------------------------------------

    def ejecutar_naive_bayes(self):
        datos = self.obtener_datos_divididos()
        if not datos: return
        X_train, X_test, y_train, y_test = datos
        try:
            modelo = GaussianNB().fit(X_train, y_train)
            self.log("\n--- RESULTADO DEL ENTRENAMIENTO (NAIVE BAYES) ---")
            self.log(f"Exactitud Naive Bayes: {metrics.accuracy_score(y_test, modelo.predict(X_test))*100:.2f}%")
            self.guardar_estado_metricas(modelo, X_test, y_test, False)
        except Exception as e:
            messagebox.showerror("Error", f"Error al entrenar Naive Bayes: {e}")
            return

        inputs = self.pedir_valores_prediccion()
        if inputs["status"]:
            self.mostrar_prediccion_clasificacion(modelo, np.array([inputs["data"]]), "Naive Bayes")

    def ejecutar_gradient_boosting(self):
        datos = self.obtener_datos_divididos()
        if not datos: return
        X_train, X_test, y_train, y_test = datos
        try:
            modelo = GradientBoostingClassifier(random_state=42).fit(X_train, y_train)
            self.log("\n--- RESULTADO DEL ENTRENAMIENTO (GRADIENT BOOSTING) ---")
            self.log(f"Exactitud Gradient Boosting: {metrics.accuracy_score(y_test, modelo.predict(X_test))*100:.2f}%")
            self.guardar_estado_metricas(modelo, X_test, y_test, False)
        except Exception as e:
            messagebox.showerror("Error", f"Error al entrenar Gradient Boosting: {e}")
            return

        inputs = self.pedir_valores_prediccion()
        if inputs["status"]:
            self.mostrar_prediccion_clasificacion(modelo, np.array([inputs["data"]]), "Gradient Boosting")

    def ejecutar_adaboost(self):
        datos = self.obtener_datos_divididos()
        if not datos: return
        X_train, X_test, y_train, y_test = datos
        try:
            modelo = AdaBoostClassifier(random_state=42).fit(X_train, y_train)
            self.log("\n--- RESULTADO DEL ENTRENAMIENTO (ADABOOST) ---")
            self.log(f"Exactitud AdaBoost: {metrics.accuracy_score(y_test, modelo.predict(X_test))*100:.2f}%")
            self.guardar_estado_metricas(modelo, X_test, y_test, False)
        except Exception as e:
            messagebox.showerror("Error", f"Error al entrenar AdaBoost: {e}")
            return

        inputs = self.pedir_valores_prediccion()
        if inputs["status"]:
            self.mostrar_prediccion_clasificacion(modelo, np.array([inputs["data"]]), "AdaBoost")

    def ejecutar_perceptron(self):
        datos = self.obtener_datos_divididos()
        if not datos: return
        X_train, X_test, y_train, y_test = datos
        try:
            modelo = Perceptron(max_iter=1000).fit(X_train, y_train)
            self.log(f"\nExactitud Perceptrón: {metrics.accuracy_score(y_test, modelo.predict(X_test))*100:.2f}%")
            self.guardar_estado_metricas(modelo, X_test, y_test, False)
        except Exception as e: messagebox.showerror("Error", str(e)); return
        inputs = self.pedir_valores_prediccion()
        if inputs["status"]:
            self.mostrar_prediccion_clasificacion(modelo, np.array([inputs["data"]]), "Perceptrón")

    def ejecutar_red_neuronal(self):
        datos = self.obtener_datos_divididos()
        if not datos: return
        X_train, X_test, y_train, y_test = datos
        try:
            modelo = MLPClassifier(hidden_layer_sizes=(10, 10), max_iter=1000).fit(X_train, y_train)
            self.log(f"\nExactitud Red Neuronal Sklearn: {metrics.accuracy_score(y_test, modelo.predict(X_test))*100:.2f}%")
            self.guardar_estado_metricas(modelo, X_test, y_test, False)
        except Exception as e: messagebox.showerror("Error", str(e)); return
        inputs = self.pedir_valores_prediccion()
        if inputs["status"]:
            self.mostrar_prediccion_clasificacion(modelo, np.array([inputs["data"]]), "Red Neuronal")


if __name__ == "__main__":
    root = tk.Tk()
    app = MLApp(root)
    root.mainloop()