from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.staticfiles import StaticFiles
import os
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List
import pandas as pd
import numpy as np
import uuid
import io
import json
import csv

from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestRegressor, RandomForestClassifier,
    GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
)
from sklearn.linear_model import (
    LinearRegression, LogisticRegression,
    Perceptron, Ridge, Lasso, RidgeClassifier
)
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.pipeline import make_pipeline
from sklearn import metrics


app = FastAPI(title="ML Platform API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

sessions: dict = {}


def get_session(session_id: str) -> dict:
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Sesion no encontrada")
    return sessions[session_id]


def new_session_data() -> dict:
    return {
        "df": None,
        "filename": None,
        "last_model": None,
        "last_X_test": None,
        "last_y_test": None,
        "is_regression": True,
        "last_class_labels": None,
        "last_model_name": None,
        "x_cols": [],
        "y_col": "",
        "results": [],
        "poly_transformer": None,
        "_arbol_companion": None,
        "_lasso_companion": None,
    }


class TrainRequest(BaseModel):
    session_id: str
    x_cols: List[str]
    y_col: str
    model: str


class PredictRequest(BaseModel):
    session_id: str
    values: List[float]


@app.post("/api/session")
def create_session():
    session_id = str(uuid.uuid4())
    sessions[session_id] = new_session_data()
    return {"session_id": session_id}


@app.post("/api/upload")
async def upload_csv(session_id: str = Query(...), file: UploadFile = File(...)):
    session = get_session(session_id)
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Solo se aceptan archivos CSV")
    content = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(content)).dropna()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"No se pudo leer el archivo: {e}")
    session["df"] = df
    session["filename"] = file.filename
    return {
        "filename": file.filename,
        "rows": len(df),
        "columns": df.columns.tolist(),
        "numeric_columns": df.select_dtypes(include=[np.number]).columns.tolist(),
        "preview": df.head(8).to_dict(orient="records"),
    }


REGRESSION_KEYS = {
    'decision_tree_regressor', 'linear_regression', 'polynomial_regression',
    'ridge_lasso', 'svr', 'knn_regressor',
}

@app.post("/api/train")
def train_model(req: TrainRequest):
    session = get_session(req.session_id)
    df = session["df"]
    if df is None:
        raise HTTPException(status_code=400, detail="No hay datos cargados")

    # Validate columns exist
    missing = [c for c in req.x_cols + [req.y_col] if c not in df.columns]
    if missing:
        raise HTTPException(status_code=400, detail=f"Columnas no encontradas en el dataset: {missing}")

    # Validate X columns are numeric
    non_numeric_x = [c for c in req.x_cols if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric_x:
        raise HTTPException(
            status_code=400,
            detail=f"Las columnas {non_numeric_x} contienen texto y no pueden usarse como variables X. "
                   f"Solo se permiten columnas numericas como entradas del modelo.",
        )

    # Validate Y type matches model family
    y_is_numeric = pd.api.types.is_numeric_dtype(df[req.y_col])
    if req.model in REGRESSION_KEYS and not y_is_numeric:
        raise HTTPException(
            status_code=400,
            detail=f"La columna '{req.y_col}' contiene texto. "
                   f"Los modelos de regresion requieren una variable objetivo numerica. "
                   f"Selecciona un modelo de clasificacion.",
        )

    try:
        X = df[req.x_cols].values.astype(float)
        y = df[req.y_col].values
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error al procesar las columnas: {e}")

    if len(X) < 5:
        raise HTTPException(status_code=400, detail="El dataset necesita al menos 5 filas para entrenar.")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    session["x_cols"] = req.x_cols
    session["y_col"] = req.y_col
    session["poly_transformer"] = None
    session["_arbol_companion"] = None
    session["_lasso_companion"] = None
    session["last_class_labels"] = None
    result = {}

    try:
        k = req.model

        if k == "decision_tree_regressor":
            arbol = DecisionTreeRegressor(max_depth=3).fit(X_train, y_train)
            bosque = RandomForestRegressor(n_estimators=100).fit(X_train, y_train)
            r2a = metrics.r2_score(y_test, arbol.predict(X_test))
            r2b = metrics.r2_score(y_test, bosque.predict(X_test))
            result = {"model": "Arbol & Random Forest", "type": "regression",
                      "metrics": {"r2_arbol": round(r2a, 4), "r2_bosque": round(r2b, 4)},
                      "summary": f"R2 Arbol: {r2a:.4f} | R2 Forest: {r2b:.4f}"}
            session["last_model"] = bosque
            session["_arbol_companion"] = arbol
            session["is_regression"] = True

        elif k == "linear_regression":
            modelo = LinearRegression().fit(X_train, y_train)
            r2 = metrics.r2_score(y_test, modelo.predict(X_test))
            result = {"model": "Regresion Lineal", "type": "regression",
                      "metrics": {"r2": round(r2, 4)}, "summary": f"R2: {r2:.4f}"}
            session["last_model"] = modelo
            session["is_regression"] = True

        elif k == "polynomial_regression":
            trans = PolynomialFeatures(degree=2)
            Xtr = trans.fit_transform(X_train)
            Xts = trans.transform(X_test)
            modelo = LinearRegression().fit(Xtr, y_train)
            r2 = metrics.r2_score(y_test, modelo.predict(Xts))
            result = {"model": "Regresion Polinomial", "type": "regression",
                      "metrics": {"r2": round(r2, 4)}, "summary": f"R2: {r2:.4f}"}
            session["last_model"] = modelo
            session["poly_transformer"] = trans
            session["last_X_test"] = Xts
            session["is_regression"] = True

        elif k == "ridge_lasso":
            yf_tr = y_train.astype(float)
            yf_ts = y_test.astype(float)
            ridge = Ridge(alpha=1.0).fit(X_train, yf_tr)
            lasso = Lasso(alpha=0.1, max_iter=10000).fit(X_train, yf_tr)
            r2r = metrics.r2_score(yf_ts, ridge.predict(X_test))
            r2l = metrics.r2_score(yf_ts, lasso.predict(X_test))
            result = {"model": "Ridge & Lasso", "type": "regression",
                      "metrics": {"r2_ridge": round(r2r, 4), "r2_lasso": round(r2l, 4)},
                      "summary": f"R2 Ridge: {r2r:.4f} | R2 Lasso: {r2l:.4f}"}
            session["last_model"] = ridge
            session["_lasso_companion"] = lasso
            session["is_regression"] = True

        elif k == "svr":
            yf_tr = y_train.astype(float)
            yf_ts = y_test.astype(float)
            modelo = make_pipeline(StandardScaler(), SVR(kernel="rbf", C=100, gamma="scale", epsilon=0.1))
            modelo.fit(X_train, yf_tr)
            r2 = metrics.r2_score(yf_ts, modelo.predict(X_test))
            result = {"model": "SVR", "type": "regression",
                      "metrics": {"r2": round(r2, 4)}, "summary": f"R2: {r2:.4f}"}
            session["last_model"] = modelo
            session["is_regression"] = True

        elif k == "knn_regressor":
            yf_tr = y_train.astype(float)
            yf_ts = y_test.astype(float)
            modelo = KNeighborsRegressor(n_neighbors=5).fit(X_train, yf_tr)
            r2 = metrics.r2_score(yf_ts, modelo.predict(X_test))
            result = {"model": "KNN Regressor", "type": "regression",
                      "metrics": {"r2": round(r2, 4)}, "summary": f"R2: {r2:.4f}"}
            session["last_model"] = modelo
            session["is_regression"] = True

        elif k == "logistic_regression":
            modelo = LogisticRegression(max_iter=1000).fit(X_train, y_train)
            acc = metrics.accuracy_score(y_test, modelo.predict(X_test))
            result = {"model": "Regresion Logistica", "type": "classification",
                      "metrics": {"accuracy": round(acc * 100, 2)}, "summary": f"Exactitud: {acc*100:.2f}%"}
            session["last_model"] = modelo
            session["is_regression"] = False

        elif k == "svm":
            modelo = SVC(probability=True, kernel="rbf").fit(X_train, y_train)
            acc = metrics.accuracy_score(y_test, modelo.predict(X_test))
            result = {"model": "SVM", "type": "classification",
                      "metrics": {"accuracy": round(acc * 100, 2)}, "summary": f"Exactitud: {acc*100:.2f}%"}
            session["last_model"] = modelo
            session["is_regression"] = False

        elif k == "knn_classifier":
            modelo = KNeighborsClassifier(n_neighbors=5).fit(X_train, y_train)
            acc = metrics.accuracy_score(y_test, modelo.predict(X_test))
            result = {"model": "KNN Clasificador", "type": "classification",
                      "metrics": {"accuracy": round(acc * 100, 2)}, "summary": f"Exactitud: {acc*100:.2f}%"}
            session["last_model"] = modelo
            session["is_regression"] = False

        elif k == "decision_tree_classifier":
            modelo = DecisionTreeClassifier(max_depth=4, random_state=42).fit(X_train, y_train)
            acc = metrics.accuracy_score(y_test, modelo.predict(X_test))
            result = {"model": "Arbol Clasificador", "type": "classification",
                      "metrics": {"accuracy": round(acc * 100, 2)}, "summary": f"Exactitud: {acc*100:.2f}%"}
            session["last_model"] = modelo
            session["is_regression"] = False

        elif k == "random_forest_classifier":
            modelo = RandomForestClassifier(n_estimators=150, random_state=42).fit(X_train, y_train)
            acc = metrics.accuracy_score(y_test, modelo.predict(X_test))
            result = {"model": "Random Forest Clasificador", "type": "classification",
                      "metrics": {"accuracy": round(acc * 100, 2)}, "summary": f"Exactitud: {acc*100:.2f}%"}
            session["last_model"] = modelo
            session["is_regression"] = False

        elif k == "extra_trees":
            modelo = ExtraTreesClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)
            acc = metrics.accuracy_score(y_test, modelo.predict(X_test))
            result = {"model": "Extra Trees", "type": "classification",
                      "metrics": {"accuracy": round(acc * 100, 2)}, "summary": f"Exactitud: {acc*100:.2f}%"}
            session["last_model"] = modelo
            session["is_regression"] = False

        elif k == "ridge_classifier":
            modelo = RidgeClassifier().fit(X_train, y_train)
            acc = metrics.accuracy_score(y_test, modelo.predict(X_test))
            result = {"model": "Ridge Clasificador", "type": "classification",
                      "metrics": {"accuracy": round(acc * 100, 2)}, "summary": f"Exactitud: {acc*100:.2f}%"}
            session["last_model"] = modelo
            session["is_regression"] = False

        elif k == "qda":
            modelo = QuadraticDiscriminantAnalysis().fit(X_train, y_train)
            acc = metrics.accuracy_score(y_test, modelo.predict(X_test))
            result = {"model": "QDA", "type": "classification",
                      "metrics": {"accuracy": round(acc * 100, 2)}, "summary": f"Exactitud: {acc*100:.2f}%"}
            session["last_model"] = modelo
            session["is_regression"] = False

        elif k == "naive_bayes":
            modelo = GaussianNB().fit(X_train, y_train)
            acc = metrics.accuracy_score(y_test, modelo.predict(X_test))
            result = {"model": "Naive Bayes", "type": "classification",
                      "metrics": {"accuracy": round(acc * 100, 2)}, "summary": f"Exactitud: {acc*100:.2f}%"}
            session["last_model"] = modelo
            session["is_regression"] = False

        elif k == "gradient_boosting":
            modelo = GradientBoostingClassifier(random_state=42).fit(X_train, y_train)
            acc = metrics.accuracy_score(y_test, modelo.predict(X_test))
            result = {"model": "Gradient Boosting", "type": "classification",
                      "metrics": {"accuracy": round(acc * 100, 2)}, "summary": f"Exactitud: {acc*100:.2f}%"}
            session["last_model"] = modelo
            session["is_regression"] = False

        elif k == "adaboost":
            modelo = AdaBoostClassifier(random_state=42).fit(X_train, y_train)
            acc = metrics.accuracy_score(y_test, modelo.predict(X_test))
            result = {"model": "AdaBoost", "type": "classification",
                      "metrics": {"accuracy": round(acc * 100, 2)}, "summary": f"Exactitud: {acc*100:.2f}%"}
            session["last_model"] = modelo
            session["is_regression"] = False

        elif k == "perceptron":
            modelo = Perceptron(max_iter=1000).fit(X_train, y_train)
            acc = metrics.accuracy_score(y_test, modelo.predict(X_test))
            result = {"model": "Perceptron", "type": "classification",
                      "metrics": {"accuracy": round(acc * 100, 2)}, "summary": f"Exactitud: {acc*100:.2f}%"}
            session["last_model"] = modelo
            session["is_regression"] = False

        elif k == "mlp_classifier":
            modelo = MLPClassifier(hidden_layer_sizes=(10, 10), max_iter=1000).fit(X_train, y_train)
            acc = metrics.accuracy_score(y_test, modelo.predict(X_test))
            result = {"model": "MLP (Red Neuronal)", "type": "classification",
                      "metrics": {"accuracy": round(acc * 100, 2)}, "summary": f"Exactitud: {acc*100:.2f}%"}
            session["last_model"] = modelo
            session["is_regression"] = False

        else:
            raise HTTPException(status_code=400, detail=f"Modelo desconocido: {k}")

        if k != "polynomial_regression":
            session["last_X_test"] = X_test
        session["last_y_test"] = y_test
        session["last_model_name"] = result["model"]

        session["results"].append({
            "model": result["model"],
            "type": result["type"],
            "x_cols": req.x_cols,
            "y_col": req.y_col,
            "metrics": result["metrics"],
            "summary": result["summary"],
        })

        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/predict")
def predict(req: PredictRequest):
    session = get_session(req.session_id)
    modelo = session.get("last_model")
    if modelo is None:
        raise HTTPException(status_code=400, detail="No hay modelo entrenado")

    try:
        dato = np.array([req.values])

        poly = session.get("poly_transformer")
        if poly is not None:
            dato = poly.transform(dato)

        if session["is_regression"]:
            pred = modelo.predict(dato)
            extra = {}
            if session.get("_arbol_companion"):
                extra["arbol"] = round(float(session["_arbol_companion"].predict(dato)[0]), 4)
            if session.get("_lasso_companion"):
                extra["lasso"] = round(float(session["_lasso_companion"].predict(dato)[0]), 4)
            return {"prediction": round(float(pred[0]), 4), "type": "regression", "extra": extra}
        else:
            pred = modelo.predict(dato)[0]
            result = {"prediction": str(pred), "type": "classification"}
            if hasattr(modelo, "predict_proba"):
                probs = modelo.predict_proba(dato)[0]
                result["probabilities"] = {
                    str(c): round(float(p), 6) for c, p in zip(modelo.classes_, probs)
                }
            return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/metrics/{session_id}")
def get_metrics(session_id: str):
    session = get_session(session_id)
    modelo = session.get("last_model")
    if modelo is None:
        raise HTTPException(status_code=400, detail="No hay modelo entrenado")

    X_test = session["last_X_test"]
    y_test = session["last_y_test"]

    try:
        y_pred = modelo.predict(X_test)

        if session["is_regression"]:
            return {
                "type": "regression",
                "mae": round(float(metrics.mean_absolute_error(y_test, y_pred)), 4),
                "mse": round(float(metrics.mean_squared_error(y_test, y_pred)), 4),
                "r2":  round(float(metrics.r2_score(y_test, y_pred)), 4),
            }
        else:
            labels = session.get("last_class_labels")
            if labels is not None:
                mapa = {c: i for i, c in enumerate(labels)}
                y_test_idx = np.array([mapa.get(c, c) for c in y_test])
                label_indices = np.arange(len(labels))
                cm = metrics.confusion_matrix(y_test_idx, y_pred, labels=label_indices).tolist()
                report = metrics.classification_report(
                    y_test_idx, y_pred, labels=label_indices,
                    target_names=[str(l) for l in labels], zero_division=0, output_dict=True
                )
                acc = round(float(metrics.accuracy_score(y_test_idx, y_pred)) * 100, 2)
            else:
                y_test_str = np.array([str(v) for v in y_test])
                y_pred_str = np.array([str(v) for v in y_pred])
                cm = metrics.confusion_matrix(y_test_str, y_pred_str).tolist()
                report = metrics.classification_report(y_test_str, y_pred_str, zero_division=0, output_dict=True)
                acc = round(float(metrics.accuracy_score(y_test_str, y_pred_str)) * 100, 2)
            return {
                "type": "classification",
                "accuracy": acc,
                "confusion_matrix": cm,
                "report": report,
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/preview/{session_id}")
def get_preview(session_id: str, limit: int = Query(10), offset: int = Query(0)):
    session = get_session(session_id)
    df = session["df"]
    if df is None:
        raise HTTPException(status_code=400, detail="No hay datos cargados")
    total = len(df)
    rows = df.iloc[offset:offset + limit].to_dict(orient="records")
    return {"rows": rows, "total": total, "offset": offset, "limit": limit}


@app.get("/api/results/{session_id}")
def get_results(session_id: str):
    session = get_session(session_id)
    return {"results": session["results"]}


@app.get("/api/results/{session_id}/download")
def download_results(session_id: str, format: str = "json"):
    session = get_session(session_id)
    results = session["results"]
    if not results:
        raise HTTPException(status_code=400, detail="No hay resultados guardados")

    if format == "json":
        content = json.dumps(results, indent=2, ensure_ascii=False)
        return StreamingResponse(
            io.BytesIO(content.encode("utf-8")),
            media_type="application/json",
            headers={"Content-Disposition": "attachment; filename=resultados.json"},
        )

    if format == "csv":
        rows = []
        for r in results:
            flat = {
                "modelo": r["model"], "tipo": r["type"],
                "columnas_x": ", ".join(r["x_cols"]),
                "columna_y": r["y_col"], "resumen": r["summary"],
            }
            flat.update(r["metrics"])
            rows.append(flat)
        output = io.StringIO()
        if rows:
            writer = csv.DictWriter(output, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        return StreamingResponse(
            io.BytesIO(output.getvalue().encode("utf-8")),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=resultados.csv"},
        )

    raise HTTPException(status_code=400, detail="Formato no soportado: usa json o csv")


@app.delete("/api/results/{session_id}")
def clear_results(session_id: str):
    session = get_session(session_id)
    session["results"] = []
    return {"message": "Resultados eliminados"}

# Serve compiled React frontend (production)
_static = os.path.join(os.path.dirname(__file__), "static")
if os.path.isdir(_static):
    app.mount("/", StaticFiles(directory=_static, html=True), name="static")
