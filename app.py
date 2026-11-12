from contextlib import asynccontextmanager
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import sys
import threading
from typing import Annotated, Literal

import joblib
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field


MODEL_PATH = Path(__file__).resolve().parent / "models" / "random_forest_baseline.joblib"
logger = logging.getLogger("factorypulse")
logging_lock = threading.RLock()
logging_users = 0
logging_handler = None
Temperature = Annotated[float, Field(strict=True, allow_inf_nan=False, ge=250, le=400)]
Speed = Annotated[float, Field(strict=True, allow_inf_nan=False, ge=0, le=5000)]
Torque = Annotated[float, Field(strict=True, allow_inf_nan=False, ge=0, le=150)]
Wear = Annotated[float, Field(strict=True, allow_inf_nan=False, ge=0, le=500)]


@asynccontextmanager
async def lifespan(app: FastAPI):
    global logging_users, logging_handler
    level = os.environ.get("LOG_LEVEL", "INFO").upper()
    if level not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
        raise ValueError("Invalid LOG_LEVEL; use DEBUG, INFO, WARNING, ERROR or CRITICAL")
    with logging_lock:
        if logging_users == 0:
            logging_handler = logging.StreamHandler(sys.stderr)
            logging_handler.setFormatter(JsonFormatter())
            logger.addHandler(logging_handler)
            logger.setLevel(level)
            logger.propagate = False
        elif logger.level != getattr(logging, level):
            raise ValueError("Overlapping lifespans require the same LOG_LEVEL")
        logging_users += 1
    logger.info("application_start")
    try:
        try:
            path = Path(os.environ.get("MODEL_PATH") or MODEL_PATH)
            app.state.model = joblib.load(path)
        except Exception:
            logger.error("model_load_failed")
            raise RuntimeError("Unable to load model") from None
        logger.info("model_loaded")
        try:
            yield
        finally:
            with logging_lock:
                if logging_users == 1:
                    del app.state.model
    finally:
        with logging_lock:
            logging_users -= 1
            if logging_users == 0:
                logger.removeHandler(logging_handler)
                logging_handler.close()
                logging_handler = None


class JsonFormatter(logging.Formatter):
    def format(self, record):
        return json.dumps({"timestamp": datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
                           "level": record.levelname, "event": record.getMessage()})


app = FastAPI(title="FactoryPulse inference", lifespan=lifespan)


@app.exception_handler(RequestValidationError)
async def validation_error(request: Request, exc: RequestValidationError):
    # Exclude the invalid input itself: NaN and infinity are not JSON values.
    errors = [{"loc": error["loc"], "msg": error["msg"], "type": error["type"]}
              for error in exc.errors()]
    return JSONResponse(status_code=422, content={"detail": errors})


class MachineConditions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["L", "M", "H"]
    air_temperature_k: Temperature
    process_temperature_k: Temperature
    rotational_speed_rpm: Speed
    torque_nm: Torque
    tool_wear_min: Wear


class Prediction(BaseModel):
    predicted_class: Literal[0, 1]
    failure_score: float = Field(description="Class 1 model score; not necessarily a calibrated probability.")


@app.get("/health")
def health(request: Request):
    return {"status": "ok", "model_loaded": hasattr(request.app.state, "model")}


@app.post("/predict", response_model=Prediction)
def predict(conditions: MachineConditions, request: Request):
    row = pd.DataFrame([{
        "Type": conditions.type,
        "Air temperature [K]": conditions.air_temperature_k,
        "Process temperature [K]": conditions.process_temperature_k,
        "Rotational speed [rpm]": conditions.rotational_speed_rpm,
        "Torque [Nm]": conditions.torque_nm,
        "Tool wear [min]": conditions.tool_wear_min,
    }])
    try:
        model = request.app.state.model
        index = list(model.classes_).index(1)
        result = Prediction(predicted_class=int(model.predict(row)[0]),
                            failure_score=float(model.predict_proba(row)[0, index]))
    except Exception:
        logger.error("prediction_failed")
        return JSONResponse(status_code=500, content={"detail": "Prediction failed"})
    logger.info("prediction_completed")
    return result
