from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal

import joblib
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field


MODEL_PATH = Path(__file__).resolve().parent / "models" / "random_forest_baseline.joblib"
Temperature = Annotated[float, Field(strict=True, allow_inf_nan=False, ge=250, le=400)]
Speed = Annotated[float, Field(strict=True, allow_inf_nan=False, ge=0, le=5000)]
Torque = Annotated[float, Field(strict=True, allow_inf_nan=False, ge=0, le=150)]
Wear = Annotated[float, Field(strict=True, allow_inf_nan=False, ge=0, le=500)]


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = joblib.load(MODEL_PATH)
    yield
    del app.state.model


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
    model = request.app.state.model
    index = list(model.classes_).index(1)
    return Prediction(predicted_class=int(model.predict(row)[0]),
                      failure_score=float(model.predict_proba(row)[0, index]))
