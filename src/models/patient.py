"""
Patient Health Profile schema (Day 1: manual entry; OCR extraction is a
later, optional layer that would populate the same fields).
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class PatientLabData(BaseModel):
    age: int = Field(..., ge=0, le=120)
    weight_kg: Optional[float] = Field(None, gt=0)
    height_cm: Optional[float] = Field(None, gt=0)

    hba1c_percent: Optional[float] = Field(None, ge=0, le=20, description="HbA1c, %")
    fasting_glucose_mgdl: Optional[float] = Field(None, ge=0)
    bmi: Optional[float] = Field(None, ge=0)
    total_cholesterol_mgdl: Optional[float] = Field(None, ge=0)
    ldl_mgdl: Optional[float] = Field(None, ge=0)
    hdl_mgdl: Optional[float] = Field(None, ge=0)
    systolic_bp_mmhg: Optional[int] = Field(None, ge=0)
    diastolic_bp_mmhg: Optional[int] = Field(None, ge=0)
    egfr: Optional[float] = Field(None, ge=0, description="Kidney function, mL/min/1.73m2")

    current_medications: list[str] = Field(default_factory=list)
    known_conditions: list[str] = Field(default_factory=list)

    class Config:
        json_schema_extra = {
            "example": {
                "age": 54,
                "weight_kg": 82.5,
                "height_cm": 172,
                "hba1c_percent": 8.1,
                "fasting_glucose_mgdl": 152,
                "bmi": 27.9,
                "total_cholesterol_mgdl": 210,
                "ldl_mgdl": 130,
                "hdl_mgdl": 42,
                "systolic_bp_mmhg": 138,
                "diastolic_bp_mmhg": 88,
                "egfr": 78,
                "current_medications": ["Metformin 1000mg"],
                "known_conditions": ["Type 2 Diabetes"],
            }
        }
