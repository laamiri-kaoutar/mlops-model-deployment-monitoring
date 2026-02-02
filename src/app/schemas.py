from pydantic import BaseModel


class PatientData(BaseModel):
    Glucose: float
    Age: int
    BloodPressure: float
    SkinThickness: float
    BMI: float
    Insulin_log: float
    DiabetesPedigreeFunction_log: float

    # Example for Swagger UI documentation
    class Config:
        json_schema_extra = {
            "example": {
                "Glucose": 148.0,
                "Age": 50,
                "BloodPressure": 72.0,
                "SkinThickness": 35.0,
                "BMI": 33.6,
                "Insulin_log": 2.0,
                "DiabetesPedigreeFunction_log": 0.5,
            }
        }
