from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.responses import HTMLResponse, RedirectResponse
from uvicorn import run as app_run


from heart_stroke.constant.application import APP_HOST, APP_PORT
from heart_stroke.pipeline.prediction_pipeline import (HeartData,
                                                       HeartStrokeClassifier)
from heart_stroke.pipeline.train_pipeline import TrainPipeline

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory='templates')

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class DataForm:
    def __init__(self, request: Request):
        self.request: Request = request
        self.gender: Optional[str] = None
        self.age: Optional[str] = None
        self.hypertension: Optional[str] = None
        self.heart_disease: Optional[str] = None
        self.ever_married: Optional[str] = None
        self.work_type: Optional[str] = None
        self.Residence_type: Optional[str] = None
        self.avg_glucose_level: Optional[str] = None
        self.smoking_status: Optional[str] = None
        self.bmi: Optional[str] = None
        

    async def get_stroke_data(self):
        form =  await self.request.form()
        self.gender = form.get("gender")
        self.age = form.get("age")
        self.hypertension = form.get("hypertension")
        self.heart_disease = form.get("heart_disease")
        self.ever_married = form.get("ever_married")
        self.work_type = form.get("work_type")
        self.Residence_type = form.get("Residence_type")
        self.avg_glucose_level = form.get("avg_glucose_level")
        self.smoking_status = form.get("smoking_status")
        self.bmi = form.get("bmi")

@app.get("/", tags=["authentication"])
async def index(request: Request):

    return templates.TemplateResponse(
            "index.html",{"request": request, "context": "Rendering"})


@app.get("/train")
async def trainRouteClient():
    try:
        train_pipeline = TrainPipeline()

        train_pipeline.run_pipeline()

        return Response("Training successful !!")

    except Exception as e:
        return Response(f"Error Occurred! {e}")


@app.post("/")
async def predictRouteClient(request: Request):
    try:
        form = DataForm(request)
        await form.get_stroke_data()
        
        heart_stroke_data = HeartData(gender= form.gender, 
                                   age= form.age, 
                                   hypertension= form.hypertension,
                                   heart_disease= form.heart_disease, 
                                   ever_married= form.ever_married, 
                                   work_type= form.work_type, 
                                   Residence_type= form.Residence_type, 
                                   smoking_status= form.smoking_status,
                                   avg_glucose_level= form.avg_glucose_level,
                                   bmi = form.bmi
                                   )
        
        stroke_data_df = heart_stroke_data.get_heart_stroke_input_data_frame()
        
        model_predictor = HeartStrokeClassifier()

        stroke_value = model_predictor.predict(dataframe=stroke_data_df)

        return templates.TemplateResponse(
            "index.html",
            {"request": request, "context": stroke_value},
        )
        
    except Exception as e:
        return {"status": False, "error": f"{e}"}


if __name__ == "__main__":
    app_run(app, host=APP_HOST, port=APP_PORT)
