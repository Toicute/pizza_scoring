import sys 
sys.path.append('core/yolov7')

from fastapi import FastAPI
import uvicorn

from api.endpoint import scoring_endpoint
from api.dependencies import lifespan


app = FastAPI(lifespan=lifespan)

app.include_router(scoring_endpoint.router, prefix="/scoring", tags=["scoring"])

@app.get("/")
def read_root():
    return {"message": "Welcome to Pizza scoring API!"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)