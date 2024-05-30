from contextlib import asynccontextmanager

from fastapi import FastAPI

from core.engine import Yolov7Segmentation


engine = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the engine
    engine["yolov7-seg"] = Yolov7Segmentation(model_weight='checkpoint/yolov7-mask.pt',
                                              model_config='core/yolov7/data/hyp.scratch.mask.yaml')
    yield
    engine.clear()