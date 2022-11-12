from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from app.config import logger
from app.routers.exploration_router import router as exploration_router
from app.routers.dataset_router import router as dataset_router
from app.routers.preprocessor_router import router as preprocessor_router
from app.scripts import download_corpora
import pandas as pd

app = FastAPI(
    title='Tesis MCPI',
    description='Proyecto de desarrollo para la tesis de la Maestría en '
    + 'Ciencias del Procesamiento de la Información.',
    version='1.0.0',
    debug=True
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(exploration_router)
app.include_router(dataset_router)
app.include_router(preprocessor_router)


@app.on_event('startup')
def on_startup():
    logger.info('App startup init.')
    download_corpora.start()


@app.on_event('shutdown')
def on_shutdown():
    logger.info('App shutdown init.')


@app.exception_handler(FileNotFoundError)
async def catch_file_not_found(request: Request, exc: FileNotFoundError):
    return JSONResponse(status_code=400, content={'detail': 'FILE_NOT_FOUND'})


@app.exception_handler(pd.errors.EmptyDataError)
async def catch_pandas_empty_data(request: Request, exc: pd.errors.EmptyDataError):
    return JSONResponse(status_code=400, content={'detail': 'FILE_EMPTY'})
