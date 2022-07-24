from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import logger
from app.routers.exploration_router import router as exploration_router
from app.routers.dataset_router import router as dataset_router

app = FastAPI(
    title='Tesis MCPI',
    description='Proyecto de desarrollo para la tesis de la Maestría en Ciencias del Procesamiento de la Información.',
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


@app.on_event('startup')
def on_startup():
    logger.info('App startup init.')


@app.on_event('shutdown')
def on_shutdown():
    logger.info('App shutdown init.')
