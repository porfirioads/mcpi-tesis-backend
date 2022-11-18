from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from app.config import logger
from app.routers.dataset_router import router as dataset_router
from app.scripts import download_corpora

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

app.include_router(dataset_router)


@app.on_event('startup')
def on_startup():
    logger.info('on_startup')
    download_corpora.start()


@app.on_event('shutdown')
def on_shutdown():
    logger.info('on_shutdown')


@app.exception_handler(Exception)
async def catch_exception(request: Request, exc: Exception):
    logger.debug(exc)
    return JSONResponse(status_code=400, content={'detail': 'UNKNOWN_ERROR'})


@app.exception_handler(FileNotFoundError)
async def catch_file_not_found_error(request: Request, exc: FileNotFoundError):
    logger.debug(exc)
    return JSONResponse(status_code=400, content={'detail': 'FILE_NOT_FOUND'})
