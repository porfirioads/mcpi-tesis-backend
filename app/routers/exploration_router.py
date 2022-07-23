from fastapi import APIRouter

from app.schemas.exploration_schemas import ExplorationObject


router = APIRouter(prefix='/exploration', tags=['Exploration'])


@router.get('/', response_model=ExplorationObject)
def home():
    return 'Hello exploration'
