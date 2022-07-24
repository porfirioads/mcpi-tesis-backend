from fastapi import APIRouter

from app.schemas.exploration_schemas import Exploration


router = APIRouter(prefix='/exploration', tags=['Exploration'])


@router.get('', response_model=Exploration)
async def home():
    return Exploration(
        name='Hello exploration',
        description='This is the hello exploration description.'
    )
