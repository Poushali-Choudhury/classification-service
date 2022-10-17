from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel

from algorithm import train_process, predict

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


class Item(BaseModel):
    text: str
    user_id: int


@app.post("/predict/")
def prediction(item: Item):
    result = predict(item.user_id, item.text)
    return {"result": result}


@app.get("/train/{user_id}")
async def train(user_id: int):
    train_process(user_id)
    return {"result": "success"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Optional[str] = None):
    return {"item_id": item_id, "q": q}
