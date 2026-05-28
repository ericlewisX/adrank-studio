from pydantic import BaseModel


class RankRequest(BaseModel):
    historical_ctr: float
    bid_amount: float
    session_depth: int
    time_of_day: int
    device_type: str