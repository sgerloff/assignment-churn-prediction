from pydantic import BaseModel

from typing import Optional


class Payload(BaseModel):
    USER_ID: Optional[int]
    REGISTRATION_AT: Optional[str]
    TOTAL_VISIT_COUNT: Optional[int]
    LAST_VISIT_AT: Optional[str]
    TOTAL_POST_COUNT: Optional[int]
    LAST_POST_AT: Optional[str]
    TOTAL_LIKES_RECEIVED: Optional[int]
    LAST_LIKE_RECEIVED_AT: Optional[str]
    TOTAL_COMMENTS_RECEIVED: Optional[int]
    LAST_COMMENT_RECEIVED_AT: Optional[str]
    TOTAL_LIKES_GIVEN: Optional[int]
    LAST_LIKE_GIVEN_AT: Optional[str]
    TOTAL_COMMENTS_WRITTEN: Optional[int]
    LAST_COMMENT_WRITTEN_AT: Optional[str]
    CHURNED: bool = False
    