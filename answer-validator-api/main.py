"""
answer-validator-api
Port: 8006

Single source of truth for what constitutes a valid, grounded answer.
Exposes POST /validate_answer per the Project LEDGER service contracts.

Run:
    uvicorn main:app --host 0.0.0.0 --port 8006 --reload
"""

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from schemas import AnswerRequest, ValidatedResponse, RejectedResponse
from validators import validate_answer
from logger_config import log_success, log_error

app = FastAPI(title="answer-validator-api", version="1.0.0")


@app.get("/health")
def health():
    return {"status": "ok", "service": "answer-validator-api", "port": 8006}


@app.post("/validate_answer", response_model=None)
def validate_answer_endpoint(payload: AnswerRequest):
    answer_dict = payload.model_dump()

    ok, reason, generic = validate_answer(answer_dict)

    if ok:
        log_success(answer_dict["answer_type"], answer_dict.get("evidence", []))
        return JSONResponse(
            status_code=200,
            content=ValidatedResponse(answer=answer_dict).model_dump(),
        )

    log_error(reason, answer_type=answer_dict.get("answer_type"), generic=generic)
    return JSONResponse(
        status_code=422,
        content=RejectedResponse(reason=reason).model_dump(),
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8006, reload=True)
