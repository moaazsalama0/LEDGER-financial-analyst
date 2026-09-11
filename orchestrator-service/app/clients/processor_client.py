import httpx

from app.clients.base import post_file_with_retry
from app.config import settings


async def call_doc_processor(
    client: httpx.AsyncClient,
    filename: str,
    file_bytes: bytes,
    content_type: str | None,
    trace_id: str,
) -> dict:
    """
    Calls doc-processor-api's EXISTING POST /process with the uploaded PDF
    as multipart/form-data, using "file" as the upload field name (the
    standard FastAPI UploadFile convention used elsewhere in this project).

    NOTE: the Contracts PDF documents doc-processor-api's *output* schema
    only — it does not document the exact request/field name /process
    expects. If doc-processor-api's actual handler names its UploadFile
    parameter something other than "file", update FILE_FIELD_NAME below;
    no other orchestrator code needs to change.
    """
    FILE_FIELD_NAME = "file"

    url = f"{settings.PROCESSOR_SERVICE_URL}{settings.PROCESSOR_PROCESS_ENDPOINT}"
    files = {FILE_FIELD_NAME: (filename, file_bytes, content_type or "application/pdf")}

    return await post_file_with_retry(client, url, files, service_name="doc-processor-api", trace_id=trace_id)
