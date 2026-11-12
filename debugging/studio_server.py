"""Browser access for LangSmith Studio's connection to the local server."""

from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware


# This LangGraph release drops allow_private_network from its JSON CORS config.
# A custom HTTP app applies Starlette's option directly, before the API's CORS.
app = Starlette(
    middleware=[
        Middleware(
            CORSMiddleware,
            allow_origins=["https://smith.langchain.com"],
            allow_methods=["*"],
            allow_headers=["*"],
            allow_private_network=True,
            expose_headers=[
                "x-pagination-total",
                "x-pagination-next",
                "content-location",
            ],
        )
    ]
)
