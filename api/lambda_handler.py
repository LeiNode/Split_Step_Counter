"""AWS Lambda entry point: adapts API Gateway / Function URL events to the FastAPI app."""

from mangum import Mangum

from api.main import app

handler = Mangum(app, lifespan="off")
