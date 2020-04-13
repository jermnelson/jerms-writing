__author__ = "Jeremy Nelson"

import datetime
import sys
from starlette.applications import Starlette
from starlette.responses import JSONResponse, PlainTextResponse 
from starlette.routing import Route
import uvicorn

app = Starlette(debug=True)

@app.route('/')
async def homepage(request):
    return JSONResponse({"about": { "author": __author__,
                                    "name": "jerms-writing",
                                    "description": "A machine-learning experiment on handwriting classification"},
                         "timestamp": datetime.datetime.utcnow().isoformat(),
                         "python": f"{sys.version_info.major}.{sys.version_info.minor}"})

@app.route('/user')
async def user(request):
    username = request.path_params['username']
    return PlainTextResponse(f"Hello {username}")


if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0', port=9560)
