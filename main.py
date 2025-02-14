from fastapi import FastAPI,status
from fastapi.responses import JSONResponse
import uvicorn
from schema import Query
from utils import reranker
from fastapi.middleware.cors import CORSMiddleware

app=FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins (update to specific domains in production)
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods
    allow_headers=["*"],  # Allows all headers
)

@app.post('/query')
def response(query:Query):
    try:     
        data=reranker(query.query)
        
        return JSONResponse(
            content={"status":status.HTTP_200_OK,"succeed":True,"message":"Request done succesfully","data":data},
            status_code=status.HTTP_200_OK)

    except Exception as e:
        return JSONResponse(
            content={"status":status.HTTP_404_NOT_FOUND,"succeed":False,"message":"Request Failed","data":None},
            status_code=status.HTTP_404_NOT_FOUND)

if __name__=="__main__":
    uvicorn.run("main=app",host="0.0.0.0",port=8787)
