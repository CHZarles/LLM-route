import uvicorn
from transformers import AutoModel, AutoTokenizer

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

pretrained = "./model/chatglm-6b-int4"
tokenizer = AutoTokenizer.from_pretrained(pretrained, trust_remote_code=True)
model = AutoModel.from_pretrained(pretrained, trust_remote_code=True).half().cuda()
model = model.eval()
app = FastAPI()

app.add_middleware(CORSMiddleware)

with open("websocket_demo.html") as f:
    html = f.read()


@app.get("/")
async def get():
    return HTMLResponse(html)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    input: JSON String of {"query": "", "history": []}
    output: JSON String of {"response": "", "history": [], "status": 200}
        status 200 stand for response ended, else not
    """
    # 接受WebSocket连接
    await websocket.accept()
    try:
        while True:
            json_request = await websocket.receive_json()
            print("receive json_request from request is\n", type(json_request))
            query = json_request["query"]
            history = json_request["history"]
            print("receive history from request is\n", history)
            for response, history in model.stream_chat(
                tokenizer, query, history=history
            ):
                print("response is\n", response)
                print("history is\n", history)
                await websocket.send_json(
                    {
                        "response": response,
                        "history": history,
                        # "history": history,
                        "status": 202,
                    }
                )
            await websocket.send_json({"status": 200})
    except WebSocketDisconnect:
        pass


def main():
    uvicorn.run(f"{__name__}:app", host="localhost", port=8000, workers=1)


if __name__ == "__main__":
    main()
