import zmq
import contextlib
import json
import traceback
from dataclasses import dataclass
from typing import Any, Optional, Callable


def make_error_response(
    id: Optional[int | float | str],
    error_code: int,
    error_message: str,
    error_data: Optional[Any] = None,
) -> dict:
    e = {"code": error_code, "message": error_message}
    if error_data is not None:
        e["data"] = error_data
    return {"jsonrpc": "2.0", "id": id, "error": e}


def make_invalid_request_error(id: Optional[int | float | str] = None) -> dict:
    return make_error_response(id, -32600, "Invalid request")


THROW_MARKER = object()


@dataclass(frozen=True)
class JsonRpcRequest:
    id: Optional[int | float | str]
    notification: bool
    method: str
    params: list[Any] | dict[str, Any]

    def to_json(self) -> dict:
        json = {"jsonrpc": "2.0", "method": self.method}
        if self.params:
            json["params"] = self.params
        if not self.notification:
            json["id"] = self.id
        return json

    def make_success_response(self, result: Any = None) -> dict:
        return {"jsonrpc": "2.0", "id": self.id, "result": result}

    def make_error_response(
        self, error_code: int, error_message: str, error_data: Any = None
    ) -> dict:
        return make_error_response(self.id, error_code, error_message, error_data)

    def get_param(
        self,
        name: Optional[str] = None,
        *,
        index: Optional[int] = None,
        default: Any = THROW_MARKER,
    ) -> Any:
        if isinstance(index, int) and isinstance(self.params, list):
            if default is THROW_MARKER or 0 <= index < len(self.params):
                return self.params[index]
            else:
                return default
        elif isinstance(name, str) and isinstance(self.params, dict):
            if default is THROW_MARKER:
                return self.params[name]
            else:
                return self.params.get(name, default)
        raise ValueError


class JsonRpcServer(contextlib.AbstractContextManager):
    socket: zmq.Socket
    handler: Callable[[JsonRpcRequest], dict]

    def __init__(self, addr: str, handler: Callable[[JsonRpcRequest], dict]):
        self.handler = handler
        context = zmq.Context()
        self.socket = context.socket(zmq.REP)
        self.socket.bind(addr)

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False

    def close(self):
        self.socket.close()

    def run(self):
        print("Running rpc server")
        while True:
            try:
                data = self.socket.recv_json()
            except Exception as e:
                print(f"<error: {e}>")
                self.socket.send_json(make_error_response(None, -32700, f"Error: {e}"))
                continue

            print(data)
            response = self.__handle_request(data)
            if response:
                print(response)
                self.socket.send_json(response)

    def __handle_request(self, request):
        if isinstance(request, list) and len(data) > 0:
            response = []
            for req in request:
                res = self.__handle_single_request(req)
                if res:
                    response.append(res)
        elif isinstance(request, dict):
            response = self.__handle_single_request(request)
        else:
            response = make_invalid_request_error()
        return response

    def __handle_single_request(self, request):
        protocol = request.get("jsonrpc")
        method = request.get("method")
        params = request.get("params", {})

        if (
            protocol != "2.0"
            or not isinstance(method, str)
            or not isinstance(params, (list, dict))
        ):
            return make_invalid_request_error()

        if "id" not in request:
            notification = True
            id = None
        else:
            notification = False
            id = request["id"]
            if id is not None and not isinstance(id, (str, int, float)):
                return make_invalid_request_error()

        req_obj = JsonRpcRequest(id, notification, method, params)
        try:
            response = self.handler(req_obj)
        except BaseException as e:
            traceback.print_exception(e)
            response = req_obj.make_error_response(-32603, f"Error: {e!r}")

        if notification:
            return None
        else:
            return response


class JsonRpcClient(contextlib.AbstractContextManager):
    socket: zmq.Socket
    counter: int

    def __init__(self, addr: str):
        context = zmq.Context()
        self.socket = context.socket(zmq.REQ)
        self.socket.connect(addr)
        self.counter = 1

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False

    def close(self):
        self.socket.close()

    def rpc_notify(self, method: str, params: list[Any] | dict[str, Any]):
        # req = JsonRpcRequest(None, True, method, params)
        # self.socket.send_json(req.to_json())
        raise NotImplementedError

    def rpc_call(self, method: str, params: list[Any] | dict[str, Any]) -> Any:
        print(f"rpc_call: {method} {params}")

        req = JsonRpcRequest(self.counter, False, method, params)
        self.counter += 1
        print(req)
        self.socket.send_json(req.to_json())

        res = self.socket.recv_json()
        print(res)
        return self.__handle_single_response(req, res)

    def __handle_single_response(self, req: JsonRpcRequest, res: dict) -> Any:
        protocol = res.get("jsonrpc")
        id = res.get("id")

        if protocol != "2.0" or id != req.id:
            raise ValueError("invalid response")

        if "error" in res:
            raise ValueError(f"error response: {res['error']}")

        return res["result"]


def main():
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "client":
        with JsonRpcClient("ipc:///tmp/crazyflie") as client:
            if len(sys.argv) >= 3:
                method = sys.argv[2]
            else:
                method = "test"
            if len(sys.argv) >= 4:
                params = json.loads(sys.argv[3])
            else:
                params = {"foo": "bar"}
            result = client.rpc_call(method, params)
            print(f"result: {result}")
    else:
        with JsonRpcServer(
            "ipc:///tmp/crazyflie", lambda req: req.make_success_response(42)
        ) as server:
            server.run()


if __name__ == "__main__":
    main()
