import zmq
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


@dataclass(frozen=True)
class JsonRpcRequest:
    id: Optional[int | float | str]
    notification: bool
    name: str
    params: list[Any] | dict[str, Any]

    def make_success_response(self, result: Optional[Any] = None) -> dict:
        return {"jsonrpc": "2.0", "id": self.id, "result": result}

    def make_error_response(
        self, error_code: int, error_message: str, error_data: Optional[Any] = None
    ) -> dict:
        return make_error_response(self.id, error_code, error_message, error_data)


class JsonRpcServer:

    socket: zmq.Socket
    handler: Callable[[JsonRpcRequest], dict]

    def __init__(self, addr: str, handler: Callable[[JsonRpcRequest], dict]):
        self.handler = handler
        context = zmq.Context()
        self.socket = context.socket(zmq.REP)
        self.socket.bind(addr)

    def __recv_json(self):
        return self.socket.recv_json()

    def __send_json(self, json):
        self.socket.send_json(json)

    def run(self):
        while True:
            try:
                data = self.__recv_json()
            except Exception as e:
                print(f"<error: {e}>")
                self.__send_json(make_error_response(None, -32700, f"Error: {e}"))
                continue

            print(data)
            response = self.__handle_request(data)
            if response:
                self.__send_json(response)

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
            response = req_obj.make_error_response(-32603, f"Error: {e}")

        if notification:
            return None
        else:
            return response


def main():
    server = JsonRpcServer(
        "ipc:///tmp/crazyflie", lambda req: req.make_success_response(42)
    )
    server.run()


if __name__ == "__main__":
    main()
