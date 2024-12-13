import zmq
import contextlib
import json
import traceback
from dataclasses import dataclass
from typing import Any, Callable, Optional, Union, List, Dict


def make_error_response(
    id: Union[int, float, str, None],
    error_code: int,
    error_message: str,
    error_data: Optional[Any] = None,
) -> Dict[str, Any]:
    e = {"code": error_code, "message": error_message}
    if error_data is not None:
        e["data"] = error_data
    return {"jsonrpc": "2.0", "id": id, "error": e}


def make_invalid_request_error(
    id: Union[int, float, str, None] = None
) -> Dict[str, Any]:
    return make_error_response(id, -32600, "Invalid request")


THROW_MARKER = object()


@dataclass(frozen=True)
class JsonRpcRequest:
    id: Union[int, float, str, None]
    notification: bool
    method: str
    params: Union[List[Any], Dict[str, Any]]

    def to_json(self) -> Dict[str, Any]:
        json = {"jsonrpc": "2.0", "method": self.method}
        if self.params:
            json["params"] = self.params
        if not self.notification:
            json["id"] = self.id
        return json

    def make_success_response(self, result: Any = None) -> Dict[str, Any]:
        return {"jsonrpc": "2.0", "id": self.id, "result": result}

    def make_error_response(
        self, error_code: int, error_message: str, error_data: Any = None
    ) -> Dict[str, Any]:
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
    handler: Callable[[JsonRpcRequest], Dict[str, Any]]

    def __init__(self, addr: str, handler: Callable[[JsonRpcRequest], Dict[str, Any]]):
        self.handler = handler
        context = zmq.Context()
        self.socket = context.socket(zmq.REP)
        print(f"Binding rpc server to {addr}")
        self.socket.bind(addr)

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False

    def close(self):
        self.socket.close()

    def run(self):
        print("Running rpc server")
        try:
            while True:
                try:
                    data = self.socket.recv_json()
                except Exception as e:
                    print(f"<error: {e}>")
                    self.socket.send_json(
                        make_error_response(None, -32700, f"Error: {e}")
                    )
                    continue

                print(f"recv: {data}")
                response = self.__handle_request(data)
                if response:
                    print(f"send: {response}")
                    self.socket.send_json(response)
        except KeyboardInterrupt:
            import sys

            print("Keyboard interrupt, exiting...")
            sys.exit(1)

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
        print(f"Connecting rpc client to {addr}")
        self.socket.connect(addr)
        self.counter = 1

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False

    def close(self):
        self.socket.close()

    def rpc_notify(self, method: str, params: Union[List[Any], Dict[str, Any]]):
        # req = JsonRpcRequest(None, True, method, params)
        # self.socket.send_json(req.to_json())
        raise NotImplementedError

    def rpc_call(self, method: str, params: Union[List[Any], Dict[str, Any]]) -> Any:
        print(f"rpc_call: {method} {params}")

        req = JsonRpcRequest(self.counter, False, method, params)
        self.counter += 1
        req_json = req.to_json()
        print(f"send: {req_json}")
        self.socket.send_json(req_json)

        res = self.socket.recv_json()
        print(f"recv: {res}")
        return self.__handle_single_response(req, res)

    def __handle_single_response(self, req: JsonRpcRequest, res: Dict[str, Any]) -> Any:
        protocol = res.get("jsonrpc")
        id = res.get("id")

        if protocol != "2.0" or id != req.id:
            raise ValueError("invalid response")

        if "error" in res:
            raise ValueError(f"error response: {res['error']}")

        return res["result"]


def main():
    """Usage:
    python zmq_rpc.py ['server'|'client'] [endpoint] [method] [params]

    server/client decides whether to start a rpc server or make a rpc call - optional, default is server
    endpoint is the zmq endpoint to connect/bind to (must contain ://) - optional, default is tcp://localhost:22272
    method is only used by the client - optional, default is test
    params is only used by the client - optional, default is {"foo": "bar"}
    """
    import sys

    args = list(sys.argv)
    args.pop(0)

    if args and args[0] in ("client", "server"):
        side = args.pop(0)
    else:
        side = "server"

    if args and "://" in args[0]:
        endpoint = args.pop(0)
    else:
        endpoint = "tcp://localhost:22272"

    if side == "client":
        method = args.pop(0) if args else "test"
        params = json.loads(args.pop(0)) if args else {"foo": "bar"}
        if args:
            print(f"ignoring args '{' '.join(args)}'")
        with JsonRpcClient(endpoint) as client:
            result = client.rpc_call(method, params)
            print(f"result: {result}")
    else:
        if args:
            print(f"ignoring args '{' '.join(args)}'")
        with JsonRpcServer(
            endpoint,
            lambda req: req.make_success_response(
                [None, True, 42, 1.337, "foo", {"a": "b", "21": 21}]
            ),
        ) as server:
            server.run()


if __name__ == "__main__":
    main()
