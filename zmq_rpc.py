import zmq
from json import JSONDecodeError


def error_response(id, error_code, error_message, error_data=None):
    e = {"code": error_code, "message": error_message}
    if error_data is not None:
        e["data"] = error_data
    return {"jsonrpc": "2.0", "id": id, "error": e}


def invalid_request_error():
    return error_response(None, -32600, "Invalid request")


def success_response(id, result):
    return {"jsonrpc": "2.0", "id": id, "result": result}


def do_call(id, method, params):
    # TODO: calls
    return success_response(id, 42)


def handle_single_request(request):
    protocol = request.get("jsonrpc")
    method = request.get("method")
    params = request.get("params", {})

    if (
        protocol != "2.0"
        or not isinstance(method, str)
        or not isinstance(params, (list, dict))
    ):
        return invalid_request_error()

    if "id" not in request:
        notification = True
        id = None
    else:
        notification = False
        id = request["id"]
        if id is not None and not isinstance(id, (str, int, float)):
            return invalid_request_error()

    try:
        response = do_call(id, method, params)
    except:
        response = error_response(id, -32603, "Internal error")

    if notification:
        return None
    else:
        return response


def handle_request(request):
    if isinstance(request, list) and len(data) > 0:
        response = []
        for req in request:
            res = handle_single_request(req)
            if res:
                response.append(res)
    elif isinstance(request, dict):
        response = handle_single_request(request)
    else:
        response = invalid_request_error()
    return response


def main():
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("ipc:///tmp/crazyflie")

    while True:
        try:
            data = socket.recv_json()
        except JSONDecodeError:
            print("<json parse error>")
            socket.send_json(error_response(None, -32700, "Parse error"))
            continue

        print(data)
        response = handle_request(data)
        if response:
            socket.send_json(response)


if __name__ == "__main__":
    main()
