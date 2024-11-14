from zmq_rpc import JsonRpcServer, JsonRpcRequest
from typing import Any
import contextlib
import logging
import time
import cflib
import cflib.crazyflie
import cflib.crazyflie.syncCrazyflie
import cflib.crazyflie.syncLogger
import cflib.crazyflie.log


logging.basicConfig(level=logging.DEBUG)


class CrazyflieRpcConnector(contextlib.AbstractContextManager):
    _crazyflies: dict[str, cflib.crazyflie.syncCrazyflie.SyncCrazyflie]
    _log_data: dict[str, dict[str, Any]]

    def __init__(self):
        self._crazyflies = {}
        self._log_data = {}

    def close(self):
        for cf in self._crazyflies.values():
            cf.close_link()
        self._crazyflies.clear()
        self._log_data.clear()

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False

    def _rpc_handler(self, req: JsonRpcRequest) -> dict:
        if req.method not in {"close"} and not req.method.startswith("_"):
            attr = getattr(self, req.method, None)
            if attr:
                if isinstance(req.params, dict):
                    value = attr(**req.params)
                else:
                    value = attr(*req.params)
                return req.make_success_response(value)

        return req.make_error_response(-32601, f"Method not found: {req.method}")

    def open_link(self, url):
        if not isinstance(url, str) or len(url) == 0:
            raise ValueError(f"invalid url: {url}")
        if url in self._crazyflies:
            raise ValueError(1, f"url {url} already in use")

        scf = cflib.crazyflie.syncCrazyflie.SyncCrazyflie(
            url
        )  # TODO: what about the cache?
        scf.open_link()
        scf.wait_for_params()
        self._crazyflies[url] = scf

    def close_link(self, url):
        if url not in self._crazyflies:
            raise ValueError(f"unknown url: {url}")

        scf = self._crazyflies[url]
        scf.cf.loc.send_emergency_stop()
        scf.close_link()
        del self._crazyflies[url]
        self._log_data.pop(url, None)

    def get_values(self, url):
        if url not in self._crazyflies:
            raise ValueError(f"unknown url: {url}")

        scf = self._crazyflies[url]
        scf.cf.param.request_update_of_all_params()
        return scf.cf.param.values

    def get_value(self, url, name):
        if url not in self._crazyflies:
            raise ValueError(f"unknown url: {url}")
        if not isinstance(name, str):
            raise ValueError(f"invalid name: {name}")

        scf = self._crazyflies[url]
        return scf.cf.param.get_value(name)

    def register_log(self, url, name, variables, period=10):
        if url not in self._crazyflies:
            raise ValueError(f"unknown url: {url}")
        if not isinstance(name, str):
            raise ValueError(f"invalid name: {name}")

        log_config = cflib.crazyflie.log.LogConfig(name, period_in_ms=period)
        for variable in variables:
            log_config.add_variable(variable)
        self._crazyflies[url].cf.log.add_config(log_config)
        log_config.data_received_cb.add_callback(
            lambda *args: self._receive_data(url, *args)
        )
        log_config.start()

    def _receive_data(self, url, timestamp, data, logconf):
        existing_data = self._log_data.setdefault(url, {})
        existing_data.update(data)

    def get_log_var(self, url, name):
        if url not in self._crazyflies:
            raise ValueError(f"unknown url: {url}")
        if url not in self._log_data or name not in self._log_data[url]:
            raise ValueError(f"invalid name: {name}")

        return self._log_data[url][name]

    def takeoff(self, url, height, duration):
        if url not in self._crazyflies:
            raise ValueError(f"unknown url: {url}")

        scf = self._crazyflies[url]
        scf.cf.high_level_commander.takeoff(height, duration)
        time.sleep(duration)

    def land(self, url, height, duration):
        if url not in self._crazyflies:
            raise ValueError(f"unknown url: {url}")

        scf = self._crazyflies[url]
        scf.cf.high_level_commander.land(height, duration)
        time.sleep(duration)


def main():
    cflib.crtp.init_drivers()
    with CrazyflieRpcConnector() as rpc:
        with JsonRpcServer("ipc:///tmp/crazyflie", rpc._rpc_handler) as server:
            server.run()


if __name__ == "__main__":
    main()
