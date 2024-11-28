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
from cflib.positioning.motion_commander import MotionCommander
from cflib.crazyflie.high_level_commander import HighLevelCommander


logging.basicConfig(level=logging.DEBUG)


class CrazyflieRpcConnector(contextlib.AbstractContextManager):
    _crazyflies: dict[str, cflib.crazyflie.syncCrazyflie.SyncCrazyflie]
    _log_data: dict[str, dict[str, Any]]
    _motion_commander: dict[str, MotionCommander | HighLevelCommander]

    def __init__(self):
        self._motion_commander = {}
        self._crazyflies = {}
        self._log_data = {}

    def close(self):
        for cf in self._crazyflies.values():
            cf.cf.loc.send_emergency_stop()
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

    def open_link(self, url, absolute_positioning=False):
        if not isinstance(url, str) or len(url) == 0:
            raise ValueError(f"invalid url: {url}")
        if url in self._crazyflies:
            raise ValueError(1, f"url {url} already in use")
        if not isinstance(absolute_positioning, bool):
            raise ValueError(
                f"invalid parameter absolute_positioning: {absolute_positioning}"
            )

        # TODO: what about the cache?
        scf = cflib.crazyflie.syncCrazyflie.SyncCrazyflie(url)
        scf.open_link()
        scf.wait_for_params()
        self._crazyflies[url] = scf

        if not absolute_positioning:
            self._motion_commander[url] = MotionCommander(scf)
        else:
            self._motion_commander[url] = scf.cf.high_level_commander

    def close_link(self, url):
        if url not in self._crazyflies:
            raise ValueError(f"unknown url: {url}")

        scf = self._crazyflies[url]
        scf.close_link()
        del self._crazyflies[url]
        self._log_data.pop(url, None)
        del self._motion_commander[url]

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

        if isinstance(variables, str):
            variables = [variables]

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

    def takeoff(self, url, height=1.0):
        if url not in self._crazyflies:
            raise ValueError(f"unknown url: {url}")

        mc = self._motion_commander[url]
        if isinstance(mc, MotionCommander):
            mc.take_off(height=height, velocity=0.5)
        else:
            mc.takeoff(absolute_height_m=height, duration_s=height / 0.5)

    def land(self, url, height=0.0):
        if url not in self._crazyflies:
            raise ValueError(f"unknown url: {url}")

        mc = self._motion_commander[url]
        if isinstance(mc, MotionCommander):
            mc.land(velocity=0.5)
        else:
            mc.land(absolute_height_m=height, duration_s=height / 0.5)

    def left(self, url, distance):
        if url not in self._crazyflies:
            raise ValueError(f"unknown url: {url}")

        mc = self._motion_commander[url]
        if isinstance(mc, MotionCommander):
            mc.left(distance, velocity=0.5)
        else:
            mc.go_to(
                x=0,
                y=distance,
                z=0,
                yaw=0,
                duration_s=distance / 0.5,
                relative=True,
                linear=True,
            )

    def right(self, url, distance):
        if url not in self._crazyflies:
            raise ValueError(f"unknown url: {url}")

        mc = self._motion_commander[url]
        if isinstance(mc, MotionCommander):
            mc.right(distance, velocity=0.5)
        else:
            mc.go_to(
                x=0,
                y=-distance,
                z=0,
                yaw=0,
                duration_s=distance / 0.5,
                relative=True,
                linear=True,
            )

    def up(self, url, distance):
        if url not in self._crazyflies:
            raise ValueError(f"unknown url: {url}")

        mc = self._motion_commander[url]
        if isinstance(mc, MotionCommander):
            mc.up(distance, velocity=0.5)
        else:
            mc.go_to(
                x=0,
                y=0,
                z=distance,
                yaw=0,
                duration_s=distance / 0.5,
                relative=True,
                linear=True,
            )

    def down(self, url, distance):
        if url not in self._crazyflies:
            raise ValueError(f"unknown url: {url}")

        mc = self._motion_commander[url]
        if isinstance(mc, MotionCommander):
            mc.down(distance, velocity=0.5)
        else:
            mc.go_to(
                x=0,
                y=0,
                z=-distance,
                yaw=0,
                duration_s=abs(distance) / 0.5,
                relative=True,
                linear=True,
            )

    def forward(self, url, distance):
        if url not in self._crazyflies:
            raise ValueError(f"unknown url: {url}")

        mc = self._motion_commander[url]
        if isinstance(mc, MotionCommander):
            mc.forward(distance, velocity=0.5)
        else:
            mc.go_to(
                x=distance,
                y=0,
                z=0,
                yaw=0,
                duration_s=abs(distance) / 0.5,
                relative=True,
                linear=True,
            )

    def backward(self, url, distance):
        if url not in self._crazyflies:
            raise ValueError(f"unknown url: {url}")

        mc = self._motion_commander[url]
        if isinstance(mc, MotionCommander):
            mc.back(distance, velocity=0.5)
        else:
            mc.go_to(
                x=-distance,
                y=0,
                z=0,
                yaw=0,
                duration_s=abs(distance) / 0.5,
                relative=True,
                linear=True,
            )


def main():
    cflib.crtp.init_drivers()
    with CrazyflieRpcConnector() as rpc:
        with JsonRpcServer("tcp://localhost:22272", rpc._rpc_handler) as server:
            server.run()


if __name__ == "__main__":
    main()
