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
from cflib.positioning.position_hl_commander import PositionHlCommander


logging.basicConfig(level=logging.DEBUG)

_COMMANDERS = ("motion", "high_level", "position_high_level")


class CrazyflieRpcConnector(contextlib.AbstractContextManager):
    _crazyflies: dict[str, cflib.crazyflie.syncCrazyflie.SyncCrazyflie]
    _log_data: dict[str, dict[str, Any]]
    _commander: dict[str, MotionCommander | HighLevelCommander | PositionHlCommander]

    def __init__(self):
        self._crazyflies = {}
        self._log_data = {}
        self._commander = {}

    def close(self):
        print("Cleaning up crazyflie connector")
        for cf in self._crazyflies.values():
            cf.cf.loc.send_emergency_stop()
            cf.close_link()
        self._log_data.clear()
        self._commander.clear()
        self._crazyflies.clear()

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

    def open_link(self, url, reinit=False, commander="position_high_level"):
        if not isinstance(url, str) or len(url) == 0:
            raise ValueError(f"invalid url: {url}")
        if not isinstance(reinit, bool):
            raise ValueError(f"invalid parameter reinit: {reinit}")
        if not isinstance(commander, str) or commander not in _COMMANDERS:
            raise ValueError(
                f"invalid parameter commander: {commander}, expected one of {_COMMANDERS}"
            )
        if url in self._crazyflies:
            if reinit:
                try:
                    self.close_link(url)
                except Exception as e:
                    # might happen when the existing drone has turned off?
                    print(
                        f"ignoring exception from close_link while re-opening the link: {e}"
                    )
            else:
                raise ValueError(1, f"url {url} already in use")

        # TODO: what about the cache?
        scf = cflib.crazyflie.syncCrazyflie.SyncCrazyflie(url)
        scf.open_link()
        scf.wait_for_params()
        self._crazyflies[url] = scf

        if commander == "motion":
            self._commander[url] = MotionCommander(scf.cf)
        elif commander == "high_level":
            self._commander[url] = scf.cf.high_level_commander
        elif commander == "position_high_level":
            self._commander[url] = PositionHlCommander(scf.cf)
        else:
            raise AssertionError("unknown commander")

    def close_link(self, url):
        if url not in self._crazyflies:
            raise ValueError(f"unknown url: {url}")

        scf = self._crazyflies[url]
        scf.close_link()
        del self._crazyflies[url]
        del self._commander[url]
        self._log_data.pop(url, None)

    def get_all_values(self, url):
        if url not in self._crazyflies:
            raise ValueError(f"unknown url: {url}")

        scf = self._crazyflies[url]
        scf.cf.param.request_update_of_all_params()
        return scf.cf.param.values

    def get_value(self, url, name):
        if url not in self._crazyflies:
            raise ValueError(f"unknown url: {url}")
        if not isinstance(name, str):
            raise ValueError(f"invalid parameter name: {name}")

        scf = self._crazyflies[url]
        return scf.cf.param.get_value(name)

    def get_values(self, url, names):
        if url not in self._crazyflies:
            raise ValueError(f"unknown url: {url}")

        if isinstance(names, str):
            names = [names]

        scf = self._crazyflies[url]
        data = {}
        for name in names:
            if not isinstance(name, str):
                raise ValueError(f"invalid parameter name: {name}")
            data[name] = scf.cf.param.get_value(name)

        return data

    def set_value(self, url, name, value):
        if url not in self._crazyflies:
            raise ValueError(f"unknown url: {url}")
        if not isinstance(name, str):
            raise ValueError(f"invalid parameter name: {name}")

        scf = self._crazyflies[url]
        scf.cf.param.set_value(name, value)

    def set_values(self, url, values):
        if url not in self._crazyflies:
            raise ValueError(f"unknown url: {url}")
        if not isinstance(values, dict):
            raise ValueError(f"invalid parameter values: {values}")

        scf = self._crazyflies[url]
        for name, value in values.items():
            if not isinstance(name, str):
                raise ValueError(f"invalid parameter name: {name}")
            scf.cf.param.set_value(name, value)

    def register_log(self, url, name, variables, period=10):
        if url not in self._crazyflies:
            raise ValueError(f"unknown url: {url}")
        if not isinstance(name, str):
            raise ValueError(f"invalid name: {name}")

        if isinstance(variables, str):
            variables = [variables]

        log_config = cflib.crazyflie.log.LogConfig(name, period_in_ms=period)
        for variable in variables:
            if not isinstance(variable, str):
                raise ValueError(f"invalid variable name: {variable}")
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
        if url not in self._crazyflies or url not in self._log_data:
            raise ValueError(f"unknown url: {url}")
        if not isinstance(name, str) or name not in self._log_data[url]:
            raise ValueError(f"invalid variable name: {name}")

        return self._log_data[url][name]

    def get_log_vars(self, url, names):
        if url not in self._crazyflies or url not in self._log_data:
            raise ValueError(f"unknown url: {url}")

        if isinstance(names, str):
            names = [names]

        log_data = self._log_data[url]
        data = {}
        for name in names:
            if name not in log_data:
                raise ValueError(f"invalid name: {name}")
            data[name] = log_data[name]

        return data

    def get_position_high_level_commander_pos(self, url):
        if url not in self._crazyflies or url not in self._log_data:
            raise ValueError(f"unknown url: {url}")

        mc = self._commander[url]
        if not isinstance(mc, PositionHlCommander):
            raise ValueError(
                f"crazyflie was configured with a different commander than 'position_high_level': {type(mc).__name__}"
            )
        x, y, z = mc.get_position()
        return {"x": x, "y": y, "z": z}

    def takeoff(self, url, height=1.0):
        if url not in self._crazyflies:
            raise ValueError(f"unknown url: {url}")
        if not isinstance(height, (int, float)):
            raise ValueError(f"invalid target height {height}")

        mc = self._commander[url]
        if isinstance(mc, MotionCommander):
            # height is the relative height
            mc.take_off(height=height, velocity=0.5)
        elif isinstance(mc, HighLevelCommander):
            # height is the absolute target height
            mc.takeoff(absolute_height_m=height, duration_s=2.0)
        elif isinstance(mc, PositionHlCommander):
            # height is the absolute target height
            mc.take_off(height=height, velocity=0.5)
        else:
            raise AssertionError("unknown commander")

    def land(self, url, height=0.0):
        if url not in self._crazyflies:
            raise ValueError(f"unknown url: {url}")
        if not isinstance(height, (int, float)):
            raise ValueError(f"invalid target height {height}")

        mc = self._commander[url]
        if isinstance(mc, MotionCommander):
            # no height parameter
            mc.land(velocity=0.5)
        elif isinstance(mc, HighLevelCommander):
            # height is the absolute target height
            mc.land(absolute_height_m=height, duration_s=2.0)
        elif isinstance(mc, PositionHlCommander):
            # height is the absolute target height
            mc.land(landing_height=height, velocity=0.5)
        else:
            raise AssertionError("unknown commander")

    def left(self, url, distance):
        if url not in self._crazyflies:
            raise ValueError(f"unknown url: {url}")
        if not isinstance(distance, (int, float)) or distance <= 0:
            raise ValueError(f"invalid distance {distance}")

        mc = self._commander[url]
        if isinstance(mc, MotionCommander):
            mc.left(distance, velocity=0.5)
        elif isinstance(mc, HighLevelCommander):
            mc.go_to(
                x=0,
                y=distance,
                z=0,
                yaw=0,
                duration_s=distance / 0.5,
                relative=True,
                linear=True,
            )
        elif isinstance(mc, PositionHlCommander):
            mc.left(distance, velocity=0.5)
        else:
            raise AssertionError("unknown commander")

    def right(self, url, distance):
        if url not in self._crazyflies:
            raise ValueError(f"unknown url: {url}")
        if not isinstance(distance, (int, float)) or distance <= 0:
            raise ValueError(f"invalid distance {distance}")

        mc = self._commander[url]
        if isinstance(mc, MotionCommander):
            mc.right(distance, velocity=0.5)
        elif isinstance(mc, HighLevelCommander):
            mc.go_to(
                x=0,
                y=-distance,
                z=0,
                yaw=0,
                duration_s=distance / 0.5,
                relative=True,
                linear=True,
            )
        elif isinstance(mc, PositionHlCommander):
            mc.right(distance, velocity=0.5)
        else:
            raise AssertionError("unknown commander")

    def up(self, url, distance):
        if url not in self._crazyflies:
            raise ValueError(f"unknown url: {url}")
        if not isinstance(distance, (int, float)) or distance <= 0:
            raise ValueError(f"invalid distance {distance}")

        mc = self._commander[url]
        if isinstance(mc, MotionCommander):
            mc.up(distance, velocity=0.5)
        elif isinstance(mc, HighLevelCommander):
            mc.go_to(
                x=0,
                y=0,
                z=distance,
                yaw=0,
                duration_s=distance / 0.5,
                relative=True,
                linear=True,
            )
        elif isinstance(mc, PositionHlCommander):
            mc.up(distance, velocity=0.5)
        else:
            raise AssertionError("unknown commander")

    def down(self, url, distance):
        if url not in self._crazyflies:
            raise ValueError(f"unknown url: {url}")
        if not isinstance(distance, (int, float)) or distance <= 0:
            raise ValueError(f"invalid distance {distance}")

        mc = self._commander[url]
        if isinstance(mc, MotionCommander):
            mc.down(distance, velocity=0.5)
        elif isinstance(mc, HighLevelCommander):
            mc.go_to(
                x=0,
                y=0,
                z=-distance,
                yaw=0,
                duration_s=distance / 0.5,
                relative=True,
                linear=True,
            )
        elif isinstance(mc, PositionHlCommander):
            mc.down(distance, velocity=0.5)
        else:
            raise AssertionError("unknown commander")

    def forward(self, url, distance):
        if url not in self._crazyflies:
            raise ValueError(f"unknown url: {url}")
        if not isinstance(distance, (int, float)) or distance <= 0:
            raise ValueError(f"invalid distance {distance}")

        mc = self._commander[url]
        if isinstance(mc, MotionCommander):
            mc.forward(distance, velocity=0.5)
        elif isinstance(mc, HighLevelCommander):
            mc.go_to(
                x=distance,
                y=0,
                z=0,
                yaw=0,
                duration_s=distance / 0.5,
                relative=True,
                linear=True,
            )
        elif isinstance(mc, PositionHlCommander):
            mc.forward(distance, velocity=0.5)
        else:
            raise AssertionError("unknown commander")

    def backward(self, url, distance):
        if url not in self._crazyflies:
            raise ValueError(f"unknown url: {url}")
        if not isinstance(distance, (int, float)) or distance <= 0:
            raise ValueError(f"invalid distance {distance}")

        mc = self._commander[url]
        if isinstance(mc, MotionCommander):
            mc.back(distance, velocity=0.5)
        elif isinstance(mc, HighLevelCommander):
            mc.go_to(
                x=-distance,
                y=0,
                z=0,
                yaw=0,
                duration_s=distance / 0.5,
                relative=True,
                linear=True,
            )
        elif isinstance(mc, PositionHlCommander):
            mc.back(distance, velocity=0.5)
        else:
            raise AssertionError("unknown commander")


def main():
    cflib.crtp.init_drivers()
    with CrazyflieRpcConnector() as rpc:
        with JsonRpcServer("tcp://localhost:22272", rpc._rpc_handler) as server:
            server.run()


if __name__ == "__main__":
    main()
