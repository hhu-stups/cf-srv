import zmq
import cflib
import logging
import sys
import time
from threading import Event

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.positioning.motion_commander import MotionCommander
from cflib.utils import uri_helper


URI = uri_helper.uri_from_env(default='radio://0/80/2M/E7E7E7E7E7')

DEFAULT_HEIGHT = 0.5
BOX_LIMIT = 0.5

def param_deck_flow(_, value_str):
    value = int(value_str)
    print(value)
    if value:
        deck_attached_event.set()
        print('Deck is attached!')
    else:
        print('Deck is NOT attached!')

def take_off_simple(scf):
    with MotionCommander(scf, default_height=DEFAULT_HEIGHT) as mc:
        time.sleep(3)
        mc.stop()

def do_a_stunt():
    with SyncCrazyflie(URI, cf=Crazyflie(rw_cache='./cache')) as scf:
        take_off_simple(scf)

DEFAULT_DELTA = 0.2

if __name__ == '__main__':
    cflib.crtp.init_drivers()

    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("ipc:///tmp/crazyflie-bridge")


    with SyncCrazyflie(URI, cf=Crazyflie(rw_cache='./cache')) as scf:
        # take_off_simple(scf)
        mc = MotionCommander(scf)

        while (True):
            message = socket.recv_multipart()
            socket.send(b"1")
            print("Received request: %s" % message)
            print(type(message))
            #print(int(message.decode("utf-8")))
            whole_command = [frame.decode("utf-8") for frame in message]
            x = int(whole_command[0])

            if x == 0:
                take_off_simple(scf)
            elif x == 1:
                mc.take_off()
            elif x == 2:
                mc.land()
            elif x == 10:
                mc.forward(float(whole_command[1]))
            elif x == 11:
                mc.back(float(whole_command[1]))
            elif x == 12:
                mc.left(float(whole_command[1]))
            elif x == 13:
                mc.right(float(whole_command[1]))
            elif x == 20:
                mc.up(float(whole_command[1]))
            elif x == 21:
                mc.down(float(whole_command[1]))
            elif x == 30:
                mc.turn_left(float(whole_command[1]))
            elif x == 31:
                mc.turn_right(float(whole_command[1]))
            elif x == 40:
                scf.cf.param.set_value("sound.effect", int(whole_command[1]))
            elif x == 41:
                scf.cf.param.set_value("sound.effect", 0)



    #cflib.crtp.init_drivers()
    #with SyncCrazyflie(URI, cf=Crazyflie(rw_cache='./cache')) as scf:

    #    take_off_simple(scf)

