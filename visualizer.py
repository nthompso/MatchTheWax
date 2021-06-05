#!/usr/bin/env python
from threading import Lock

from flask import Flask, render_template, session, request, \
    copy_current_request_context

from flask_socketio import SocketIO, emit, join_room, leave_room, \
    close_room, rooms, disconnect

import sys
import asyncio
import queue
import sys
import soundfile as sf

import sounddevice as sd

from Milkcrate import TheMilkcrate

# Set this variable to "threading", "eventlet" or "gevent" to test the
# different async modes, or leave it set to None for the application to choose
# the best option based on installed packages.
async_mode = 'eventlet'

app = Flask(__name__)
app.config['SECRET_KEY'] = 'normdog802'
socketio = SocketIO(app, async_mode=async_mode)
thread = None
thread_lock = Lock()



def background_thread():
    tmc  = TheMilkcrate()
    CHANNELS = 1
    RATE = 44100
    DURATION = 6
    CHUNK = RATE * DURATION
    song_name = 'None'

    def callback(indata, frame_count, time_info, status):
        nonlocal song_name
        (max_delta, delta, shift_max, flag) = tmc.match_the_wax(indata)
        song_name = max_delta

    stream = sd.InputStream(callback=callback, channels=CHANNELS, samplerate=RATE, blocksize=CHUNK)

    with stream:
        while True:
            socketio.emit('my_response', {'data': song_name, 'count': 1})
            socketio.sleep(3)
            

@app.route('/')
def index():
    return render_template('index.html', async_mode=socketio.async_mode)


@socketio.event
def connect():
    global thread
    with thread_lock:
        if thread is None:
            thread = socketio.start_background_task(background_thread)
    emit('my_response', {'data': 'Connected', 'count': 0})


if __name__ == '__main__':
    socketio.run(app)