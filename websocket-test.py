#!/usr/bin/env python3
# This works with ws.mjs running server first

import websockets
import json
from websockets.sync.client import connect

with connect("ws://localhost:5678") as websocket:
	websocket.send(json.dumps([1, 0, -1, 0, 1, 0, -1, -1, 1]))
