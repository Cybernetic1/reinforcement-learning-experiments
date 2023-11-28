import asyncio
import datetime
from typing import Iterator
import websockets
import random

websocket_connections = set()
# global_socket = lambda: None

async def register(websocket):
	print('register event received')
	websocket_connections.add(websocket) # Add this client's socket
	# global_socket = websocket
	async for data in websocket:
		reply = f"Data recieved as: {data}!"
		print("Data recieved =", data)
		await websocket.send(reply)

async def poll_log():
	await asyncio.sleep(0.3) # Settle
	while True:
		await asyncio.sleep(0.3) # Slow things down
		
		# Send a dynamic message to the client after random delay
		r = random.randint(1, 10)
		if (r == 5): # Only send 10% of the time
			a_msg = "srv -> cli: " + str(random.randint(1,10000))
			print("sending msg: " + a_msg)
			websockets.broadcast(websocket_connections, a_msg) # Send to all connected clients

if __name__ == "__main__":
	sock_server = websockets.serve(register, 'localhost', 5678)
	asyncio.get_event_loop().run_until_complete(sock_server)
	print("Websockets server starting up ...")
	asyncio.get_event_loop().run_until_complete(poll_log())

exit(0)

import asyncio

import websockets

# create handler for each connection

async def handler(websocket, path):
	async for data in websocket:
		reply = f"Data recieved as: {data}!"
		print("Data recieved =", data)
		await websocket.send(reply)

start_server = websockets.serve(handler, "localhost", 5678)

print("serving Web Socket ws://localhost:5678 ...")

async def test():
	await asyncio.sleep(1) # Settle
		
	# Send a dynamic message to the client
	a_msg = "srv -> cli: Hello"
	print("sending msg: " + a_msg)
	websockets.broadcast(websocket_connections, a_msg) # Send to all connected clients

asyncio.get_event_loop().run_until_complete(test())

asyncio.get_event_loop().run_until_complete(start_server)

asyncio.get_event_loop().run_forever()
