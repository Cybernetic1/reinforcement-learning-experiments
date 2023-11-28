// This works with GUI.html and gym-tictactoe/tic_tac_toe_plain.py
// To start server:
//     node ws.mjs

import WebSocket, { WebSocketServer } from 'ws';

const wss = new WebSocketServer({ port: 5678 });

console.log("Listening Websocket on localhost:5678")
console.log("Broadcasting received messages to all other clients...")

wss.on('connection', function connection(ws) {
  ws.on('error', console.error);

  ws.on('message', function message(data, isBinary) {
    wss.clients.forEach(function each(client) {
      if (client !== ws && client.readyState === WebSocket.OPEN) {
        client.send(data, { binary: isBinary });
      }
    });
  });
});

/*
import { WebSocketServer } from 'ws';

const wss = new WebSocketServer({ port: 5678 });

wss.on('connection', function connection(ws) {
  ws.on('error', console.error);

  ws.on('message', function message(data) {
    console.log('received: %s', data);
    ws.send(data);
  });

  // ws.send(JSON.stringify([1, -1, 1, -1, 1, -1, -1, 1, -1]));
});
*/
