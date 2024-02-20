#!/bin/env python
from app import create_app, socketio
from app.utilities.scheduler import Scheduler
import configparser


config = configparser.ConfigParser()
config.read('config.ini')

scheduler = Scheduler()
scheduler.run()

app = create_app(debug=True)

if __name__ == '__main__':
    socketio.run(app, host=config['server']['ip'], port=config['server']['port'], debug=True, use_reloader=False)