from app import app
import time

# The port number should be the same as the front end
try:
    app.run(use_reloader=True, debug=True, port = 9950)
except:
    print("Some thing wrong!")






