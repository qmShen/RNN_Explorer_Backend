from app import app

# The port number should be the same as the front end
try:
    app.run(use_reloader=True, debug=True, port = '9930')
except:
    print("Some thing wrong!")
    print("Some thing wrong!")