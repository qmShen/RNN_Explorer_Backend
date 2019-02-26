from app import app

# The port number should be the same as the front end
try:
    app.run(use_reloader=False, debug=True)
except:
    print("Some thing wrong!")