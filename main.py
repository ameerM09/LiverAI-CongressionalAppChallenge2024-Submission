from website import create_web_app

web_app = create_web_app()

if __name__ == "__main__":
    web_app.run(debug = True, port="8080")