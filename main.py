from frontend.app import MathAIChatbot

if __name__ == "__main__":
    chatbot = MathAIChatbot()
    app = chatbot.launch_interface()
    app.launch(
        server_name="localhost",
        server_port=8502,
        share=False  # Cambiar a True para generar URL pública
    )