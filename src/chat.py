# src/chat.py
from search import search_prompt
from dotenv import load_dotenv

def main():
    print("Faça sua pergunta (Ctrl+C para sair):\n")
    while True:
        try:
            question = input("PERGUNTA: ").strip()
            if not question:
                continue
            answer, pages = search_prompt(question)
            print(f"RESPOSTA: {answer}")
            if pages:
                print(f"FONTES: páginas {', '.join(map(str, pages))}\n")
            else:
                print()
        except KeyboardInterrupt:
            print("\nSaindo...")
            break
        except Exception as e:
            print(f"Erro: {e}\n")

if __name__ == "__main__":
    main()
