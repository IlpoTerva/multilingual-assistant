from pipelines.build_vector_index1 import build_index
from analytics.retrieve import embed
from app.chat import agent_answer
from app.chat_lang import generate_chat_response, GRAPH
from IPython.display import Image, display

def main():
    import sys
    if "--build" in sys.argv:
        
        build_index() #building vector database
    else:
        history = []
        print("Loading vector index and embedding model...")
        embed(["test"])  # warm up the model

        print("Operator assistant ready. Ask a question (Ctrl-C to quit).")
        while True:
            try:
                q = input("\n> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if q:
                #reply, messages = agent_answer(q, history=history) #ollama chat
                reply = generate_chat_response(q, history=history) #langgraph version
                if "--debug" in sys.argv:
                    print(reply["tool_calls"])
                print("\n--- ANSWER ---")
                ans = reply["answer"]
                print(f"\n{ans}\n")
                history += [{"role": "user", "content": q}, {"role": "assistant", "content": reply}]


if __name__ == "__main__":
    main()