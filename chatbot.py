from matching import get_best_match

while True:
    user_input = input("You: ")
    if user_input.lower() in ("exit", "quit"):
        break
    answer = get_best_match(user_input)
    print("Bot:", answer)
