import re

def negation_preprocessing(text):
    negation_words = {
        "not", "never", "no", "nor", "neither",
        "isn't", "wasn't", "doesn't", "didn't",
        "won't", "can't", "couldn't", "hardly", "barely"
    }
    stop_words = {"but", "however", "although", "though", "yet"}

    tokens = re.findall(r"\b\w+\b", text.lower())
    result = []
    negating = 0

    for token in tokens:
        if token in negation_words:
            negating = 3
            result.append(token)
        elif token in stop_words:
            negating = 0
            result.append(token)
        elif negating > 0:
            result.append("not_" + token)
            negating -= 1
        else:
            result.append(token)

    return " ".join(result)

examples = [
    "the film was not good at all",
    "i never enjoyed a movie like this",
    "the acting was not bad but the story was boring",
    "this is not the worst film i have seen",
]

for s in examples:
    print(f"BEFORE: {s}")
    print(f"AFTER:  {negation_preprocessing(s)}")
    print()