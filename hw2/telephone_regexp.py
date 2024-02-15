import re
import sys


def main(file_path="test_dollar_phone_corpus.txt"):
    with open(file_path, encoding="utf8") as file:
        all_phone = file.read()

    pattern = r"(?:(?:\(\d{3}\) \d{3}-\d{4})|(?:\d{3}-\d{3}-\d{4}))"
    result = re.findall(pattern, all_phone)

    with open("telephone_output.txt", "w") as file:
        file.write("\n".join(result))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: dollar_program.py <file_path>")
    else:
        file_path = sys.argv[1]
        main(file_path)
