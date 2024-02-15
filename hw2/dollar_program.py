import re
import sys


def main(file_path="test_dollar_phone_corpus.txt"):
    with open(file_path, encoding="utf8") as file:
        all_dollar = file.read()

    pattern1 = r"\$(?:(?:\d{1,3}(?:\,\d{3})*)|(?:\d+))+(?:\.\d+)?(?: million| billion| trillion| thousand| hundred)?(?: dollars?| cents?)?"
    pattern2 = r"[^$]((?:(?:\d{1,3}(?:,\d{3})*)|(?:\d+)|(?:a|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)-?(?:(?:one|two|three|four|five|six|seven|eight|nine))?)+(?: million| billion| trillion| thousand| hundred)*(?: dollars?| cents?))"

    result_dollar_1 = re.findall(pattern1, all_dollar)
    result_dollar_2 = re.findall(pattern2, all_dollar, flags=re.IGNORECASE)

    result_dollar = result_dollar_1 + result_dollar_2

    with open("dollar_output.txt", "w") as file:
        file.write("\n".join(result_dollar))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: dollar_program.py <file_path>")
    else:
        file_path = sys.argv[1]
        main(file_path)
