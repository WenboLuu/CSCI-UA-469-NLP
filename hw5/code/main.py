import nltk
import subprocess
from tqdm import tqdm
from nltk.chunk import tree2conlltags

# Initialize stemmer
stemmer = nltk.stem.SnowballStemmer("english")
pattern = "NP: {<DT>?<JJ>*<NN>}"
regexp_parser = nltk.RegexpParser(pattern)


def predict_bio_tags(sentence, with_tag=True):
    """
    Predict BIO tags for each token in a sentence.

    Args:
    - sentence (list of tuples): A sentence represented as a list of tuples,
                                 where each tuple is (token, POS tag).

    Returns:
    - List of tuples: Each tuple consists of (token, POS tag, predicted BIO tag).
    """
    if with_tag:
        token_postags = [(token, postag) for token, postag, _ in sentence]
    else:
        token_postags = [(token, postag) for token, postag in sentence]

    chunked_sentence = regexp_parser.parse(token_postags)
    bio_predicted = tree2conlltags(chunked_sentence)  # Convert tree to BIO tags
    # only return the BIO tags
    bio_tagged_sentence = [bio_tag for _, _, bio_tag in bio_predicted]

    return bio_tagged_sentence


def prefix_extract(word):
    if not word:
        return None
    prefixes = [
        "anti",
        "dis",
        "extra",
        "inter",
        "pre",
        "re",
        "sub",
        "un",
        "in",
        "im",
        "ir",
        "il",
        "over",
        "under",
        "trans",
        "mis",
        "non",
        "co",
        "com",
        "con",
        "de",
        "auto",
        "bio",
        "geo",
        "psycho",
    ]
    for prefix in prefixes:
        if word.startswith(prefix):
            return prefix
    return None


def suffix_extract(word):
    if not word:
        return None
    suffixes = [
        "able",
        "ible",
        "ation",
        "ment",
        "ness",
        "ity",
        "ty",
        "ly",
        "ing",
        "ed",
        "ize",
        "ise",
        "ful",
        "less",
        "ous",
        "ive",
        "al",
        "er",
        "or",
        "ism",
        "ist",
        "ship",
        "hood",
        "th",
        "en",
        "ify",
        "ward",
        "wise",
    ]
    for suffix in suffixes:
        if word.endswith(suffix):
            return suffix
    return None


def is_capitalized(word):
    return word[0].isupper()


def stem(word):
    return stemmer.stem(word)


def read_sentences(file_path):
    sentences, sentence = [], []
    with open(file_path, "r") as f:
        for line in f:
            if line.strip():
                word = line.strip().split("\t")
                sentence.append(word)
            else:
                if sentence:
                    sentences.append(sentence)
                    sentence = []
    if sentence:
        sentences.append(sentence)
    return sentences


def extract_features(sentences, with_tag=True):
    features = []
    for sentence in tqdm(sentences):
        bio_tags_predicted = predict_bio_tags(sentence, with_tag)
        for i, token in enumerate(sentence):
            word, pos = token[:2]
            bio_tag = token[2] if with_tag else None
            feature = {
                "WORD": word,
                "STEM": stem(word),
                "POS": pos,
                "LENGTH": len(word),
                "PREFIX": prefix_extract(word),
                "SUFFIX": suffix_extract(word),
                "POSITION": round(i / len(sentence), 2),
                "CAPITALIZED": is_capitalized(word),
                "PREVIOUS_WORD": sentence[i - 1][0] if i >= 1 else None,
                "PREVIOUS_STEM": stem(sentence[i - 1][0]) if i >= 1 else None,
                "PREVIOUS_POS": sentence[i - 1][1] if i >= 1 else None,
                "PREVIOUS2_WORD": sentence[i - 2][0] if i >= 2 else None,
                "PREVIOUS2_STEM": stem(sentence[i - 2][0]) if i >= 2 else None,
                "PREVIOUS2_POS": sentence[i - 2][1] if i >= 2 else None,
                "NEXT_WORD": sentence[i + 1][0] if i < len(sentence) - 1 else None,
                "NEXT_STEM": (
                    stem(sentence[i + 1][0]) if i < len(sentence) - 1 else None
                ),
                "NEXT_POS": sentence[i + 1][1] if i < len(sentence) - 1 else None,
                "NEXT2_WORD": sentence[i + 2][0] if i < len(sentence) - 2 else None,
                "NEXT2_STEM": (
                    stem(sentence[i + 2][0]) if i < len(sentence) - 2 else None
                ),
                "NEXT2_POS": sentence[i + 2][1] if i < len(sentence) - 2 else None,
                # "BIO_PREDICTED": bio_tags_predicted[i],
                # "BIO_PREVIOUS": bio_tags_predicted[i - 1] if i >= 1 else None,
                # "BIO_NEXT": (
                #     bio_tags_predicted[i + 1] if i < len(sentence) - 1 else None
                # ),
            }
            if with_tag:
                feature.update(
                    {
                        "PREVIOUS_TAG": (
                            "@@" if i >= 1 else None
                        ),  # @@ should be the placeholder for the BIO tags
                        "PREVIOUS2_TAG": "@@" if i >= 2 else None,
                    }
                )
                features.append((feature, bio_tag))
            else:
                features.append((feature, None))
        # mark for new line
        features.append(("NEWLINE", None))
    return features


def write_features(features, file_path, with_tag=True):
    with open(file_path, "w") as f:
        f.write("\n")  # for format sake
        for feature, bio_tag in features:
            if feature == "NEWLINE":
                f.write("\n")
                continue
            for key, value in feature.items():
                if key == "WORD":
                    f.write(f"{feature['WORD']}\t")
                else:
                    f.write(f"{key}={value}\t")
            if with_tag:
                f.write(f"{bio_tag}\n")
            else:
                f.write("\n")


sentences = read_sentences("../data/WSJ_02-21.pos-chunk")
features = extract_features(sentences)
write_features(features, "../bin/train.features")

sentences = read_sentences("../data/WSJ_24.pos")
features = extract_features(sentences, with_tag=False)
write_features(features, "../bin/dev.features", with_tag=False)

# Execute user.bat located at ..\bin\train_and_test.bat for automatic training and testing
subprocess.run(["..\\bin\\train_and_dev.bat"], shell=True)

# if output test
sentences = read_sentences("../data/WSJ_23.pos")
features = extract_features(sentences, with_tag=False)
write_features(features, "../bin/test.features", with_tag=False)

# Execute user.bat located at ..\bin\output_test.bat for automatic training and testing
subprocess.run(["..\\bin\\output_test.bat"], shell=True)