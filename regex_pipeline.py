# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import re
import lexicon_filtering
import nltk
import string
from typing import List, Dict, Tuple

tokenizer = nltk.data.load("tokenizers/punkt/PY3/english.pickle")
lexicon_map, multi_word_phrases = lexicon_filtering.get_hardcoded_lexicon()
all_AA_keys = set(
    list(lexicon_map.keys()) + multi_word_phrases
)  # The list of all indicators

# List of negation words that are not permitted.
negation_words = [
    "weren't",
    "wasn't",
    "don't",
    "aren't",
    "can't",
    "neither",
    "if",
    "couldn't",
    "not",
    "shouldn't",
    "wouldn't",
    "stop",
    "people think",
    "you think",
    "nobody",
    "no one",
]

# List of exaggerators as described in line 246 of the paper.
exaggerator_synonyms = (
    "(?:a\s|an\s)*("
    + "|".join(
        [
            "soo*\s",
            "re*a*lly*\s",
            "ve*ry*\s",
            "extre*mely*\s",
            "su*per*\s",
            "pre*tty*\s",
            "the\smost\s",
            "one\sof\sthe\smost\s",
            "abso*lu*tely\s",
            "such\sa\s",
            "alwa*ys*\s",
            "ju*st*\s",
            "espe*cia*lly*\s",
            "friggin.\s",
            "fuckin.\s",
            "friggin\s",
            "fuckin\s",
            "by\sfar\sthe\smo*st\s",
            "probably\sthe\smo*st\s",
            "too*\s",
            "a\slittle\s",
            "a\s*lo*t\s",
            "more\s",
            "quite\spossibly\sthe\smo*st\s",
            "actually\s",
            "ki*nd*\sof\s",
            "freakin.\s",
            "freakin\s",
            "bit\s",
            "currently\s",
            "recently\s",
            "lately\s",
            "honestly\s",
            "truly\s",
            "unbelievably\s",
            "insanely\s",
            "seriously\s",
        ]
    )
    + ")*(?:a\s|an\s)*"
)

# Additional sub-patterns used in CARE patterns
singular_subjective_pronouns = "(" + "|".join(["he", "she"]) + ")"
plural_subjective_pronouns = "(" + "|".join(["they", "you", "u"]) + ")"
singular_demonstrative_pronouns = "(" + "|".join(["that", "this"]) + ")"
plural_demonstrative_pronouns = "(" + "|".join(["these", "those"]) + ")"
beginning = r"(\.|!|but\s|however\s|oh\sno\s|oh\s|oh\sman\s|oh\ssnap\s|omg\s|wow\s|jesus|holy\scrap\s|for\ssome\sreason\s|,|^)\s*(?:funny\senough\s|holy\sshit\s|damn\s|oh\sshit\s)*"
ending = "\s*([^\s]*)\s*([^\s]*)\s*([^\s]*)"
# ending = "\s*([a-z]*)\s*([a-z]*)\s*([a-z]*)"

# Map of CARE pattern names to their respective regular expressions.
regex_name_to_pattern = {
    "individual": beginning
    + "(i)(\s|\sam\s|'m\s|m\s|'ve\s|\shave\s|\shave\snever\s.een\s)"
    + exaggerator_synonyms
    + ending,
    "individual_feel": beginning
    + "(i\sfeel\s)(like\s)*"
    + exaggerator_synonyms
    + ending,
    "we": beginning + "(we)(\sare|'re|re|have|'ve)\s" + exaggerator_synonyms + ending,
    "we_feel": beginning + "(we\sfeel\s)(like\s)" + exaggerator_synonyms + ending,
    "heshe": beginning
    + singular_subjective_pronouns
    + "(\sis|'s|s)\s"
    + exaggerator_synonyms
    + ending,
    "it": beginning + "(it)" + "(\sis|'s|s)\s" + exaggerator_synonyms + ending,
    "theyyou": beginning
    + plural_subjective_pronouns
    + "(\sare|'re|re)\s"
    + exaggerator_synonyms
    + ending,
    "this_is": beginning
    + "(this|that)\s(?:story\s|situation\s)*(is\s|was\s|\s)"
    + exaggerator_synonyms
    + ending,
    "hisher_story": beginning
    + "(his|her)\s(?:story\s|situation\s)*(is\s|was\s|\s)"
    + exaggerator_synonyms
    + ending,
    "noun_is": beginning
    + "(?:the\s)"
    + "([a-z']+)"
    + "\s(is)\s"
    + exaggerator_synonyms
    + ending,
    "this_really": beginning
    + singular_demonstrative_pronouns
    + "\s(re*a*lly*)\s"
    + "(is\s|was\s|\s)*"
    + ending,
    "this_makes_me": beginning
    + singular_demonstrative_pronouns
    + "\s(makes\sme\sfeel|made\sme|made\sme\sfeel|makes\sme)\s"
    + exaggerator_synonyms
    + ending,
    "these_are": beginning
    + plural_demonstrative_pronouns
    + "\s(are|were|)\s"
    + exaggerator_synonyms
    + ending,
    "these_really": beginning
    + plural_demonstrative_pronouns
    + "\s(really)"
    + "\s(are\s|were\s|)*"
    + ending,
    "these_make_me": beginning
    + plural_demonstrative_pronouns
    + "\s(make\sme|make\sme\sfeel|made\sme|made\sme\sfeel)\s"
    + exaggerator_synonyms
    + ending,
    "made_me": beginning
    + "(makes\sme|made\sme)\s(feel\s)*"
    + exaggerator_synonyms
    + ending,
    "feeling": beginning + "()()(feeling\s)" + exaggerator_synonyms + ending,
    "my_heart": beginning + "(my\sheart\sis)" + exaggerator_synonyms + ending,
    "sovery": beginning
    + "()()("
    + "|".join(["soo*\s", "very\s", "extremely\s"])
    + ")+"
    + ending,
    "what_a": beginning + "(what\s)(a|an)\s" + exaggerator_synonyms + ending,
    "how": beginning + "()()(how\s)" + exaggerator_synonyms + ending,
    "some_people": beginning
    + "(some\speople\s|humans\s|society\s)(is\s|are\s|make\sme\s)"
    + exaggerator_synonyms
    + ending,
    "freeform": beginning + "()()()" + ending,
}

# Helper function to skip duplicate affects that can occur from matching multiple patterns.
def get_set(
    matches: List, affects: List[str], indicators: List[str]
) -> Tuple[List[str], List[str], List[str]]:
    output_matches = []
    output_indicators = []

    seen = set()
    for i, affect in enumerate(affects):
        if affect in seen:
            continue
        else:
            seen.add(affect)
            output_matches.append(matches[i])
            output_indicators.append(indicators[i])
    return output_matches, list(seen), output_indicators


# Function for getting a list of all matches, all affects, and all indicators from a given piece of text.
def get_regex_match_all(text: str) -> List[str]:
    if type(text) == list:
        sentences = text
    else:
        sentences = tokenizer.tokenize(text)

    all_matches = []
    all_affects = []
    all_indicators = []

    for sentence in sentences:
        matches, affects, indicators = get_regex_match(sentence)
        if len(affects) > 0:
            matches, affects, indicators = get_set(matches, affects, indicators)
            all_affects.extend(affects)
            all_matches.extend(matches)
            all_indicators.extend(indicators)
    return all_affects


# Check that the pattern and keyword combination is not forbidding.
def is_valid_regex_pattern(regex_name: str, affect: str, keyword: str) -> bool:
    if regex_name in lexicon_filtering.affect_to_prohibited_patterns[affect]:
        return False
    if regex_name == "freeform" and len(keyword.split(" ")) == 1:
        return False
    return True


# Clean the text of punctuation, numbers, and extra spaces, and make lower case.
def clean_text(text: str) -> str:
    # remove numbers
    text_nonum = re.sub(r"\d+", "", text)
    # remove punctuations and convert characters to lower case
    text_nopunct = "".join(
        [
            char.lower()
            for char in text_nonum
            if char not in string.punctuation or char == "'" or char == ","
        ]
    )
    # substitute multiple whitespace with single whitespace
    # Also, removes leading and trailing whitespaces
    text_no_doublespace = re.sub("\s+", " ", text_nopunct).strip()
    return text_no_doublespace


# Apply regular expression matching to a single sentence.
def get_regex_match(sentence: str) -> Tuple[List[str], List[str], List[str]]:
    matches = []
    affects = []
    indicators = []

    if "but" in sentence:
        sentence = sentence[sentence.index("but") + 4 :]
    if "however" in sentence:
        sentence = sentence[sentence.index("however") + 8 :]

    sentence = clean_text(sentence)

    for regex_name, regex_pattern in regex_name_to_pattern.items():
        regex = re.compile(regex_pattern)
        match = regex.search(sentence.lower())

        if match is not None and len(match.groups()) > 0:
            # Make sure that the given group is a noun if the regular expression name is 'noun_is'.
            if regex_name == "noun_is":
                if match.groups()[0] != "":
                    if nltk.pos_tag([match.groups()[0]])[0][1] != "NN":
                        if (
                            match.groups()[1] != ""
                            and nltk.pos_tag([match.groups()[1]])[0][1] != "NN"
                        ):
                            continue
                elif match.groups()[0] == "":
                    if (
                        match.groups()[1] != ""
                        and nltk.pos_tag([match.groups()[1]])[0][1] != "NN"
                    ):
                        continue
            index = 4  # This is the index of the group defining the start of the indicator phrase
            if index > len(match.groups()):
                continue

            indicator = match.groups()[index : len(match.groups())]
            indicator = [
                x.rstrip().lstrip() for x in indicator if x != "" and x is not None
            ]

            for negator in negation_words:
                if negator in indicator:
                    joined_indicator = " ".join(indicator)
                    if (
                        "can't stop laughing" in joined_indicator
                        or "cannot stop laughing" in joined_indicator
                    ):
                        continue
                    else:
                        indicator = []

            keyword = ""

            for i, word in enumerate(indicator):
                if keyword in lexicon_map:
                    print(
                        is_valid_regex_pattern(
                            regex_name, lexicon_map[keyword], keyword
                        )
                    )

                word = word.replace(",", "").rstrip().lstrip()
                if word in all_AA_keys:
                    if word in multi_word_phrases:
                        two_words = " ".join(indicator[:-1])
                        if two_words in lexicon_map:
                            keyword = two_words
                        three_words = two_words + " " + indicator[-1]
                        if three_words in lexicon_map:
                            keyword = three_words
                    elif word in lexicon_map:
                        keyword = word

                if keyword != "" and is_valid_regex_pattern(
                    regex_name, lexicon_map[keyword], keyword
                ):
                    matches.append(
                        " ".join(
                            [
                                x.rstrip().lstrip()
                                for x in match.groups()
                                if x is not None and x != "" and x != " "
                            ]
                        )
                    )
                    affects.append(lexicon_map[keyword])
                    indicators.append(regex_name + ": " + keyword)
                    return matches, affects, indicators
    return matches, affects, indicators
