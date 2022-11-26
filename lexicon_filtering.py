# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from typing import List, Tuple, Dict

# Map of keyword in the CARE lexicon to pattern combinations that are prohibited.
affect_to_prohibited_patterns = {
    "disgusted": [],
    "saddened": ["heshe", "theyyou"],
    "amused": ["theyyou", "it"],
    "angered": [],
    "disappointed": ["heshe", "theyyou"],
    "entertained": ["individual", "individual_feel", "we", "we_feel"],
    "interested": ["hesheit", "theyyou"],
    "impressed": [],
    "excited": ["heshe", "theyyou", "some_people"],
    "inspired": [],
    "annoyed": [],
    "admiring": [
        "individual_feel",
        "we_feel",
        "heshe",
        "it",
        "theyyou",
        "this_is",
        "hisher_story",
        "noun_is",
        "this_really",
        "these_are",
        "these_really",
        "feeling",
        "what_a",
        "some_people",
    ],
    "scared": ["theyyou", "heshe"],
    "worried": [],
    "anxious": [],
    "adoring": [
        "individual",
        "individual_feel",
        "we",
        "we_feel",
        "this_makes_me",
        "these_make_me",
        "made_me",
        "feeling",
    ],
    "approving": [
        "individual_feel",
        "we",
        "we_feel",
        "this_makes_me",
        "these_make_me",
        "made_me",
        "feeling",
    ],
    "awed": ["heshe", "theyyou", "hisher_story", "some_people"],
    "attracted": [
        "individual",
        "individual_feel",
        "it",
        "we",
        "we_feel",
        "this_is",
        "hisher_story",
        "noun_is",
        "this_really",
        "this_makes_me",
        "these_are",
        "these_really",
        "these_make_me",
        "made_me",
        "feeling",
        "sovery",
        "how",
        "some_people",
    ],
}

# Map of each class to keywords. This is the inverse mapping of the CARE lexicon, as defined in the paper.
affect_to_words = {
    "disgusted": [
        "gross",
        "grosses me out",
        "disgusting",
        "disgusted",
        "disgusts me",
        "nasty",
        "disgust",
        "repulsive",
        "repulses me",
    ],
    "saddened": [
        "depressing",
        "that really sucks",
        "saddening",
        "saddens me",
        "sad",
        "sorry for your",
        "sorry for them",
        "sorry to hear",
        "heartbreaking",
        "heartbroken",
        "tragic",
        "painful to watch",
        "painful to see",
        "hard to see",
        "hard to watch",
        "unfortunate",
        "depressed",
        "depresses me",
    ],
    "amused": [
        "hilarious",
        "funny",
        "cracks me up",
        "laugh",
        "never laughed so",
        "can't stop laughing",
        "cannot stop laughing",
        "the funniest thing",
    ],
    "angered": [
        "why i hate",
        "fake",
        "mislead",
        "infuriated",
        "infuriating",
        "infuriates me",
        "infuriate",
        "fed up",
        "furious",
        "frustrate me",
        "frustrates me",
        "frustrated",
        "frustrating",
        "mad",
        "angry",
        "angers me",
        "pissed me off",
        "pisses me off",
        "fuck the",
        "fuck this",
        "fuck them",
    ],
    "disappointed": [
        "disappointing",
        "disappointed",
        "let down",
        "a bummer",
        "letting down",
    ],
    "entertained": ["entertaining"],
    "interested": [
        "intriguing",
        "intrigues me",
        "interesting",
        "curious to see",
        "talented",
        "curious to know",
        "intrigued",
    ],
    "impressed": [
        "brilliant",
        "impressed",
        "impressive",
        "proud of you",
        "impressive",
        "impresses me",
    ],
    "excited": [
        "happy",
        "ecstatic",
        "excited",
        "stoked",
        "exciting",
        "jazzed",
        "excites me",
        "excite",
        "looking forward to",
    ],
    "inspired": [
        "forward to trying",
        "inspired",
        "inspiring",
        "inspiration",
        "inspires me",
        "uplift",
        "uplifts me",
        "inspire",
        "creative",
        "motivated",
        "encouraged",
        "motivates me",
        "encourages me",
        "motivation",
        "encouragement",
    ],
    "annoyed": [
        "sick of",
        "annoy",
        "annoys me",
        "annoying",
        "annoyed",
        "annoyance",
        "irritates me",
        "irritating",
        "agitates me",
        "agitated",
        "agitation",
        "tired of this",
        "getting ridiculous",
        "tired of seeing",
        "tired of hearing",
    ],
    "admiring": ["admire you", "of admiration for", "admirable"],
    "scared": [
        "scare me",
        "scared",
        "scares me",
        "freaks me out",
        "freak me out",
        "freaky",
        "creepy",
    ],
    "worried": ["worried", "worries me", "concerning", "concerns me"],
    "anxious": ["anxious", "gives me anxiety", "nervous"],
    "adoring": [
        "adorable",
        "the cutest",
        "cute",
        "adorbs",
        "sweet",
        "cutest thing",
    ],
    "approving": [
        "love this",
        "love that",
        "dope",
        "fabulous",
        "high five",
        "excellent",
        "amazing",
        "damn good",
        "fantastic",
        "epic",
        "wonderful",
        "awesome",
        "the best",
        "the greatest",
    ],
    "awed": [
        "magnificent",
        "awe inspiring",
        "awe-inspiring",
        "spectacular",
        "breathtaking",
        "majestic",
        "incredible",
        "in awe",
        "awe-inspired",
    ],
    "attracted": ["beautiful", "gorgeous", "handsome"],
}

# Creates the word to affect lexicon and collects a list of multi-word indicators.
def get_hardcoded_lexicon() -> Tuple[Dict[str, str], List[str]]:
    words_to_affect = {x: k for k, v in affect_to_words.items() for x in v}
    multi_word_phrases = [k.split(" ")[0] for k in words_to_affect.keys() if " " in k]
    return words_to_affect, multi_word_phrases