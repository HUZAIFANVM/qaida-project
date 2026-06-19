"""
Lesson configuration for all 16 Qaida chapters.

Each lesson has different column names in the xlsx file.
This config maps them to a unified schema.
"""

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

LESSON_CONFIG = {
    1: {
        "xlsx": "1_Finalized quaida alphabet.xlsx",
        "audio_dir": "1_Finalized quaida alphabet",
        "arabic_col": "letter",
        "translit_col": "Transliteration",
        "type_name": "alphabet",
    },
    2: {
        "xlsx": "2_alphabet_shapes_final copy.xlsx",  # original is locked
        "audio_dir": "2_alphabet_shapes_final.xlsx",   # dir named like xlsx
        "arabic_col": "shape",
        "translit_col": "Transliteration",
        "type_name": "shapes",
    },
    3: {
        "xlsx": "3_harakat finalized.xlsx",
        "audio_dir": "3_harakat finalized",
        "arabic_col": "Arabic",
        "translit_col": "Transliteration",
        "type_name": "harakat",
        "extra_cols": {"sub_type": "Type", "base_letter": "letter"},
    },
    4: {
        "xlsx": "4_finalized Long Voices.xlsx",
        "audio_dir": "4_finalized Long Voices",
        "arabic_col": "Arabic",
        "translit_col": "Transliteration",
        "type_name": "long voices",
        "extra_cols": {"sub_type": "Type", "base_letter": "Letter"},
    },
    5: {
        "xlsx": "5_finalized tanween.xlsx",
        "audio_dir": "5_finalized tanween",
        "arabic_col": "ArAbic",         # note casing
        "translit_col": "TrAnsliterAtion",  # note casing
        "type_name": "tanween",
        "extra_cols": {"sub_type": "Type"},
    },
    6: {
        "xlsx": "6_Finalized Sukoon .xlsx",
        "audio_dir": "6_Finalized Sukoon",
        "arabic_col": "Arabic",
        "translit_col": "Transliteration",
        "type_name": "sukoon",
        "extra_cols": {"sub_type": "Type"},
    },
    7: {
        "xlsx": "7_finalized shaddah.xlsx",
        "audio_dir": "7_finalized shaddah",
        "arabic_col": "Arabic",
        "translit_col": "Transliteration",
        "type_name": "shaddah",
    },
    8: {
        "xlsx": "8_Finalized qalqalah.xlsx",
        "audio_dir": "8_Finalized qalqalah",
        "arabic_col": "word",
        "translit_col": "Transliteration",
        "type_name": "qalqalah",
    },
    9: {
        "xlsx": "9_finalized Madd.xlsx",
        "audio_dir": "9_finalized Madd",
        "arabic_col": "Arabic",
        "translit_col": "Transliteration",
        "type_name": "madd",
        "extra_cols": {"sub_type": "Type"},
    },
    10: {
        "xlsx": "10_Finalized leen letters.xlsx",
        "audio_dir": "10_Finalized leen letters",
        "arabic_col": "Arabic",
        "translit_col": "Transliteration",
        "type_name": "leen letters",
        "extra_cols": {"sub_type": "Type", "base_letter": "letter"},
    },
    11: {
        "xlsx": "11_finalized Heavy_Letters.xlsx",
        "audio_dir": "11_finalized Heavy_Letters",
        "arabic_col": "Heavy Letters",
        "translit_col": "Transliteration",
        "type_name": "heavy letters",
    },
    12: {
        "xlsx": "12_finalized Heavy and Light Alif.xlsx",
        "audio_dir": "12_finalized Heavy and Light Alif",
        "arabic_col": "Words",
        "translit_col": "Transliteration",
        "type_name": "heavy and light alif",
    },
    13: {
        "xlsx": "13_Finalized List Heavy and Light Laam.xlsx",
        "audio_dir": "13_Finalized Heavy and Light Laam",
        "arabic_col": "Heavy and Light Laam",
        "translit_col": "Transliteration",
        "type_name": "heavy and light laam",
    },
    14: {
        "xlsx": "14_Finalized_Heavy_and_Light_Raa.xlsx",
        "audio_dir": "14_Finalized_Heavy_and_Light_Raa",
        "arabic_col": "Word",
        "translit_col": "Transliteration",
        "type_name": "heavy and light raa",
    },
    15: {
        "xlsx": "15_finalized_Gunnah.xlsx",
        "audio_dir": "15_finalized_Gunnah",
        "arabic_col": "Ghunnah",
        "translit_col": "Transliteration",
        "type_name": "gunnah",
    },
    16: {
        "xlsx": "16_finalized_hurof mukatat .xlsx",
        "audio_dir": "16_finalized_hurof mukatat",
        "arabic_col": "Words",
        "translit_col": "Transliteration",
        "type_name": "hurof mukatat",
    },
}

# Lesson 4 has 3 rows with NULL audio_id (letter ذ zhal).
# Orphan WAV files 04_067/068/069 correspond to these rows.
LESSON4_NULL_AUDIO_FIX = {
    "QUAIDA_00221": "04_067",  # ذٰ zhaa
    "QUAIDA_00222": "04_068",  # ذٖ zhee
    "QUAIDA_00223": "04_069",  # ذٗ zhoo
}
