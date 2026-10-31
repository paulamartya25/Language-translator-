"""
Comprehensive language registry with NLLB-200 FLORES+ codes for all
Indian and international languages.
"""

# All 22 official Indian languages (Scheduled languages of India) + others
INDIAN_LANGUAGES = {
    "Hindi": {
        "nllb_code": "hin_Deva",
        "gtts_lang": "hi",
        "flag": "🇮🇳",
        "script": "Devanagari",
        "region": "North India",
    },
    "Bengali": {
        "nllb_code": "ben_Beng",
        "gtts_lang": "bn",
        "flag": "🇮🇳",
        "script": "Bengali",
        "region": "East India",
    },
    "Telugu": {
        "nllb_code": "tel_Telu",
        "gtts_lang": "te",
        "flag": "🇮🇳",
        "script": "Telugu",
        "region": "South India",
    },
    "Marathi": {
        "nllb_code": "mar_Deva",
        "gtts_lang": "mr",
        "flag": "🇮🇳",
        "script": "Devanagari",
        "region": "West India",
    },
    "Tamil": {
        "nllb_code": "tam_Taml",
        "gtts_lang": "ta",
        "flag": "🇮🇳",
        "script": "Tamil",
        "region": "South India",
    },
    "Urdu": {
        "nllb_code": "urd_Arab",
        "gtts_lang": "ur",
        "flag": "🇮🇳",
        "script": "Nastaliq",
        "region": "North India",
    },
    "Gujarati": {
        "nllb_code": "guj_Gujr",
        "gtts_lang": "gu",
        "flag": "🇮🇳",
        "script": "Gujarati",
        "region": "West India",
    },
    "Kannada": {
        "nllb_code": "kan_Knda",
        "gtts_lang": "kn",
        "flag": "🇮🇳",
        "script": "Kannada",
        "region": "South India",
    },
    "Odia": {
        "nllb_code": "ory_Orya",
        "gtts_lang": "or",
        "flag": "🇮🇳",
        "script": "Odia",
        "region": "East India",
    },
    "Malayalam": {
        "nllb_code": "mal_Mlym",
        "gtts_lang": "ml",
        "flag": "🇮🇳",
        "script": "Malayalam",
        "region": "South India",
    },
    "Punjabi": {
        "nllb_code": "pan_Guru",
        "gtts_lang": "pa",
        "flag": "🇮🇳",
        "script": "Gurmukhi",
        "region": "North India",
    },
    "Assamese": {
        "nllb_code": "asm_Beng",
        "gtts_lang": "as",
        "flag": "🇮🇳",
        "script": "Bengali",
        "region": "Northeast India",
    },
    "Maithili": {
        "nllb_code": "mai_Deva",
        "gtts_lang": "hi",
        "flag": "🇮🇳",
        "script": "Devanagari",
        "region": "North India",
    },
    "Sanskrit": {
        "nllb_code": "san_Deva",
        "gtts_lang": "sa",
        "flag": "🇮🇳",
        "script": "Devanagari",
        "region": "Classical",
    },
    "Nepali": {
        "nllb_code": "npi_Deva",
        "gtts_lang": "ne",
        "flag": "🇳🇵",
        "script": "Devanagari",
        "region": "Himalayan",
    },
    "Konkani": {
        "nllb_code": "kok_Deva",
        "gtts_lang": "hi",
        "flag": "🇮🇳",
        "script": "Devanagari",
        "region": "West India",
    },
    "Sindhi": {
        "nllb_code": "snd_Arab",
        "gtts_lang": "sd",
        "flag": "🇮🇳",
        "script": "Arabic",
        "region": "West India",
    },
    "Kashmiri": {
        "nllb_code": "kas_Arab",
        "gtts_lang": "hi",
        "flag": "🇮🇳",
        "script": "Nastaliq",
        "region": "North India",
    },
    "Manipuri": {
        "nllb_code": "mni_Beng",
        "gtts_lang": "hi",
        "flag": "🇮🇳",
        "script": "Bengali/Meitei",
        "region": "Northeast India",
    },
    "Dogri": {
        "nllb_code": "dgo_Deva",
        "gtts_lang": "hi",
        "flag": "🇮🇳",
        "script": "Devanagari",
        "region": "North India",
    },
    "Bodo": {
        "nllb_code": "brx_Deva",
        "gtts_lang": "hi",
        "flag": "🇮🇳",
        "script": "Devanagari",
        "region": "Northeast India",
    },
    "Santali": {
        "nllb_code": "sat_Olck",
        "gtts_lang": "hi",
        "flag": "🇮🇳",
        "script": "Ol Chiki",
        "region": "East India",
    },
}

INTERNATIONAL_LANGUAGES = {
    "English": {
        "nllb_code": "eng_Latn",
        "gtts_lang": "en",
        "flag": "🇺🇸",
        "script": "Latin",
        "region": "Global",
    },
    "Japanese": {
        "nllb_code": "jpn_Jpan",
        "gtts_lang": "ja",
        "flag": "🇯🇵",
        "script": "CJK",
        "region": "East Asia",
    },
    "Chinese (Simplified)": {
        "nllb_code": "zho_Hans",
        "gtts_lang": "zh-cn",
        "flag": "🇨🇳",
        "script": "CJK",
        "region": "East Asia",
    },
    "Chinese (Traditional)": {
        "nllb_code": "zho_Hant",
        "gtts_lang": "zh-tw",
        "flag": "🇹🇼",
        "script": "CJK",
        "region": "East Asia",
    },
    "German": {
        "nllb_code": "deu_Latn",
        "gtts_lang": "de",
        "flag": "🇩🇪",
        "script": "Latin",
        "region": "Europe",
    },
    "French": {
        "nllb_code": "fra_Latn",
        "gtts_lang": "fr",
        "flag": "🇫🇷",
        "script": "Latin",
        "region": "Europe",
    },
    "Dutch": {
        "nllb_code": "nld_Latn",
        "gtts_lang": "nl",
        "flag": "🇳🇱",
        "script": "Latin",
        "region": "Europe",
    },
    "Spanish": {
        "nllb_code": "spa_Latn",
        "gtts_lang": "es",
        "flag": "🇪🇸",
        "script": "Latin",
        "region": "Global",
    },
    "Portuguese": {
        "nllb_code": "por_Latn",
        "gtts_lang": "pt",
        "flag": "🇵🇹",
        "script": "Latin",
        "region": "Global",
    },
    "Russian": {
        "nllb_code": "rus_Cyrl",
        "gtts_lang": "ru",
        "flag": "🇷🇺",
        "script": "Cyrillic",
        "region": "Europe/Asia",
    },
    "Arabic": {
        "nllb_code": "arb_Arab",
        "gtts_lang": "ar",
        "flag": "🇸🇦",
        "script": "Arabic",
        "region": "Middle East",
    },
    "Korean": {
        "nllb_code": "kor_Hang",
        "gtts_lang": "ko",
        "flag": "🇰🇷",
        "script": "Hangul",
        "region": "East Asia",
    },
    "Italian": {
        "nllb_code": "ita_Latn",
        "gtts_lang": "it",
        "flag": "🇮🇹",
        "script": "Latin",
        "region": "Europe",
    },
    "Turkish": {
        "nllb_code": "tur_Latn",
        "gtts_lang": "tr",
        "flag": "🇹🇷",
        "script": "Latin",
        "region": "Middle East/Europe",
    },
    "Vietnamese": {
        "nllb_code": "vie_Latn",
        "gtts_lang": "vi",
        "flag": "🇻🇳",
        "script": "Latin",
        "region": "Southeast Asia",
    },
    "Thai": {
        "nllb_code": "tha_Thai",
        "gtts_lang": "th",
        "flag": "🇹🇭",
        "script": "Thai",
        "region": "Southeast Asia",
    },
    "Swedish": {
        "nllb_code": "swe_Latn",
        "gtts_lang": "sv",
        "flag": "🇸🇪",
        "script": "Latin",
        "region": "Europe",
    },
    "Polish": {
        "nllb_code": "pol_Latn",
        "gtts_lang": "pl",
        "flag": "🇵🇱",
        "script": "Latin",
        "region": "Europe",
    },
    "Indonesian": {
        "nllb_code": "ind_Latn",
        "gtts_lang": "id",
        "flag": "🇮🇩",
        "script": "Latin",
        "region": "Southeast Asia",
    },
    "Greek": {
        "nllb_code": "ell_Grek",
        "gtts_lang": "el",
        "flag": "🇬🇷",
        "script": "Greek",
        "region": "Europe",
    },
    "Hebrew": {
        "nllb_code": "heb_Hebr",
        "gtts_lang": "iw",
        "flag": "🇮🇱",
        "script": "Hebrew",
        "region": "Middle East",
    },
    "Swahili": {
        "nllb_code": "swh_Latn",
        "gtts_lang": "sw",
        "flag": "🇰🇪",
        "script": "Latin",
        "region": "Africa",
    },
    "Malay": {
        "nllb_code": "zsm_Latn",
        "gtts_lang": "ms",
        "flag": "🇲🇾",
        "script": "Latin",
        "region": "Southeast Asia",
    },
    "Filipino": {
        "nllb_code": "fil_Latn",
        "gtts_lang": "tl",
        "flag": "🇵🇭",
        "script": "Latin",
        "region": "Southeast Asia",
    },
    "Ukrainian": {
        "nllb_code": "ukr_Cyrl",
        "gtts_lang": "uk",
        "flag": "🇺🇦",
        "script": "Cyrillic",
        "region": "Europe",
    },
    "Czech": {
        "nllb_code": "ces_Latn",
        "gtts_lang": "cs",
        "flag": "🇨🇿",
        "script": "Latin",
        "region": "Europe",
    },
    "Romanian": {
        "nllb_code": "ron_Latn",
        "gtts_lang": "ro",
        "flag": "🇷🇴",
        "script": "Latin",
        "region": "Europe",
    },
}

# Combined registry
ALL_LANGUAGES = {**INDIAN_LANGUAGES, **INTERNATIONAL_LANGUAGES}

# Whisper language code to NLLB code mapping
WHISPER_TO_NLLB = {
    "en": "eng_Latn",
    "hi": "hin_Deva",
    "bn": "ben_Beng",
    "te": "tel_Telu",
    "mr": "mar_Deva",
    "ta": "tam_Taml",
    "ur": "urd_Arab",
    "gu": "guj_Gujr",
    "kn": "kan_Knda",
    "ml": "mal_Mlym",
    "pa": "pan_Guru",
    "or": "ory_Orya",
    "as": "asm_Beng",
    "ne": "npi_Deva",
    "sd": "snd_Arab",
    "sa": "san_Deva",
    "ja": "jpn_Jpan",
    "zh": "zho_Hans",
    "de": "deu_Latn",
    "fr": "fra_Latn",
    "nl": "nld_Latn",
    "es": "spa_Latn",
    "pt": "por_Latn",
    "ru": "rus_Cyrl",
    "ar": "arb_Arab",
    "ko": "kor_Hang",
    "it": "ita_Latn",
    "tr": "tur_Latn",
    "vi": "vie_Latn",
    "th": "tha_Thai",
    "sv": "swe_Latn",
    "pl": "pol_Latn",
    "id": "ind_Latn",
    "el": "ell_Grek",
    "he": "heb_Hebr",
    "sw": "swh_Latn",
    "ms": "zsm_Latn",
    "uk": "ukr_Cyrl",
    "cs": "ces_Latn",
    "ro": "ron_Latn",
}

# Whisper detected language name mapping
WHISPER_LANG_NAMES = {
    "en": "English",
    "hi": "Hindi",
    "bn": "Bengali",
    "te": "Telugu",
    "mr": "Marathi",
    "ta": "Tamil",
    "ur": "Urdu",
    "gu": "Gujarati",
    "kn": "Kannada",
    "ml": "Malayalam",
    "pa": "Punjabi",
    "or": "Odia",
    "as": "Assamese",
    "ne": "Nepali",
    "sd": "Sindhi",
    "sa": "Sanskrit",
    "ja": "Japanese",
    "zh": "Chinese",
    "de": "German",
    "fr": "French",
    "nl": "Dutch",
    "es": "Spanish",
    "pt": "Portuguese",
    "ru": "Russian",
    "ar": "Arabic",
    "ko": "Korean",
    "it": "Italian",
    "tr": "Turkish",
    "vi": "Vietnamese",
    "th": "Thai",
    "sv": "Swedish",
    "pl": "Polish",
    "id": "Indonesian",
    "el": "Greek",
    "he": "Hebrew",
    "sw": "Swahili",
    "ms": "Malay",
    "uk": "Ukrainian",
    "cs": "Czech",
    "ro": "Romanian",
}


def get_nllb_code(lang_name: str) -> str:
    """Get NLLB code for a language name."""
    if lang_name in ALL_LANGUAGES:
        return ALL_LANGUAGES[lang_name]["nllb_code"]
    return None


def get_gtts_lang(lang_name: str) -> str:
    """Get gTTS language code for a language name."""
    if lang_name in ALL_LANGUAGES:
        return ALL_LANGUAGES[lang_name]["gtts_lang"]
    return None


def whisper_to_nllb(whisper_code: str) -> str:
    """Convert Whisper language code to NLLB FLORES code."""
    return WHISPER_TO_NLLB.get(whisper_code, "eng_Latn")


def get_language_name_from_whisper(whisper_code: str) -> str:
    """Get human-readable language name from Whisper language code."""
    return WHISPER_LANG_NAMES.get(whisper_code, whisper_code.upper())
