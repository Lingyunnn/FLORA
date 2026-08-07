"""
This file is part of FLORA, an unsupervised system for automatic knowledge graph (KG) alignment. 
The file is licensed under the Creative Commons Attribution 4.0 International License (CC BY 4.0) by Yiwen Peng, Thomas Bonald, Fabian Suchanek and Lingyun Huang.

Description: Shared literal parsing, normalization, filtering, and streaming helpers for literal processing.
"""

import re
import unicodedata
from urllib.parse import unquote


# Regex for literals
literalRegex = re.compile('"([^"]*)"(@([a-z-]+))?(\\^\\^(.*))?')

# Regex for int values
intRegex = re.compile('^"?[+-]?[0-9]+"?$')

# Regex for float values
floatRegex = re.compile('^"?([+-])?([0-9.]+)"?$')
sciFloatRegex = re.compile('^"?([+-])?([0-9.]+[Ee][+-]?[0-9]+)"?$')

# Regex for numbers: post code, phone number, etc.
# A normalized-number candidate must contain at least one digit.
numberRegex = re.compile(r'(?=.*\d)[\d\W]+')
identifierRegex = re.compile(r'([a-zA-Z]+)(\d+)') # TBD if needed

DATE_DATATYPES = {'xsd:date', 'xsd:gYear', 'xsd:gYearMonth', 'xsd:dateTime', 'xsd:gMonthDay'}


def isLiteral(term):
    return re.match(literalRegex, term) or re.match(floatRegex, term)


def normalize_datatype(datatype):
    """Normalize the datatype to a standard format. e.g., "http://www.w3.org/2001/XMLSchema#string" -> "xsd:string"."""
    if datatype is None:
        return None
    datatype = datatype.strip()
    if datatype.startswith('<') and datatype.endswith('>'):
        datatype = datatype[1:-1]
    xmlschema_prefix = 'http://www.w3.org/2001/XMLSchema#'
    if datatype.startswith(xmlschema_prefix):
        datatype = 'xsd:' + datatype[len(xmlschema_prefix):]
    return datatype


def unicode(txt):
    # e.g., "Coamo,_Puerto_Rico" -> "Coamo_u002C_Puerto_Rico"
    encoded_str = re.sub(r'[^a-zA-Z0-9_]', lambda x: "_u{:04X}_".format(ord(x.group())), txt)
    return encoded_str


def decode_unicode(encoded_str):
    decoded_string = re.sub(r'_u([0-9A-F]{4})_', lambda x: chr(int(x.group(1), 16)), encoded_str)
    if '\\u' in decoded_string:
        try:
            s = decoded_string.encode('utf-8').decode('unicode_escape')
            return s.encode('utf-16', 'surrogatepass').decode('utf-16')
        except Exception:
            return decoded_string
    else:
        return decoded_string


def numeric_normalization(term):
    term = term.strip('"')
    # Normalize the input string by removing all non-digit characters, except the sign
    # sign = term[0] if term.startswith('-') or term.startswith('+') else None
    normalized = re.sub(r'[^0-9]', '', term)
    return normalized


def splitLiteral(term):
    """ Returns String value, int value, language, and datatype of a term (or None, None, None, None). No good backslash handling """
    literal, _, lang, _, datatype = re.match(literalRegex, term).groups()
    datatype = normalize_datatype(datatype)
    # Dates
    if datatype in DATE_DATATYPES:
        return (literal, None, lang, datatype)

    # Numbers
    floatmatch = re.match(floatRegex, literal)
    scifloatmatch = re.match(sciFloatRegex, literal)
    if (floatmatch or scifloatmatch) and lang is None:
        try:
            # some identifiers are integers
            Value = int(literal.strip('"'))
            if len(str(Value)) != len(literal.strip('"')):
                # e.g., "06" /= 6, "+3" /= 3
                return (literal, None, lang, datatype) # datatype 'none'
            return (literal, Value, lang, datatype)
        except:
            try:
                # e.g., code version 1.0 /= 1
                Value = float(literal.strip('"'))
                return (literal, Value, lang, datatype)
            except ValueError:
                # e.g. "23.78.9" version
                return (literal, None, lang, datatype)

    matchNumber = re.fullmatch(numberRegex, literal)
    # Check if the string is a number type, e.g., "818/762-1221"
    if matchNumber:
        Value = numeric_normalization(literal)
        if len(Value) > 0:
            return (literal, 'normalized_' + Value, lang, datatype)
    # Strings: lowecasing, order-agnostic, decode unicode
    # Pre-processing for string literals
    de_literal = decode_unicode(literal)
    return (de_literal, None, lang, 'xsd:string')


def reorder_string_with_brackets(input):
    bracket_content = re.findall(r'\(.*?\)', input)
    content_without_brackets = re.sub(r'\(.*?\)', '', input).split()
    return (' '.join(set(content_without_brackets)) + ' ' + ' '.join(bracket_content)).strip()


def is_punctuation_only_literal(txt):
    """Check if a string literal consists only of punctuation characters."""
    txt = decode_unicode(txt).strip()
    return bool(txt) and not any(char.isalnum() for char in txt)


def is_human_readable(txt):
    if not txt or len(txt) == 0:
        return False
    # not a web link
    if txt.startswith('http'):
        return False
    non_alpha_ratio = sum(not c.isalpha() for c in txt) / len(txt)
    return non_alpha_ratio < 0.5


def is_faiss_literal_candidate(txt):
    """Filter string literals before FAISS search."""
    if not txt:
        return False
    txt = decode_unicode(txt).strip()
    if not txt:
        return False
    return is_human_readable(txt)


def normalize_string_literal(value):
    value = decode_unicode(value) # decode unicode characters
    value = unquote(value) # decode URL-encoded characters
    value = value.strip() # remove spaces
    value = value.lower() # convert to lowercase

    # Normalize common separators.
    value = value.replace('_', ' ') # replace underscores with spaces
    value = re.sub(r'\s+', ' ', value) # replace multiple spaces with a single space
    return value.strip()


def is_latin_alpha(char):
    if not char.isalpha():
        return False
    try:
        return 'LATIN' in unicodedata.name(char)
    except ValueError:
        return False


def is_english_string_literal(literal, lang=None, min_latin_ratio=0.8):
    """Check if a string literal is likely to be English based on its language tag and the ratio of Latin letters."""
    if lang is not None:
        lang = lang.lower()
        if lang != 'en' and not lang.startswith('en-'):
            return False

    letters = [char for char in decode_unicode(literal) if char.isalpha()]
    if not letters:
        return True

    latin_letters = sum(1 for char in letters if is_latin_alpha(char))
    return latin_letters / len(letters) >= min_latin_ratio


def iter_literal_objects(kb):
    if hasattr(kb, 'iter_literal_object_ids') and hasattr(kb, 'entity_for_id'): # for CompactGraph
        yield from (kb.entity_for_id(object_id) for object_id in kb.iter_literal_object_ids())
        return
    yield from (object for object in kb.objects() if isLiteral(object))


def iter_ttl_literal_facts(path, fast_line_parser=True):
    """Yield (subject, literal_object) from TTL files.
    The fast parser is intended for large one-triple-per-line TTL files. 
    Set fast_line_parser=False to use FLORA's general Turtle parser instead.
    """
    if not fast_line_parser:
        import utils

        for subject, _, obj in utils.triplesFromTurtleFile(path):
            if isLiteral(obj):
                yield subject, obj
        return

    with open(path, "rt", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("@prefix") or line.startswith("#"):
                continue
            parts = line.split(None, 2)
            if len(parts) != 3 or not parts[2].endswith("."):
                continue
            obj = parts[2][:-1].strip()
            if obj.startswith('"') and isLiteral(obj):
                yield parts[0], obj


def iter_ttl_object_literals(path, fast_line_parser=True):
    """Yield literal objects from one-triple-per-line TTL files."""
    for _, obj in iter_ttl_literal_facts(path, fast_line_parser=fast_line_parser):
        yield obj