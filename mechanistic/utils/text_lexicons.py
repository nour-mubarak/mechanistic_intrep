import re

MALE_EN = [r"\bman\b", r"\bmen\b", r"\bmale\b", r"\bboy\b", r"\bboys\b",
           r"\bguy\b", r"\bhe\b", r"\bhim\b", r"\bhis\b", r"\bmr\.?\b",
           r"\bhusband\b", r"\bbrother\b", r"\bfather\b", r"\bdad\b", r"\bson\b"]
FEMALE_EN = [r"\bwoman\b", r"\bwomen\b", r"\bfemale\b", r"\bgirl\b", r"\bgirls\b",
             r"\blady\b", r"\bshe\b", r"\bher\b", r"\bmrs\.?\b", r"\bwife\b",
             r"\bsister\b", r"\bmother\b", r"\bmom\b", r"\bdaughter\b"]

MALE_AR = [r"\bرجل\b", r"\bرجال\b", r"\bذكر\b", r"\bشاب\b", r"\bصبي\b", r"\bفتى\b",
           r"\bهو\b", r"\bله\b", r"\bزوج\b", r"\bأب\b", r"\bوالد\b", r"\bابن\b", r"\bأخ\b", r"\bسيد\b"]
FEMALE_AR = [r"\bامرأة\b", r"\bإمرأة\b", r"\bنساء\b", r"\bأنثى\b", r"\bفتاة\b", r"\بنت\b",
             r"\بهي\b", r"\bلها\b", r"\bزوجة\b", r"\bأم\b", r"\bوالدة\b", r"\bابنة\b", r"\bأخت\b", r"\bسيدة\b"]

male_re = re.compile("|".join(MALE_EN + MALE_AR), flags=re.IGNORECASE)
female_re = re.compile("|".join(FEMALE_EN + FEMALE_AR), flags=re.IGNORECASE)

def caption_gender_label(text_en: str, text_ar: str):
    text = f"{text_en or ''} || {text_ar or ''}"
    has_m = bool(male_re.search(text))
    has_f = bool(female_re.search(text))
    if has_m and has_f: return "mixed"
    if has_m: return "male"
    if has_f: return "female"
    return "none"
