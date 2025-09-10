from mechanistic.utils import text_lexicons as lex

def align_token_spans(en_txt: str, ar_txt: str):
    spans = {"male": [], "female": []}
    if en_txt:
        for m in lex.male_re.finditer(en_txt):
            spans["male"].append({"lang":"en","start":m.start(),"end":m.end(),"text":en_txt[m.start():m.end()]})
        for f in lex.female_re.finditer(en_txt):
            spans["female"].append({"lang":"en","start":f.start(),"end":f.end(),"text":en_txt[f.start():f.end()]})
    if ar_txt:
        for m in lex.male_re.finditer(ar_txt):
            spans["male"].append({"lang":"ar","start":m.start(),"end":m.end(),"text":ar_txt[m.start():m.end()]})
        for f in lex.female_re.finditer(ar_txt):
            spans["female"].append({"lang":"ar","start":f.start(),"end":f.end(),"text":ar_txt[f.start():f.end()]})
    return spans
