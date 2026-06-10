import re

def extract(question, sym_name):
    pattern = rf"\b{sym_name}\s*(?:=(?:\s*[a-zA-Z0-9_]+\s*=)*|\s+is\s+)\s*([-+]?(?:\d+[.,]\d+|\d+[.,]?|[.,]\d+)(?:\s*(?:[x×*]\s*10\^?|[eE])\s*[-+]?\d+)?)\s*([cm\u03bcuµnpkMG]?)([a-zA-Z]*)"
    match = re.search(pattern, question, re.IGNORECASE)
    if match:
        val_str = match.group(1)
        prefix = match.group(2)
        rest = match.group(3)
        
        # normalize
        val_str = re.sub(r'\s*(?:[x×*]\s*10\^?|[eE])', 'e', val_str)
        val_str = val_str.replace(' ', '').replace(',', '.')
        val = float(val_str)
        
        multipliers = {
            'c': 1e-2, 'm': 1e-3, 'u': 1e-6, 'µ': 1e-6, '\u03bc': 1e-6,
            'n': 1e-9, 'p': 1e-12, 'k': 1e3, 'M': 1e6, 'G': 1e9
        }
        if prefix and rest and prefix in multipliers:
            val *= multipliers[prefix]
        return val
    return None

print("16 cm:", extract("r = 16 cm", "r"))
print("16 m:", extract("r = 16 m", "r"))
print("16 mm:", extract("r = 16 mm", "r"))
print("16 uC:", extract("q = 16 uC", "q"))
print("16 kg:", extract("m = 16 kg", "m"))
