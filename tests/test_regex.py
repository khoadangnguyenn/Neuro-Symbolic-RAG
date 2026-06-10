import re

def extract(question, sym_name):
    pattern = rf"\b{sym_name}\s*(?:=(?:\s*[a-zA-Z0-9_]+\s*=)*|\s+is\s+)\s*([-+]?\d*(?:[.,]\d+)?(?:\s*(?:[x×*]\s*10\^?|[eE])\s*[-+]?\d+)?)\s*([m\u03bcuµnpkMG]?)[a-zA-Z]*"
    match = re.search(pattern, question, re.IGNORECASE)
    if match:
        val_str = match.group(1)
        prefix = match.group(2)
        
        # normalize
        val_str = re.sub(r'\s*(?:[x×*]\s*10\^?|[eE])', 'e', val_str)
        val_str = val_str.replace(' ', '').replace(',', '.')
        val = float(val_str)
        
        multipliers = {
            'm': 1e-3, 'u': 1e-6, 'µ': 1e-6, '\u03bc': 1e-6,
            'n': 1e-9, 'p': 1e-12, 'k': 1e3, 'M': 1e6, 'G': 1e9
        }
        if prefix in multipliers:
            val *= multipliers[prefix]
        return val
    return None

q = "Three electric charges, q1 = q2 = q3 = 2.4 × 10^-19 C, are placed at the three vertices of an equilateral triangle ABC with side length 16 cm in air. Determine the net electric force vector acting on q3"

print("q1:", extract(q, "q1"))
print("q2:", extract(q, "q2"))
print("q3:", extract(q, "q3"))

