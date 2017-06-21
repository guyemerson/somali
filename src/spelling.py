import re
from functools import partial

def intervocalic(regex_str):
    """
    Modify a regex string, to only match when it is
    both preceded and followed by a vowel
    """
    # This uses lookahead (?=...) and lookbehind (?<=...)
    # so that the context is not consumed by the regex
    return '(?<=[aeiouwy]){}(?=[aeiouwy])'.format(regex_str)

# These regexes will be applied in this order
regexes = [('lt', 'l'),  # These are simple substitutions
           ('dh', 'd'),
           ('h?t', 'd'),
           ('kh', 'q'),
           ('ch', 'j'),
           ('p', 'b'),
           ('k', 'g'),
           ('hn', 'n'),
           ('x', 'h'),
           ('z', 's'),
           (intervocalic('r'), 'd'),  # only if in between vowels
           (intervocalic('w'), 'b'),
           # The following map vowels down to just 'a', 'i', 'u'
           ('[ae][iy]', 'i'),  # Vowel followed by 'i' or 'y' should be replaced with 'i'
           ('[ao][uw]', 'u'),  # Vowel followed by 'u' or 'w' should be replaced with 'u'
           ('[ey]', 'i'),  # Replace any remaining 'e', 'y' with 'i'
           ('[ow]', 'u'),  # Replace any remaining 'o', 'w' with 'u'
           # Finally, replace repeated letters
           (r'(.)\1+', r'\1')]

# Construct functions that take a string and return the transformed string
rules = [partial(re.compile(in_regex).sub, out_regex)
         for in_regex, out_regex in regexes]


# Informal test
if __name__ == "__main__":
    # Check that certain inputs are mapped to the same or different outputs
    same = [('dhaqan', 'daqan'),
            ('khatar', 'qataar'),
            ('dhaqan', 'daqan'),
            ('khatar', 'qataar'),
            ('jawabey', 'chawabey'),
            ('qubee', 'qubay'),
            ('jacayl', 'jaceyl'),
            ('ari', 'adhi'),
            ('mobil', 'mobel'),
            ('agteyt', 'akteyt'),
            ('mii', 'miyi'),
            ('khutarada', 'khudarada'),
            ('kanecada', 'kanecadu'),
            ('hargeisa', 'hargaysa'),
            ('jawab', 'jawap'),
            ('nawad', 'nabad'),
            ('dhaw', 'dhow'),
            ('qubeyo', 'qobeyo'),
            ('laiskailaliyo', 'layskailaliyo'),
            ('cayrin', 'cayriin'),
            ('xage', 'xagee'),
            ('maxa', 'maxaa'),
            ('xolo', 'xoolo'),
            ('carur', 'caruur'),
            ('welti', 'welli'),
            ('bahti', 'batti'),
            ('bahni', 'banni')]
    
    different = [('a', 'i'),
                 ('a', 'u'),
                 ('i', 'u')]
    
    def process(msg):
        for r in rules:
            msg = r(msg)
        return msg
    
    for x,y in same:
        if process(x) != process(y):
            print(x,y)
    
    for x,y in different:
        if process(x) == process(y):
            print(x,y)
