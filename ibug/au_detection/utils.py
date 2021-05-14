__all__ = ['get_au_name']


_au_names = {
    1: 'Inner Brow Raiser',
    2: 'Outer Brow Raiser',
    4: 'Brow Lowerer',
    5: 'Upper Lid Raiser',
    6: 'Cheek Raiser',
    7: 'Lid Tightener',
    9: 'Nose Wrinkler',
    10: 'Upper Lip Raiser',
    11: 'Nasolabial Deepener',
    12: 'Lip Corner Puller',
    13: 'Cheek Puffer',
    14: 'Dimpler',
    15: 'Lip Corner Depressor',
    16: 'Lower Lip Depressor',
    17: 'Chin Raiser',
    18: 'Lip Puckerer',
    20: 'Lip stretcher',
    22: 'Lip Funneler',
    23: 'Lip Tightener',
    24: 'Lip Pressor',
    25: 'Lips part',
    26: 'Jaw Drop',
    27: 'Mouth Stretch',
    28: 'Lip Suck',
    41: 'Lid droop',
    42: 'Slit',
    43: 'Eyes Closed',
    44: 'Squint',
    45: 'Blink',
    46: 'Wink',
    51: 'Head turn left',
    52: 'Head turn right',
    53: 'Head up',
    54: 'Head down',
    55: 'Head tilt left',
    56: 'Head tilt right',
    57: 'Head forward',
    58: 'Head back',
    61: 'Eyes turn left',
    62: 'Eyes turn right',
    63: 'Eyes up',
    64: 'Eyes down'
}


def get_au_name(au: int) -> str:
    return _au_names[au]
