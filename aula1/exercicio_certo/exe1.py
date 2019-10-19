import numpy as np

p = {
    "exame=+|doenca=+" : 0.95,
    "exame=-|doenca=+" : 0.05,
    "exame=-|doenca=-" : 0.95,
    "exame=+|doenca=-" : 0.05,
    "doenca=+"         : 1/50000,
    "doenca=-"         : 49999/50000
}

# p[doenca=+|exame=+]
# = (p[exame=+|doenca=+] * p[doenca=+]) / p[exame=+]
# p['exame=+'] = p['exame=+,doenca=+'] + p['exame=+,doenca=-']

p['exame=+,doenca=+'] = p["exame=+|doenca=+"] * p['doenca=+']
p['exame=+,doenca=-'] = p["exame=+|doenca=-"] * p['doenca=-']
p['exame=+'] = p['exame=+,doenca=+'] + p['exame=+,doenca=-']


(p["exame=+|doenca=+"] * p["doenca=+"]) / p["exame=+"]
# = 0.037%