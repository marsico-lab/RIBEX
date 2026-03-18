import pandas as pd

#extracted from Bressin et. al 2019 Supplimentary Tables S5 and S6
GOterms = [   # ['GO_term', 'UniProd-KW', 'name']
    (None, "KW-0010", "Activator"),
    (None, "KW-0013", "ADP-ribosylation"),
    ("GO:0006325", "KW-0156", "Chromatin regulator"),
    ("GO:0007059", "KW-0159", "Chromosome partition"),
    ("GO:0000786", "KW-0544", "Nucleosome core"),
    ("GO:0005694", "KW-0158", "Chromosome"),
    ("GO:0004519", "KW-0255", "Endonuclease"),
    ("GO:0004518", "KW-0267", "Excision nuclease"),
    ("GO:0004527", "KW-0269", "Exonuclease"),
    ("GO:0004386", "KW-0347", "Helicase"),
    ("GO:0006314", "KW-0404", "Intron homing"),
    ("GO:0016853", "KW-0413", "Isomerase"),
    ("GO:0004518", "KW-0540", "Nuclease"),
    ("GO:0005681", "KW-0747", "Spliceosome"),
    ("GO:0003916", "KW-0799", "Topoisomerase"),
    (None, "KW-0804", "Transcription"),
    (None, "KW-0805", "Transcription regulation"),
    ("GO:0006353", "KW-0806", "Transcription termination"),
    ("GO:0006417", "KW-0810", "Translation regulation"),
    ("GO:0003677", "KW-0238", "DNA-binding"),
    ("GO:0006974", "KW-0227", "DNA damage"),
    ("GO:0006281", "KW-0228", "DNA excision"),
    ("GO:0015074", "KW-0229", "DNA integration"),
    ("GO:0006281", "KW-0234", "DNA repair"),
    ("GO:0006260", "KW-0235", "DNA replication"),
    ("GO:0008156", "KW-0236", "DNA replication inhibitor"),
    ("GO:0071897", "KW-0237", "DNA synthesis"),
    ("GO:0006310", "KW-0233", "DNA recombination"),
    ("GO:0003887", "KW-0239", "DNA-directed DNA polymerase"),
    ("GO:0000428", "KW-0240", "DNA-directed RNA polymerase"),
    ("GO:0003723", "KW-0694", "RNA-binding"),
    ("GO:0003964", "KW-0695", "RNA-directed DNA polymerase"),
    ("GO:0003968", "KW-0696", "RNA-directed RNA polymerase"),
    ("GO:0042245", "KW-0692", "RNA repair"),
    ("GO:0019843", "KW-0699", "rRNA-binding"),
    ("GO:0006364", "KW-0698", "rRNA processing"),
    ("GO:0006397", "KW-0507", "mRNA processing"),
    ("GO:0008380", "KW-0508", "mRNA splicing"),
    ("GO:0051028", "KW-0509", "mRNA transport"),
    ("GO:0000049", "KW-0820", "tRNA-binding"),
    ("GO:0008033", "KW-0819", "tRNA processing"),
    (None, "KW-0687", "Ribonucleoprotein"),
    ("GO:0005840", "KW-0689", "Ribosomal protein"),
    ("GO:0019072", "KW-0231", "Viral genome packaging"),
    ("GO:0039694", "KW-0693", "Viral RNA replication"),
    ("GO:0042254", "KW-0690", "Ribosome biogenesis"),
    ("GO:0000166", "KW-0547", "Nucleotide-binding")
]

GOterms = pd.DataFrame(GOterms, columns=['GO_term', 'UniProd-KW', 'name'])

# TODO: 
# - some have no GO equivalent
# - some have the same GO term? (0004518 0006281)
