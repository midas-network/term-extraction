import sys

import spacy

import pandas as pd

import re

from utils.extraction_utils import build_corpus_words_only

searchTerms = {
               "(anaplasma|anaplasmosis)",
               "anthrax",
               "argentine hemorrhagic fever",
               "avian (influenza|flu)",
               "babesia",
               "babesiosis"
               "bacillary dysentery",
               "bacillus anthracis",
               "bacterial vaginosis",
               "blinding trachoma",
               "bolivian hemorrhagic fever",
               "brazilian hemorrhagic fever",
               "bv",
               "c\.? diff",
               "campylobacter",
               "campylobacteriosis",
               "(candida|candidiasis)",
               "cat scratch disease",
               "chancroid",
               "(chickenpox|varicella)",
               "chikungunya( fever)?",
               "chlamydia",
               "cholera",
               "cjd",
               "(clostridium|clostridioides) difficile",

               "coronavirus",
               "covid ?19",
               "creutzfeldt-jakob disease",
               "crimean-congo hemorrhagic fever",
               "(cryptosporidiosis|cryptosporidium( infection)?)",
               "csd",
               "cyclospora",
               "cyclosporiasis",
               "(cytomegalovirus( infection)?|cmv)",
               "cytomegalovirus",
               "(dhf|dengue( fever)?)",
               "diphtheria",
               "dysentery",
               "((escherichia|e\.? coli) infection|enterohemorrhagic (e\.?|escherichia) coli|ehec|etec|stec)",
               "eastern equine encephalitis",
               "(ebola|ebv)",
               "(ehrlichia|ehrlichiosis)",
               "enterotoxic e\.? coli",
               "epstein-barr virus infection",
               "escherichia",
               "fever and rash illnesses",
               "fungal infection",
               "gastroenteritis",
               "(giardia infection|giardiasis)",
               "glandular fever",
               "(gonorrhea|gonorrhoea)",
               "hand, foot, and mouth disease",
               "hantavirus (infection|pulmonary syndrome)",
               "(hai|healthcare acquired infection|healthcare associated infection)",
               "heartland virus",
               "(hus|hemolytic uremic syndrome)",
               "hemorrhagic fever with renal syndrome",
               "hendra virus infection",
               "hep a",
               "hep b",
               "hep c",
               "hep e",
               "hepatitis a",
               "hepatitis b",
               "hepatitis c",
               "hepatitis e",
               "(herpes|herpes simplex virus|hsv)",
               "hfmd",
               "histoplasmosis",
               "(hiv|human immunodeficiency virus|aids|acquired immune deficiency syndrome)",
               "(human papillomavirus infection|hpv infection)",
               "(ili|influenza-like illness)",
               "infectious mononucleosis",
               "(influenza|flu)",
               "invasive candidiasis",
               "japanese encephalitis",
               "kyasanur forest disease",
               "lassa (hemorrhagic )?fever",
               "(legionella|legionellosis|legionnaires' disease)",
               "(leishmania|leishmaniasis)",
               "(leprosy|leprae|hansen disease)",
               "leptospirosis",
               "(listeria|listeriosis)",
               "lockjaw",
               "lyme disease",
               "lymphocytic choriomeningitis",
               "(malaria|plasmodium)",
               "marburg virus disease",
               "measles",
               "meningitis",
               "meningococcal( disease)?",
               "meningococcemia",
               "(mers|middle east respiratory syndrome)",
               "methicillin-resistant staphylococcus aureus",
               "(monkeypox|mpox|monkey pox)",
               "mononucleosis",
               "(mrsa|methicillin-resistant staphylococcus aureus)",
               "mumps",
               "nipah virus infection",
               "norovirus",
               "norwegian scabies",
               "o157:h7",
               "omsk hemorrhagic fever",
               "(pertussis|whooping cough)",
               "(plague|pestis)",
               "pneumonia",
               "(polio|poliomyelitis)",
               "powassan virus",
               "q fever",
               "rabies",
               "(rickettsia|rickettsial)",
               "rift valley fever",

               "(rmsf|rocky mountain spotted fever)",
               "rotavirus( infection)?",
               "rotavirus",
               "(rubella|german measles)",
               "(salmonella|salmonellosis)",
               "(sars|severe acute respiratory syndrome)",
               "scabies",
               "scrub typhus",
               "shiga toxin-producing (e\.?|escherichia) coli",
               "(shigella|shigellosis)",
               "(smallpox|small pox)",
               "st\.? louis encephalitis",
               "staph infections",
               "(staphylococcus|staphylococcal infections)",
               "(streptococcus|streptococcal)",
               "syphilis",
               "(tb|tuberculosis)",
               "tetanus",
               "(tss|toxic shock syndrome)",
               "toxo",
               "(toxoplasma|toxoplasmosis)",
               "trachoma",
               "trich",
               "trichomonas",
               "trichomoniasis",
               "trichonella",
               "trichonellosis",
               "tularemia",
               "typhoid( fever)?",
               "typhus fever",
               "vancomycin resistant staphylococcus aureus",
               "vcjd",
               "venezuelan hemorrhagic fever",
               "visa",
               "vrsa",
               "west nile fever",
               "west nile neuroinvasive disease",
               "(wnv|west nile virus)",
               "western equine encephalitis",
               "yeast infection",
               "yellow fever",
               "zika (fever|virus)",
               "(zoonosis|zoonoses|zoonotic infections)",
               "zoonotic influenza"}

def extract_terms_from_datasource(datasource, output_format):
    if datasource == 'projects':
        fields = ["grantAbstract"]
    elif datasource == 'papers':
        fields = ["pubmedKeywords", "meshTerms", "paperAbstract"]
    else:
        raise Exception("Datasource not found: ", datasource)

    if output_format == 'csv':
        line_prefix = ""
        column_delimiter = ","
        line_delimiter = "\n"
        pairs_header = "Term,Surrounding text, Text\n"
        counts_header = "Term,Count\n"
        footer = ""
    elif output_format == 'html':
        term_css_style = " .keyword {background-color: yellow; font-weight:bold;}"
        pairs_style = "<style>table,th,td {padding: 10px; border: 1px solid black; border-collapse: collapse;}" \
                      + term_css_style + "</style>"
        counts_style = "<style>table,th,td {border: 1px solid black; border-collapse: collapse;}</style>"
        begin_span = "<span class=\"keyword\">"
        end_span = "</span>"
        line_prefix = "<tr><td>"
        column_delimiter = "</td><td>"
        line_delimiter = "</td></tr>"
        pairs_header = "<html><head>" + pairs_style + "</head><body><table><tr><th>Term</th><th>Text</th></tr>"
        #pairs_header_surround = "<html><head>" + pairs_style + "</head><body><table><tr><th>Term</th><th>Surrounding text</th><th>Text</th></tr>"
        counts_header = "<html><thead>" + counts_style + "</thead><body><table><tr><th>Term</th><th>Count</th></tr>"
        footer = "</table></body></html>"
    else:
        raise Exception("Please supply csv or html as a parameter for the result file you would like.  Output format not found: ", output_format)

    documents = build_corpus_words_only(datasource, fields, do_stemming=False,
                                     do_remove_common=False)

    pairs_filename = 'output/term_pairs_' + datasource + '.' + output_format
    counts_filename = 'output/term_counts_' + datasource + '.' + output_format

    counts = {}
    tpp_f = open(pairs_filename, 'w')
    tpp_f.write(pairs_header)
    for searchTerm in searchTerms:
        print(searchTerm)
        for docText in documents["text"]:
            found_term = re.search(r"\b" + r"{}".format(searchTerm.lower()) + r"s?\b", docText)
            if found_term is not None:
                start_pos = max(0, found_term.span()[0] - 20)
                end_pos = min(len(docText), found_term.span()[1] + 20)
                if output_format == 'html':
                    tpp_f.write(line_prefix)
                    tpp_f.write(searchTerm + column_delimiter)
                    if "Surrounding" in pairs_header:
                        tpp_f.write(docText[start_pos:end_pos].strip() + column_delimiter)
                    start_idx = 0
                    for idx in re.finditer(r"\b" + r"{}".format(searchTerm.lower()) + r"s?\b", docText):
                        if idx is not None:
                            match_idx = max(start_idx, idx.span()[0])
                            end_idx = min(idx.span()[1], len(docText))
                            tpp_f.write(docText[start_idx:match_idx] + begin_span + docText[match_idx:end_idx] + end_span)
                            start_idx = end_idx
                    if start_idx < len(docText):
                        tpp_f.write(docText[start_idx:len(docText)])
                    tpp_f.write(line_delimiter)
                else:
                    tpp_f.write(searchTerm + column_delimiter + docText[start_pos:end_pos].strip() + column_delimiter + docText + line_delimiter)

                if searchTerm in counts.keys():
                    counts[searchTerm] += 1
                else:
                    counts[searchTerm] = 1
    tpp_f.write(footer)
    tpp_f.close()
    counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)

    f = open(counts_filename, 'w')
    f.write(counts_header)
    for item in counts:
        if output_format == 'html':
            f.write(line_prefix)
        f.write(item[0] + column_delimiter + str(item[1]) + line_delimiter)
    f.write(footer)
    f.close()

if __name__ == '__main__':
    result_format = sys.argv[1]
    extract_terms_from_datasource('projects', result_format)
    extract_terms_from_datasource('papers', result_format)
    exit(0)
