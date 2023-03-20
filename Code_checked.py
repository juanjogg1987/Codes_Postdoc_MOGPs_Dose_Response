#! /usr/bin/env python3

#The user should provide a nucleotide string
user_str = input("Please enter a nucleotide string:")

def stop_codon():

    rev_seq = user_str[::-1]
    rev_com = ""
    stop_found = False

    #The function will reverse the string given by the user and provide reverse complement

    for nt in rev_seq:
        if nt == "A":
            rev_com = rev_com + "T"
        elif nt == "T":
            rev_com = rev_com + "A"
        elif nt == "C":
            rev_com = rev_com + "G"
        elif nt == "G":
            rev_com = rev_com + "C"

    #The return value will be a Boolean. False in the case that there is no a stop codon or True if a codon is found.

    codon_forward = False
    codon_reverse = False
    for nt in range(0, len(user_str), 3):
        codon_f = user_str[nt:nt + 3]

        if codon_f == "TAA" or codon_f == "TAG" or codon_f == "TGA":
            codon_forward = True

    print(user_str, " stop codon ",codon_forward )

    for nt in range(0, len(rev_com), 3):
        codon_rc = rev_com[nt:nt + 3]

        if codon_rc == "TAA" or codon_rc == "TAG" or codon_rc == "TGA":
            codon_reverse = True

    print(rev_com," stop codon ", codon_reverse)

    return codon_forward, codon_reverse  #Return two a booleans: one for string givan and one for the string reversed

print(stop_codon())
