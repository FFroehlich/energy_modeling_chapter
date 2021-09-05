#!/bin/bash

jupyter nbconvert --to=latex chapter.ipynb
pdflatex chapter.tex
bibtex chapter.aux
pdflatex chapter.tex
pdflatex chapter.tex

rm *.bbl *.aux *.blg *.log *.out
