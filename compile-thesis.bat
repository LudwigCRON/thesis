@ECHO off

REM use %APPDATA% is more appropriate here
SET miktex_path=%USERPROFILE%/AppData/Local/Programs/MiKTeX 2.9/bin

ECHO "Compiling your PhD Thesis...please wait...!"
CALL %miktex_path%/pdflatex.exe -interaction=nonstopmode thesis.tex
CALL %miktex_path%/biber.exe thesis
CALL %miktex_path%/makeindex.exe thesis.aux
CALL %miktex_path%/makeindex.exe thesis.idx
CALL %miktex_path%/makeindex.exe thesis.nlo -s nomencl.ist -o thesis.nls
CALL %miktex_path%/pdflatex.exe -interaction=batchmode thesis.tex
CALL %miktex_path%/makeindex.exe thesis.nlo -s nomencl.ist -o thesis.nls
CALL %miktex_path%/pdflatex.exe -synctex=1 -interaction=batchmode thesis.tex
ECHO "Success!"
EXIT
