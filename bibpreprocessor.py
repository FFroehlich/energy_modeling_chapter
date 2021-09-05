"""This preprocessor replaces bibliography code in markdowncell
"""

#-----------------------------------------------------------------------------
# Copyright (c) 2015, Julius Schulz
#
# Distributed under the terms of the Modified BSD License.
#
#-----------------------------------------------------------------------------

from nbconvert.preprocessors import *
import re
import os
import sys


class BibTexPreprocessor(Preprocessor):

    def create_bibfile(self, filename):
        if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))
        f = open(filename, "w")
        for r in self.references:
            f.write(create_bibentry(r, self.references[r]))
        f.close()

    def preprocess(self, nb, resources):
        try:
          self.references = nb["metadata"]["cite2c"]["citations"]
          self.create_bibfile(
              f'{resources["output_files_dir"]}/{resources["unique_key"]}.bib'
          )
        except KeyError:
          print("Did not find cite2c")
        for index, cell in enumerate(nb.cells):
            nb.cells[index], resources = self.preprocess_cell(cell, resources, index)
        return nb, resources

    def preprocess_cell(self, cell, resources, index):
        """
        Preprocess cell

        Parameters
        ----------
        cell : NotebookNode cell
            Notebook cell being processed
        resources : dictionary
            Additional resources used in the conversion process.  Allows
            preprocessors to pass variables into the Jinja engine.
        cell_index : int
            Index of the cell being processed (see base.py)
        """
        if cell.cell_type == "markdown":
            if "<div class=\"cite2c-biblio\"></div>" in cell.source:
                replaced = re.sub("<div class=\"cite2c-biblio\"></div>", r"\\bibliography{"+resources["output_files_dir"]+"/"+resources["unique_key"]+r"} \n ", cell.source)
                cell.source = replaced
        return cell, resources

def create_bibentry(refkey, reference):
    entry = "@article{" + refkey + ",\n"

    entry += "  author = {"
    entry += " and ".join(map(lambda a: a["family"] + ", " + a["given"],
                              reference["author"]))
    entry += "}, \n"

    if "title" in reference:
        entry += f"  title = {{{reference['title']}}}, \n"
    if "issued" in reference:
        entry += f"  year = {{{reference['issued']['year']}}}, \n"
    if "container-title" in reference:
        entry += f"  journal = {{{reference['container-title']}}}, \n"
    if "page" in reference:
        pages = re.sub('-', '--', reference['page'])
        entry += f"  pages = {{{pages}}}, \n"
    if "volume" in reference:
        entry += f"  volume = {{{reference['volume']}}}, \n"
    if "issue" in reference:
        entry += f"  issue = {{{reference['issue']}}}, \n"
    if "DOI" in reference:
        entry += f"  doi = {{{reference['DOI']}}}, \n"

    entry += "}\n\n"
    return entry
