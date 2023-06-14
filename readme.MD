## To run:

1. cd project directory
2. source venv/bin/activate
3. python update_data.py
3. python term_counter.py (csv|html)

## In Pycharm:
Make sure that you are using the correct python interpreter, specifically the one in the virtual environment.

To do this, go to:

Pycharm -> Settings -> Project -> Python Interpreter -> Add -> Existing environment -> Interpreter

and set it to project_dir/venv/bin/python

### Todo for Jeff:
~~
1. Update update_data.py to grab grants from the database
2. Update extraction_utils.build_corpus_words_only to accept a filename as a parameter instead of using the hardcoded papers.json
3. Update term_counter.py to pass in the filename to extraction_utils.build_corpus_words_only
4. Update term_counter.py to pass in the correct fields (field names in the grants.json that you want to extract text from
5. change the output file names to something better, and include the datasource in the name (e.g. "term_counts_grants.json")
6. change do_manual() to something more descriptive (e.g. extract_terms_from_datasource)
7. call do_manual() once for papers, and once for grants
8. Look in our regex term list in term_counter.py and try to group diseases into a single line.  For example:
   1. "lassa fever" and "lassa hemorrhagic fever" are counted seperately.  A suggested improvement would be:
   2. "(lassa fever|lassa hemorrhagic fever)"
   3. Another example could be "meningococcal disease", and "meningococcal", perhaps (meningococcal disease|meningococcal) is better
   4. Verify these combos with Alice?
9. Instead of CSV output, add HTML output.  In the HTML, instead of printing the 2nd column in the huge file, just highlight the search terms in RED or something.  Have fun with it.
   1. I'm going to oversimplify this probably, but really you could just write:
~~
```
 <html><body><table><tr><th>search term></th><th>result</th></tr>
 then for every term that is a match: <tr><td>term</td><td>this is <span class="highlight">all</span> the text</td></tr>
 finally: </table></body></html>
 ```
