ascii:
	# find every file that contains non-ASCII characters
	# and convert these files to ASCII
	for file in `git grep -P -n -l "[\x80-\xFF]" -- "*.py"`; \
	do \
		iconv -f utf-8 -t ascii//translit $$file > temp.py && mv temp.py $$file; \
	done;

test:
	pytest -v --duration=10 alphacsc
