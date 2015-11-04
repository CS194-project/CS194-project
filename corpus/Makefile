# FILES are the original files downloaded from the corpus website.
# FILES_BIG are the files which is the combination of 10 copies of correponding FILES.
# combination is the combination of all FILES_ORIGIN

FILES := dickens mozilla mr nci ooffice osdb reymont samba sao webster xml x-ray
FILES_BIG := $(addsuffix .big, $(FILES))

all: $(FILES) $(FILES_BIG) combined

$(FILES):
	wget -O- http://sun.aei.polsl.pl/~sdeor/corpus/$@.bz2 | bzip2 -d > $@

%.big: %
	echo $(FILES_BIG)
	for i in $$(seq 1 10); \
		do cat $< >> $@; \
	done;

combined: $(FILES)
	cat $^ >> combined

.PHONY: clean distclean
clean:
	rm -rf $(addsuffix .bz2,$(FILES)) $(addsuffix .gz,$(FILES)) $(addsuffix .gz,$(FILES_BIG)) combined.gz

distclean: clean
	rm -rf $(FILES) $(FILES_BIG) combined
