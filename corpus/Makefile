# FILES_ORIGIN are the original files downloaded from the corpus website.
# FILES_BIG are the files which is the combination of 10 acopies of correponding FILES_ORIGIN.
# combination is the combination of all FILES_ORIGIN

FILES := dickens mozilla mr nci ooffice osdb reymont samba sao webster xml x-ray
FILES_ORIGIN := $(addsuffix .orig, $(FILES))
FILES_BIG    := $(addsuffix .big, $(FILES))

all: $(FILES_ORIGIN) $(FILES_BIG) combined

$(FILES_ORIGIN):
	echo generating $@
	wget -O- http://sun.aei.polsl.pl/~sdeor/corpus/$(subst .orig,,$@).bz2 | bzip2 -d > $@

$(FILES_BIG): %.big : %.orig
	echo generating $@
	for i in $$(seq 1 10); \
		do cat $< >> $@; \
	done;

combined: $(FILES_ORIGIN)
	echo generating $@
	cat $^ >> combined

.PHONY: clean distclean
clean:
	rm -rf $(addsuffix .bz2,$(subst .orig,,$(FILES_ORIGIN))) $(addsuffix .gz,$(FILES_ORIGIN)) $(addsuffix .gz,$(FILES_BIG)) combined.gz

distclean: clean
	rm -rf $(FILES_ORIGIN) $(FILES_BIG) combined
