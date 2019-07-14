.PHONY: amr amr/lpps.pdf psd/lpps.pdf evaluation txt repp udpipe clean release update

%.mrp: %.amr
	../mtool/main.py --read amr --write mrp $< $@;

amr: amr/amr-consensus.mrp amr/bolt.mrp amr/dfa.mrp amr/lorelei.mrp \
     amr/proxy.mrp amr/xinhua.mrp amr/lpps.mrp

amr/lpps.mrp: amr/lpp.amr lpps.ids lpps.txt
	for i in $$(cat lpps.ids); do \
	  ../mtool/main.py --text lpps.txt --quiet --read amr \
	    --id $$i --write mrp amr/lpp.amr; \
	done > $@;

amr/lpps.pdf: psd/lpps.mrp
	for i in $$(cat lpps.ids); do \
	  ../mtool/main.py --text lpps.txt --read mrp \
	    --id $$i --write dot \
	    ./amr/lpps.mrp ./amr/dot/$$i.dot; \
	done
	for i in ./amr/dot/*.dot; do \
	  j=$$(basename $$i .dot); \
	  dot -Tpdf $$i > ./amr/pdf/$${j}.pdf; \
	done

dm/brown.mrp: dm/brown.sdp brown.txt
	../mtool/main.py --text brown.txt --read dm \
	  --write mrp ./dm/brown.sdp $@;

dm/lpps.mrp: dm/lpps.sdp lpps.txt
	../mtool/main.py --text lpps.txt --read dm \
	  --write mrp ./dm/lpps.sdp $@;

dm/wsj.mrp: dm/wsj.sdp wsj.txt
	../mtool/main.py --text wsj.txt --read dm \
	  --write mrp ./dm/wsj.sdp $@;

eds/brown.mrp: eds/brown.eds wsj.txt
	../mtool/main.py --text brown.txt --read eds \
	  --write mrp $< $@;

eds/lpps.mrp: eds/lpps.eds lpps.txt
	../mtool/main.py --text lpps.txt --read eds \
	  --write mrp $< $@;

eds/wsj.mrp: eds/wsj.eds wsj.txt
	../mtool/main.py --text wsj.txt --read eds \
	  --write mrp $< $@;

psd/brown.mrp: psd/brown.sdp brown.txt
	../mtool/main.py --text brown.txt --read psd \
	  --write mrp $< $@;

psd/lpps.mrp: psd/lpps.sdp lpps.txt
	../mtool/main.py --text lpps.txt --read psd \
	  --write mrp $< $@;

psd/wsj.mrp: psd/wsj.sdp ../training/wsj.txt
	../mtool/main.py --text ../training/wsj.txt --read psd \
	  --write mrp $< $@;

psd/lpps.pdf: psd/lpps.mrp
	for i in $$(cat lpps.ids); do \
	  ../mtool/main.py --text lpps.txt --read mrp \
	    --id $$i --write dot \
	    ./psd/lpps.mrp ./psd/dot/$$i.dot; \
	done
	for i in ./psd/dot/*.dot; do \
	  j=$$(basename $$i .dot); \
	  dot -Tpdf $$i > ./psd/pdf/$${j}.pdf; \
	done

ucca/ewt.mrp: ucca/ewt.txt ucca/ewt/xml/*
	../mtool/main.py \
	  --text ucca/ewt.txt --read ucca --write mrp \
	  ./ucca/ewt/xml/files.txt ./ucca/ewt.mrp;

ucca/lpps.mrp: lpps.txt ucca/lpps/xml/*
	../mtool/main.py \
	  --text lpps.txt --read ucca --write mrp \
	  ./ucca/lpps/xml/files.txt ./ucca/lpps.mrp;

ucca/wiki.mrp: ucca/wiki.txt ucca/wiki/xml/*
	../mtool/main.py \
	  --text ucca/wiki.txt --read ucca --write mrp \
	  ./ucca/wiki/xml/files.txt ./ucca/wiki.mrp;

input.mrp: amr/amr-consensus.mrp amr/bolt.mrp amr/dfa.mrp \
            amr/lorelei.mrp amr/lpps.mrp amr/proxy.mrp amr/xinhua.mrp \
	    dm/brown.mrp dm/lpps.mrp dm/wsj.mrp \
	    eds/brown.mrp eds/lpps.mrp eds/wsj.mrp \
	    psd/brown.mrp psd/lpps.mrp psd/wsj.mrp \
	    ucca/ewt.mrp ucca/lpps.mrp ucca/wiki.mrp
	../mtool/main.py --read mrp --source lpps --write evaluation \
	  amr/lpps.mrp > input.mrp
	for i in wsj brown; do \
	  ../mtool/main.py --read mrp --source $${i} --write evaluation \
	    dm/$${i}.mrp; \
	done >> input.mrp
	for i in ewt wiki; do \
	  ../mtool/main.py --read mrp --source $${i} --write evaluation \
	    ucca/$${i}.mrp; \
	done >> input.mrp
	for i in amr-consensus bolt dfa lorelei proxy xinhua; do \
	  ../mtool/main.py --read mrp --source $${i} --write evaluation \
	    amr/$${i}.mrp; \
	done >> input.mrp

txt: input.mrp
	../mtool/main.py --read mrp --write txt $< $@;
	split -l 1000 -d -a 1 input.txt input;
	for i in input?; do mv $${i} $${i}.txt; done

repp:
	for i in input?.txt; do \
	  name=$$(basename $$i .txt); \
	  ( cd ../companion; \
	    ./repp.py ../evaluation/$${name}.txt | \
	    awk -F\\t '/^#/ {print; i = 1;} \
	               /^[0-9]/ {printf("%d\t%s\t_\t_\t_\t_\t_\t_\t_\t_\n", i++, $$3);} \
	               /^$$/ {print;}' \
	    > ../evaluation/$${name}.conllu; ) \
	done

udpipe:
	for i in {0..6}; do sbatch parse.slurm input$${i}.conllu udpipe$${i}.conllu; done

udpipe.mrp: udpipe?.conllu
	for i in udpipe?.conllu; do \
	  ../mtool/main.py --read conllu --write mrp \
	    --text input.txt $${i}; \
	done > $@;

jayeol:
	tar zpScvf ../jayeol.tgz amr/*.amr udpipe.mrp udpipe?.conllu
	scp ../jayeol.tgz sh.hpc.uio.no:/ltg/www/nlpl/mrp

amr/lpps.amr:
	( cd amr; ./filter.awk lpp.amr > lpps.amr; )

jamr.mrp: jamr/*.txt amr/lpps.amr
	for i in ./jamr/*.txt; do \
	  name=$$(basename $${i} .txt); \
	  ../mtool/main.py --read amr \
	    --alignment $${i} --overlay jamr/$${name}.mrp \
	    ./amr/$${name}.amr; \
	done;
	cat jamr/*.mrp > jamr.mrp;

isi.mrp: isi/*.txt
	for i in ./isi/*.txt; do \
	  name=$$(basename $${i} .txt); \
	  ../mtool/main.py --read amr \
	    --alignment $${i} --overlay isi/$${name}.mrp \
	    ./amr/$${name}.amr; \
	done;
	cat isi/*.mrp > isi.mrp;

clean:
	rm amr/dot/*.dot amr/pdf/*.pdf psd/dot/*.dot psd/pdf/*.pdf \
	  input.mrp input.txt input?.txt input?.conllu udpipe?.conllu udpipe?.conllu.npz

release:
	cp README.txt ../public/evaluation/README.txt;
	tar zpScvf ../evaluation10.tgz --transform='s@^@mrp/2019/evaluation/@'\
	  README.txt Makefile \
	  input.mrp udpipe.mrp;
	scp ../evaluation10.tgz sh.hpc.uio.no:/ltg/www/nlpl/mrp/ldc;


