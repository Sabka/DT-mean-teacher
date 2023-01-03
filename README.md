# DT-mean-teacher

Kod je adaptovany z https://github.com/iSarmad/MeanTeacher-SNTG-HybridNet, poodstranovala som casti kt. nie su 
potrebne pre mean teachera a tie, ktore pouzivaju cudu, lebo ju nemám a hádzali erorry.

Chýba ešte priečinok data-local so samotným datasetom, kt. je ale veľmi veľký 500MB.

Ako si vytvoriť dataset?
- v priečinku data-local/bin spustíme `python3 unpack_cifar10.py . .`, tým sa dáta stiahnu
- pomocou skriptu `./data-local/bin/prepare_cifar10.sh`

Ako si vytvoriť dataset pre binary mean teachera?
- ( odporúčam si teraz premenovať `data-local` na `data-local-10-labels` a vytvoriť prázdny `data-local` )
- pôvodný `data-local` obsahoval dataset rozdelený na 10 labelov, ak chceme binárny, použijeme skript 
  `prepare_dataset_to_binMT.py`
   ( kde ak sme vynechali predošlý krok upravíme hodnoty `old = 'data-local-10-labels'` a `cur = 'data-local'  ` na cesty k priečinkom so starým datasetom a prázdnemu priečinku kde bude nový dataset, inak len spustíme)
- po spustení by mali byť dáta pripravené pre trénovanie binary mean teachera


Trénovanie vypisuje na konzolu, možno bude dobré posielať výstup niekam do súboru.
