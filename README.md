# Metaphotical-ratings-LLMs-replication-study

Il codice contenuto nella repository è stato scritto per realizzare l'esperimento previsto dal mio progetto di tesi in Linguistica computazionale. "Il titolo della tesi è: Large Language Model e simulazione della cognizione umana: il caso della valutazione automatica di espressioni metaforiche". Qua di seguito espongo in breve il framework dell'esperimento e l'esperimento stesso, ma lascio per chi avesse voglia anche il testo integrale della tesi: [GitHub](https://github.com/orboboro/Metaphorical-ratings-LLMs-replication-study/integral_thesis_text.pdf)



La domanda di ricerca principale a cui la tesi si propone di rispondere consiste nel chiedersi se la capacità dei Large Language Models di simulare la cognizione umana sia arrivata a un punto tale che possiamo sensatamente pensare di sostituire i partecipanti umani negli esperimenti psicolinguistici con dei modelli di linguaggio.

Esiste un filone di studi già avviato che indaga questa possibilità si inserisce proprio in questo filone, e in cui la tesi si inserisce. La rilevanza di questo ambito di ricerca deriva dal fatto che la sostituzione integrale o parziale dei tester umani potrebbe giovare alla ricerca linguistica, perché abbatterebbe i costi e i tempi della raccolta dei dati, permettendo di creare basi di dati molto più grandi e in molto meno tempo.
![slide1](/presentation_slides/slide2.png)




Prima ancora di mettere alla prova empiricamente la capacità dei modelli di linguaggio, dobbiamo fare un passo indietro e chiederci: è possibile in linea di principio che questi modelli comprendano effettivamente il linguaggio cosi come noi umani lo facciamo?

Di fatto no, non è così: gli umani imparano la lingua interagendo con l'ambiente circostante e legando parole a percezioni o referenti; i modelli invece sono privi di un apparato sensoriale e sono addestrati sulla pura forma linguistica, sganciata dalla dimensione referenziale. Questo fa si che solo gli umani forniscano un grounding percettivo ai significati, mentre i modelli si costruiscono una rappresentazione distribuzionale dei significati. Per esempio, per un modello la parola "ape" non corrisponde a quell'essere che ronza, che è giallo e nero e che può farci provare dolore pungendoci, ma la parola "ape" è solo quella forma che spesso occorre con "fiore", "giallo", "alveare" etc.
![slide1](/presentation_slides/slide3.png)



Ad ogni modo, se anche i modelli di linguaggio non capiscono i significati delle parole, ci si può comunque chiedere quanto bene possano simulare di farlo. In particolare, nell'esperimento riportato nella tesi ci si occupa di significati metaforici. Il fine era capire quanto le valutazioni semantiche prodotte dai certi modelli di linguaggio su certe espressioni metaforiche fossero vicine a quelle prodotte da umani sulle stesse espressioni.

Le espressioni in questione e i corrispondenti giudizi umani sono stati tratti dal Figurative Archive. Per quanto riguarda i modelli, ne sono stati scelti 2 della famiglia di Meta: Llama 3.3 con 70 miliardi di parametri e Llama 3.1 con 8 miliardi di parametri, entrambi nella loro versione instruction tuned.

Nella pratica l'esperimento è consistito nel replicare la raccolta dei giudizi semantici sulle metafore del Figurative Archive usando i suddetti modelli. I giudizi sintetici sono poi stati confrontati con quelli umani. Passiamo allora a i principali risultati dell'analisi.
![slide1](/presentation_slides/slide4.png)



Innanzitutto si è voluto escludere l'ipotesi di data leakage, ovvero il fenomeno che consiste nella sovrapposizione fra test set e training set. Se infatti un modello viene testato su dati che compaiono già nel suo addestramento, allora anche se ottiene delle buone performance non possiamo escludere che si stia solo 'ricordando' per così dire dei dati già visti, invece di generalizzare ciò che ha imparato durante l'addestramento.

Per scongiurare questo pericolo si sono divise le espressioni metaforiche tra quelle già comparse in studi precedenti al knowledge cutoff dei modelli e quelle invece inedite, ovvero post cutoff. I risultati ottenuti permettono di escludere l'ipotesi di data leakage, poiché non solo i coefficienti di correlazione su item pre-cutoff non sono più alti di quelli su item-post cutoff, ma sono addirittura sistematicamente più bassi.
![slide1](/presentation_slides/slide5.png)



In tutte e 6 le dimensioni sotto indagine, sia Llama 70 che Llama 8 hanno prodotto rating che correlavano pósitivamente con quelli umani. Tuttavia quasi tutti i coefficienti, restano nella fascia medio-bassa. La sola eccezione sono i coefficienti relativi alla body relatedness, che si collocano nella fascia alta, con quello di Llama 70 che supera la soglia ideale dello 0.8, per altro con un p-value molto basso. Questo risultato lascia adito all'ipotesi secondo cui una rappresentazione puramente distribuzionale dei significati può comunque risultare efficace anche in task che richiedono la gestione di informazioni embodied, come è appunto giudicare la body relatedness di una metafora.
![slide1](/presentation_slides/slide6.png)



Tuttavia, c'è una parte di risultati che racconta un'altra storia. Usando come spartiacque il valore mediano dei rating umani per la dimensione di body relatedness, si sono distinte le metafore molto embodied e quelle poco embodied, A questo punto si è osservato se ci fosse una differenza nelle correlazioni tra i due gruppi. E' emerso che gli item metaforici molto
embodied mettono in crisi il modello, che produce giudizi scarsamente correlati a quelli umani; il modello correla invece molto meglio sulle metafore giudicate poco body related dagli umani. Alla luce di questi dati sembra allora che effettivamente il fatto che i modelli non abbiano una rappresentazione anche embodied dei significati mini la loro capacità di replicare il comportamento verbale umano.

Un'ulteriore evidenza di questo sta nella differenza tra il valore mediano di body relatedness negli esseri umani e nei modelli: negli umani è 3.81 (praticamente a metà della scala Likert). mentre per il modello è 1.63. Ciò significa che il modello giudiça metà delle metafore molto poco corporee. Evidentemente, quindi, c'è una discrepanza significativa nella percezione di questa dimensione semantica.
![slide1](/presentation_slides/slide7.png)



L'ultima tappa dell'analisi dei dati è consistita in uno studio di sostituzione. E in questo caso la domanda centrale è questa: se svolgiamo l'analisi dei dati utilizzando i dati sintetici, arriviamo a conclusioni diverse rispetto a quelle a cui arriviamo usando i dati umani?

Per rispondere a questa domanda ho confrontato le matrici di correlazione tra dimensioni per i dati umani con le matrici di correlazione sintetiche. Nella slide per semplicità ho riportato le matrici correlazioni relative solo a uno degli studi replicati, ma le stesse osservazioni si applicano anche agli altri. Se guardiamo i coefficienti vediamo che i dati sintetici replicano in
modo coerente la struttura delle relazioni dei dati reali. Il segno delle correlazioni infatti viene sempre mantenuto, anche se i coefficienti sintetici tendono ad essere più estremi. Ad esempio, sia dai dati umani che da quelli sintetici emerge che c'è una correlazione negativa tra familiarità e difficoltà, ma se nel caso dei dati umani it coefficiente di correlazione è -0.60, nei dataset sintetici è -0.95.
![slide1](/presentation_slides/slide8.png)



Curiosamente confrontando metafore letterarie e di uso quotidiano si osserva una dinamica opposta, ma in fondo coerente. Cioè anche qui i dati sintetici či dicono fondamentalmente le stesse cose dei dati umani: le metafore d'uso quotidiano sono più familiari, più ricche di significato e meno difficili da capire rispetto a quelle letterarie. In questo caso però i dati sintetici non accentuano le differenze, ma le attenuano.
![slide1](/presentation_slides/slide9.png)



Traiamo a questo punto le conclusioni:

Quanto al problema del data leakage, i dati raccolti sembrano scongiurario. Tuttavia, non si può escludere del tutto un effetto di familiarità con il tipo di task, visto che questo task valutativo risultava già ampiamente documentato prima del knowledge cutoff.

Quanto alla questione della competenza embodied dei modelli, osserviamo che da un lato i modelli mostrano una sensibilità sorprendente verso la dimensione di body relatedness; dall'altro lato, però, proprio sugli item più embodied i giudizi sintetici riportano gli scarti maggiori rispetto a quelli umani. Quindi quello che possiamo trarre è questo: la rappresentazione distribuzionale dei significati sembra rendere i modelli capaci di recuperare certe informazioni embodied dalla forma linguistica, ma non sostituisce pienamente un grounding percettivo.

Veniamo poi alla questione centrale, quella che poneva la domanda di ricerca da cui siamo
partiti: quanto è fattibile la sostituzione dei partecipanti umani con dei modelli linguistici?

Allora, abbiamo visto che modelli di linguaggio possono produrre giudizi che convergono con quelli umani, ma tale convergenza non è ancora abbastanza forte né uniforme lungo tutte le dimensioni del significato perché sia legittimo usare in modo generalizzato i janguage models in esperimenti psicolinguistici. L'unica correlazione al di sopra della sogli ideale infatti è stata quella di Llama 70 per la body relatedness.
![slide1](/presentation_slides/slide10.png)

Inoltre, lo studio di sostituzione ha dimostrato che i modelli tendono a preservare la struttura relazionale dei dati umani, pur amplificando o attenuando l'intensità di queste relazioni.

Nella pratica, quindi, la prospettiva che si delinea non è tanto quella di una sostituzione, ma di una integrazione controllata, mantenendo come condizione incrollabile quella di non interrompere mai il confronto con i dati umani, perché una ricerca sperimentale condotta esclusivamente su dati sintetici rischierebbe di portare alla deriva le nostre teorie sull'elaborazione linguistica e sulla cognizione umana in generale.
