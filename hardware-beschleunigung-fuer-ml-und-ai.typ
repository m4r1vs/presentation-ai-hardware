#let title-page(title: [], subtitle: [], fill: yellow, body) = {
  // set page( numbering: "1")
  set page(margin: (top: 1.5in, rest: 2in))
  set text(font: "Ubuntu", size: 10pt, lang: "de", top-edge: 0.85em)
  set heading(numbering: "1.1.1")
  line(start: (0%, 0%), end: (8.5in, 0%), stroke: (thickness: 2pt))
  align(horizon + left)[
    #text(size: 21pt, title)\
    #v(1em)
    #subtitle
  ]

  align(bottom + left)[
    #link("mailto: marius.niveri@studium.uni-hamburg.de")[Marius Niveri]\
    #datetime(year: 2025, month: 03, day: 31).display()
  ]
  pagebreak()
  set page(
    fill: none,
    margin: auto,
    footer: context [
      #text(
        size: 8pt,
        [
          #title\
          #link("mailto: marius.niveri@studium.uni-hamburg.de")[Marius Niveri]
          #h(1fr)
          #counter(page).display(
            "1",
            both: false,
          )
        ],
      )
    ],
  )
  align(horizon, outline(indent: auto))
  pagebreak()
  body
  pagebreak()
  bibliography("bib.yaml")
}

#show: body => title-page(
  title: [Hardware-Beschleunigung für ML/AI: GPUs und TPUs],
  subtitle: [
    Seminar Supercomputer: Forschung und Innovation\
    Bei
    #link("mailto: anna.fuchs@uni-hamburg.de")[Anna Fuchs]
    und
    #link("mailto: jannek.squar@uni-hamburg.de")[Jannek Squar]
  ],
  body,
)

#set par(spacing: 2em, justify: true)

= Einführung
Der Markt für Chips zum Ausführen von KI- und ML-Programmen (der Einfachheit halber bei "KI" mitgemeint) wird aktuell stark
von hochparallelisierbaren Prozessoren dominiert. Dazu zählen die "Graphical
Processing Units" (GPU) aber auch andere teilweise noch spezialisiertere
Hardware wie etwa die "Tensor Processing Unit" (TPU).

In dieser Ausarbeitung evaluieren wir den Nutzen und die Wirtschaftlichkeit der GPU im Vergleich zur CPU.
Dafür analysieren wir kurz jeweils die Laufzeit von unterschiedlichen KI- und nicht-KI Algorithmen.
Außerdem klären wir, wodurch die Dominanz von NVIDIA über ihren Konkurrenten AMD zu begründen ist.

== Motivation
Die der KI zugrundeliegenden Algorithmen worden entwickelt, bevor die Hardware eine angemessene Geschwindigkeit
erlaubte. #footnote[z.B. der 1958 von Frank Rosenblatt veröffentlichte "Perceptron" Algorithmus] Dies zeigt, dass gerade der Fortschritt in der Hardware ein Maßgeblicher Treiber des Fortschrittes in der
KI allgemein ist. Hinzu kommen die Milliardeninvestitionen des privaten und öffentlichen Sektors@MicrAcq500Mio -- man verspricht sich
bessere Modelle durch rechenintensiveres Training zusätzlich zur Optimierung der Software.

= Warum GPUs?
Wie bereits aus anderen Voträgen in diesem Modul bekannt ist, sind Matrix-Matrix- und Vektor-Matrix-Multiplikationen die mit großem Abstand häufigsten und rechenintensivsten
Berechnungen, wenn KI-Algorithmen ausgeführt werden. In der Vergangenheit standen diese Berechnungen meist im Kontext der Rastergrafiken (z.B. VFX, Gaming, Bildbearbeitung, etc.).
Daher nennen wir hierfür optimierte Chips meist _Grafik_-Prozessor.

Im Detail unterscheiden sich jedoch die Anforderungen. Zum Beipsiel sind die Spitzenreiter der (multimodalen) Sprach-modelle häufig weit über 100GiB groß @DeepSeekDepl.
Damit diese ohne ineffiziente Auslagerung in den CPU-RAM laufen können, muss die Hardware also eine entsprechende Kapazität des dedizierten Speichers (VRAM) aufweisen.

Dies hat zur Entwicklung von speziell für KI optimierten Hardware geführt. Die Bezeichnung unterscheidet sich je nach Hersteller; Tensor Processing Unit (TPU) ist
sehr gängig.

== GPU Architektur

Hardware, die auf Matrix-Rechnungen optimiert ist, profitiert in erster Linie von der sehr guten Parallelisierbarkeit dieser Berechnungen.

*Erinnerung*:

$
  text("Gegeben") arrow(v) = mat(v_1; v_2) text("und") M = mat(M_(1,1), M_(1,2); M_(2,1), M_(2,2))
$
$
  text("gilt:  ") arrow(v) dot M = mat(v_1 dot M_(1,1) +v_2 dot M_(2,1); v_1 dot M_(1,2) + v_2 dot M_(2,2))
$

Die einzelnen Zellen im Produkt sind alle voneinander unabhängig. Das Beispiel ist also mit 4 Rechnern berechenbar und jedem Rechner reicht eine Teilmenge der Eingabewerte aus.
In der Praxis sind die Matritzen hochdimensional und die Anzahl an Zellen (z.B. Parameter) wird in Milliarden datiert.

#pagebreak()

#figure(
  image("./cpu-gpu-mem.png", width: 80%),
  caption: [Vergleich einer CPU mit einer GPU],
)<cpu-gpu-mem>

Der entscheidende Vorteil in der Matritzenberechnung ist also die Anzahl an Kernen, die parallel rechnen können. Eine GPU besteht aus mehreren unabhängigen SMs (Streaming Multiprocessor),
die wiederum aus sogentannten Kerneln bestehen. Innerhalb eines SMs können alle Kernel nur eine selbe Berechnung durchführen -- alerdings mit je eigenen Zeigern auf die
Daten im gemeinsamen Speicher. Hat eine GPU also z.B. 1024 Kerne (Kernel), so heißt das nicht, dass sie tatsächlich 1024 verschiedene Berechnungen durchführen kann.

Der Vorteil einer CPU ist also in ihrer Fähigkeit tatsächlich diverse Berechnungen parallel durchzuführen während eine GPU wenige verschiedene Berechnungen höchstparallel
berechnen kann.

== Warum NVIDIA?
NVIDIA dominiert den KI-Computing-Markt@NvTheChipBBC vor allem durch CUDA@cudaDocs, einer Schnittstelle, die speziell für die hauseigenen GPUs entwickelt wurde.
CUDA ist tief in KI-Frameworks wie TensorFlow und PyTorch integriert und bietet eine Vielzahl von Optimierungen, die für rechenintensive Anwendungen entscheidend sind.
Diese enge Verzahnung von Hardware und Software hat NVIDIA einen enormen Wettbewerbsvorteil verschafft.

AMD setzt mit ROCm@rocmDocs auf einen Open-Source-Ansatz und bietet damit eine Alternative, die mehr Flexibilität bei der Hardware-Nutzung verspricht. Allerdings ist ROCm in der Praxis
oft nicht so ausgereift wie CUDA, was zu Performance-Nachteilen führt. Auch die Hardware-Unterstützung ist begrenzter: Während CUDA mit praktisch allen modernen NVIDIA-GPUs
funktioniert, läuft ROCm nur auf ausgewählten AMD-Grafikkarten und bevorzugt unter Linux.

Obwohl ROCm durch seine Offenheit Potenzial hat und langfristig als ernsthafte Alternative wachsen könnte, bleibt CUDA aktuell die dominierende Lösung. Vor allem Unternehmen
und Forschungseinrichtungen setzen weiterhin auf NVIDIA, gerade auch weil die Dokumenation und der Community-Support aktuell deutlich stärker ist.

Daheim wird auch gerne auf Apple Hardware gesetzt, da sich die GPU auf den neueren Geräten den RAM mit der CPU teilt. Dadurch ist der Preis relativ günstig gemessen am
VRAM, den die Computer zur Verfügung stellen.@AppleMacUnifiedAI

= Unterschiede in den Modellen
Verschiedene KI-Algorithmen unterscheiden sich in ihrer Parallelisierbarkeit. So werden die klassischen _Feedforward_ Netze fast ausschließlich
durch Vektor-Matrix-Multiplikationen berechnet. Außerderm sind die Trainingsdaten meist voneinader unabhängig und können auf einem Rechencluster mit verteiltem Speicher
verarbeitet werden. Aus den berechneten Gewichten muss zwar ein gemeinsamer Mittelwert gebildet werden, dieser muss aber nicht stetik synchron sein.

Eine zusätzliche Option zur Optimierung ist die Aufteilung der Ebenen auf einzelne Prozessoren. Gregor Stange hat diese Möglichkeiten in seinem Vortrag im Detail verglichen.

Fast gegensätzlich zu den Feedforward Netzen stehen die rekurrenten neuronalen Netze. Zur Berechnung eines Gewichtes durch den Gradienten ist nicht nur das Ergebnis der
vorgeschalteten Ebene notwendig sondern auch die auf einer Ebene vorgeschalteten Gradienten. @fig-hog-rnn veranschaulicht diese Abhängigkeit.

#figure(
  image("Hopfield-net-vector.svg", width: 25%),
  caption: [RNN mit 4 Neuronen @HopRNNSVG],
)<fig-hog-rnn>

Rekurrente Netze sind gerade aufgrund ihrer schlechten Parallelisierbarkeit weniger weit verbreitet als die etwas ähnlichen Transformer Netze.
2017 wurde durch das Papier "Attention Is All You Need" @AttnIsAllYouNeed der Grundstein für effizient berechenbare Transformer gelegt. Im Jahr darauf
wurde dann GPT-1 (Generative Pretrained Tranformer) @GPT1GitHub veröffentlicht -- die Grundlage von ChatGPT.

Es wurde eine sogenannte _Aufmerksamkeitsfunktion_ vorgestellt, die parallel auf verschiedene Teilmengen der Eingabedaten ausgeführt werden kann
und dadurch sehr gut gelichzeitig auf einer GPU laufen laufen kann. #footnote[Eine grobe Übersicht der Funktionsweise steht auf
  meinen #link("https://github.com/m4r1vs/presentation-ai-hardware")[#underline[Folien (Seite 9-11)]]]

= Vergleich von Hardware

Um die thoeretischen Vermutungen, dass GPUs ausgezeichnet in der Matrixmultiplikation sind. CPUs aber dennoch in gewissen Algorithmen vorne liegen, habe
ich vier verschiedene Anwendungen auf einer GPU und zwei CPUs laufen lassen und die durchschnittliche Laufzeit notiert:

#set par(spacing: 2em, justify: false)

#set table(
  stroke: none,
  gutter: 0.2em,
  fill: (x, y) => if x == 0 or y == 0 { gray },
  inset: (right: 1.5em),
)

#show table.cell: it => {
  if it.x == 0 or it.y == 0 {
    set text(white)
    strong(it)
  } else if it.body == [] {
    pad(..it.inset)[_N/A_]
  } else {
    it
  }
}

#table(
  columns: 4,
  [], [NVIDIA Quadro P2000], [Intel Xeon E-2176M], [Intel Core i5-1345U],
  [Matrixmultiplikation #footnote[Zufällig initialisierte $1000 times 1000$-Matrix mit $1000$ Iterationen. Siehe #link("https://github.com/m4r1vs/presentation-ai-hardware")[#underline[hier]] für den Quellcode.]],
  [$1.81$ Sekunden],
  [$99.12$ Sekunden],
  [$42.07$ Sekunden],

  [Phi-4 LLM #footnote[Microsoft Phi-4 @Phi4 mittels Ollama]],
  [$3.91$ TPS #footnote[Tokens pro Sekunde]<tps>],
  [$3.64$ TPS #footnote(<tps>)],
  [$5.23$ TPS #footnote(<tps>)],

  [Iterativer DFS #footnote[Depth-First-Search in Graf mit $25$ tausend Knoten]<dfs>],
  [$2.85$ Sekunden],
  [$1.28$ Sekunden],
  [$0.84$ Sekunden],

  [Rekursiver DFS #footnote(<dfs>)],
  [$-$ _unmöglich_ #footnote[Kernel kann keinen Kernel erschaffen.]],
  [$1.19$ Sekunden],
  [$0.85$ Sekunden],
)

#set par(spacing: 2em, justify: true)

#pagebreak()

Wie erwartet, hat die GPU die CPU derselben Generation in der Matrixmultiplikation deutlich übertroffen. Das Ergebnis überträgt sich allerdings gar nicht auf den praktischen
Einsatz eines Large Language Models. Hier ist die GPU nur geringfügig schneller. Eine Erklärung könnte der knappe VRAM von 2GB im Gegensatz zu den 64GB CPU RAM sein.
Allerdings erhöht sich ser Vorsprung auch bei anderen kleineren Modellen kaum.

Auf modernerer Hardware und GPUs mit mehr VRAM ist das Ergebnis wieder unseren Erwartungen entsprechender. So Liefert eine AMD Ryzen 9 8945HS CPU 11 TPS während die
ungefähr gleichwertige NVidia RTX 4070 Mobile zwischen 40 und 50 TPS schnell ist. @LLMbenchmarkAMDNVDA @BenchPerfGPU

= Fazit

Wenn der aktuelle Trend in Richtung KI-gestützter Anwendungen fortschreitet, wird auch die Hardware Nachfrage weiter steigen. Durch die hohen GPU Kosten, ihre Knappheit
und zusätzlich hohen Energieverbrauch gibt es allerdings auch ein zunehmendes Streben nach kleineren und weniger intensiven KI Modellen. Deswegen hat zuletzt das von DeepSeek entwickelte
R1 Modell Schlagzeilen gemacht.

Speziell NVidia hat sich als Gewinner herauskristalisiert, da sie mit CUDA eine reife Schnittstelle zur Programmierung von Grafikkarten bereitstellen. Dadurch bieten
zahlreiche High-Level Bibliotheken eine starke Hardware-Beschleunigung, die aktuell allerdings oft nur richtig mit GPUs von NVidia funktioniert.

Dennoch verbreitet sich GPUs von allen Herstellern aus immer weiter in Rechencentern @GPUDataCenterShare und wird sie wird in PCs häufiger in "Neural Chip" umgetauft @CopilotPC,
bzw. es wird dedizierte KI-Hardware verbaut und vermarktet.
Ursprünglich für die Bildverarbeitung entwickelt, wird die GPU durch die Verbreitung von KI immer unersetzlicher für die Verarbeitung von allen Daten im Internet.
