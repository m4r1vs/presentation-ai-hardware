#let title-page(title: [], subtitle: [], fill: yellow, body) = {
  // set page( numbering: "1")
  set page(margin: (top: 1.5in, rest: 2in))
  set text(font: "EB Garamond", size: 10pt, lang: "de")
  set heading(numbering: "1.1.1")
  line(start: (0%, 0%), end: (8.5in, 0%), stroke: (thickness: 2pt))
  align(horizon + left)[
    #text(size: 21pt, title)\
    #v(1em)
    #subtitle
  ]

  align(bottom + left)[
    #link("mailto: marius.niveri@studium.uni-hamburg.de")[Marius Niveri (TODO)]\
    #datetime.today().display()
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
          #link("mailto: marius.niveri@studium.uni-hamburg.de")[Marius Niveri (TODO)]
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

= Einführung
Der Markt für Chips zum Ausführen von KI- und ML-Programmen wird aktuell stark
von hochparallelisierbaren Prozessoren dominiert. Dazu zählen die "Graphical
Processing Units" (GPU) aber auch andere teilweise noch spezialisiertere
Hardware wie etwa die "Tensor Processing Unit" (TPU).

In dieser Ausarbeitung evaluieren wir den Nutzen und die Wirtschaftlichkeit der GPU im Vergleich zur CPU.
Dafür analysieren wir jeweils die Laufzeit von unterschiedlichen KI- und nicht-KI Algorithmen.
Außerdem klären wir, wodurch die Dominanz von NVIDIA über ihren Konkurrenten AMD zu begründen ist.

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
    // Replace empty cells with 'N/A'
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
  [$80.27$ TPS #footnote[Tokens pro Sekunde]<tps>],
  [$0.21$ TPS #footnote(<tps>)],
  [],

  [Iterativer DFS #footnote[Depth-First-Search in Graf mit $25$ tausend Knoten]<dfs>],
  [$2.85$ Sekunden],
  [$1.28$ Sekunden],
  [],

  [Rekursiver DFS #footnote(<dfs>)],
  [$-$ _unmöglich_ #footnote[Kernel kann keinen Kernel erschaffen.]],
  [$1.19$ Sekunden],
  [],
)
