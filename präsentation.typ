#let title-page(title: [], subtitle: [], faculty, body) = {
  // set page( numbering: "1")
  set page(
    margin: (rest: 1.5in, left: 2.7in),
    paper: "presentation-16-9",
    fill: color.linear-rgb(255, 252, 240),
  )
  set text(font: "EB Garamond", size: 16pt, lang: "de")
  set heading(numbering: "1.1.1", supplement: metadata("Shown"))
  line(start: (0%, 0%), end: (8in, 0%), stroke: (thickness: 2pt))

  align(horizon + left)[
    #text(size: 32pt, title)\
    #v(1em)
    #subtitle
  ]

  align(bottom + left)[
    #link("mailto: marius.niveri@studium.uni-hamburg.de")[Marius Niveri]\
    #datetime.today().display()
    #align(
      right,
      link(faculty)[Universität Hamburg],
    )
  ]
  pagebreak()
  set text(font: "EB Garamond", size: 22pt, lang: "de")
  set page(
    margin: auto,
    footer: context [
      #text(
        size: 9pt,
        grid(
          columns: (1fr, 1fr, 1fr),
          align(
            left,
            [
              #title \
              #link("mailto: marius.niveri@studium.uni-hamburg.de")[Marius Niveri]
            ],
          ),
          align(
            center,
            counter(page).display("1", both: false),
          ),
          align(
            right,
            link(faculty)[Universität Hamburg],
          ),
        ),
      )
    ],
  )
  align(center, text(outline(indent: auto, title: none), size: 16pt))
  pagebreak()
  body
  pagebreak()
  set text(font: "EB Garamond", size: 16pt, lang: "de")
  heading(
    "Literaturverzeichnis",
    numbering: none,
    level: 1,
    supplement: metadata("Hidden"),
  )
  bibliography("bib.yaml", title: none)
}

#show heading.where(level: 1, supplement: metadata("Shown")): it => {
  block(width: 100%)[
    #set align(horizon + center)
    #set text(30pt, weight: "bold")
    #v(32%)
    #text(it.body)
  ]
  pagebreak()
}

#show: body => title-page(
  title: [Hardware-Beschleunigung für ML/AI: GPUs und TPUs],
  subtitle: [
    Seminar Supercomputer: Forschung und Innovation\
    Bei
    #link("mailto: anna.fuchs@uni-hamburg.de")[Anna Fuchs]
    und
    #link("mailto: jannek.squar@uni-hamburg.de")[Jannek Squar]\
  ],
  "https://wr.informatik.uni-hamburg.de/start",
  body,
)

#show cite: it => {
  super(it)
}

= Einführung

== Aktueller Stand
#v(16pt)

- Markt für KI-Chips aktuell stark von hochparallelisierbaren\
  Prozessoren dominiert: @NvTheChipBBC
  - GPUs, TPUs, NPUs, etc.
- Microsoft kaufte 2024 eine halbe Millionen NVIDIA Karten. @MicrAcq500Mio

#v(32pt)

#pagebreak()

== Wirtschaftlichkeit

- TODO: GPUs sind so populär für KI, weil sie wirtschaftlich so viel Sinn ergeben. Sie ergeben
  wirtschaftlich Sinn, da sie deutlich schneller und effizienter als CPUs sind.

#align(
  right,
  [
    #sym.arrow.r.double Warum sind xPUs so gut geeignet für KI- und ML- Anwendungen?
  ],
)

#pagebreak()

= Mathematik hinter dem Training

== Feedforward Neural Network

- TODO: Konkretes und knappes Beispiel, das zeigen soll, warum KI-Berechnungen (Matrixmultiplikationen in diesem Fall) so
  gut parallelisierbar sind.

#pagebreak()

== Transformer und Generatoren

- TODO: Kurze Übersicht über das Papier "Attention Is All You Need", da es den Grundstein für parallelisierbare Transformer gelegt hat.
- ChatGPT z.B. ist ein "Generative [..] Transformer".

#pagebreak()

== Alles Matritzen?

- TODO: Herausfinden, ob es KI- / ML- Modelle gibt, die nicht gut parallelisierbar sind

#pagebreak()

= Matrixmultiplikation auf einer GPU

== Implementierung in Software

- An den Vortrag von Ben anknüpfen und mit Pseudocode zeigen, wie man in Software
  einen Kernel definiert, den man auf einer GPU ausführen kann.
- Es soll die Matrix aus #link((page: 7, x: 0pt, y: 0pt))[dem Beispiel] auf Seite 7
  benutzt werden.

#pagebreak()

== Was passiert in der Hardware?

- Zeigen (vllt Animation?), was bei der Ausführung der vorherigen Software passiert.

#pagebreak()

== Unterschiede GPU #sym.arrow.l.r TPU #sym.arrow.l.r NPU

- Kurze Übersicht über die Unterschiede von der GPU zu anderen "Matrix Processing Units".

#pagebreak()

== Vergleich zur CPU

- Den Bogen zu HPC schließen und begründen, warum GPUs auf Supercomputern nicht fehlen dürfen,
  wenn sie für KI und ML bereit sein sollen.
