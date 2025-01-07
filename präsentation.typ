#let title-page(title: [], subtitle: [], fill: yellow, body) = {
  // set page( numbering: "1")
  set page(
    margin: (rest: 1.5in, left: 2.7in),
    paper: "presentation-16-9",
    fill: color.linear-rgb(255, 252, 240),
  )
  set text(font: "EB Garamond", size: 14pt, lang: "de")
  set heading(numbering: "1.1.1")
  line(start: (0%, 0%), end: (8in, 0%), stroke: (thickness: 2pt))

  align(horizon + left)[
    #text(size: 34pt, title)\
    #v(1em)
    #subtitle
  ]

  align(bottom + left)[
    #link("mailto: marius.niveri@studium.uni-hamburg.de")[Marius Niveri]\
    #datetime.today().display()
    #align(
      right,
      link("https://wr.informatik.uni-hamburg.de/start")[Universität Hamburg],
    )
  ]
  pagebreak()
  set text(font: "EB Garamond", size: 18pt, lang: "de")
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
            link("https://wr.informatik.uni-hamburg.de/start")[Universität Hamburg],
          ),
        ),
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
    #link("mailto: jannek.squar@uni-hamburg.de")[Jannek Squar]\
  ],
  body,
)

= Einführung
Der Markt für Chips zum Ausführen von KI- und ML-Programmen wird aktuell stark
von hochparallelisierbaren Prozessoren dominiert. Dazu zählen die
"Graphical Processing Units" (GPUs) aber auch speziellere xPUs. @NvTheChipBBC

== Aktueller Stand
Eine Der @JenH200Sam

Um die Ursache dieser Entwicklung zu erklären, werden wir uns mit der Ökonomie
von KI- und ML- Hardware beschäftigen. Da die Kosteneffizienz quasi direkt von
der Energieeffizienz und Geschwindigkeit abhängt, werden wir uns im Detail
sowohl mit der Hardwarearchitektur, als auch der softwareseitigen Programmierung
solcher Chips befassen.
