#let title-page(title:[], subtitle:[], fill: yellow, body) = {
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
  set page(fill: none, margin: auto, footer: context [
    #text(size: 8pt, [
      #title\
      #link("mailto: marius.niveri@studium.uni-hamburg.de")[Marius Niveri (TODO)] 
      #h(1fr)
      #counter(page).display(
        "1",
        both: false,
      )
    ])
  ])
  align(horizon, outline(indent: auto))
  pagebreak()
  body
}

#show: body => title-page(
  title: [Hardware-Beschleunigung für ML/AI: GPUs und TPUs],
  subtitle: [
    Seminar Supercomputer: Forschung und Innovation\
    Bei #link("mailto: anna.fuchs@uni-hamburg.de")[Anna Fuchs] und #link("mailto: jannek.squar@uni-hamburg.de")[Jannek Squar]
  ],
  body
)

= Einführung
  Geht man 
